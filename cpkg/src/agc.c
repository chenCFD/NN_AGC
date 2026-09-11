#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "agc.h"
#include "arch.h"
#include "gru.h"

/* 內建的模型，定義在 dump_nnagc.py 產生的 nnagc_data_xxx.c 裡 */
extern const NnagcModel nnagc_model;

typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[FRAME_SIZE_AGC];
} CommonState;

static CommonState common;

static void check_init() {
  int i;

  if (common.init) return;
  common.kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE_AGC, NULL, NULL, NULL, 0);
  for (i=0;i<FRAME_SIZE_AGC;i++)
    common.half_window[i] = sin(.5*M_PI*SQUARE(sin(.5*M_PI*(i+.5)/FRAME_SIZE_AGC)));
  common.init = 1;
}

static void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE_AGC];
  kiss_fft_cpx y[WINDOW_SIZE_AGC];
  check_init();
  for (i=0;i<WINDOW_SIZE_AGC;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(common.kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE_AGC;i++) out[i] = y[i];
}

static void apply_window(float *x) {
  int i;
  check_init();
  for (i=0;i<FRAME_SIZE_AGC;i++) {
    x[i] *= common.half_window[i];
    x[WINDOW_SIZE_AGC - 1 - i] *= common.half_window[i];
  }
}

/* 50% overlap STFT，輸出 magnitude spectrum (對應 AGC_STFT_GRU 的輸入 x) */
static void frame_analysis(float *analysis_mem, float *feature, const float *in) {
  int i, f;
  float x[WINDOW_SIZE_AGC];
  kiss_fft_cpx X[FREQ_SIZE_AGC];

  RNN_COPY(x, analysis_mem, FRAME_SIZE_AGC);
  for (i=0;i<FRAME_SIZE_AGC;i++) x[FRAME_SIZE_AGC + i] = in[i];
  RNN_COPY(analysis_mem, in, FRAME_SIZE_AGC);

  apply_window(x);
  forward_transform(X, x);

  for (f=0; f<FREQ_SIZE_AGC; f++)
    feature[f] = sqrtf(X[f].r*X[f].r + X[f].i*X[f].i);
}


int agc_init(AgcState *st) {
  memset(st, 0, sizeof(*st));
  st->net.model = &nnagc_model;
  st->net.gru_state = calloc(sizeof(float), MAX_NEURONS);
  return 0;
}

AgcState *agc_create() {
  AgcState *st;
  st = (AgcState*)malloc(sizeof(AgcState));
  agc_init(st);
  return st;
}

void agc_destroy(AgcState *st) {
  free(st->net.gru_state);
  free(st);
}

/* 對應 Python: cf.feature_generation_training(mic_in)
 * 訓練資料產生階段用這個函式；跟 agc_process_frame() 不要混用同一顆 state。 */
void agc_feature_generation(AgcState *st, const short *in, float *feature) {
  int i;
  float xf[FRAME_SIZE_AGC];

  for (i=0;i<FRAME_SIZE_AGC;i++)
    xf[i] = (float) in[i];

  frame_analysis(st->analysis_mem, st->feature, xf);

  if (feature)
    RNN_COPY(feature, st->feature, FREQ_SIZE_AGC);

  st->frame_cnt++;
}

/* 對應 Python: est, gain = model(x_array, feature_array)
 * 即時串流版：每次處理一個 hop_len 長度的 frame */
void agc_process_frame(AgcState *st, short *out, const short *in) {
  int i;
  float distorted[FRAME_SIZE_AGC];
  float estimate[FRAME_SIZE_AGC];

  for (i=0;i<FRAME_SIZE_AGC;i++)
    distorted[i] = (float) in[i];

  /* 1. STFT magnitude feature (加窗, 50% overlap) -> st->feature */
  frame_analysis(st->analysis_mem, st->feature, distorted);

  /* 2. GRU + Linear + gain，並乘上「未加窗」的原始樣本 */
  compute_agc(&st->net, estimate, st->gain, distorted, st->feature);

  /* 3. 輸出轉回 int16 (四捨五入 + clip) */
  if (out) {
    for (i=0;i<FRAME_SIZE_AGC;i++) {
      float v = estimate[i];
      if (v > 32767.f) v = 32767.f;
      if (v < -32768.f) v = -32768.f;
      out[i] = (short) lrintf(v);
    }
  }

  st->frame_cnt++;
}
