#ifndef AGC_H
#define AGC_H 1

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef AISSL_EXPORT
# if defined(WIN32)
#  if defined(AISSL_BUILD) && defined(DLL_EXPORT)
#   define AISSL_EXPORT __declspec(dllexport)
#  else
#   define AISSL_EXPORT
#  endif
# elif defined(__GNUC__) && defined(AISSL_BUILD)
#  define AISSL_EXPORT __attribute__ ((visibility ("default")))
# else
#  define AISSL_EXPORT
# endif
#endif

#define SQUARE(x) ((x)*(x))
#define EPS 1e-8

#ifndef M_PI
#define M_PI 3.141592653589793238
#endif

#ifndef M_E
#define M_E 2.718281828459045
#endif

/* ------------------------------------------------------------
 * 對應 AGC_STFT_GRU(n_fft=960, hidden_size=10, win_len=960, hop_len=480)
 * 及 eval_c_transition_tmp.py 裡 fs=24000, frame_size=int(0.02*fs)=480。
 * 這組數字要跟訓練時 model.py / dump_nnagc.py 用的完全一致，
 * 修改前請再三確認。
 * ------------------------------------------------------------ */
#define SAMPLE_RATE_AGC   24000
#define FRAME_SIZE_AGC    480                  /* hop_len：每次處理的時域樣本數 */
#define WINDOW_SIZE_AGC   (2*FRAME_SIZE_AGC)   /* win_len / n_fft = 960 */
#define FREQ_SIZE_AGC     (FRAME_SIZE_AGC+1)   /* n_fft/2 + 1 = 481 */

typedef float cnn_weight;
// typedef int8_t cnn_weight;
// typedef int16_t cnn_weight;

/* ------------------------------------------------------------
 * 權重 struct，欄位順序需與 dump_nnagc.py 產生的 struct literal 一致：
 *   GRULayer   : { bias, input_weights, recurrent_weights, nb_inputs, nb_neurons }
 *   LinearLayer: { bias, input_weights, nb_inputs, nb_neurons }
 * ------------------------------------------------------------ */
typedef struct {
  const cnn_weight *bias;
  const cnn_weight *input_weights;
  int nb_inputs;
  int nb_neurons;
} LinearLayer;

typedef struct {
  const cnn_weight *bias;
  const cnn_weight *input_weights;
  const cnn_weight *recurrent_weights;
  int nb_inputs;
  int nb_neurons;
} GRULayer;

/* 對應 dump_nnagc.py 產生的:
 *   const NnagcModel nnagc_model = { &gru, &lin, n_fft, hidden_size, win_len, hop_len };
 */
typedef struct {
  const GRULayer    *gru;
  const LinearLayer *lin;
  int n_fft;
  int hidden_size;
  int win_len;
  int hop_len;
} NnagcModel;

/* GRU 執行期狀態 */
typedef struct {
  const NnagcModel *model;
  float *gru_state;   /* calloc(MAX_NEURONS), 見 gru.h */
  int frame_cnt;
} NnagcState;

/* 對外的完整狀態 */
typedef struct {
  int frame_cnt;
  float analysis_mem[FRAME_SIZE_AGC];  /* STFT 50% overlap 緩衝 */
  float feature[FREQ_SIZE_AGC];        /* 這個 frame 的 STFT magnitude */
  float gain[FRAME_SIZE_AGC];          /* 這個 frame 的 gain (log2->linear 後) */

  NnagcState net;
} AgcState;

/**
 * Initializes a pre-allocated AgcState
 * (使用內建的 nnagc_model, 定義在 nnagc_data_xxx.c)
 */
AISSL_EXPORT int agc_init(AgcState *st);

/**
 * Allocate and initialize an AgcState
 * 回傳的 pointer 必須用 agc_destroy() 釋放
 */
AISSL_EXPORT AgcState *agc_create();

/**
 * Free an AgcState produced by agc_create()
 */
AISSL_EXPORT void agc_destroy(AgcState *st);

/**
 * 只做 STFT 特徵抽取，等同舊版 agc_0.h 的 agc_feature_generation()。
 * 這是 c_lib.py（訓練資料產生）透過 ctypes 呼叫的入口，簽名維持不變：
 *   void agc_feature_generation(void *st, const short *x, float *feature)
 *
 * in      : 長度 FRAME_SIZE_AGC 的原始時域樣本 (未加窗)
 * feature : 輸出, 長度 FREQ_SIZE_AGC 的 STFT magnitude, 可傳 NULL 代表只更新內部 state
 *
 * 注意：同一顆 AgcState 不要交替呼叫 agc_feature_generation() 和
 * agc_process_frame()，兩者都會更新 st->analysis_mem 的 overlap 狀態，
 * 混用會讓 STFT 對齊錯亂。
 */
AISSL_EXPORT void agc_feature_generation(AgcState *st, const short *in, float *feature);

/**
 * 處理一個 frame: STFT -> GRU -> Linear -> gain -> 輸出音訊
 * in  : 長度 FRAME_SIZE_AGC 的原始時域樣本
 * out : 輸出, 長度 FRAME_SIZE_AGC 的估測音訊 (= in * gain), 可傳 NULL 代表不需要輸出音訊
 */
AISSL_EXPORT void agc_process_frame(AgcState *st, short *out, const short *in);

#ifdef __cplusplus
}
#endif

#endif /* AGC_H */
