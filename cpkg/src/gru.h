#ifndef GRU_H_
#define GRU_H_

#include "agc.h"
#include "opus_types.h"

// #define WEIGHTS_SCALE (1.f/256)
#define WEIGHTS_SCALE 1.f
#define MAX_NEURONS   256

/**
 * 執行一個 frame 的 GRU + Linear + gain 計算
 *   net       : GRU 執行期狀態 (state 需跨 frame 保留)
 *   estimate  : 輸出, 長度 hop_len, = distorted * gain, 可傳 NULL
 *   gain_out  : 輸出, 長度 hop_len, 可傳 NULL 代表不需要
 *   distorted : 這個 frame 對應的原始時域樣本 (未加窗), 長度 hop_len
 *   x         : 這個 frame 的 STFT magnitude, 長度 n_fft/2+1
 */
void compute_agc(NnagcState *net, float *estimate, float *gain_out,
                  const float *distorted, const float *x);

#endif /* GRU_H_ */
