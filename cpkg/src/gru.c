#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "tansig_table.h"
#include "gru.h"

#ifndef M_LN2
#define M_LN2 0.69314718055994530942f
#endif

static OPUS_INLINE float tansig_approx(float x)
{
    int i;
    float y, dy;
    float sign=1;
    /* Tests are reversed to catch NaNs */
    if(isnan(x))
    {
       printf("tansig_approx: nan detected\n");
    }

    if (!(x<8))
        return 1;
    if (!(x>-8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
       return 0;
#endif
    if (x<0)
    {
       x=-x;
       sign=-1;
    }
    i = (int)floor(.5f+25*x);
    x -= .04f*i;
    y = tansig_table[i];
    dy = 1-y*y;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}

static OPUS_INLINE float sigmoid_approx(float x)
{
   return .5 + .5*tansig_approx(.5*x);
}

static OPUS_INLINE float relu(float x)
{
   return (x < 0) ? 0 : x;
}


static void compute_linear(const LinearLayer *layer, float *output, const float *input)
{
   int i, j;
   int N, M;
   M = layer->nb_inputs;
   N = layer->nb_neurons;

   for (i=0;i<N;i++)
   {
      float sum=layer->bias[i];
      for (j=0;j<M;j++)
         sum += layer->input_weights[i*M + j]*input[j];
      output[i] = WEIGHTS_SCALE*sum;
   }
}


static void compute_gru(const GRULayer *gru, float *state, const float *input)
{
   int i, j;
   int N, M;
   int stride_in, stride_h;
   float r[MAX_NEURONS];
   float z[MAX_NEURONS];
   float h[MAX_NEURONS];
   celt_assert(gru->nb_neurons<=MAX_NEURONS);
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   stride_in = N*M;
   stride_h = N*N;

   /* Compute reset gate. */
   for (i=0;i<N;i++) {
      float sum = gru->bias[i]+gru->bias[3*N+i];
      for (j=0;j<M;j++)
         sum += gru->input_weights[i*M + j]*input[j];
      for (j=0;j<N;j++)
         sum += gru->recurrent_weights[i*N + j]*state[j];
      r[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
   }

   /* Compute update gate. */
   for (i=0;i<N;i++) {
      float sum = gru->bias[N+i] + gru->bias[4*N+i];
      for (j=0;j<M;j++)
         sum += gru->input_weights[stride_in + i*M + j]*input[j];
      for (j=0;j<N;j++)
         sum += gru->recurrent_weights[stride_h + i*N + j]*state[j];
      z[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
   }

   /* Compute new gate. */
   for (i=0;i<N;i++){
      float sum = gru->bias[2*N+i];
      for (j=0;j<M;j++)
         sum += gru->input_weights[2*stride_in + i*M + j]*input[j];
      float sum_r = gru->bias[5*N+i];
      for (j=0;j<N;j++)
         sum_r += gru->recurrent_weights[2*stride_h + i*N + j]*state[j];
      sum += (r[i]*sum_r);
      sum = tansig_approx(WEIGHTS_SCALE*sum);
      h[i] = (1-z[i])*sum + z[i]*state[i];
      state[i] = h[i];
   }
}


/* 對應 Python AGC_STFT_GRU.forward():
 *   x, hidden = self.gru(x)
 *   x = self.relu(x)
 *   x = self.lin(x)
 *   x = x[:, :, 0]
 *   x = self.upsample(x)              # nearest, scale_factor = hop_len
 *   gain = torch.exp(gain * ln(2))    # gain 為 log2 domain -> linear
 *   estimate = distorted * gain
 */
void compute_agc(NnagcState *net, float *estimate, float *gain_out,
                  const float *distorted, const float *x)
{
   int i;
   const NnagcModel *model = net->model;
   const int hop_len = model->hop_len;
   float relu_state[MAX_NEURONS];
   float lin_out[1];
   float g;

   celt_assert(model->gru->nb_neurons <= MAX_NEURONS);

   /* GRU: 用這個 frame 的 STFT magnitude (x) 更新 hidden state */
   compute_gru(model->gru, net->gru_state, x);

   /* x = relu(x)：對 GRU 輸出/更新後的 hidden state 做 ReLU */
   for (i=0; i<model->gru->nb_neurons; i++)
      relu_state[i] = relu(net->gru_state[i]);

   /* x = self.lin(x)  hidden_size -> 1 */
   compute_linear(model->lin, lin_out, relu_state);

   /* Upsample(nearest, scale_factor=hop_len)：同一個 gain 值
    * 複製到這個 frame 對應的 hop_len 個時域樣本上；
    * gain = exp(x * ln2)  (log2 domain -> linear domain) */
   g = expf(lin_out[0] * (float)M_LN2);

   for (i=0; i<hop_len; i++) {
      if (gain_out)
         gain_out[i] = g;
      if (estimate)
         estimate[i] = distorted[i] * g;
   }

   net->frame_cnt++;
}
