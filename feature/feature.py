import os, sys
import multiprocessing as mp
from datetime import datetime
import numpy as np
from os.path import join
from soundfile import read
from c_lib import Functions


fs = 24000
n_mic = 1
frame_size = int(0.02*fs)
freq_size = frame_size + 1
lib = "libnnagc.so.1"
cf = Functions(lib, 
               n_mic=n_mic,
               frame_len=frame_size,
               freq_len = freq_size,
               fs=fs)


mic, au_fs = read('distorted_test.wav', dtype='int16')
assert au_fs==fs, "Input audio must be 24kHz"

feature_list = []
x_list = []  # 用於儲存每一幀的 x

N_frame = len(mic) // cf.frame_size

# 每 480 個取樣點做一次特徵提取
for nf in range(N_frame):
    mic_in = mic[nf * cf.frame_size : (nf + 1) * cf.frame_size]
    
    # 儲存每一幀的 x
    x_list.append(mic_in.copy())
    
    # 呼叫 C 函數生成 feature
    cf.feature_generation_training(mic_in)
    tmp = cf.feature.copy()
    feature_list.append(tmp)

# 將所有特徵與 x 組合並輸出
feature_array = np.array(feature_list)
x_array = np.array(x_list)

np.save("feature_array.npy", feature_array)
np.save("x_array.npy", x_array)


print(feature_array.shape)
print(x_array.shape)

    
