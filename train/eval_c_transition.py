from model import AGC_STFT_GRU
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

import time
import soundfile as sf
import numpy as np
from c_lib import Functions

model = AGC_STFT_GRU(960, 10, 960, 480)   
package = torch.load('ckpts_ver18/h40/best.pth.tar')
model.load_state_dict(package['state_dict'])

model.cpu()
model.eval()

# initial c feature
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


frame_time_max = 0

start = time.time()

db = 60

mic, au_fs = sf.read('test/54-24-f.wav', dtype='int16')
assert au_fs == fs, "Input audio must be 24kHz"

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

feature_array = torch.from_numpy(feature_array)
x_array = torch.from_numpy(x_array)

feature_array = feature_array.unsqueeze(0)
x_array = x_array.unsqueeze(0)

print(x_array.shape)
print(feature_array.shape)

est, gain = model(x_array, feature_array)

est_numpy = est[0].cpu().detach().numpy()/32768

sf.write('test/est_54-24-f.wav', est_numpy, 24000)