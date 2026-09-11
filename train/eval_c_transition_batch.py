import os
import time
import torch
import numpy as np
import soundfile as sf
from model import AGC_STFT_GRU
from c_lib import Functions

# ================= 1. 路徑與參數設定 =================
INPUT_DIR = "/mnt/d/audio/Neural-AGC/nn-agc/test/final_test/selected"           # 輸入 WAV 檔的資料夾
OUTPUT_DIR = "/mnt/d/audio/Neural-AGC/nn-agc/test/final_test/selected_nnagc_ver18" # 輸出結果的資料夾
CKPT_PATH = 'ckpts/h40/best.pth.tar'
LIB_PATH = "libnnagc.so.1"

fs = 24000
n_mic = 1
frame_size = int(0.02 * fs)  # 480
freq_size = frame_size + 1   # 481

# 確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. 載入模型 =================
model = AGC_STFT_GRU(960, 10, 960, 480)
package = torch.load(CKPT_PATH)
model.load_state_dict(package['state_dict'])

model.cpu()
model.eval()

# ================= 3. 搜尋資料夾內的 WAV 檔 =================
wav_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith('.wav')]

if not wav_files:
    print(f"在 '{INPUT_DIR}' 資料夾中沒有找到任何 WAV 檔案。")
    exit()

print(f"找到 {len(wav_files)} 個 WAV 檔，開始執行批量推理...\n")
start_time = time.time()

# ================= 4. 逐一處理音檔 =================
for idx, file_name in enumerate(wav_files, 1):
    file_path = os.path.join(INPUT_DIR, file_name)
    base_name = os.path.splitext(file_name)[0]
    
    print(f"[{idx}/{len(wav_files)}] 處理中：{file_name}")

    # 4.1 讀取音訊
    mic, au_fs = sf.read(file_path, dtype='int16')
    if au_fs != fs:
        print(f"  ⚠️ 警告：{file_name} 的取樣率為 {au_fs}Hz（非 {fs}Hz），已跳過。")
        continue

    # 每處理一個新檔案即重新初始化 C 函式庫，確保 C 端狀態不會跨檔殘留
    cf = Functions(
        LIB_PATH,
        n_mic=n_mic,
        frame_len=frame_size,
        freq_len=freq_size,
        fs=fs
    )

    # 4.2 特徵提取
    feature_list = []
    x_list = []
    N_frame = len(mic) // cf.frame_size

    for nf in range(N_frame):
        mic_in = mic[nf * cf.frame_size : (nf + 1) * cf.frame_size]
        x_list.append(mic_in.copy())

        cf.feature_generation_training(mic_in)
        tmp = cf.feature.copy()
        feature_list.append(tmp)

    if len(feature_list) == 0:
        print(f"  ⚠️ 警告：{file_name} 音訊長度不足一幀，已跳過。")
        continue

    # 4.3 轉為 Tensor 並進行模型推理
    feature_array = torch.from_numpy(np.array(feature_list)).unsqueeze(0)
    x_array = torch.from_numpy(np.array(x_list)).unsqueeze(0)

    with torch.no_grad():
        est, gain = model(x_array, feature_array)

    # 4.4 儲存模型推導出的預估音訊 (.wav)
    est_numpy = est[0].cpu().detach().numpy() / 32768.0
    est_output_path = os.path.join(OUTPUT_DIR, f"est_{base_name}.wav")
    sf.write(est_output_path, est_numpy, fs)

print(f"\n所有音檔處理完成！總耗時：{time.time() - start_time:.2f} 秒。")