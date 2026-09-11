import os
import glob
import random
import shutil

def copy_random_wavs(src_folder, dst_folder, sample_size=20000):
    # 確保目標資料夾存在
    os.makedirs(dst_folder, exist_ok=True)

    # 讀取所有 wav 檔案
    wav_files = glob.glob(os.path.join(src_folder, "*.wav"))

    # 檢查檔案數量是否足夠
    if len(wav_files) < sample_size:
        raise ValueError(f"來源資料夾只有 {len(wav_files)} 個檔案，不足以抽取 {sample_size} 個")

    # 隨機抽樣
    selected_files = random.sample(wav_files, sample_size)

    # 複製檔案到目標資料夾
    for file in selected_files:
        shutil.copy(file, dst_folder)

    print(f"已成功複製 {sample_size} 個 wav 檔案到 {dst_folder}")

# 使用範例
src_folder = "/mnt/d/audio/DNS-Challenge/datasets_fullband/datasets_fullband/clean_fullband/read_speech"   # 請替換成來源資料夾路徑
dst_folder = "/mnt/d/audio/Neural-AGC/feature/dataset/speech"  # 請替換成目標資料夾路徑
copy_random_wavs(src_folder, dst_folder, 20000)
