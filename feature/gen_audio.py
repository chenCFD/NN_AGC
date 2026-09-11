import soundfile as sf
import os
import librosa
import numpy as np
from scipy import signal
import random
from scipy.signal import butter, lfilter
from c_lib import Functions
from tqdm import tqdm
import multiprocessing as mp
from scipy.ndimage import gaussian_filter1d

def rms_normalize(audio, target_db=-30):
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    return audio * gain

def highpass_filter(data, sr, cutoff_freq, order=5):
    nyquist = 0.5 * sr  # 奈奎斯特頻率
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

os.makedirs("dataset/check/valid/original/", exist_ok=True)
os.makedirs("dataset/check/train/original/", exist_ok=True)
os.makedirs("dataset/valid/output/", exist_ok=True)
os.makedirs("dataset/train/output/", exist_ok=True)

os.makedirs("dataset/check/valid/distorted/", exist_ok=True)
os.makedirs("dataset/check/train/distorted/", exist_ok=True)
os.makedirs("dataset/valid/input_time/", exist_ok=True)
os.makedirs("dataset/valid/input_freq/", exist_ok=True)
os.makedirs("dataset/train/input_time/", exist_ok=True)
os.makedirs("dataset/train/input_freq/", exist_ok=True)


samples = 20
target_samplerate = 24000
frame_size = 480
hop_length = 480
output_duration =4

speech_path = "dataset/speech/"
speech_files = [f for f in os.listdir(speech_path) if f.endswith(".wav")]

rir_path = "dataset/rir/"
rir_files = [f for f in os.listdir(rir_path) if f.endswith(".wav")]

noise_path = "dataset/noise/"
noise_files = [f for f in os.listdir(noise_path) if f.endswith(".wav")]

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
save_length = 200
total_time_seg = 200
valid_ratio = 0.1

def load_data(i,param):
    print(i)
    speech_wav, sr = sf.read(speech_path + speech_files[i * (len(speech_files)//samples)])
    speech_wav = speech_wav[:output_duration * sr]
    speech_wav = librosa.resample(speech_wav, orig_sr=sr, target_sr=target_samplerate)
    speech_wav = speech_wav[:output_duration * target_samplerate]
    # print(speech_wav.shape)
    # sf.write('original_speech.wav', speech_wav, target_samplerate)
    
    # 計算每幀的 RMS 能量
    rms = librosa.feature.rms(y=speech_wav, frame_length=frame_size, hop_length=hop_length)[0]

    # 簡單的 VAD 判斷：能量大於閾值則視為有聲音
    vad = rms > 0.01
    
    vad_indices = np.where(vad.astype(int) == 1)[0]
    
    no_sound_flag = 0
    
    if len(vad_indices) > 0:
        index_vad = vad_indices[0]
    else:
        no_sound_flag = 1
    
    # 找出連續 VAD=1 的區段
    segments = []
    start = None
    for j, is_voiced in enumerate(vad):
        if is_voiced and start is None:
            start = j
        elif not is_voiced and start is not None:
            segments.append((start, j))
            start = None
    
    # 若最後一段結束時仍在有聲音的狀態
    if start is not None:
        segments.append((start, len(vad)))
    
    # 設定目標 RMS
    target_db = -28
    target_rms = 10 ** (target_db / 20)  # 將 dB 轉為 RMS 值
    
    # 對每個區段進行 RMS 正規化
    for start, end in segments:
        # 計算該區段的取樣點範圍
        start_sample = start * hop_length
        end_sample = min(len(speech_wav), end * hop_length + frame_size)
        segment = speech_wav[start_sample:end_sample]
        
        # 計算區段 RMS
        current_rms = np.sqrt(np.mean(segment ** 2))
        
        # 避免除以零
        if current_rms > 0:
            # 計算正規化係數
            gain = target_rms / current_rms
            speech_wav[start_sample:end_sample] *= gain
        
    # debug original speech -> ok
    # sf.write('original_speech_normalized.wav', speech_wav, target_samplerate)
    
    # ratio of only speech, speech + rir, speech + rir + noise
    seed = random.randint(1,10)
    
    # pure speech or add rir
    if seed > 3:
        rir, sr_rir = sf.read(rir_path + rir_files[i * (len(rir_files)//samples)])
        rir = rir[:int(sr_rir * 0.05)]
        rir = librosa.resample(rir, orig_sr=sr_rir, target_sr=target_samplerate)
        rir = rir / np.sqrt(np.sum(rir**2))
        convolved_speech = signal.fftconvolve(speech_wav, rir, mode='full')[:len(speech_wav)]
        
        # do high pass filter
        convolved_speech = highpass_filter(convolved_speech, target_samplerate, 20, 4)
        
        # print(convolved_speech.shape)
        
        # # debug convolved speech
        # # sf.write('convolved_speech.wav', convolved_speech, target_samplerate)
        
        normalized_speech = rms_normalize(convolved_speech, -28)
        
        # print(normalized_speech.shape)
        
        # # debug normalize -> ok
        # sf.write('speech_normalize_debug.wav', normalized_speech, target_samplerate)
    else:
        target_speech = speech_wav
    
    
    # speech+rir or speech+rir+noise
    if seed > 4:
        noise, noise_sr = sf.read(noise_path + noise_files[(i % len(noise_files))])
        noise = noise[:output_duration * noise_sr]
        # print("********",len(noise.shape))
        if len(noise.shape) > 1:
            noise = noise[:,0]
        noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=target_samplerate)
        noise_db = random.uniform(-28, -48)
        noise = rms_normalize(noise, noise_db)
        
        # print(noise.shape)
        while noise.shape[0] < target_samplerate * output_duration:
            noise = np.concatenate([noise, noise])
        
        noise = noise[:target_samplerate * output_duration]

        
        target_speech = normalized_speech + noise
    elif seed == 4:
        target_speech = normalized_speech
    
    # # debug normalize -> ok
    # sf.write('original_test.wav', target_speech, target_samplerate)
    
    
    # # finish target
    if i < samples * valid_ratio:    
        sf.write('dataset/check/valid/original/original_'+str(i)+'.wav', target_speech, target_samplerate)
    else:
        sf.write('dataset/check/train/original/original_'+str(i)+'.wav', target_speech, target_samplerate)
    # save target [500, 480]
    output = target_speech.copy()
    output = (output*32768).astype(np.int16)
    output = output.reshape(total_time_seg, frame_size)
    
    if i < samples * valid_ratio:
        for out_index in range(int(total_time_seg/save_length)):
            np.save('dataset/valid/output/output_'+str(i)+'_'+str(out_index), output[out_index *save_length : (out_index +1)*save_length])
    else:
        for out_index in range(int(total_time_seg/save_length)):
            np.save('dataset/train/output/output_'+str(i)+'_'+str(out_index), output[out_index *save_length : (out_index +1)*save_length])
    
    
    # 定義兩個區間
    # take both range
    select_first_range = random.randint(1, 2)
    if select_first_range == 1:
        range1 = [-32, -1]
        range2 = [5, 8]
    else:
        range2 = [-32, -1]
        range1 = [5, 8]
    
    # 在選到的區間內隨機取值
    gain_db_1 = random.uniform(range1[0], range1[1])
    gain_db_2 = random.uniform(range2[0], range2[1])

    
    # gain_db = random.uniform(-30, 10)
    gain1 = 10 ** (gain_db_1 / 20)
    gain2 = 10 ** (gain_db_2 / 20)
    # print(gain1)
    
    type_select = random.randint(1, 2)
    print(type_select)
    if type_select == 1:
        # apply all
        target_speech *= gain1
    else:
        # select segment
        total_block = len(speech_wav) // frame_size
    
        gain_array = np.ones(total_block)
        
        if no_sound_flag == 1:
            select_block = random.randint(1, 3)
        if no_sound_flag == 0:
            select_block = random.randint(1, 4)
        
        if select_block == 1:
            start_point = random.randint(int(total_block * 0.2), int(total_block * 0.4))
            duration_point = random.randint(int(total_block * 0.2), int(total_block * 0.5))
        
        elif select_block == 2:
            start_point = 0
            duration_point = random.randint(int(total_block * 0.2), int(total_block * 0.6))
        
        elif select_block == 3:
            duration_point = random.randint(int(total_block * 0.2), int(total_block * 0.6))
        
        else:
            print('select_block = start from vad')
            if i % 2 == 0:  
                start_point = index_vad
            else:
                start_point = index_vad - 1
            duration_point = random.randint(int(total_block * 0.2), int(total_block * 0.5))
        
        # print(start_point)
        # print(duration_point)
        if select_block == 3:
            gain_array[:] = gain2
            gain_array[int(total_block) - duration_point : int(total_block)] = gain1
            # transition
            # gain_array = gaussian_filter1d(gain_array, sigma=1)
            
        else:
            gain_array[:] = gain2
            gain_array[start_point : start_point + duration_point] = gain1
            # transition
            # gain_array = gaussian_filter1d(gain_array, sigma=1)

        for t in range(len(gain_array)):
            target_speech[frame_size*t : frame_size * (t+1)] *= gain_array[t]
    
    
    if i < samples * valid_ratio:
        sf.write('dataset/check/valid/distorted/distorted_'+str(i)+'.wav', target_speech, target_samplerate)
    else:
        sf.write('dataset/check/train/distorted/distorted_'+str(i)+'.wav', target_speech, target_samplerate)
        
    
    mic = target_speech.copy()
    mic = (mic*32768).astype(np.int16)
    
    print(mic[0:5])
    
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
    
    # print(x_array.shape)
    print(x_array[0,0:5])
    # print(feature_array.shape)
    print(feature_array[0,0:5])
    print(x_array.shape)
    print(feature_array.shape)
    
    
    if i < samples * valid_ratio:
    # save x, feature npy [500, 480], [500, 481]
        for in_index in range(int(total_time_seg/save_length)):
            np.save('dataset/valid/input_time/input_time_'+str(i)+'_'+str(in_index), x_array[in_index *save_length : (in_index +1)*save_length])
            np.save('dataset/valid/input_freq/input_freq_'+str(i)+'_'+str(in_index), feature_array[in_index *save_length : (in_index +1)*save_length])
    else:
    # save x, feature npy [500, 480], [500, 481]
        for in_index in range(int(total_time_seg/save_length)):
            np.save('dataset/train/input_time/input_time_'+str(i)+'_'+str(in_index), x_array[in_index *save_length : (in_index +1)*save_length])
            np.save('dataset/train/input_freq/input_freq_'+str(i)+'_'+str(in_index), feature_array[in_index *save_length : (in_index +1)*save_length])
    
    
    print('===finish===')

# Parameters for RIR generation
class Params():
    def __init__(self, i):
        self.rand_speaker=random.randint(0, 20)

if __name__ == '__main__':
    total_num=20
    
    iterator = []
    for i in tqdm(range(total_num)):
        iterator.append([i, Params(i)])
    
    
    print("Start agc data generation...")
    nproc = mp.cpu_count() - 4
    pool = mp.Pool(processes=nproc)
    pool.starmap(load_data, iterator)
    
    pool.close()
    pool.join()
    
    print("agc data generation finished")

