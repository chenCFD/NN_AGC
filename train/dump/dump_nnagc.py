import os
import sys
import csv
import math
import torch
import numpy as np
from torch import nn
from torch.nn import GRU, Linear

# ============================================================
#  dump_nnagc.py
#  將 AGC_STFT_GRU (nnagc) 的權重 checkpoint (best.pth.tar)
#  同時 dump 成三種版本的 C 語言靜態陣列：
#     nnagc_data_float.c  (原始 float，不量化)
#     nnagc_data_int16.c  (int16 量化)
#     nnagc_data_int8.c   (int8  量化)
#  並把每個 tensor 的統計數據 / scale / 精度輸出成 quant_stats.csv
# ============================================================

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'best.pth.tar'
out_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
csv_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(out_dir, 'quant_stats.csv')

# 量化模式定義: bits=None 代表不量化 (float)
QUANT_MODES = {
    'float': {'bits': None, 'c_type': 'float'},
    'int16': {'bits': 16, 'c_type': 'int16_t'},
    'int8': {'bits': 8, 'c_type': 'int8_t'},
}


# ------------------------------------------------------------
#  模型定義 (需與訓練時一致，若已另存成檔案可改成 import)
# ------------------------------------------------------------
class AGC_STFT_GRU(nn.Module):

    def __init__(self, n_fft, hidden_size, win_len, hop_len):
        super(AGC_STFT_GRU, self).__init__()
        self.n_fft = n_fft
        input_size = self.n_fft // 2 + 1
        self.hidden_size = hidden_size
        self.win_len = win_len
        self.hop_len = hop_len
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.lin = nn.Linear(hidden_size, 1)
        self.upsample = nn.Upsample(scale_factor=hop_len, mode='nearest')
        self.window = torch.hann_window(win_len, periodic=True)

    def forward(self, distorted, x):
        distorted = distorted.view(distorted.shape[0], distorted.shape[1] * distorted.shape[2])
        x, hidden = self.gru(x)
        x = self.relu(x)
        x = self.lin(x)
        x = x[:, :, 0]
        x = x.unsqueeze(1)
        x = self.upsample(x)
        gain = x.squeeze(1)
        gain = gain[:, :distorted.shape[1]]
        gain = torch.exp(gain * math.log(2.))
        estimate = distorted * gain
        return estimate, gain

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['n_fft'], package['hidden_size'], package['win_len'], package['hop_len'])
        model.load_state_dict(package['state_dict'])
        return model


# ------------------------------------------------------------
#  用來收集所有 tensor 的統計數據，最後寫成 CSV
# ------------------------------------------------------------
stats_rows = []


def tensor_stats(vector):
    v = vector.detach().numpy().reshape(-1)
    absmax = float(np.abs(v).max()) if v.size else 0.0
    return {
        'num_params': int(v.size),
        'min': float(v.min()) if v.size else 0.0,
        'max': float(v.max()) if v.size else 0.0,
        'absmax': absmax,
        'mean': float(v.mean()) if v.size else 0.0,
        'std': float(v.std()) if v.size else 0.0,
    }


def compute_scale(absmax, bits):
    """回傳 (scale, resolution)。bits=None 代表不量化。"""
    if bits is None:
        return 1.0, None
    qmax = 2 ** (bits - 1) - 1  # int8 -> 127, int16 -> 32767
    if absmax == 0:
        return 1.0, 0.0
    scale = qmax / absmax
    resolution = absmax / qmax  # 每一格代表的浮點區間 = 最大量化誤差的上限
    return scale, resolution


# ------------------------------------------------------------
#  印出一個 flatten 過的權重陣列 (依 mode 決定 float / int16 / int8)
# ------------------------------------------------------------
def printVector(f, vector, name, mode, layer_name, tensor_role):
    v = np.reshape(vector.detach().numpy(), (-1))
    bits = QUANT_MODES[mode]['bits']
    c_type = QUANT_MODES[mode]['c_type']
    st = tensor_stats(vector)
    scale, resolution = compute_scale(st['absmax'], bits)

    if bits is None:
        q = v
    else:
        qmax = 2 ** (bits - 1) - 1
        qmin = -2 ** (bits - 1)
        q = np.clip(np.round(v * scale), qmin, qmax).astype(np.int64)

    stats_rows.append({
        'layer': layer_name,
        'tensor': tensor_role,
        'mode': mode,
        'num_params': st['num_params'],
        'min': st['min'],
        'max': st['max'],
        'absmax': st['absmax'],
        'mean': st['mean'],
        'std': st['std'],
        'scale': scale,
        'resolution': resolution if resolution is not None else '',
        'max_quant_error': (resolution / 2) if resolution is not None else '',
    })

    f.write('/* absmax={:.6f}  scale={:.6f}{} */\n'.format(
        st['absmax'], scale,
        '  resolution={:.6e}'.format(resolution) if resolution is not None else ''))
    f.write('static const {} {}[{}] = {{\n   '.format(c_type, name, len(v)))
    for i in range(0, len(v)):
        if bits is None:
            f.write('{}'.format(v[i]))
        else:
            f.write('{}'.format(int(q[i])))
        if i != len(v) - 1:
            f.write(',')
        else:
            break
        if i % 8 == 7:
            f.write("\n   ")
        else:
            f.write(" ")
    f.write('\n};\n\n')


# ------------------------------------------------------------
#  GRU layer dump
# ------------------------------------------------------------
def dump_gru_module(self, f, name, mode):
    print("printing layer {} ({})".format(name, mode))
    input_weights = self.weight_ih_l0.data.clone()
    recurrent_weights = self.weight_hh_l0.data.clone()
    bias = torch.cat((self.bias_ih_l0, self.bias_hh_l0))
    printVector(f, input_weights, name + '_weights', mode, name, 'weight_ih')
    printVector(f, recurrent_weights, name + '_recurrent_weights', mode, name, 'weight_hh')
    printVector(f, bias, name + '_bias', mode, name, 'bias')
    f.write('const GRULayer {} = {{\n  {}_bias,\n  {}_weights,\n  {}_recurrent_weights,\n  {}, {}\n}};\n\n'
             .format(name, name, name, name,
                     input_weights.shape[1],
                     input_weights.shape[0] // 3))


GRU.dump_data = dump_gru_module


# ------------------------------------------------------------
#  Linear layer dump
# ------------------------------------------------------------
def dump_linear_module(self, f, name, mode):
    print("printing layer {} ({})".format(name, mode))
    weight = self.weight.data.clone()  # out_dims, in_dims
    output_dim, input_dim = weight.shape
    bias = self.bias.clone()
    printVector(f, bias, name + '_bias', mode, name, 'bias')
    printVector(f, weight, name + '_weights', mode, name, 'weight')
    f.write('const LinearLayer {} = {{\n  {}_bias,\n  {}_weights,\n  {}, {}\n}};\n\n'
            .format(name, name, name, input_dim, output_dim))


Linear.dump_data = dump_linear_module


# ------------------------------------------------------------
#  針對單一 mode dump 出一份完整的 .c 檔
# ------------------------------------------------------------
def dump_one_mode(model, mode, n_fft, hidden_size, win_len, hop_len, cfile):
    try:
        os.remove(cfile)
    except OSError:
        pass

    f = open(cfile, 'w')
    f.write('/*This file is automatically generated from a Pytorch model*/\n')
    f.write('/*Quantization mode: {}*/\n\n'.format(mode))
    f.write('#include "nnagc.h"\n#include "aissl.h"\n\n')

    model.gru.dump_data(f, 'gru', mode)
    model.lin.dump_data(f, 'lin', mode)

    f.write('const NnagcModel nnagc_model = {\n')
    f.write('  &gru,\n')
    f.write('  &lin,\n')
    f.write('  {}, /* n_fft */\n'.format(n_fft))
    f.write('  {}, /* hidden_size */\n'.format(hidden_size))
    f.write('  {}, /* win_len */\n'.format(win_len))
    f.write('  {} /* hop_len */\n'.format(hop_len))
    f.write('};\n')

    f.close()
    print("done -> {}".format(cfile))


# ------------------------------------------------------------
#  依序 dump 三種模式，並把統計數據存成 CSV
# ------------------------------------------------------------
def dump_nnagc(model, n_fft, hidden_size, win_len, hop_len):
    model.to("cpu")
    os.makedirs(out_dir, exist_ok=True)

    for mode in ('float', 'int16', 'int8'):
        cfile = os.path.join(out_dir, 'gru_data_{}.c'.format(mode))
        dump_one_mode(model, mode, n_fft, hidden_size, win_len, hop_len, cfile)

    fieldnames = ['layer', 'tensor', 'mode', 'num_params', 'min', 'max',
                  'absmax', 'mean', 'std', 'scale', 'resolution', 'max_quant_error']
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)
    print("stats -> {}".format(csv_path))


if __name__ == '__main__':
    package = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    if isinstance(package, dict) and 'state_dict' in package:
        model = AGC_STFT_GRU.load_model_from_package(package)
        n_fft, hidden_size = package['n_fft'], package['hidden_size']
        win_len, hop_len = package['win_len'], package['hop_len']
    else:
        # checkpoint 只有純 state_dict，沒有其他 meta 資訊，
        # 請依你實際訓練設定手動修改下面四個參數
        n_fft, hidden_size, win_len, hop_len = 480, 10, 480, 240
        model = AGC_STFT_GRU(n_fft, hidden_size, win_len, hop_len)
        model.load_state_dict(package)

    dump_nnagc(model, n_fft, hidden_size, win_len, hop_len)
