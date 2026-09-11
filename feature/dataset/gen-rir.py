import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import pandas as pd
import random
import os
from tqdm import tqdm

# ================= CONFIG ==================

OUT_DIR = "rir_dataset"
META_FILE = "metadata.csv"

FS = 48000
N_PER_CLASS = 100

RT60_RANGES = {
    "small": (0.2, 0.5),
    "medium": (0.6, 1.0),
    "large": (1.2, 2.0),
}

ROOM_DIM_RANGE = {
    "x": (4, 10),
    "y": (4, 10),
    "z": (2.5, 4)
}

SIGNAL_LENGTH_SEC = 3.0   # must be long enough to capture full decay

os.makedirs(OUT_DIR, exist_ok=True)

metadata = []

# ===========================================

def random_room_dim():
    return [
        random.uniform(*ROOM_DIM_RANGE["x"]),
        random.uniform(*ROOM_DIM_RANGE["y"]),
        random.uniform(*ROOM_DIM_RANGE["z"]),
    ]


def random_positions(room_dim):

    src = [
        random.uniform(0.5, room_dim[0] - 0.5),
        random.uniform(0.5, room_dim[1] - 0.5),
        random.uniform(1.0, room_dim[2] - 0.5),
    ]

    mic = [
        random.uniform(0.5, room_dim[0] - 0.5),
        random.uniform(0.5, room_dim[1] - 0.5),
        random.uniform(1.0, room_dim[2] - 0.5),
    ]

    return src, mic


def generate_single_rir(rt60_target, idx, label):

    room_dim = random_room_dim()

    absorption, max_order = pra.inverse_sabine(
        rt60_target, room_dim
    )

    room = pra.ShoeBox(
        room_dim,
        fs=FS,
        materials=pra.Material(absorption),
        max_order=max_order,
    )

    src_pos, mic_pos = random_positions(room_dim)

    # ---------- unit impulse input ----------

    signal_len = int(SIGNAL_LENGTH_SEC * FS)
    impulse = np.zeros(signal_len)
    impulse[0] = 1.0

    room.add_source(src_pos, signal=impulse)

    mic_array = pra.MicrophoneArray(
        np.array(mic_pos).reshape(3, 1),
        FS
    )

    room.add_microphone_array(mic_array)

    # ---------- simulate propagation ----------

    room.simulate()

    rir = room.mic_array.signals[0]

    # ---------- save wav ----------

    filename = f"{label}_{idx:04d}.wav"
    filepath = os.path.join(OUT_DIR, filename)

    sf.write(filepath, rir, FS)

    distance = np.linalg.norm(np.array(src_pos) - np.array(mic_pos))

    return {
        "filename": filename,
        "rt60_target": rt60_target,
        "room_x": room_dim[0],
        "room_y": room_dim[1],
        "room_z": room_dim[2],
        "src_x": src_pos[0],
        "src_y": src_pos[1],
        "src_z": src_pos[2],
        "mic_x": mic_pos[0],
        "mic_y": mic_pos[1],
        "mic_z": mic_pos[2],
        "distance": distance,
        "label": label,
    }


# ================== MAIN ===================

for label, (rt_min, rt_max) in RT60_RANGES.items():

    print(f"\nGenerating {label} RIRs...")

    for i in tqdm(range(N_PER_CLASS)):

        rt60 = random.uniform(rt_min, rt_max)

        info = generate_single_rir(rt60, i, label)

        metadata.append(info)


df = pd.DataFrame(metadata)
df.to_csv(os.path.join(OUT_DIR, META_FILE), index=False)

print("\nDone!")
print(f"Saved to: {OUT_DIR}")
