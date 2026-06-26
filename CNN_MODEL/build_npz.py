"""
build_npz.py — 2s per rep, 1s discarded, filtered, no sliding window
Output: master_windows.npz  X:(N,FIXED_WIN,4)  y:(N,)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample, iirnotch, butter, filtfilt

SESSION_FILES = [f"session{i}.csv" for i in range(1, 11)]
CHANNELS      = ['Channel1','Channel2','Channel3','Channel4']
N_GESTURES    = 5
REPS          = 20
REP_SECS      = 3
TAKE_SECS     = 2
REST_SECS     = 30
BLOCK_SECS    = REPS * REP_SECS + REST_SECS
TOTAL_SECS    = N_GESTURES * BLOCK_SECS
FIXED_WIN     = 4000
NOMINAL_SR    = 2000

def filter_window(data, fs):
    # notch at 50Hz + bandpass 20-450Hz — same as session_recorder/predictor
    b_n, a_n   = iirnotch(50.0, Q=30, fs=fs)
    nyq        = fs / 2
    b_bp, a_bp = butter(4, [20/nyq, min(450/nyq, 0.99)], btype='band')
    out = np.zeros_like(data, dtype=np.float32)
    for ch in range(data.shape[1]):
        x = filtfilt(b_n, a_n, data[:, ch].astype(np.float64))
        x = filtfilt(b_bp, a_bp, x)
        out[:, ch] = x.astype(np.float32)
    return out

def segment_file(path):
    df   = pd.read_csv(path)
    data = df[CHANNELS].values.astype(np.float32)
    fs   = len(data) / TOTAL_SECS

    rep_samps   = int(REP_SECS  * fs)
    take_samps  = int(TAKE_SECS * fs)
    block_samps = int(BLOCK_SECS * fs)

    print(f"  {path.name}: {len(data)} samples | fs={fs:.1f}Hz")

    X, y = [], []
    for g in range(N_GESTURES):
        g_start = g * block_samps
        for rep in range(REPS):
            rep_start = g_start + rep * rep_samps
            rep_end   = rep_start + take_samps
            if rep_end > len(data):
                continue
            seg = data[rep_start:rep_end]
            seg = filter_window(seg, fs)              # ← filter
            seg = resample(seg, FIXED_WIN).astype(np.float32)
            X.append(seg)
            y.append(g)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

all_X, all_y = [], []
for fname in SESSION_FILES:
    path = Path(fname)
    if not path.exists():
        print(f"  skipping {fname} (not found)")
        continue
    X, y = segment_file(path)
    all_X.append(X)
    all_y.append(y)
    print(f"    → {len(X)} windows")

X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)

np.savez('master_windows.npz', X=X_all, y=y_all)
print(f"\nSaved master_windows.npz")
print(f"  X : {X_all.shape}")
print(f"  y : {y_all.shape}")
print(f"  Class dist: {np.bincount(y_all)}")