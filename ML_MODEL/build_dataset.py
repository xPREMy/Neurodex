"""
=============================================================
build_dataset.py — Extract features from session_recorder CSVs
=============================================================
Protocol:
  5 gestures × (20 reps × 3s active + 30s rest)
  Recording starts immediately — no lead-in.

Usage:
  python build_dataset.py

Output:
  outputs/features_all.csv  ← all sessions combined
=============================================================
"""

import sys
import glob
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
os.makedirs('events',  exist_ok=True)

# ════════════════════════════════════════════════════════
# ✏️  CONFIGURE YOUR FILES HERE
# ════════════════════════════════════════════════════════

SESSION_FILES = [
    "session1.csv",
    "session2.csv",
    "session3.csv",
    "session4.csv",
    "session5.csv",
    "session6.csv",
    "session7.csv",
    "session8.csv",
    "session9.csv",
    "session10.csv",
]

# ════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════
NOMINAL_SR  = 2000
REPS        = 20
REP_SECS    = 3
REST_SECS   = 30
BLOCK_SECS  = REPS * REP_SECS + REST_SECS
TOTAL_SECS  = 5 * BLOCK_SECS

WAMP_THRESH = 0.005

CHANNELS = ['Channel1', 'Channel2', 'Channel3', 'Channel4']

GESTURE_NAMES = {
    1: 'thumb_flexion',
    2: 'index_flexion',
    3: 'middle_flexion',
    4: 'ring_flexion',
    5: 'little_flexion',
}

# Fraction of reps to DROP per gesture (0.0 = keep all, 0.4 = drop 40%)
# Tune these values based on confusion matrix results
DROP_FRACTION = {
    3: 0.4,   # middle_flexion  → keeps ~12/20 reps
    4: 0.4,   # ring_flexion    → keeps ~12/20 reps
    5: 0.4,   # little_flexion  → keeps ~12/20 reps
}

ALL_FEATURES = [
    'MAV_ch1','WAMP_ch1','VAR_ch1','WL_ch1','MDF_ch1','MNF_ch1',
    'MAV_ch2','WAMP_ch2','VAR_ch2','WL_ch2','MDF_ch2','MNF_ch2',
    'MAV_ch3','WAMP_ch3','VAR_ch3','WL_ch3','MDF_ch3','MNF_ch3',
    'MAV_ch4','WAMP_ch4','VAR_ch4','WL_ch4','MDF_ch4','MNF_ch4',
]

# ════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ════════════════════════════════════════════════════════
def extract_features(event_data, SR):
    feats = {}
    for i, ch in enumerate(CHANNELS):
        x      = event_data[:, i].astype(np.float64)
        N      = len(x)
        ch_num = i + 1

        feats[f'MAV_ch{ch_num}']  = float(np.mean(np.abs(x)))
        feats[f'WAMP_ch{ch_num}'] = int(np.sum(np.abs(np.diff(x)) >= WAMP_THRESH))
        feats[f'VAR_ch{ch_num}']  = float(np.sum(x**2) / (N - 1))
        feats[f'WL_ch{ch_num}']   = float(np.sum(np.abs(np.diff(x))))

        freqs, psd = scipy_signal.welch(x, fs=SR, nperseg=256)
        band = (freqs >= 20) & (freqs <= 450)
        fb, pb = freqs[band], psd[band]

        if len(fb) == 0 or np.sum(pb) == 0:
            feats[f'MDF_ch{ch_num}'] = 0.0
            feats[f'MNF_ch{ch_num}'] = 0.0
            continue

        cumsum  = np.cumsum(pb)
        mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        feats[f'MDF_ch{ch_num}'] = float(fb[min(mdf_idx, len(fb)-1)])
        feats[f'MNF_ch{ch_num}'] = float(np.sum(fb * pb) / np.sum(pb))

    return feats

# ════════════════════════════════════════════════════════
# PROCESS ONE SESSION CSV
# ════════════════════════════════════════════════════════
def process_session(csv_path, subject_id):
    print(f"\n{'─'*55}")
    print(f"Subject {subject_id} | {os.path.basename(csv_path)}")
    print(f"{'─'*55}")

    df   = pd.read_csv(csv_path)
    cols = [c for c in CHANNELS if c in df.columns]
    data = df[cols].values.astype(np.float32)

    total_samples = len(data)
    SR            = total_samples / TOTAL_SECS
    REP_SAMPLES   = int(round(REP_SECS  * SR))
    BLOCK_SAMPS   = int(round(BLOCK_SECS * SR))

    print(f"  Samples    : {total_samples:,}")
    print(f"  Actual SR  : {SR:.1f} Hz  (nominal {NOMINAL_SR})")
    print(f"  Rep window : {REP_SAMPLES} samples ({REP_SECS}s)")
    print(f"  Block size : {BLOCK_SAMPS} samples ({BLOCK_SECS}s)")

    all_feats = []

    for g_idx in range(5):
        gesture_id   = g_idx + 1
        gesture_name = GESTURE_NAMES[gesture_id]
        g_start      = g_idx * BLOCK_SAMPS
        drop_frac    = DROP_FRACTION.get(gesture_id, 0.0)

        reps_extracted = 0
        reps_dropped   = 0

        for rep_idx in range(REPS):
            ev_s = g_start + rep_idx * REP_SAMPLES
            ev_e = ev_s + REP_SAMPLES

            if ev_e > total_samples:
                print(f"  ⚠️  G{gesture_id} R{rep_idx+1}: out of bounds — skipping")
                continue

            # ── DROP logic ───────────────────────────────
            if drop_frac > 0.0 and np.random.rand() < drop_frac:
                reps_dropped += 1
                continue
            # ─────────────────────────────────────────────

            ev_data = data[ev_s:ev_e, :]

            ev_df = pd.DataFrame(ev_data, columns=cols)
            ev_df.to_csv(
                f"events/S{subject_id:02d}_G{gesture_id}_R{rep_idx+1:02d}.csv",
                index=False)

            f = extract_features(ev_data, SR)
            f['subject_id']   = subject_id
            f['gesture_id']   = gesture_id
            f['gesture_name'] = gesture_name
            f['rep_num']      = rep_idx + 1
            all_feats.append(f)
            reps_extracted += 1

        drop_note = f"  (dropped {reps_dropped})" if reps_dropped else ""
        status = "✅" if reps_extracted > 0 else "⚠️ "
        print(f"  {status} G{gesture_id} ({gesture_name}): {reps_extracted}/{REPS} reps{drop_note}")

    return pd.DataFrame(all_feats)

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def main():
    np.random.seed(42)  # reproducible drops

    print("╔══════════════════════════════════════════════════╗")
    print("║         EMG Dataset Builder                      ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"\n  Sessions ({len(SESSION_FILES)}):")
    for f in SESSION_FILES: print(f"    {f}")
    print(f"\n  Drop fractions: { {GESTURE_NAMES[k]: v for k, v in DROP_FRACTION.items()} }")
    print()

    all_dfs = []
    for subject_id, csv_path in enumerate(SESSION_FILES, start=1):
        if not os.path.exists(csv_path):
            print(f"\n⚠️  Not found: {csv_path} — skipping")
            continue
        feat_df = process_session(csv_path, subject_id)
        if feat_df is not None and len(feat_df) > 0:
            all_dfs.append(feat_df)
            print(f"  → {len(feat_df)} rows extracted")

    if not all_dfs:
        print("❌ No data processed.")
        return

    meta   = ['subject_id', 'gesture_id', 'gesture_name', 'rep_num']
    master = pd.concat(all_dfs, ignore_index=True)
    feats  = [c for c in master.columns if c not in meta]
    master = master[meta + feats]

    out = 'outputs/features_all.csv'
    master.to_csv(out, index=False)

    print(f"\n{'='*55}")
    print(f"✅ Done : {out}  {master.shape}")
    print(f"\n  Reps per subject per gesture:")
    summary = (master.groupby(['subject_id', 'gesture_name'])['rep_num']
               .count().unstack(fill_value=0))
    print(summary.to_string())
    print(f"\n👉 Next: python train_model.py")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
