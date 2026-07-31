"""
=============================================================
EMG Full Pipeline — Paper: Arteaga et al. (2020)
=============================================================
Handles multiple CSV files (one per subject OR multiple per subject)
Steps:
  1. Load CSV(s)
  2. Filter (notch 50Hz + bandpass 20-450Hz)
  3. Segment gestures (RMS-based activity detection)
  4. Extract features (MAV, WAMP, VAR, WL, MDF, MNF) × 6 channels
  5. Combine all subjects → master feature CSV
  6. Ready for KNN / SVM / ANN

Usage:
  python emg_full_pipeline.py

Place all your CSV files in the same folder.
Edit FILE_CONFIG below to map filenames → subject_id + gesture info.
=============================================================
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import binary_closing, binary_opening
import os, sys, warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════╗
# ║              EDIT THIS SECTION                       ║
# ╠══════════════════════════════════════════════════════╣
#
# Two possible formats:
#
# FORMAT A: One file per subject (all 6 gestures inside)
#   → like your data6.csv
#
# FORMAT B: Multiple files per subject (e.g. 1 file per gesture)
#   → like if data1.csv=gesture1, data2.csv=gesture2 etc.
#
# Set FORMAT below, then fill FILE_CONFIG accordingly.

FORMAT = "A"   # "A" or "B"

# FORMAT A config:
# filename → subject_id
# Add all your files here!
FILES_FORMAT_A = {
    "newtry_jivitesh.csv"  : 1,   # subject 1 — all 6 gestures
    "newtry_jivitesh2.csv": 2,   # subject 2 — all 6 gestures (add when ready)
     "newtry_jivitesh3.csv": 3,   # subject 3
     "newtry_jivitesh4.csv": 4,
    "newtry_jivitesh5.csv": 5,
    "newtry_jivitesh6.csv":6,
    "newtry_jivitesh7.csv":7,
    "newtry_jivitesh8.csv":8,
    "newtry_jivitesh9.csv":10,
    "newtry_jivitesh10.csv":11,
    
    # "rishita_data1.csv":13,
    # "saipriya_data1.csv":14,
    # "rishita_data2.csv":15,
    # "rishita_data3.csv":16,
    # "rishita_data5.csv":17,
    # "rishita_data6.csv":18
}

# FORMAT B config:
# filename → (subject_id, gesture_id)
FILES_FORMAT_B = {
    # "subject2_gesture1.csv": (2, 1),
    # "subject2_gesture2.csv": (2, 2),
    # ... add your files here
}

# ─── Gesture time boundaries for FORMAT A ───────────────
# These are from RMS analysis of data6.csv
# If other subjects have different timing, add their boundaries here!
# subject_id → [(g1_start, g1_end), (g2_start, g2_end), ...]
GESTURE_BOUNDARIES = {
    # 1: [          # Subject 1 (data6.csv) boundaries
    #     (0,    240),
    #     (330,  600),
    #     (690,  960),
    #     (1050, 1320),
    #     (1380, 1680),
    #     (1740, 2100),
    # ],
    #  2: [          # data1.csv
    #     (0,    270),
    #     (300,  600),
    #     (630,  900),
    #     (930,  1260),
    #     (1290, 1590),
    #     (1680, 1950),
    # ],
    # 3: [          # data2.csv
    #     (0,    240),
    #     (330,  630),
    #     (690,  990),
    #     (1020, 1350),
    #     (1380, 1680),
    #     (1740, 2010),
    # ],
    # 4: [          # data3.csv
    #     (0,    300),
    #     (330,  600),
    #     (660,  960),
    #     (1020, 1350),
    #     (1380, 1680),
    #     (1740, 2040),
    # ],
    # 5: [          # data4.csv
    #     (0,    270),
    #     (330,  660),
    #     (690,  990),
    #     (1050, 1380),
    #     (1410, 1710),
    #     (1740, 2040),
    # ],
    # 6: [          # data5.csv
    #     (0,    270),
    #     (330,  630),
    #     (690,  990),
    #     (1050, 1320),
    #     (1410, 1710),
    #     (1770, 2040),
    # ],
    # 7: [
    # (0,    90),    # Gesture 1
    # (120,  210),   # Gesture 2
    # (240,  330),   # Gesture 3
    # (360,  450),   # Gesture 4
    # (480,  570),   # Gesture 5
    # (600,  690),   # Gesture 6
    # ],
#     1: [
#      (0,    65),    # Gesture 1
#      (89,  152),   # Gesture 2
#      (177,  236),   # Gesture 3
#      (266,  326),   # Gesture 4
#      (356,  416),   # Gesture 5
#      (439,  510),   # Gesture 6
#      ],
#      2: [              # data1.csv = subject 2
#     (0,60),   # Gesture 1
#     (90,150),  # Gesture 2
#     (190,245),  # Gesture 3
#     (250,315),  # Gesture 4
#     (336,400),  # Gesture 5
#     (426, 480 ),  # Gesture 6
# ],
#      3: [
#      (0,    60),    # Gesture 1
#      (90,  150),   # Gesture 2
#      (177,  233),   # Gesture 3
#      (257,  323),   # Gesture 4
#      (350,  413),   # Gesture 5
#      (440,  500),   # Gesture 6
#      ],
#      4: [
#      (0,    62),    # Gesture 1
#      (90,  150),   # Gesture 2
#      (181,  243),   # Gesture 3
#      (262,  325),   # Gesture 4
#      (351,  412),   # Gesture 5
#      (440,  510),   # Gesture 6
#      ],
#      5: [
#      (0,    60),    # Gesture 1
#      (90,  153),   # Gesture 2
#      (178,  240),   # Gesture 3
#      (270,  330),   # Gesture 4
#      (360,  435),   # Gesture 5
#      (445,  510),   # Gesture 6
#      ],
#      6: [
#      (0,   63),    # Gesture 1
#      (90,  150),   # Gesture 2
#      (180,  240),   # Gesture 3
#      (265,  329),   # Gesture 4
#      (360,  420),   # Gesture 5
#      (450,  510),   # Gesture 6
#      ],
#      7: [
#      (0,    50),    # Gesture 1
#      (75,  130),   # Gesture 2
#      (159,  206),   # Gesture 3
#      (229,  284),   # Gesture 4
#      (313,  370),   # Gesture 5
#      (400,  460),   # Gesture 6
#      ],
#      8: [
#      (0,    55),    # Gesture 1
#      (85,  143),   # Gesture 2
#      (170,  230),   # Gesture 3
#      (252,  310),   # Gesture 4
#      (336,  390),   # Gesture 5
#      (415,  475),   # Gesture 6
#      ],
#    10: [
#      (0,    60),    # Gesture 1
#      (86,  147),   # Gesture 2
#      (176,  235),   # Gesture 3
#      (264,  320),   # Gesture 4
#      (347,  410),   # Gesture 5
#      (434,  494),   # Gesture 6
#      ],
#      11: [
#      (0,    60),    # Gesture 1
#      (86,  144),   # Gesture 2
#      (170,  230),   # Gesture 3
#      (258,  320),   # Gesture 4
#      (345,  402),   # Gesture 5
#      (429,  490),   # Gesture 6
#      ],
#      12: [
#      (0,    45),    # Gesture 1
#      (66,  110),   # Gesture 2
#      (130,  179),   # Gesture 3
#      (204,  260),   # Gesture 4
#      (288,  346),   # Gesture 5
#      (374,  432),   # Gesture 6
#      ],
#      13: [
#          (0,62),
#          (88,149),
#          (179,240),
#          (270,330),
#          (355,414),
#          (445,505),

#      ],
#      14:[
#          (0,63),
#          (90,149),
#          (170,230),
#          (251,306),
#          (330,395),
#          (416,478),
#      ],
#      15:[
#         (0,60),
#         (83,120),
#         (191,205),
#         (264,327),
#         (348,411),
#         (436,501),
#      ],
#      16:[
#          (0,62),
#          (86,120),
#          (174,205),
#          (262,322),
#          (355,405),
#          (435,497)
#      ],
#      17:[
#          (0,60),
#           (90,120),
#           (180,205),
#           (270,330),
#           (360,420),
#           (449,505)
#              ],
#     18:[
#         (0,60),
#         (90,120),
#         (180,205),
#         (270,330),
#         (360,420),
#         (450,505)
#     ]
   1:[
       (0,58.6),
       (84,141),
       (170,226),
       (257,316),
       (340,401),
       (428,485)
   ],
   2:[
       (0,58.8),
       (89,145),
       (172,235),
       (263,322),
       (347,410),
       (435,493)
   ],
   3:[
      (0,66.3),
      (86,143),
      (170,237),
      (260,314),
      (340,403),
      (425,480) 
   ],
   4:[
       (0,62),
       (87,148),
       (175,241),
       (260,320),
       (360,420),
       (440,495)
   ],
   5:[
      (0,63),
      (83,144),
      (173,233),
      (262,320),
      (348,408),
      (438,500), 
   ],
  6:[
      (0,62.7),
      (89.9,149),
      (172,233),
      (263,323),
      (349,408),
      (433,490)
  ],
  7:[
      (0,61.8),
      (87,145),
      (175,235),
      (262,323),
      (350,410),
      (441,500)
  ],
  8:[
      (0,61.6),
      (89,149),
      (174,235),
      (263,323),
      (353,410),
      (440,500)
  ],
  9:[
     (0,62),
     (88.9,150),
     (182,236),
     (265,317),
     (350,410),
     (441,500) 
  ],
  10:[
      (0,61),
      (87,143),
      (171,232),
      (259,318),
      (345,407),
      (434,490)
  ]

}

# ╚══════════════════════════════════════════════════════╝

# ─── Constants ──────────────────────────────────────────
SR            = 2000
EVENT_SAMPLES = 3*SR    # 1500 samples per event
WAMP_THRESH   = 0.005     # 5mV
CHANNELS      = ['Channel1','Channel2','Channel3',
                  'Channel4'
                  ]  
                #   'Channel5','Channel6'
GESTURE_NAMES = {
    1:'thumb_flexion', 2:'index_flexion',  3:'middle_flexion',
     4:'ring_flexion',  5:'little_flexion' 
    #  6:'hand_closed'
}
os.makedirs('events',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ════════════════════════════════════════════════════════
# STEP 1: LOAD
# ════════════════════════════════════════════════════════
def load_csv(filepath):
    df = pd.read_csv(filepath)
    # Auto-detect channels present
    present = [c for c in CHANNELS if c in df.columns]
    duration = len(df) / SR
    print(f"  Loaded: {os.path.basename(filepath)}")
    print(f"  Samples: {len(df):,}  |  Duration: {duration:.1f}s  |  Channels: {len(present)}")
    return df, present

# ════════════════════════════════════════════════════════
# STEP 2: FILTER
# ════════════════════════════════════════════════════════
def filter_channel(x, sr=SR):
    """Notch 50Hz (India) + Bandpass 20-450Hz, zero-phase Butterworth"""
    # Notch
    b_n, a_n = signal.iirnotch(50.0, Q=30, fs=sr)
    x = signal.filtfilt(b_n, a_n, x)
    # Bandpass
    nyq = sr / 2
    b_bp, a_bp = signal.butter(4, [20/nyq, min(450/nyq, 0.99)], btype='band')
    x = signal.filtfilt(b_bp, a_bp, x)
    return x

def filter_dataframe(df, channels):
    df_f = df.copy()
    for ch in channels:
        df_f[ch] = filter_channel(df[ch].values)
    return df_f

# ════════════════════════════════════════════════════════
# STEP 3: SEGMENT
# ════════════════════════════════════════════════════════
def detect_events(sig, t_start, t_end, gesture_id):
    """
    Detect active 3s windows within [t_start, t_end] seconds.
    Returns list of (global_start_sample, global_end_sample).
    """
    s0   = int(t_start * SR)
    s1   = int(t_end   * SR)
    seg  = sig[s0:s1]

    # RMS energy
    win  = int(SR * 0.3)
    step = int(SR * 0.05)
    rms  = np.array([
        np.sqrt(np.mean(seg[i:i+win]**2))
        for i in range(0, len(seg)-win, step)
    ])
    t_loc = np.arange(len(rms)) * 0.05
    thresh = max(np.mean(rms) + 1.5*np.std(rms),0.002)
    active = rms > thresh
    active = binary_closing(active, structure=np.ones(12))
    active = binary_opening(active, structure=np.ones(6))

    trans  = np.diff(active.astype(int))
    starts = np.where(trans ==  1)[0]
    ends   = np.where(trans == -1)[0]

    if len(ends) > 0 and len(starts) > 0:
        if ends[0] < starts[0]: ends = ends[1:]
        n = min(len(starts), len(ends))
        starts, ends = starts[:n], ends[:n]

    events = []
    for s, e in zip(starts, ends):
        dur = t_loc[e] - t_loc[s]
        if 0.3 <= dur <= 5:
            center  = s0 + int(t_loc[s] * SR) + int(dur * SR / 2)
            ev_s    = center - EVENT_SAMPLES // 2
            ev_e    = center + EVENT_SAMPLES // 2
            if ev_s >= 0 and ev_e <= len(sig):
                events.append((ev_s, ev_e))

    status = "✅" if len(events) >= 15 else "⚠️ "
    print(f"  {status} G{gesture_id} ({t_start}-{t_end}s): "
          f"{len(events)} events detected (expected 20)")
    return events

# ════════════════════════════════════════════════════════
# STEP 4: FEATURE EXTRACTION
# ════════════════════════════════════════════════════════
def extract_features(event_data, channels):
    """
    event_data: np.array (EVENT_SAMPLES, n_channels)
    Returns dict of features: 6 features × n_channels
    """
    feats = {}
    for i, ch in enumerate(channels):
        x  = event_data[:, i]
        N  = len(x)
        ch_num = i + 1

        # MAV
        feats[f'MAV_ch{ch_num}']  = np.mean(np.abs(x))
        # WAMP (threshold 5mV as in paper)
        feats[f'WAMP_ch{ch_num}'] = int(np.sum(np.abs(np.diff(x)) >= WAMP_THRESH))
        # VAR
        feats[f'VAR_ch{ch_num}']  = np.sum(x**2) / (N - 1)
        # WL
        feats[f'WL_ch{ch_num}']   = np.sum(np.abs(np.diff(x)))


        # ZC — Zero Crossings
        # feats[f'ZC_ch{ch_num}'] = int(
        #     np.sum(np.diff(np.sign(x)) != 0))

        # # SSC — Slope Sign Changes
        # feats[f'SSC_ch{ch_num}'] = int(np.sum(
        #     ((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:])) > 0))

        # # RMS — Root Mean Square
        # feats[f'RMS_ch{ch_num}'] = float(
        #     np.sqrt(np.mean(x**2)))

        # # DASDV — Difference Absolute Standard Deviation
        # feats[f'DASDV_ch{ch_num}'] = float(
        #     np.sqrt(np.mean(np.diff(x)**2)))
        




        # PSD for MDF + MNF
        freqs, psd = signal.welch(x, fs=SR, nperseg=256)
        band = (freqs >= 20) & (freqs <= 450)
        fb, pb = freqs[band], psd[band]



        if len(fb) == 0 or np.sum(pb) == 0:
            feats[f'MDF_ch{ch_num}'] = 0.0
            feats[f'MNF_ch{ch_num}'] = 0.0
            # feats[f'PKF_ch{ch_num}'] = 0.0
            # feats[f'MNP_ch{ch_num}'] = 0.0
            # feats[f'TTP_ch{ch_num}'] = 0.0
            continue

        # MDF — median frequency
        cumsum  = np.cumsum(pb)
        mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        feats[f'MDF_ch{ch_num}'] = float(fb[min(mdf_idx, len(fb)-1)])

        # MNF — mean frequency
        feats[f'MNF_ch{ch_num}'] = float(np.sum(fb * pb) / np.sum(pb))




        # PKF — Peak Frequency
        # feats[f'PKF_ch{ch_num}'] = float(
        #     fb[np.argmax(pb)])

        # MNP — Mean Power
        # feats[f'MNP_ch{ch_num}'] = float(np.mean(pb))

        # TTP — Total Power
        # feats[f'TTP_ch{ch_num}'] = float(np.sum(pb))

    return feats

# ════════════════════════════════════════════════════════
# PROCESS ONE FILE (FORMAT A)
# ════════════════════════════════════════════════════════
def process_file_A(filepath, subject_id):
    """Process one file containing all 6 gestures for one subject."""
    print(f"\n{'─'*55}")
    print(f"Subject {subject_id} | File: {os.path.basename(filepath)}")
    print(f"{'─'*55}")

    df, channels = load_csv(filepath)

    print(f"\n  Filtering...")
    df_f = filter_dataframe(df, channels)
    sig_mat = df_f[channels].values   # (N, n_channels)

    # Get boundaries for this subject
    if subject_id not in GESTURE_BOUNDARIES:
        print(f"\n  ❌ No GESTURE_BOUNDARIES defined for subject {subject_id}!")
        print(f"  Run boundary_finder.py on this file first, then add to GESTURE_BOUNDARIES.")
        return None

    boundaries = GESTURE_BOUNDARIES[subject_id]
    print(f"\n  Segmenting gestures...")

    all_feats = []
    for g_idx, (t_start, t_end) in enumerate(boundaries):
        gesture_id = g_idx + 1
        if gesture_id == 6:      # ← yeh add karo
           continue      
        events     = detect_events(sig_mat[:, 0], t_start, t_end, gesture_id)

        for rep_num, (ev_s, ev_e) in enumerate(events, start=1):
            ev_data = sig_mat[ev_s:ev_e, :]

            # Save raw event CSV
            ev_df = pd.DataFrame(ev_data, columns=channels)
            ev_df.to_csv(
                f"events/S{subject_id:02d}_G{gesture_id}_R{rep_num:02d}.csv",
                index=False
            )

            # Extract features
            f = extract_features(ev_data, channels)
            f['subject_id']   = subject_id
            f['gesture_id']   = gesture_id
            f['gesture_name'] = GESTURE_NAMES[gesture_id]
            f['rep_num']      = rep_num
            all_feats.append(f)

    return pd.DataFrame(all_feats)

# ════════════════════════════════════════════════════════
# PROCESS ONE FILE (FORMAT B — single gesture per file)
# ════════════════════════════════════════════════════════
def process_file_B(filepath, subject_id, gesture_id):
    """Process one file containing one gesture (20 reps) for one subject."""
    print(f"\n{'─'*55}")
    print(f"Subject {subject_id} | Gesture {gesture_id} | File: {os.path.basename(filepath)}")
    print(f"{'─'*55}")

    df, channels = load_csv(filepath)
    df_f = filter_dataframe(df, channels)
    sig  = df_f[channels[0]].values
    sig_mat = df_f[channels].values

    # Whole file is one gesture — use full duration
    t_end = len(sig) / SR
    print(f"\n  Segmenting gestures...")
    events = detect_events(sig, 0, t_end, gesture_id)

    all_feats = []
    for rep_num, (ev_s, ev_e) in enumerate(events, start=1):
        ev_data = sig_mat[ev_s:ev_e, :]

        ev_df = pd.DataFrame(ev_data, columns=channels)
        ev_df.to_csv(
            f"events/S{subject_id:02d}_G{gesture_id}_R{rep_num:02d}.csv",
            index=False
        )

        f = extract_features(ev_data, channels)
        f['subject_id']   = subject_id
        f['gesture_id']   = gesture_id
        f['gesture_name'] = GESTURE_NAMES[gesture_id]
        f['rep_num']      = rep_num
        all_feats.append(f)

    return pd.DataFrame(all_feats)

# ════════════════════════════════════════════════════════
# BOUNDARY FINDER — run this to get boundaries for new files
# ════════════════════════════════════════════════════════
def find_boundaries(filepath):
    """
    Print RMS timeline for a new file so you can manually
    identify gesture group boundaries and add to GESTURE_BOUNDARIES.
    """
    print(f"\n🔍 Boundary finder: {filepath}")
    df = pd.read_csv(filepath)
    ch1 = df['Channel1'].values
    window, step = SR, SR//2
    rms, times = [], []
    for i in range(0, len(ch1)-window, step):
        rms.append(np.sqrt(np.mean(ch1[i:i+window]**2)))
        times.append(i/SR)
    rms = np.array(rms)

    print(f"\n{'Time':>8} | {'Max RMS':>8} | Pattern")
    print("─"*50)
    for t in range(0, int(times[-1]), 30):
        mask = (np.array(times) >= t) & (np.array(times) < t+30)
        if not mask.any(): continue
        seg  = rms[mask]
        bar  = "█" * min(int(seg.max()/0.005), 20)
        rest = " (REST ← boundary here?)" if seg.max() < 0.005 else ""
        print(f"{t:>7}s | {seg.max():>8.4f} | {bar}{rest}")

    print("\n👉 Look for REST periods (low RMS) — those are your 30s gaps between gestures.")
    print("   Add boundaries to GESTURE_BOUNDARIES dict in this script.")

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║         EMG FULL PIPELINE — Arteaga et al.          ║")
    print("╚══════════════════════════════════════════════════════╝")

    all_dfs = []

    if FORMAT == "A":
        for filename, subject_id in FILES_FORMAT_A.items():
            if not os.path.exists(filename):
                print(f"\n⚠️  File not found: {filename} — skipping.")
                continue
            feat_df = process_file_A(filename, subject_id)
            if feat_df is not None:
                all_dfs.append(feat_df)

    elif FORMAT == "B":
        for filename, (subject_id, gesture_id) in FILES_FORMAT_B.items():
            if not os.path.exists(filename):
                print(f"\n⚠️  File not found: {filename} — skipping.")
                continue
            feat_df = process_file_B(filename, subject_id, gesture_id)
            if feat_df is not None:
                all_dfs.append(feat_df)

    if not all_dfs:
        print("\n❌ No files processed. Check file names in FILE_CONFIG.")
        return

    # ── Combine all subjects ──────────────────────────────
    master = pd.concat(all_dfs, ignore_index=True)

    # ── Reorder columns ───────────────────────────────────
    meta  = ['subject_id','gesture_id','gesture_name','rep_num']
    feats = [c for c in master.columns if c not in meta]
    master = master[meta + feats]

    # ── Save master CSV ───────────────────────────────────
    out = 'outputs/features_all_subjects.csv'
    master.to_csv(out, index=False)

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"{'═'*55}")
    print(f"Output: {out}")
    print(f"Shape:  {master.shape}  "
          f"({master.shape[1]-4} features × {master.shape[0]} events)")

    print(f"\nEvents per subject per gesture:")
    summary = master.groupby(['subject_id','gesture_name'])['rep_num'].count().unstack()
    print(summary.to_string())

    print(f"\nFeature columns ({len(feats)}):")
    # Print in table format: feature × channel
    feat_names = ['MAV','WAMP','VAR','WL','MDF','MNF']
    ch_count   = len([c for c in master.columns if 'MAV' in c])
    print(f"  {feat_names} × {ch_count} channels = {len(feats)} total")

    print(f"\n👉 Next step: run classifier.py on outputs/features_all_subjects.csv")
    print(f"{'═'*55}\n")

    return master


if __name__ == "__main__":
    # Special mode: just find boundaries for a new file
    if len(sys.argv) == 3 and sys.argv[1] == "--find-boundaries":
        find_boundaries(sys.argv[2])
    else:
        master = main()
        if master is not None:
            print("Preview:")
            print(master.head(3).to_string())
