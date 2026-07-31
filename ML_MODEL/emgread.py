"""
=============================================================
Real-Time EMG Gesture Prediction — CSV Replay Mode
=============================================================
Replaces BLE streaming with CSV file replay.

CSV format (Chords Web / NPG Lite export):
  Columns: Counter, Channel1, Channel2, Channel3, Channel4,
           Channel5, Channel6
  Values : Already normalized (±1)

Pipeline identical to emg_full_pipeline.py:
  SR            = 2000 Hz
  EVENT_SAMPLES = 3s = 6000 samples
  CHANNELS      = 4 active (Channel1–Channel4)
  Filter        = Notch 50Hz + Bandpass 20-450Hz
  Features      = MAV, WAMP, VAR, WL, MDF, MNF × 4 channels

Usage:
  python emg_csv_predict.py                     # prompts for CSV path
  python emg_csv_predict.py my_recording.csv    # direct path argument
=============================================================
"""

import sys
import os
import time
import threading
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

from scipy import signal as scipy_signal
from collections import deque

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ════════════════════════════════════════════════════════
# CONFIG  (must match emg_full_pipeline.py exactly)
# ════════════════════════════════════════════════════════
SR              = 2000
NUM_CHANNELS    = 6           # CSV has 6 channels
N_ACTIVE_CH     = 4           # only use Channel1–Channel4
EVENT_SAMPLES   = 3 * SR      # 6000 samples = 3-second window
STEP_SEC        = 0.5         # prediction every 0.5 s
WAMP_THRESH     = 0.005
ACTIVITY_THRESH = 0.002

# CSV column names (Chords Web / NPG Lite format)
CSV_COUNTER_COL = 'Counter'
CSV_CHANNEL_COLS = [
    'Channel1', 'Channel2', 'Channel3', 'Channel4',
    'Channel5', 'Channel6',
]

# Feature list — exactly as in emg_full_pipeline.py (4 active channels)
ALL_FEATURES = [
    'MAV_ch1',  'WAMP_ch1', 'VAR_ch1',  'WL_ch1',  'MDF_ch1',  'MNF_ch1',
    'MAV_ch2',  'WAMP_ch2', 'VAR_ch2',  'WL_ch2',  'MDF_ch2',  'MNF_ch2',
    'MAV_ch3',  'WAMP_ch3', 'VAR_ch3',  'WL_ch3',  'MDF_ch3',  'MNF_ch3',
    'MAV_ch4',  'WAMP_ch4', 'VAR_ch4',  'WL_ch4',  'MDF_ch4',  'MNF_ch4',
    # 'MAV_ch1','WAMP_ch1','VAR_ch1','WL_ch1','ZC_ch1','SSC_ch1',
    # 'RMS_ch1','DASDV_ch1','MDF_ch1','MNF_ch1','PKF_ch1','MNP_ch1',
    # 'MAV_ch2','WAMP_ch2','VAR_ch2','WL_ch2','ZC_ch2','SSC_ch2',
    # 'RMS_ch2','DASDV_ch2','MDF_ch2','MNF_ch2','PKF_ch2','MNP_ch2',
    # 'MAV_ch3','WAMP_ch3','VAR_ch3','WL_ch3','ZC_ch3','SSC_ch3',
    # 'RMS_ch3','DASDV_ch3','MDF_ch3','MNF_ch3','PKF_ch3','MNP_ch3',
    # 'MAV_ch4','WAMP_ch4','VAR_ch4','WL_ch4','ZC_ch4','SSC_ch4',
    # 'RMS_ch4','DASDV_ch4','MDF_ch4','MNF_ch4','PKF_ch4','MNP_ch4',
]

GESTURE_NAMES = {
    0: 'Thumb Flexion',
    1: 'Index Flexion',
    2: 'Middle Flexion',
    3: 'Ring Flexion',
    4: 'Little Flexion',
    5: 'Hand Closed',
}
GESTURE_EMOJIS = {0:'👍', 1:'☝️', 2:'🖕', 3:'💍', 4:'🤙', 5:'✊'}
GESTURE_COLS   = {
    0:'#1D9E75', 1:'#BA7517', 2:'#D4537E',
    3:'#D85A30', 4:'#378ADD', 5:'#639922',
}

# ════════════════════════════════════════════════════════
# GLOBAL STATE
# ════════════════════════════════════════════════════════
sample_buffer = deque(maxlen=EVENT_SAMPLES * 3)
latest_pred   = {'gesture': None, 'proba': None, 'active': False}
replay_status = {'running': False, 'progress': 0.0,
                 'total': 0, 'current': 0, 'msg': 'Loading...'}
lock          = threading.Lock()
running       = True

# ════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════
def load_model():
    missing = [f for f in ('best_model.pkl', 'scaler.pkl')
               if not os.path.exists(f)]
    if missing:
        print(f"\n❌ Missing files: {missing}")
        print("   Run classifier.py first to generate best_model.pkl and scaler.pkl")
        return None, None
    clf    = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print(f"✅ Model  : {type(clf).__name__}")
    print(f"✅ Scaler : ready")
    print(f"✅ Features: {len(ALL_FEATURES)}")
    return clf, scaler

# ════════════════════════════════════════════════════════
# LOAD CSV
# ════════════════════════════════════════════════════════
def load_csv(filepath):
    """
    Load Chords Web CSV.  Values are already ±1 normalised — no conversion needed.
    Returns np.array of shape (N, N_ACTIVE_CH).
    """
    df = pd.read_csv(filepath)

    # Verify expected columns exist
    missing_cols = [c for c in CSV_CHANNEL_COLS[:N_ACTIVE_CH]
                    if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV missing columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    data = df[CSV_CHANNEL_COLS[:N_ACTIVE_CH]].values.astype(np.float32)
    duration = len(data) / SR
    print(f"✅ CSV loaded : {os.path.basename(filepath)}")
    print(f"   Samples    : {len(data):,}  |  Duration: {duration:.1f}s")
    print(f"   Channels   : {N_ACTIVE_CH} active  (Channel1–Channel{N_ACTIVE_CH})")
    return data

# ════════════════════════════════════════════════════════
# FILTER — identical to filter_channel() in pipeline
# ════════════════════════════════════════════════════════
def filter_channel(x):
    b_n, a_n = scipy_signal.iirnotch(50.0, Q=30, fs=SR)
    x = scipy_signal.filtfilt(b_n, a_n, x)
    nyq = SR / 2
    b_bp, a_bp = scipy_signal.butter(
        4, [20 / nyq, min(450 / nyq, 0.99)], btype='band')
    x = scipy_signal.filtfilt(b_bp, a_bp, x)
    return x

# ════════════════════════════════════════════════════════
# FEATURE EXTRACTION — identical to extract_features() in pipeline
# ════════════════════════════════════════════════════════
def extract_features_rt(window):
    """window: (EVENT_SAMPLES, N_ACTIVE_CH)"""
    feats = {}
    for i in range(N_ACTIVE_CH):
        x      = filter_channel(window[:, i])
        N      = len(x)
        ch_num = i + 1

        feats[f'MAV_ch{ch_num}']  = float(np.mean(np.abs(x)))
        feats[f'WAMP_ch{ch_num}'] = int(np.sum(np.abs(np.diff(x)) >= WAMP_THRESH))
        feats[f'VAR_ch{ch_num}']  = float(np.sum(x ** 2) / (N - 1))
        feats[f'WL_ch{ch_num}']   = float(np.sum(np.abs(np.diff(x))))



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

        freqs, psd = scipy_signal.welch(x, fs=SR, nperseg=256)
        band = (freqs >= 20) & (freqs <= 450)
        fb, pb = freqs[band], psd[band]

        # if len(fb) == 0 or np.sum(pb) == 0:
        #     feats[f'MDF_ch{ch_num}'] = 0.0
        #     feats[f'MNF_ch{ch_num}'] = 0.0
        #     continue
        if len(fb) == 0 or np.sum(pb) == 0:
            feats[f'MDF_ch{ch_num}'] = 0.0
            feats[f'MNF_ch{ch_num}'] = 0.0
            # feats[f'PKF_ch{ch_num}'] = 0.0
            # feats[f'MNP_ch{ch_num}'] = 0.0
            # feats[f'TTP_ch{ch_num}'] = 0.0
            continue

        cumsum  = np.cumsum(pb)
        mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        feats[f'MDF_ch{ch_num}'] = float(fb[min(mdf_idx, len(fb) - 1)])
        feats[f'MNF_ch{ch_num}'] = float(np.sum(fb * pb) / np.sum(pb))



        feats[f'MNF_ch{ch_num}'] = float(np.sum(fb * pb) / np.sum(pb))




        # # PKF — Peak Frequency
        # feats[f'PKF_ch{ch_num}'] = float(
        #     fb[np.argmax(pb)])

        # # MNP — Mean Power
        # feats[f'MNP_ch{ch_num}'] = float(np.mean(pb))

        # # TTP — Total Power
        # feats[f'TTP_ch{ch_num}'] = float(np.sum(pb))

    return feats

def select_features(feats_dict):
    return np.array([feats_dict[f] for f in ALL_FEATURES], dtype=np.float32)

# ════════════════════════════════════════════════════════
# ACTIVITY DETECTION — identical to is_active() in BLE script
# ════════════════════════════════════════════════════════
def is_active(window):
    ch1      = window[:, 0]
    win_size = int(SR * 0.3)
    rms_vals = np.array([
        np.sqrt(np.mean(ch1[i:i + win_size] ** 2))
        for i in range(0, len(ch1) - win_size, win_size // 2)
    ])
    if len(rms_vals) == 0:
        return False
    thresh = max(
        np.mean(rms_vals) + 1.5 * np.std(rms_vals),
        ACTIVITY_THRESH
    )
    return bool(np.max(rms_vals) > thresh)

# ════════════════════════════════════════════════════════
# CSV REPLAY THREAD
# Pushes rows into sample_buffer at ~2000 Hz (simulated real-time)
# ════════════════════════════════════════════════════════
def csv_replay_thread(data: np.ndarray):
    """
    Streams CSV rows into sample_buffer at the real 2000 Hz rate.
    One iteration per sample, sleeping in batches to avoid busy-wait.
    """
    global running, replay_status

    total = len(data)
    replay_status['total']   = total
    replay_status['running'] = True
    replay_status['msg']     = 'Replaying CSV...'

    # Push rows in batches of 20 (10 ms of data) to stay close to real-time
    BATCH = 20
    sleep_per_batch = BATCH / SR          # 0.01 s per batch

    print(f"▶  Replaying {total:,} samples at {SR} Hz "
          f"(≈ {total/SR:.1f}s of data)...")

    idx = 0
    t_start = time.perf_counter()

    while idx < total and running:
        end = min(idx + BATCH, total)
        with lock:
            for row in data[idx:end]:
                sample_buffer.append(row.tolist())

        idx = end
        replay_status['current']  = idx
        replay_status['progress'] = idx / total

        # Throttle to simulate real-time pace
        elapsed   = time.perf_counter() - t_start
        expected  = idx / SR
        lag       = expected - elapsed
        if lag > 0:
            time.sleep(lag)

    replay_status['running'] = False
    replay_status['msg']     = 'Replay complete ✓'
    print("\n✅ CSV replay finished.")

# ════════════════════════════════════════════════════════
# PREDICTION LOOP — runs every STEP_SEC
# ════════════════════════════════════════════════════════
def prediction_loop(clf, scaler):
    global latest_pred, running
    print("🧠 Prediction loop started")

    while running:
        time.sleep(STEP_SEC)

        with lock:
            buf = list(sample_buffer)

        if len(buf) < EVENT_SAMPLES:
            continue

        window = np.array(buf[-EVENT_SAMPLES:])   # (6000, 4)

        if not is_active(window):
            with lock:
                latest_pred = {'gesture': None, 'proba': None, 'active': False}
            continue

        try:
            feats_dict   = extract_features_rt(window)
            feats_sel    = select_features(feats_dict)
            feats_scaled = scaler.transform(feats_sel.reshape(1, -1))
            pred         = int(clf.predict(feats_scaled)[0])
            proba        = clf.predict_proba(feats_scaled)[0]

            # Debug line — comment out if not needed
            print(f"  MAV_ch1={feats_dict['MAV_ch1']:.5f}  "
                  f"WL_ch1={feats_dict['WL_ch1']:.3f}  "
                  f"VAR_ch1={feats_dict['VAR_ch1']:.6f}  "
                  f"→ {GESTURE_NAMES[pred]}  "
                  f"({max(proba)*100:.0f}% conf)")

            if max(proba) < 0.40:
                with lock:
                    latest_pred = {
                        'gesture': None, 'proba': proba, 'active': False}
                continue

            with lock:
                latest_pred = {
                    'gesture': pred, 'proba': proba, 'active': True}

        except Exception as e:
            print(f"  ⚠ Prediction error: {e}")

# ════════════════════════════════════════════════════════
# DISPLAY
# ════════════════════════════════════════════════════════
def launch_display(csv_filename: str):
    fig = plt.figure(figsize=(14, 7), facecolor='#12121f')
    fig.canvas.manager.set_window_title(
        f'EMG Gesture Prediction — {os.path.basename(csv_filename)}')

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.5, wspace=0.3,
        left=0.06, right=0.97, top=0.90, bottom=0.10)

    # ── 4 channel signal plots ────────────────────────────
    ch_cols   = ['#1D9E75', '#534AB7', '#D85A30', '#BA7517']
    ax_sigs   = [fig.add_subplot(gs[0, i]) for i in range(4)]
    sig_lines = []
    for i, ax in enumerate(ax_sigs):
        ax.set_facecolor('#0a0a18')
        ax.set_title(f'Ch {i+1}', color='#888', fontsize=9, pad=3)
        ax.set_xlim(0, EVENT_SAMPLES)
        ax.set_ylim(-1.2, 1.2)
        ax.tick_params(colors='#333', labelsize=6)
        for sp in ax.spines.values():
            sp.set_color('#222')
        ax.axhline(0, color='#222', lw=0.5)
        line, = ax.plot([], [], color=ch_cols[i], lw=0.6)
        sig_lines.append(line)

    # ── Confidence bar chart ──────────────────────────────
    ax_conf = fig.add_subplot(gs[1, :3])
    ax_conf.set_facecolor('#0a0a18')
    ax_conf.set_title('Prediction Confidence', color='#aaa', fontsize=10, pad=5)
    ax_conf.set_xlim(0, 1)
    ax_conf.set_ylim(-0.5, 5.5)
    ax_conf.tick_params(colors='#444', labelsize=9)
    for sp in ax_conf.spines.values():
        sp.set_color('#222')
    ax_conf.axvline(0.5, color='#333', lw=0.8, ls='--')
    ax_conf.set_xlabel('Confidence', color='#555', fontsize=8)

    ylabels = [f"{GESTURE_EMOJIS[i]}  {GESTURE_NAMES[i]}" for i in range(6)]
    bars = ax_conf.barh(
        range(6), [0] * 6,
        color=[GESTURE_COLS[i] for i in range(6)],
        height=0.55, alpha=0.85)
    ax_conf.set_yticks(range(6))
    ax_conf.set_yticklabels(ylabels, color='#ccc', fontsize=9)
    bar_pcts = [
        ax_conf.text(0.01, i, '', va='center',
                     fontsize=9, color='white', fontweight='bold')
        for i in range(6)
    ]

    # ── Prediction box ────────────────────────────────────
    ax_pred = fig.add_subplot(gs[1, 3])
    ax_pred.set_facecolor('#0a0a18')
    ax_pred.axis('off')
    t_emoji = ax_pred.text(
        0.5, 0.62, '✋', fontsize=46, ha='center', va='center',
        transform=ax_pred.transAxes)
    t_label = ax_pred.text(
        0.5, 0.28, 'Waiting...', fontsize=11,
        ha='center', va='center', color='#555',
        fontweight='bold', transform=ax_pred.transAxes)
    t_conf = ax_pred.text(
        0.5, 0.10, '', fontsize=9,
        ha='center', va='center', color='#444',
        transform=ax_pred.transAxes)

    # ── Progress bar (bottom of figure) ──────────────────
    ax_prog = fig.add_axes([0.06, 0.02, 0.91, 0.025])
    ax_prog.set_facecolor('#0a0a18')
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis('off')
    prog_bg  = ax_prog.barh(0, 1, color='#1a1a2e', height=1)[0]
    prog_bar = ax_prog.barh(0, 0, color='#1D9E75', height=1)[0]
    prog_txt = ax_prog.text(
        0.5, 0.5, '', ha='center', va='center',
        color='#aaa', fontsize=8, transform=ax_prog.transAxes)

    fig.suptitle(
        f'EMG Gesture Recognition — CSV: {os.path.basename(csv_filename)}',
        color='#ddd', fontsize=11, fontweight='bold')
    t_status = fig.text(
        0.5, 0.965, '⏳ Loading...', ha='center',
        color='#888', fontsize=9)

    def update(frame):
        with lock:
            buf   = list(sample_buffer)
            pred  = dict(latest_pred)
            repst = dict(replay_status)

        # ── Signal plots ──────────────────────────────────
        if len(buf) >= EVENT_SAMPLES:
            win = np.array(buf[-EVENT_SAMPLES:])
            for i, ln in enumerate(sig_lines):
                ln.set_data(range(EVENT_SAMPLES), win[:, i])

        # ── Progress bar ──────────────────────────────────
        prog = repst['progress']
        prog_bar.set_width(prog)
        cur_s  = repst['current'] / SR if SR else 0
        tot_s  = repst['total']   / SR if SR else 0
        prog_txt.set_text(
            f"{cur_s:.1f}s / {tot_s:.1f}s  ({prog*100:.0f}%)")

        # ── Status text ───────────────────────────────────
        if not repst['running'] and repst['current'] >= repst['total'] > 0:
            t_status.set_text('✅ Replay complete')
            t_status.set_color('#1D9E75')
        elif pred['active']:
            t_status.set_text('🟢 Active — predicting')
            t_status.set_color('#1D9E75')
        else:
            t_status.set_text('🟡 Replaying — no active gesture detected')
            t_status.set_color('#BA7517')

        # ── Prediction panel ──────────────────────────────
        if pred['active'] and pred['proba'] is not None:
            g, proba = pred['gesture'], pred['proba']
            for j, (bar, pct) in enumerate(zip(bars, bar_pcts)):
                bar.set_width(proba[j])
                if proba[j] > 0.04:
                    pct.set_text(f'{proba[j]*100:.0f}%')
                    pct.set_x(proba[j] + 0.01)
                else:
                    pct.set_text('')
            t_emoji.set_text(GESTURE_EMOJIS[g])
            t_emoji.set_color(GESTURE_COLS[g])
            t_label.set_text(GESTURE_NAMES[g])
            t_label.set_color(GESTURE_COLS[g])
            t_conf.set_text(f'{max(proba)*100:.0f}% confident')
            t_conf.set_color('#888')
        else:
            for bar, pct in zip(bars, bar_pcts):
                bar.set_width(0)
                pct.set_text('')
            t_emoji.set_text('✋')
            t_emoji.set_color('#333')
            t_label.set_text('Rest / No gesture')
            t_label.set_color('#444')
            t_conf.set_text('')

        return (sig_lines + list(bars) + bar_pcts +
                [t_emoji, t_label, t_conf, t_status, prog_bar, prog_txt])

    ani = FuncAnimation(fig, update, interval=200,
                        blit=False, cache_frame_data=False)
    plt.show()
    return ani

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def main():
    global running

    # ── Argument / prompt for CSV path ───────────────────
    parser = argparse.ArgumentParser(
        description='Real-time EMG gesture prediction from CSV replay')
    parser.add_argument(
        'csv_file', nargs='?', default=None,
        help='Path to Chords Web CSV file (Counter + Channel1–Channel6)')
    args = parser.parse_args()

    csv_path = "jivitesh_middle.csv"
    if csv_path is None:
        csv_path = input("📂 Enter path to CSV file: ").strip().strip('"').strip("'")

    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Real-Time EMG Gesture — CSV Replay Mode           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  CSV     : {os.path.basename(csv_path):<43}║")
    print(f"║  SR      : {SR} Hz                              ║")
    print(f"║  Window  : 3s ({EVENT_SAMPLES} samples)                   ║")
    print(f"║  Channels: {N_ACTIVE_CH} active / {NUM_CHANNELS} in CSV                  ║")
    print(f"║  Features: {len(ALL_FEATURES)} total                          ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Load model ────────────────────────────────────────
    clf, scaler = load_model()
    if clf is None:
        sys.exit(1)

    # ── Load CSV ──────────────────────────────────────────
    try:
        data = load_csv(csv_path)
    except Exception as e:
        print(f"❌ CSV load error: {e}")
        sys.exit(1)

    replay_status['total'] = len(data)

    # ── Start replay thread ───────────────────────────────
    t_replay = threading.Thread(
        target=csv_replay_thread, args=(data,), daemon=True)
    t_replay.start()

    # ── Start prediction thread ───────────────────────────
    t_pred = threading.Thread(
        target=prediction_loop, args=(clf, scaler), daemon=True)
    t_pred.start()

    # ── Wait until buffer has enough data to start predicting
    print("Buffering", end='', flush=True)
    for _ in range(60):
        with lock:
            n = len(sample_buffer)
        if n >= EVENT_SAMPLES:
            break
        time.sleep(0.5)
        print('.', end='', flush=True)
    print(f"  ({n:,} samples ready)\n")

    # ── Launch GUI on main thread ─────────────────────────
    ani = launch_display(csv_path)      # blocks until window closed
    running = False
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
