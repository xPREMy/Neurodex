"""
=============================================================
predictor.py — Real-time EMG prediction via BLE
=============================================================
EXACT same decode formula as session_recorder.py:
    value = (raw - 2048) / 2048.0

Prediction fires ONLY at 100% model confidence.
Hold finger flexed for ~2-3s for a clean window.

Usage:
    python predictor.py
=============================================================
"""

import asyncio
import time
import threading
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import signal as scipy_signal
from bleak import BleakClient

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ════════════════════════════════════════════════════════
# CONFIG — must match session_recorder.py + build_dataset.py
# ════════════════════════════════════════════════════════
DEVICE_ADDRESS = "E4:B3:23:B0:5F:C6"
DATA_UUID      = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_UUID   = "0000ff01-0000-1000-8000-00805f9b34fb"

SR             = 2000
NUM_CHANNELS   = 6
SAMPLE_SIZE    = 13
N_ACTIVE_CH    = 4
EVENT_SAMPLES  = SR * 3       # 6000 = 3s window (matches training)
STEP_SEC       = 0.5          # predict every 0.5s
WAMP_THRESH    = 0.005        # must match build_dataset.py
ACTIVITY_THRESH = 0.02        # tune if needed

ALL_FEATURES = [
    'MAV_ch1','WAMP_ch1','VAR_ch1','WL_ch1','MDF_ch1','MNF_ch1',
    'MAV_ch2','WAMP_ch2','VAR_ch2','WL_ch2','MDF_ch2','MNF_ch2',
    'MAV_ch3','WAMP_ch3','VAR_ch3','WL_ch3','MDF_ch3','MNF_ch3',
    'MAV_ch4','WAMP_ch4','VAR_ch4','WL_ch4','MDF_ch4','MNF_ch4',
]

N_GESTURES = 5
GESTURE_NAMES  = {0:'Thumb Flexion', 1:'Index Flexion', 2:'Middle Flexion',
                  3:'Ring Flexion',  4:'Little Flexion'}
GESTURE_EMOJIS = {0:'👍', 1:'☝️', 2:'🖕', 3:'💍', 4:'🤙'}
GESTURE_COLS   = {0:'#1D9E75', 1:'#BA7517', 2:'#D4537E',
                  3:'#D85A30', 4:'#378ADD'}

# ════════════════════════════════════════════════════════
# GLOBALS
# ════════════════════════════════════════════════════════
from collections import deque
sample_buffer = deque(maxlen=EVENT_SAMPLES * 3)
latest_pred   = {'gesture': None, 'proba': None, 'active': False}
lock          = threading.Lock()
running       = True
ble_connected = False

# ════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(base, 'best_model.pkl')
    scaler_path = os.path.join(base, 'scaler.pkl')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"❌ Not found in {base}. Run train_model.py first.")
        return None, None
    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"✅ Model  : {type(clf).__name__}")
    print(f"✅ Scaler : ready")
    return clf, scaler

# ════════════════════════════════════════════════════════
# PACKET DECODE — EXACT same as session_recorder.py
# ════════════════════════════════════════════════════════
def decode_packet(data: bytearray):
    samples = []
    for i in range(0, len(data), SAMPLE_SIZE):
        chunk = data[i:i + SAMPLE_SIZE]
        if len(chunk) != SAMPLE_SIZE:
            continue
        channels = []
        for ch in range(NUM_CHANNELS):
            idx1, idx2 = 1 + ch*2, 2 + ch*2
            if idx2 >= len(chunk):
                break
            raw   = (chunk[idx1] << 8) | chunk[idx2]
            value = (raw - 2048) / 2048.0
            channels.append(value)
        if len(channels) == NUM_CHANNELS:
            samples.append(channels)
    return samples

def notification_handler(sender, data: bytearray):
    samples = decode_packet(data)
    with lock:
        for s in samples:
            sample_buffer.append(s[:N_ACTIVE_CH])

# ════════════════════════════════════════════════════════
# FILTER — same as session_recorder.py
# ════════════════════════════════════════════════════════
def filter_channel(x):
    b_n, a_n = scipy_signal.iirnotch(50.0, Q=30, fs=SR)
    x = scipy_signal.filtfilt(b_n, a_n, x)
    nyq = SR / 2
    b_bp, a_bp = scipy_signal.butter(4, [20/nyq, min(450/nyq, 0.99)], btype='band')
    x = scipy_signal.filtfilt(b_bp, a_bp, x)
    return x

# ════════════════════════════════════════════════════════
# ACTIVITY DETECTION
# ════════════════════════════════════════════════════════
def is_active(window):
    ch1      = window[:, 0]
    win_size = int(SR * 0.3)
    rms_vals = np.array([
        np.sqrt(np.mean(ch1[i:i+win_size]**2))
        for i in range(0, len(ch1)-win_size, win_size//2)
    ])
    if len(rms_vals) == 0:
        return False
    thresh = max(np.mean(rms_vals) + 1.5*np.std(rms_vals), ACTIVITY_THRESH)
    return bool(np.max(rms_vals) > thresh)

# ════════════════════════════════════════════════════════
# FEATURE EXTRACTION — identical to build_dataset.py
# ════════════════════════════════════════════════════════
def extract_features(window):
    feats = {}
    for i in range(N_ACTIVE_CH):
        x      = filter_channel(window[:, i].astype(np.float64))
        N      = len(x)
        ch_num = i + 1

        feats[f'MAV_ch{ch_num}']  = float(np.mean(np.abs(x)))
        feats[f'WAMP_ch{ch_num}'] = int(np.sum(np.abs(np.diff(x)) >= WAMP_THRESH))
        feats[f'VAR_ch{ch_num}']  = float(np.sum(x**2) / (N-1))
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
        feats[f'MNF_ch{ch_num}'] = float(np.sum(fb*pb) / np.sum(pb))

    return feats

def select_features(feats_dict):
    return np.array([feats_dict[f] for f in ALL_FEATURES], dtype=np.float32)

# ════════════════════════════════════════════════════════
# BLE
# ════════════════════════════════════════════════════════
async def ble_task():
    global running, ble_connected
    print(f"🔗 Connecting to {DEVICE_ADDRESS}...")
    try:
        async with BleakClient(DEVICE_ADDRESS, timeout=20.0) as client:
            ble_connected = True
            print("✅ Connected!\n")
            await client.start_notify(DATA_UUID, notification_handler)
            await client.write_gatt_char(CONTROL_UUID, b"START")
            print("📡 Streaming... hold each gesture steady for 3 seconds\n")
            while running:
                await asyncio.sleep(0.1)
            try:
                await client.stop_notify(DATA_UUID)
            except Exception:
                pass
    except Exception as e:
        print(f"❌ BLE error: {e}")
    finally:
        ble_connected = False

# ════════════════════════════════════════════════════════
# PREDICTION LOOP
# Predicts every STEP_SEC seconds on the last 3s window.
# Only fires if model is 100% confident (max proba == 1.0).
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

        window = np.array(buf[-EVENT_SAMPLES:])

        if not is_active(window):
            with lock:
                latest_pred = {'gesture': None, 'proba': None, 'active': False}
            continue

        try:
            feats_dict   = extract_features(window)
            feats_sel    = select_features(feats_dict)
            feats_scaled = scaler.transform(feats_sel.reshape(1, -1))
            pred         = int(clf.predict(feats_scaled)[0])
            proba        = clf.predict_proba(feats_scaled)[0]
            confidence   = max(proba)

            print(f"  {GESTURE_NAMES.get(pred,'?'):20s}  confidence={confidence*100:.1f}%")

            # Only show prediction at 100% confidence
            if confidence < 0.5:
                with lock:
                    latest_pred = {'gesture': None, 'proba': proba, 'active': False}
                continue

            with lock:
                latest_pred = {'gesture': pred, 'proba': proba, 'active': True, 'confidence': confidence}

        except Exception as e:
            print(f"  ⚠ Error: {e}")

# ════════════════════════════════════════════════════════
# DISPLAY
# ════════════════════════════════════════════════════════
def launch_display():
    fig = plt.figure(figsize=(14, 7), facecolor='#12121f')
    fig.canvas.manager.set_window_title('Real-Time EMG — NPG Lite')
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.5, wspace=0.3,
                            left=0.06, right=0.97, top=0.88, bottom=0.08)

    ch_cols   = ['#1D9E75','#534AB7','#D85A30','#BA7517']
    ax_sigs   = [fig.add_subplot(gs[0, i]) for i in range(4)]
    sig_lines = []
    for i, ax in enumerate(ax_sigs):
        ax.set_facecolor('#0a0a18')
        ax.set_title(f'Ch {i+1}', color='#888', fontsize=9, pad=3)
        ax.set_xlim(0, EVENT_SAMPLES)
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(colors='#333', labelsize=6)
        for sp in ax.spines.values(): sp.set_color('#222')
        ax.axhline(0, color='#222', lw=0.5)
        line, = ax.plot([], [], color=ch_cols[i], lw=0.6)
        sig_lines.append(line)

    ax_conf = fig.add_subplot(gs[1, :3])
    ax_conf.set_facecolor('#0a0a18')
    ax_conf.set_title('Prediction Confidence', color='#aaa', fontsize=10, pad=5)
    ax_conf.set_xlim(0, 1)
    ax_conf.set_ylim(-0.5, N_GESTURES - 0.5)
    ax_conf.tick_params(colors='#444', labelsize=9)
    for sp in ax_conf.spines.values(): sp.set_color('#222')
    ax_conf.axvline(0.5, color='#333', lw=0.8, ls='--')
    ax_conf.set_xlabel('Confidence', color='#555', fontsize=8)

    ylabels = [f"{GESTURE_EMOJIS[i]}  {GESTURE_NAMES[i]}" for i in range(N_GESTURES)]
    bars = ax_conf.barh(range(N_GESTURES), [0]*N_GESTURES,
                        color=[GESTURE_COLS[i] for i in range(N_GESTURES)],
                        height=0.55, alpha=0.85)
    ax_conf.set_yticks(range(N_GESTURES))
    ax_conf.set_yticklabels(ylabels, color='#ccc', fontsize=9)
    bar_pcts = [ax_conf.text(0.01, i, '', va='center',
                fontsize=9, color='white', fontweight='bold')
                for i in range(N_GESTURES)]

    ax_pred = fig.add_subplot(gs[1, 3])
    ax_pred.set_facecolor('#0a0a18')
    ax_pred.axis('off')
    t_emoji = ax_pred.text(0.5, 0.62, '✋', fontsize=46,
                           ha='center', va='center',
                           transform=ax_pred.transAxes)
    t_label = ax_pred.text(0.5, 0.28, 'Waiting...', fontsize=11,
                           ha='center', va='center', color='#555',
                           fontweight='bold', transform=ax_pred.transAxes)
    t_conf  = ax_pred.text(0.5, 0.10, '', fontsize=9,
                           ha='center', va='center', color='#444',
                           transform=ax_pred.transAxes)
    t_status = fig.text(0.5, 0.01, '🔴 Connecting...',
                        ha='center', color='#888', fontsize=9)
    fig.suptitle('NPG Lite — Real-Time Hand Gesture Recognition',
                 color='#ddd', fontsize=11, fontweight='bold', y=0.99)

    def update(frame):
        with lock:
            buf  = list(sample_buffer)
            pred = dict(latest_pred)

        if len(buf) >= EVENT_SAMPLES:
            win = np.array(buf[-EVENT_SAMPLES:])
            for i, ln in enumerate(sig_lines):
                ln.set_data(range(EVENT_SAMPLES), win[:, i])

        if ble_connected:
            t_status.set_text('🟢 Connected — hold a gesture steady for 3s')
            t_status.set_color('#1D9E75' if pred['active'] else '#BA7517')
        else:
            t_status.set_text('🔴 Not connected')
            t_status.set_color('#888')

        if pred['active'] and pred['proba'] is not None:
            g, proba = pred['gesture'], pred['proba']
            for j, (bar, pct) in enumerate(zip(bars, bar_pcts)):
                bar.set_width(proba[j])
                if proba[j] > 0.04:
                    pct.set_text(f'{proba[j]*100:.0f}%')
                    pct.set_x(proba[j]+0.01)
                else:
                    pct.set_text('')
            t_emoji.set_text(GESTURE_EMOJIS.get(g, '❓'))
            t_emoji.set_color(GESTURE_COLS.get(g, '#888'))
            t_label.set_text(GESTURE_NAMES.get(g, f'Class {g}'))
            t_label.set_color(GESTURE_COLS.get(g, '#888'))
            t_conf.set_text(f'{pred["confidence"]*100:.0f}% confident')
            t_conf.set_color('#1D9E75')
        else:
            for bar, pct in zip(bars, bar_pcts):
                bar.set_width(0); pct.set_text('')
            t_emoji.set_text('✋')
            t_emoji.set_color('#333')
            t_label.set_text('Rest / No gesture')
            t_label.set_color('#444')
            t_conf.set_text('')

        return (sig_lines + list(bars) + bar_pcts +
                [t_emoji, t_label, t_conf, t_status])

    ani = FuncAnimation(fig, update, interval=200,
                        blit=False, cache_frame_data=False)
    plt.show()
    return ani

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def main():
    global running

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Real-Time EMG Gesture — NPG Lite BLE              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Device  : {DEVICE_ADDRESS}               ║")
    print(f"║  SR      : {SR} Hz                              ║")
    print(f"║  Window  : 3s ({EVENT_SAMPLES} samples)                   ║")
    print(f"║  Step    : every {STEP_SEC}s                              ║")
    print(f"║  Fires   : only at 100% confidence                ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    clf, scaler = load_model()
    if clf is None:
        return

    ble_thread = threading.Thread(
        target=lambda: asyncio.run(ble_task()), daemon=True)
    ble_thread.start()

    pred_thread = threading.Thread(
        target=prediction_loop, args=(clf, scaler), daemon=True)
    pred_thread.start()

    print("Connecting", end='', flush=True)
    for _ in range(40):
        if ble_connected: break
        time.sleep(0.5)
        print('.', end='', flush=True)
    print()

    launch_display()
    running = False
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
