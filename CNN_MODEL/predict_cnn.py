"""
predictor_cnn.py — Real-time EMG prediction via BLE using CNN
Auto-detects muscle activation, captures 2s, predicts, 1s cooldown.
"""

import asyncio, time, threading, os, warnings
import numpy as np
import joblib
import torch
import torch.nn as nn
from collections import deque
from scipy.signal import resample, iirnotch, butter, filtfilt
from bleak import BleakClient
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ── CONFIG ────────────────────────────────────────────────────
DEVICE_ADDRESS  = "E4:B3:23:B0:5F:C6"
DATA_UUID       = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_UUID    = "0000ff01-0000-1000-8000-00805f9b34fb"

NUM_CHANNELS    = 6
SAMPLE_SIZE     = 13
N_ACTIVE_CH     = 4
NOMINAL_SR      = 2000
FIXED_WIN       = 4000
CAPTURE_SECS    = 2.0
COOLDOWN_SECS   = 0.5
ACTIVITY_THRESH = 0.015
CONF_THRESHOLD  = 0.60

GESTURE_NAMES  = {0:'Thumb Flexion', 1:'Index Flexion', 2:'Middle Flexion',
                  3:'Ring Flexion',  4:'Little Flexion'}
GESTURE_EMOJIS = {0:'👍', 1:'☝️',  2:'🖕', 3:'💍', 4:'🤙'}
GESTURE_COLS   = {0:'#1D9E75', 1:'#BA7517', 2:'#D4537E',
                  3:'#D85A30', 4:'#378ADD'}
N_GESTURES     = 5

# ── GLOBALS ───────────────────────────────────────────────────
BUFFER_SAMPS  = NOMINAL_SR * 6
sample_buffer = deque(maxlen=BUFFER_SAMPS)
latest_pred   = {'gesture': None, 'proba': None, 'active': False, 'confidence': 0}
lock          = threading.Lock()
running       = True
ble_connected = False

# ── FILTER — exact same as session_recorder / build_npz ───────
def filter_window(data, fs=NOMINAL_SR):
    b_n, a_n   = iirnotch(50.0, Q=30, fs=fs)
    nyq        = fs / 2
    b_bp, a_bp = butter(4, [20/nyq, min(450/nyq, 0.99)], btype='band')
    out = np.zeros_like(data, dtype=np.float32)
    for ch in range(data.shape[1]):
        x = filtfilt(b_n, a_n, data[:, ch].astype(np.float64))
        x = filtfilt(b_bp, a_bp, x)
        out[:, ch] = x.astype(np.float32)
    return out

# ── MODEL ─────────────────────────────────────────────────────
class EMG_CNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32,  kernel_size=15, padding=7), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32,   64,  kernel_size=11, padding=5), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64,  128,  kernel_size=7,  padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256,  kernel_size=5,  padding=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(256, 256,  kernel_size=3,  padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x.permute(0, 2, 1))

def load_model():
    base        = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(base, 'cnn_emg.pt')
    scaler_path = os.path.join(base, 'cnn_scaler.pkl')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ cnn_emg.pt or cnn_scaler.pkl not found.")
        return None, None
    ckpt   = torch.load(model_path, map_location='cpu')
    model  = EMG_CNN(in_ch=ckpt['in_channels'], n_classes=ckpt['n_classes'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scaler = joblib.load(scaler_path)
    print(f"✅ CNN loaded | win={ckpt['window_size']} ch={ckpt['in_channels']}")
    return model, scaler

# ── BLE DECODE ────────────────────────────────────────────────
def decode_packet(data: bytearray):
    samples = []
    for i in range(0, len(data), SAMPLE_SIZE):
        chunk = data[i:i+SAMPLE_SIZE]
        if len(chunk) != SAMPLE_SIZE: continue
        channels = []
        for ch in range(NUM_CHANNELS):
            idx1, idx2 = 1 + ch*2, 2 + ch*2
            if idx2 >= len(chunk): break
            raw = (chunk[idx1] << 8) | chunk[idx2]
            channels.append((raw - 2048) / 2048.0)
        if len(channels) == NUM_CHANNELS:
            samples.append(channels)
    return samples

def notification_handler(sender, data: bytearray):
    samples = decode_packet(data)
    with lock:
        for s in samples:
            sample_buffer.append(s[:N_ACTIVE_CH])

# ── ACTIVITY DETECTION ────────────────────────────────────────
def is_active(window):
    check = np.array(window[-int(NOMINAL_SR * 0.1):])
    return float(np.sqrt(np.mean(check ** 2))) > ACTIVITY_THRESH

# ── INFERENCE ─────────────────────────────────────────────────
def predict(window_np, model, scaler):
    window_np = filter_window(window_np)                              # ← filter first
    w         = resample(window_np, FIXED_WIN).astype(np.float32)    # resample to 4000
    w_norm    = scaler.transform(w.reshape(1, -1)).reshape(1, FIXED_WIN, N_ACTIVE_CH)
    tensor    = torch.tensor(w_norm)
    with torch.no_grad():
        proba = torch.softmax(model(tensor), dim=1).numpy()[0]
    return int(np.argmax(proba)), proba

# ── BLE TASK ──────────────────────────────────────────────────
async def ble_task():
    global running, ble_connected
    print(f"🔗 Connecting to {DEVICE_ADDRESS}...")
    try:
        async with BleakClient(DEVICE_ADDRESS, timeout=20.0) as client:
            ble_connected = True
            print("✅ Connected!\n")
            await client.start_notify(DATA_UUID, notification_handler)
            await client.write_gatt_char(CONTROL_UUID, b"START")
            while running:
                await asyncio.sleep(0.1)
            try: await client.stop_notify(DATA_UUID)
            except: pass
    except Exception as e:
        print(f"❌ BLE error: {e}")
    finally:
        ble_connected = False

# ── PREDICTION LOOP ───────────────────────────────────────────
def prediction_loop(model, scaler):
    global latest_pred, running
    capture_samps = int(NOMINAL_SR * CAPTURE_SECS)
    print("🧠 Prediction loop started...")

    while running:
        time.sleep(0.5)

        with lock:
            buf = list(sample_buffer)

        if len(buf) < capture_samps:
            continue

        window = np.array(buf[-capture_samps:])  # last 2s

        if not is_active(window):
            with lock:
                latest_pred = {'gesture': None, 'proba': None,
                               'active': False, 'confidence': 0}
            continue

        try:
            pred, proba = predict(window, model, scaler)
            conf = float(proba[pred])
            print(f"  → {GESTURE_NAMES[pred]:20s}  conf={conf*100:.1f}%")
            with lock:
                if conf >= CONF_THRESHOLD:
                    latest_pred = {'gesture': pred, 'proba': proba,
                                   'active': True, 'confidence': conf}
                else:
                    latest_pred = {'gesture': None, 'proba': proba,
                                   'active': False, 'confidence': conf}
        except Exception as e:
            print(f"  ⚠ {e}")

# ── DISPLAY ───────────────────────────────────────────────────
def launch_display():
    DISP_SAMPS = int(NOMINAL_SR * 2)
    fig = plt.figure(figsize=(14, 7), facecolor='#12121f')
    fig.canvas.manager.set_window_title('Real-Time EMG — CNN')
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.5, wspace=0.3,
                            left=0.06, right=0.97, top=0.88, bottom=0.08)
    ch_cols   = ['#1D9E75','#534AB7','#D85A30','#BA7517']
    ax_sigs   = [fig.add_subplot(gs[0, i]) for i in range(4)]
    sig_lines = []
    for i, ax in enumerate(ax_sigs):
        ax.set_facecolor('#0a0a18')
        ax.set_title(f'Ch {i+1}', color='#888', fontsize=9, pad=3)
        ax.set_xlim(0, DISP_SAMPS); ax.set_ylim(-1.1, 1.1)
        ax.tick_params(colors='#333', labelsize=6)
        for sp in ax.spines.values(): sp.set_color('#222')
        ax.axhline(0, color='#222', lw=0.5)
        line, = ax.plot([], [], color=ch_cols[i], lw=0.6)
        sig_lines.append(line)

    ax_conf = fig.add_subplot(gs[1, :3])
    ax_conf.set_facecolor('#0a0a18')
    ax_conf.set_title('Prediction Confidence', color='#aaa', fontsize=10, pad=5)
    ax_conf.set_xlim(0, 1); ax_conf.set_ylim(-0.5, N_GESTURES - 0.5)
    ax_conf.tick_params(colors='#444', labelsize=9)
    for sp in ax_conf.spines.values(): sp.set_color('#222')
    ax_conf.axvline(CONF_THRESHOLD, color='#555', lw=0.8, ls='--')
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
    ax_pred.set_facecolor('#0a0a18'); ax_pred.axis('off')
    t_emoji  = ax_pred.text(0.5, 0.62, '✋', fontsize=46,
                            ha='center', va='center', transform=ax_pred.transAxes)
    t_label  = ax_pred.text(0.5, 0.28, 'Waiting...', fontsize=11,
                            ha='center', va='center', color='#555',
                            fontweight='bold', transform=ax_pred.transAxes)
    t_conf   = ax_pred.text(0.5, 0.10, '', fontsize=9,
                            ha='center', va='center', color='#444',
                            transform=ax_pred.transAxes)
    t_status = fig.text(0.5, 0.01, '🔴 Connecting...',
                        ha='center', color='#888', fontsize=9)
    fig.suptitle('NPG Lite — CNN Real-Time Hand Gesture Recognition',
                 color='#ddd', fontsize=11, fontweight='bold', y=0.99)

    def update(frame):
        with lock:
            buf  = list(sample_buffer)
            pred = dict(latest_pred)

        if len(buf) >= DISP_SAMPS:
            win = np.array(buf[-DISP_SAMPS:])
            for i, ln in enumerate(sig_lines):
                ln.set_data(range(DISP_SAMPS), win[:, i])

        t_status.set_text('🟢 Connected' if ble_connected else '🔴 Not connected')
        t_status.set_color('#1D9E75' if ble_connected else '#888')

        if pred['active'] and pred['proba'] is not None:
            g, proba = pred['gesture'], pred['proba']
            for j, (bar, pct) in enumerate(zip(bars, bar_pcts)):
                bar.set_width(proba[j])
                pct.set_text(f'{proba[j]*100:.0f}%' if proba[j] > 0.04 else '')
                pct.set_x(proba[j]+0.01)
            t_emoji.set_text(GESTURE_EMOJIS.get(g, '❓'))
            t_emoji.set_color(GESTURE_COLS.get(g, '#888'))
            t_label.set_text(GESTURE_NAMES.get(g, f'Class {g}'))
            t_label.set_color(GESTURE_COLS.get(g, '#888'))
            t_conf.set_text(f'{pred["confidence"]*100:.0f}% confident')
            t_conf.set_color('#1D9E75')
        else:
            for bar, pct in zip(bars, bar_pcts):
                bar.set_width(0); pct.set_text('')
            t_emoji.set_text('✋'); t_emoji.set_color('#333')
            t_label.set_text('Rest / No gesture'); t_label.set_color('#444')
            t_conf.set_text('')

        return sig_lines + list(bars) + bar_pcts + [t_emoji, t_label, t_conf, t_status]

    ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.show()
    return ani

# ── MAIN ──────────────────────────────────────────────────────
def main():
    global running
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Real-Time EMG — CNN (auto-detect)                 ║")
    print(f"║  Capture: {CAPTURE_SECS}s | Cooldown: {COOLDOWN_SECS}s | Threshold: {CONF_THRESHOLD*100:.0f}%    ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    model, scaler = load_model()
    if model is None: return

    threading.Thread(target=lambda: asyncio.run(ble_task()), daemon=True).start()
    threading.Thread(target=prediction_loop, args=(model, scaler), daemon=True).start()

    print("Connecting", end='', flush=True)
    for _ in range(40):
        if ble_connected: break
        time.sleep(0.5); print('.', end='', flush=True)
    print()

    launch_display()
    running = False
    print("\n✅ Done.")

if __name__ == "__main__":
    main()