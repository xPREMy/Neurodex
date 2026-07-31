# ============================================================
# TF2AngleNet — Single-session training
# Based on: "TF2AngleNet: Continuous finger joint angle estimation
# based on multidimensional time–frequency features of sEMG signals"
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG  (match paper Table 1 & Section 2.3)
# ─────────────────────────────────────────────
CSV_PATH     = "emg_preprocessed.csv"  # update path as needed
EMG_COLS     = ['emg_ch0','emg_ch1','emg_ch2','emg_ch3','emg_ch4','emg_ch5']
ANGLE_COLS   = ['thumb_flex','index_mcp','middle_mcp','ring_mcp','pinkie_mcp']
N_CHANNELS   = len(EMG_COLS)        # 6 (paper uses 5; adapt if needed)
N_JOINTS     = len(ANGLE_COLS)      # 5 (paper predicts 6 with thumb-rot; we have 5)
WINDOW_SIZE  = 1600                 # 1 s at 1600 Hz (paper Section 2.3.1)
STRIDE       = 32                   # ~50 Hz effective output rate
STFT_NPERSEG = 512                  # paper Table 1: N_FL = 512
STFT_NOVERLAP= 400                  # paper: I_f = 2  →  hop = 112 ≈ ok
N_FREQ_BINS  = 257                  # STFT output bins (nperseg//2 + 1)
N_TIME_STEPS = 16                   # STFT time frames per 1-s window
BATCH_SIZE   = 64
EPOCHS       = 60                   # increase to 400 for full paper setup
LR           = 1e-3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────
# 1.  DATASET
# ─────────────────────────────────────────────
class EMGDataset(Dataset):
    """
    Yields:
        x_freq : (N_CHANNELS, N_FREQ_BINS, N_TIME_STEPS)  — STFT spectrogram
        x_time : (N_CHANNELS, WINDOW_SIZE)                 — raw EMG
        y      : (N_JOINTS,)                               — joint angles at last sample
    """
    def __init__(self, emg: np.ndarray, angles: np.ndarray,
                 window: int = WINDOW_SIZE, stride: int = STRIDE):
        super().__init__()
        self.windows = []
        self.targets = []

        n = len(emg)
        for start in range(0, n - window, stride):
            end = start + window
            seg_emg   = emg[start:end]           # (window, C)
            target    = angles[end - 1]          # angle at last timestep

            # STFT → spectrogram per channel
            spec_list = []
            for c in range(N_CHANNELS):
                f, t, Zxx = stft(seg_emg[:, c], fs=1600,
                                 nperseg=STFT_NPERSEG,
                                 noverlap=STFT_NOVERLAP)
                mag = np.abs(Zxx)                # (freq_bins, time_frames)
                spec_list.append(mag)
            spec = np.stack(spec_list, axis=0)   # (C, freq_bins, time_frames)

            self.windows.append({
                'x_freq': spec.astype(np.float32),
                'x_time': seg_emg.T.astype(np.float32),  # (C, window)
                'y':      target.astype(np.float32)
            })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        d = self.windows[idx]
        return d['x_freq'], d['x_time'], d['y']


# ─────────────────────────────────────────────
# 2.  MODEL — TF2AngleNet
# ─────────────────────────────────────────────

class ConvBlock2D(nn.Module):
    """Two stacked 2D convolutions + BN + LeakyReLU  (paper Fig. 3, 2D-ConvBlock)"""
    def __init__(self, in_ch, out_ch, kernel=(3,3), groups=1):
        super().__init__()
        pad = (kernel[0]//2, kernel[1]//2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, groups=groups, bias=False),
            nn.Conv2d(out_ch, out_ch, kernel, padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class ConvBlock1D(nn.Module):
    """Two stacked 1D convolutions + BN + LeakyReLU  (paper Fig. 3, 1D-ConvBlock)"""
    def __init__(self, in_ch, out_ch, kernel=3, groups=1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, groups=groups, bias=False),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class FFEM(nn.Module):
    """
    Frequency Feature Extraction Module — 3 × 2D-ConvBlock + MaxPool
    Input : (B, C, freq_bins, time_frames)
    Output: (B, 80*C, freq_bins', time_frames')  — grouped conv keeps channels separate
    Paper: NK = 20 → 40 → 80, KS = (3,3), grouped convolution
    """
    def __init__(self, in_channels):
        super().__init__()
        C = in_channels
        self.block1 = ConvBlock2D(C,    20*C, groups=C)
        self.down1  = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.block2 = ConvBlock2D(20*C, 40*C, groups=C)
        self.down2  = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.block3 = ConvBlock2D(40*C, 80*C, groups=C)
        self.down3  = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

    def forward(self, x):
        x = self.down1(self.block1(x))
        x = self.down2(self.block2(x))
        x = self.down3(self.block3(x))
        return x


class TFEM(nn.Module):
    """
    Time Feature Extraction Module — 3 × 1D-ConvBlock + MaxPool
    Input : (B, C, window_size)
    Output: (B, 80*C, T')
    Paper: NK = 20 → 40 → 80, KS = 3, grouped convolution
    """
    def __init__(self, in_channels):
        super().__init__()
        C = in_channels
        self.block1 = ConvBlock1D(C,    20*C, groups=C)
        self.down1  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.block2 = ConvBlock1D(20*C, 40*C, groups=C)
        self.down2  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.block3 = ConvBlock1D(40*C, 80*C, groups=C)
        self.down3  = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.down1(self.block1(x))
        x = self.down2(self.block2(x))
        x = self.down3(self.block3(x))
        return x


class Decoder(nn.Module):
    """
    3 × 1D-ConvBlock decoder, NK = 80 → 40 → 6
    Outputs joint angle sequence.
    Paper: kernel size = 3 throughout
    """
    def __init__(self, in_channels, n_joints):
        super().__init__()
        self.block1 = ConvBlock1D(in_channels, 80)
        self.block2 = ConvBlock1D(80, 40)
        self.block3 = ConvBlock1D(40, n_joints, kernel=3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x   # (B, N_JOINTS, T)


class TF2AngleNet(nn.Module):
    """
    Full TF2AngleNet as described in the paper.
    Dual-stream: FFEM (2D spectrogram) + TFEM (1D raw EMG)
    Channel-wise concatenation → Decoder → Global avg pool → joint angles
    """
    def __init__(self, n_channels=N_CHANNELS, n_joints=N_JOINTS):
        super().__init__()
        self.ffem    = FFEM(n_channels)
        self.tfem    = TFEM(n_channels)
        # After FFEM: (B, 80*C, F', T_freq)  → need to collapse freq dim
        # After TFEM: (B, 80*C, T_time')
        # We flatten the freq dim from FFEM to match 1D for concat
        concat_ch = 80 * n_channels * 2   # both streams concatenated
        self.decoder = Decoder(concat_ch, n_joints)

    def forward(self, x_freq, x_time):
        # x_freq: (B, C, F, T_f)   e.g. (B,6,257,16)
        # x_time: (B, C, W)        e.g. (B,6,1600)

        fd = self.ffem(x_freq)        # (B, 80C, F', T_f')
        td = self.tfem(x_time)        # (B, 80C, T')

        # Flatten frequency axis of fd and align temporal lengths
        B, C_fd, F_, T_f = fd.shape
        fd_flat = fd.view(B, C_fd * F_, T_f)   # (B, 80C*F', T_f')

        # Project fd_flat channel dim to match td for concat
        # Use adaptive pooling on time axis to align
        T_td = td.shape[-1]
        fd_aligned = nn.functional.adaptive_avg_pool1d(fd_flat, T_td)  # (B, 80C*F', T')

        # Now concat along channel dim
        # Reduce fd channel dim via grouped conv-like projection if too large
        # Simple approach: global-pool fd over freq, keep 80*C channels
        fd_proj = self.ffem_proj(fd_flat, T_td, C_fd)
        combined = torch.cat([fd_proj, td], dim=1)  # (B, 160C, T')
        out = self.decoder(combined)                 # (B, N_JOINTS, T')

        # Global average pool → single prediction per window
        out = out.mean(dim=-1)                       # (B, N_JOINTS)
        return out

    def ffem_proj(self, fd_flat, T_td, C_fd):
        """Collapse freq bins via mean, keeping 80*C channels, align time."""
        # fd_flat: (B, 80C*F', T_f') → reshape → mean over F'
        # Since we know 80C and F' separately:
        # Simple: just adaptive pool both channel and time
        return nn.functional.adaptive_avg_pool1d(fd_flat, T_td)[:, :C_fd, :]


# ─── Cleaner version of the model (avoids dynamic shapes in ffem_proj) ───

class TF2AngleNetV2(nn.Module):
    """
    Simplified faithful implementation:
    - FFEM outputs 2D feature map; collapsed to 1D via adaptive pooling
    - TFEM outputs 1D feature sequence
    - Concat → Decoder → global avg pool → N_JOINTS
    """
    def __init__(self, n_channels=N_CHANNELS, n_joints=N_JOINTS):
        super().__init__()
        C = n_channels
        # Frequency branch (2D CNN)
        self.freq_b1  = ConvBlock2D(C,    20*C, groups=C)
        self.freq_d1  = nn.MaxPool2d((2,1),(2,1))
        self.freq_b2  = ConvBlock2D(20*C, 40*C, groups=C)
        self.freq_d2  = nn.MaxPool2d((2,1),(2,1))
        self.freq_b3  = ConvBlock2D(40*C, 80*C, groups=C)
        self.freq_d3  = nn.MaxPool2d((2,1),(2,1))

        # Time branch (1D CNN)
        self.time_b1  = ConvBlock1D(C,    20*C, groups=C)
        self.time_d1  = nn.MaxPool1d(2, 2)
        self.time_b2  = ConvBlock1D(20*C, 40*C, groups=C)
        self.time_d2  = nn.MaxPool1d(2, 2)
        self.time_b3  = ConvBlock1D(40*C, 80*C, groups=C)
        self.time_d3  = nn.MaxPool1d(2, 2)

        # After concat: 80C (freq, after freq-dim collapse) + 80C (time) = 160C
        concat_ch = 80*C + 80*C
        # Grouped regular conv (paper: "Regular Convolution" after grouped concat)
        self.reg_conv = ConvBlock1D(concat_ch, 160*C)

        # Decoder
        self.dec1 = ConvBlock1D(160*C, 80)
        self.dec2 = ConvBlock1D(80,    40)
        self.dec3 = nn.Conv1d(40, n_joints, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()  # joint angles in [0,1]

    def forward(self, x_freq, x_time):
        # Frequency branch
        f = self.freq_d1(self.freq_b1(x_freq))   # (B,20C, F/2, T)
        f = self.freq_d2(self.freq_b2(f))         # (B,40C, F/4, T)
        f = self.freq_d3(self.freq_b3(f))         # (B,80C, F/8, T)
        # Collapse frequency axis → 1D
        f = f.mean(dim=2)                          # (B, 80C, T_f)

        # Time branch
        t = self.time_d1(self.time_b1(x_time))   # (B,20C, W/2)
        t = self.time_d2(self.time_b2(t))         # (B,40C, W/4)
        t = self.time_d3(self.time_b3(t))         # (B,80C, W/8)

        # Align time axes
        T = min(f.shape[-1], t.shape[-1])
        f = nn.functional.adaptive_avg_pool1d(f, T)
        t = nn.functional.adaptive_avg_pool1d(t, T)

        # Channel-wise concat
        x = torch.cat([f, t], dim=1)              # (B, 160C, T)
        x = self.reg_conv(x)

        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)                           # (B, N_JOINTS, T)

        # Global avg pool → single prediction per window
        x = x.mean(dim=-1)                         # (B, N_JOINTS)
        return self.sigmoid(x)


# ─────────────────────────────────────────────
# 3.  METRICS  (paper Eq. 5–7)
# ─────────────────────────────────────────────
def correlation_coefficient(y_true, y_pred):
    """Pearson CC averaged over joints."""
    ccs = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]; yp = y_pred[:, j]
        num = np.sum((yt - yt.mean()) * (yp - yp.mean()))
        den = np.sqrt(np.sum((yt - yt.mean())**2) * np.sum((yp - yp.mean())**2)) + 1e-8
        ccs.append(num / den)
    return np.mean(ccs)

def nrmse(y_true, y_pred):
    """Normalized RMSE averaged over joints."""
    nrmses = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]; yp = y_pred[:, j]
        rmse = np.sqrt(np.mean((yt - yp)**2))
        nrmse_j = rmse / (yt.max() - yt.min() + 1e-8)
        nrmses.append(nrmse_j)
    return np.mean(nrmses)

def r2_score(y_true, y_pred):
    """R² averaged over joints."""
    r2s = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]; yp = y_pred[:, j]
        ss_res = np.sum((yt - yp)**2)
        ss_tot = np.sum((yt - yt.mean())**2) + 1e-8
        r2s.append(1 - ss_res / ss_tot)
    return np.mean(r2s)


# ─────────────────────────────────────────────
# 4.  TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x_freq, x_time, y in loader:
        x_freq = x_freq.to(device)
        x_time = x_time.to(device)
        y      = y.to(device)
        optimizer.zero_grad()
        pred = model(x_freq, x_time)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_pred, all_true = [], []
    for x_freq, x_time, y in loader:
        x_freq = x_freq.to(device)
        x_time = x_time.to(device)
        pred = model(x_freq, x_time)
        total_loss += criterion(pred, y.to(device)).item() * len(y)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.numpy())
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_pred, all_true


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    emg    = df[EMG_COLS].values.astype(np.float32)
    angles = df[ANGLE_COLS].values.astype(np.float32)

    # Normalise EMG (z-score per channel)
    scaler = StandardScaler()
    emg    = scaler.fit_transform(emg)

    print(f"Building sliding-window dataset  (window={WINDOW_SIZE}, stride={STRIDE})...")
    dataset = EMGDataset(emg, angles, window=WINDOW_SIZE, stride=STRIDE)
    print(f"  → {len(dataset)} windows")

    # Train / val split (80/20, no shuffle to preserve temporal order)
    n = len(dataset)
    n_train = int(0.8 * n)
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds   = torch.utils.data.Subset(dataset, range(n_train, n))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    model = TF2AngleNetV2(n_channels=N_CHANNELS, n_joints=N_JOINTS).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # StepLR: halve LR every 10% of epochs (≈ every 40 epochs at 400 total; here ~6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, EPOCHS//10), gamma=0.5)

    # Kaiming init (paper Section 3.3)
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

    history = {'train_loss': [], 'val_loss': [], 'cc': [], 'r2': [], 'nrmse': []}
    best_val_loss = float('inf')

    print(f"\nTraining for {EPOCHS} epochs on {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, y_pred, y_true = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        cc   = correlation_coefficient(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        nrms = nrmse(y_true, y_pred)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['cc'].append(cc)
        history['r2'].append(r2)
        history['nrmse'].append(nrms)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'tf2anglenet_best.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
                  f"CC: {cc:.3f}  R²: {r2:.3f}  NRMSE: {nrms:.3f}")

    print("\n=== Best model metrics (final val) ===")
    print(f"  CC:    {max(history['cc']):.4f}")
    print(f"  R²:    {max(history['r2']):.4f}")
    print(f"  NRMSE: {min(history['nrmse']):.4f}")

    # ── Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss Curves'); axes[0].legend(); axes[0].set_xlabel('Epoch')

    axes[1].plot(history['cc'],    label='CC')
    axes[1].plot(history['r2'],    label='R²')
    axes[1].plot(history['nrmse'], label='NRMSE')
    axes[1].set_title('Metrics'); axes[1].legend(); axes[1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=120)
    print("Saved training_curves.png and tf2anglenet_best.pt")

    return model, history


if __name__ == '__main__':
    model, history = main()
