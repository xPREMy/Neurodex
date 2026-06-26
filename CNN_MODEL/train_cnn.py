"""
train_cnn.py — Train 1D CNN on master_windows.npz
Output: cnn_emg.pt, cnn_scaler.pkl
        outputs/eval/confusion_matrix.png
        outputs/eval/classification_report.txt
        outputs/eval/roc_curves.png
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

os.makedirs('outputs/eval', exist_ok=True)

GESTURE_NAMES = ['Thumb','Index','Middle','Ring','Little']
EPOCHS        = 60
BATCH_SIZE    = 32
LR            = 1e-4

# ── Load ──────────────────────────────────────────────────────
data    = np.load('master_windows.npz')
X       = data['X']   # (N, 4000, 4)
y       = data['y']
N, W, C = X.shape
print(f"Loaded: X={X.shape}  y={y.shape}")
print(f"Class dist: {np.bincount(y)}")

# ── Normalize ─────────────────────────────────────────────────
sc     = StandardScaler()
X_norm = sc.fit_transform(X.reshape(N, -1)).reshape(N, W, C).astype(np.float32)
joblib.dump(sc, 'cnn_scaler.pkl')

# ── Split ─────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X_norm, y, test_size=0.2,
                                           stratify=y, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}  |  Train={len(X_tr)}  Test={len(X_te)}")

loader = DataLoader(
    TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
Xe = torch.tensor(X_te).to(device)
ye = torch.tensor(y_te, dtype=torch.long).to(device)

# ── Model ─────────────────────────────────────────────────────
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
        return self.net(x.permute(0, 2, 1))  # (B,W,C)→(B,C,W)

model   = EMG_CNN(in_ch=C).to(device)
opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

# ── Train ─────────────────────────────────────────────────────
best_acc = 0.0
for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {ep:02d}/{EPOCHS}", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    sched.step()

    model.eval()
    with torch.no_grad():
        acc = (model(Xe).argmax(1) == ye).float().mean().item()
    print(f"Epoch {ep:02d}/{EPOCHS}  loss={total_loss/len(loader):.4f}  test_acc={acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'cnn_emg_best.pt')

# ── Final eval ────────────────────────────────────────────────
model.load_state_dict(torch.load('cnn_emg_best.pt'))
model.eval()
with torch.no_grad():
    logits = model(Xe)
    preds  = logits.argmax(1).cpu().numpy()
    proba  = torch.softmax(logits, dim=1).cpu().numpy()

print(f"\nBest test acc: {best_acc*100:.2f}%")

# classification report
report = classification_report(y_te, preds, target_names=GESTURE_NAMES)
print(report)
with open('outputs/eval/classification_report.txt', 'w') as f:
    f.write(f"Best test accuracy: {best_acc*100:.2f}%\n\n")
    f.write(report)

# confusion matrix
cm  = confusion_matrix(y_te, preds)
fig, ax = plt.subplots(figsize=(7, 6))
im  = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(GESTURE_NAMES); ax.set_yticklabels(GESTURE_NAMES)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
thresh = cm.max() / 2
for i in range(5):
    for j in range(5):
        ax.text(j, i, cm[i,j], ha='center', va='center',
                color='white' if cm[i,j] > thresh else 'black')
plt.tight_layout()
plt.savefig('outputs/eval/confusion_matrix.png', dpi=120)
plt.close()

# ROC curves
y_bin = label_binarize(y_te, classes=list(range(5)))
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('ROC Curves')
for i, ax in enumerate(axes.flat):
    if i >= 5: ax.axis('off'); continue
    fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
    ax.plot(fpr, tpr, lw=2, label=f'AUC={auc(fpr,tpr):.3f}')
    ax.plot([0,1],[0,1],'k--', lw=0.8)
    ax.set_title(GESTURE_NAMES[i])
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/eval/roc_curves.png', dpi=120)
plt.close()

# save model
torch.save({
    'model_state': model.state_dict(),
    'in_channels': C,
    'n_classes':   5,
    'window_size': W,
}, 'cnn_emg.pt')
print("\nSaved: cnn_emg.pt, cnn_scaler.pkl")
print("Eval:  outputs/eval/confusion_matrix.png, roc_curves.png, classification_report.txt")