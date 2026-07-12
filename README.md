# Neurodex

An open-source EMG-based hand gesture recognition framework using NPG Lite (ESP32-C6), BLE communication, and a complete Python machine learning pipeline for real-time finger gesture classification.

---

## Overview

Neurodex acquires surface EMG signals from forearm muscles via BLE, processes them through a signal processing pipeline, extracts time and frequency domain features, and classifies individual finger gestures using machine learning — all in real-time.

This project replicates and extends the methodology of:
> *Arteaga et al. (2020) — "EMG-driven hand model based on the classification of individual finger movements", Biomedical Signal Processing and Control*

---

## Features

- Real-time EMG signal acquisition via BLE (2000 Hz, 4 channels)
- Complete signal preprocessing pipeline (notch + bandpass filtering)
- 13 time and frequency domain feature extraction per channel (52 total)
- Benchmarking of 22 classifier configurations — SVM, ANN, KNN, Random Forest, Extra Trees, XGBoost
- **94% offline classification accuracy** across 5 individual finger gestures
- Real-time gesture prediction with live visualization
- CSV replay mode for offline testing
- Class-balanced training pipeline

---

## Hardware Used

- **NPG Lite** (Upside Down Labs) — ESP32-C6 based BioAmp device
- Surface EMG electrodes — placed on forearm (FDS/FDP muscles)
- 4 active channels, 2 reference electrodes

---

## Gestures Classified

| Gesture | Label |
|---------|-------|
| 👍 Thumb Flexion | Class 1 |
| ☝️ Index Flexion | Class 2 |
| 🖕 Middle Flexion | Class 3 |
| 💍 Ring Flexion | Class 4 |
| 🤙 Little Flexion | Class 5 |

---

## Signal Processing Pipeline

```
Raw EMG (2000 Hz, 4 channels)
         ↓
Notch Filter (50 Hz — India power line)
         ↓
Bandpass Filter (20–450 Hz, Butterworth 4th order, zero-phase)
         ↓
RMS-based Adaptive Segmentation
(threshold = mean + 1.5×std, floor = 0.002)
         ↓
Feature Extraction (13 features × 4 channels = 52 total)
         ↓
Class Balancing + 70/30 Stratified Split
         ↓
ML Classification → Gesture Prediction
```

---

## Features Extracted

### Time Domain (8 per channel)
| Feature | Description |
|---------|-------------|
| MAV | Mean Absolute Value — muscle activation level |
| WAMP | Willison Amplitude — motor unit firing rate (threshold: 5mV) |
| VAR | Variance — signal power |
| WL | Waveform Length — signal complexity |
| ZC | Zero Crossings — frequency information |
| SSC | Slope Sign Changes — firing complexity |
| RMS | Root Mean Square — signal energy |
| DASDV | Difference Absolute Standard Deviation Value |

### Frequency Domain (5 per channel)
| Feature | Description |
|---------|-------------|
| MDF | Median Frequency — dominant frequency, fatigue sensitive |
| MNF | Mean Frequency — power-weighted average frequency |
| PKF | Peak Frequency — maximum power frequency |
| MNP | Mean Power — average spectral power |
| TTP | Total Power — total spectral energy |

---

## Classifiers Benchmarked

| Type | Configurations | Best Test Accuracy |
|------|---------------|-------------------|
| SVM | 5 kernels (Linear, Quadratic, Cubic, Fine/Med Gaussian) | 90.4% |
| ANN | 4 configs (1-2 hidden layers, tanh/logistic) | **94.0%** |
| KNN | 5 configs (K=1,10 × Euclidean/Cosine/Weighted/Cubic) | 87.3% |
| Random Forest | 2 configs (100, 400 estimators) | 92.8% |
| Extra Trees | 2 configs (100, 300 estimators) | 94.0% |
| AdaBoost | 2 configs (100, 300 estimators) | 72.3% |
| XGBoost | 2 configs (100, 300 estimators) | 89.2% |

**Best Model: ANN3 (15×8 neurons, tanh activation, adam solver) — 94% test accuracy**

---

## Dataset

```
Subjects        : 11
Gestures        : 5 individual finger flexions
Repetitions     : ~20 per gesture per subject
Sampling Rate   : 2000 Hz
Channels        : 4 (forearm surface EMG)
Total events    : ~550 (after class balancing)
Features        : 52 (13 × 4 channels)
```

---

## Project Structure

```
Neurodex/
├── preprocess.py          # EMG pipeline: load → filter → segment → features
├── classifier.py          # Train + evaluate 22 classifier configurations
├── realtime_predict.py    # Live BLE prediction with visualization
├── emg_csv_predict.py     # CSV replay mode for offline testing
├── outputs/
│   ├── features_all_subjects.csv   # Master feature matrix
│   ├── classifier_results.csv      # All classifier accuracies
│   ├── confusion_matrices/         # Per-classifier confusion matrices
│   └── roc_curves/                 # Per-gesture ROC curves
├── best_model.pkl         # Saved best classifier
├── scaler.pkl             # Saved StandardScaler
└── data*.csv              # Raw EMG recordings (Chords Web format)
```

---

## Installation

```bash
git clone https://github.com/xPREMy/Neurodex.git
cd Neurodex
pip install -r requirements.txt
```

### Requirements
```
numpy
pandas
scipy
scikit-learn
matplotlib
bleak
joblib
xgboost
```

---

## Usage

### Step 1 — Collect Data
Record EMG signals using [Chords Web](https://chords.upsidedownlabs.tech) with NPG Lite.
Save CSV files as `data1.csv`, `data2.csv`, etc.

### Step 2 — Preprocess + Extract Features
```bash
python preprocess.py
# Output: outputs/features_all_subjects.csv
```

### Step 3 — Train Classifiers
```bash
python classifier.py
# Output: classifier_results.csv, confusion matrices, ROC curves
#         best_model.pkl, scaler.pkl
```

### Step 4 — Real-Time Prediction (BLE)
```bash
python realtime_predict.py
# Connects to NPG Lite, streams live EMG, predicts gesture in real-time
```

### Step 5 — Offline Testing (CSV Replay)
```bash
python emg_csv_predict.py recording.csv
# Replays CSV at 2000 Hz, runs prediction pipeline, shows live visualization
```

---

## Results


```
Best classifier : ANN3 (2 hidden layers: 15×8, tanh, adam)
Test accuracy   : 94.0%
CV mean         : 91.0%

Per-gesture performance:
  Thumb   → Precision: 90%  Recall: 93%
  Index   → Precision: 97%  Recall: 100%
  Middle  → Precision: 89%  Recall: 89%
  Ring    → Precision: 89%  Recall: 89%
  Little  → Precision: 92%  Recall: 85%

```
<p align="center">
  <video src="https://github.com/user-attachments/assets/b15664a9-fd4d-4ad7-a9e8-f584189f4b11" width="49%" controls></video>
  <video src="https://github.com/user-attachments/assets/a38a191f-77dc-4289-84b7-adb99c6093ab" width="49%" controls></video>
</p>

---

## Reference

Arteaga, M.V., Castiblanco, J.C., Mondragon, I.F., Colorado, J.D., Alvarado-Rojas, C. (2020).
*EMG-driven hand model based on the classification of individual finger movements.*
Biomedical Signal Processing and Control, 58, 101834.
https://doi.org/10.1016/j.bspc.2019.101834

---

## Future Scope

- LOSO (Leave-One-Subject-Out) cross-validation for generalization testing
- Deep learning approaches (CNN, LSTM) for raw signal classification
- Prosthetic robotic hand control via classified gestures
- Subject-specific fine-tuning for improved real-time accuracy
- Wavelet-based feature extraction for adjacent finger discrimination
- Multi-subject online learning / transfer learning

---

## Contributors

Built at **BIRD Robotics Lab, IIT Jodhpur** as part of B.Tech research internship.

---

## License

MIT License

```bash
git clone https://github.com/xPREMy/Neurodex.git
cd Neurodex
pip install -r requirements.txt
