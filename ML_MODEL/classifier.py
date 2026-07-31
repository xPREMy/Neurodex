"""
=============================================================
EMG Classifier — Paper: Arteaga et al. (2020)
=============================================================
Runs 14 configurations matching paper Table 3:
  SVM:  5 kernels
  ANN:  4 configs
  KNN:  5 configs

Input:  outputs/features_all_subjects.csv
Output: outputs/classifier_results.csv
        outputs/confusion_matrices/
        outputs/roc_curves/

Usage:
  python classifier.py
=============================================================
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.preprocessing           import StandardScaler, label_binarize
from sklearn.model_selection         import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm                     import SVC
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.neural_network          import MLPClassifier
from sklearn.metrics                 import (accuracy_score, classification_report,
                                             confusion_matrix, roc_curve, auc)
from sklearn.multiclass              import OneVsRestClassifier

os.makedirs('outputs/confusion_matrices', exist_ok=True)
os.makedirs('outputs/roc_curves',         exist_ok=True)

GESTURE_NAMES = {
    1:'Thumb', 2:'Index', 3:'Middle',
    4:'Ring',  5:'Little'
    # , 6:'Fist'
}
MAX_REPS = 20   # cap at 20 per gesture per subject (paper protocol)

# ════════════════════════════════════════════════════════
# LOAD + PREPARE DATA
# ════════════════════════════════════════════════════════
def load_data(path="outputs/features_all_subjects.csv"):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")

    df = pd.read_csv(path)

    # Apply paper protocol constraint
    df = (
        df.groupby(['subject_id', 'gesture_id'])
          .apply(lambda x: x.nsmallest(MAX_REPS, 'rep_num'))
          .reset_index(drop=True)
    )
    df = df[df['gesture_id'] != 6]
    # Features selected after correlation analysis
    # selected_features = [
    #     'VAR_ch1',
    #     'WL_ch1',
    #     'VAR_ch2',
    #     'WL_ch2',
    #     'MNF_ch2',
    #     'VAR_ch3',
    #     'WL_ch3',
    #     'MNF_ch3',
    #     'VAR_ch4',
    #     'WL_ch4',
    #     'MNF_ch4'
    # ]
    MIN_PER_CLASS = df.groupby('gesture_id')['rep_num'].count().min()
    print(f"Balancing to {MIN_PER_CLASS} events per gesture")
    df = (df.groupby('gesture_id')
        .apply(lambda x: x.sample(MIN_PER_CLASS, random_state=42))
        .reset_index(drop=True))
    meta = ['subject_id', 'gesture_id', 'gesture_name', 'rep_num']
    X = df.drop(columns=meta).values.astype(np.float32)
    y = df['gesture_id'].values-1

    print(f"\nDataset Loaded: {X.shape[0]} samples × {X.shape[1]} features")
    print("\nSelected Features:")
    # print(selected_features)
    print(f"\nClasses: {np.unique(y)}")

    return X, y,df

# ════════════════════════════════════════════════════════
# 14 CLASSIFIER CONFIGS (paper Table 3)
# ════════════════════════════════════════════════════════
def get_classifiers():
    return {
        # ── SVM (5 kernels, OvO multi-class) ──────────────
        'SVM1_Linear':         SVC(kernel='linear',  C=1,    decision_function_shape='ovo', probability=True, random_state=42),
        'SVM2_Quadratic':      SVC(kernel='poly',    degree=2, C=1, decision_function_shape='ovo', probability=True, random_state=42),
        'SVM3_Cubic':          SVC(kernel='poly',    degree=3, C=1, decision_function_shape='ovo', probability=True, random_state=42),
        'SVM4_FineGaussian':   SVC(kernel='rbf',     gamma='scale', C=1, class_weight='balanced',decision_function_shape='ovo', probability=True, random_state=42),
        'SVM5_MedGaussian':    SVC(kernel='rbf',     gamma='auto',  C=1,   decision_function_shape='ovo', probability=True, random_state=42),

        # ── ANN (4 configs) ────────────────────────────────
        # ANN1: 1 hidden layer (15), tan-sigmoid → tanh
        'ANN1_15_tanh':        MLPClassifier(hidden_layer_sizes=(15,),   activation='tanh',    max_iter=1000, random_state=42),
        # ANN2: 2 hidden layers (15,8), log-sigmoid → logistic
        'ANN2_15x8_logistic':  MLPClassifier(hidden_layer_sizes=(15, 8), activation='logistic',max_iter=1000, random_state=42),
        # ANN3: 2 hidden layers (15,8), tan-sigmoid
        'ANN3_15x8_tanh':      MLPClassifier(hidden_layer_sizes=(15, 8), activation='tanh',    max_iter=4000, random_state=42),
        # ANN4: 2 hidden layers (15,8), tan-sigmoid + logistic output
        'ANN4_15x8_mixed':     MLPClassifier(hidden_layer_sizes=(15, 8), activation='tanh',    max_iter=1000, random_state=42, solver='lbfgs'),

        # ── KNN (5 configs) ────────────────────────────────
        'KNN1_Fine_K1_Eucl':   KNeighborsClassifier(n_neighbors=1,  metric='euclidean'),
        'KNN2_Med_K10_Eucl':   KNeighborsClassifier(n_neighbors=10, metric='euclidean'),
        'KNN3_Med_K10_Cosine': KNeighborsClassifier(n_neighbors=10, metric='cosine'),
        'KNN4_Med_K10_Weight': KNeighborsClassifier(n_neighbors=10, metric='euclidean', weights='distance'),
        'KNN5_Med_K10_Cubic':  KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=3),
         # ── RANDOM FOREST ────────────────────────────────
        'RF_100':
            RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=412
            ),

        'RF_400':
            RandomForestClassifier(
                n_estimators=400,
                random_state=142
            ),

        # ── EXTRA TREES (Ensemble Trees) ─────────────────
        'ExtraTrees_100':
            ExtraTreesClassifier(
                n_estimators=100,
                random_state=42
            ),

        'ExtraTrees_400':
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=42
            ),

        # ── ADABOOST ─────────────────────────────────────
        'AdaBoost_100':
            AdaBoostClassifier(
                n_estimators=100,
                random_state=42
            ),

        'AdaBoost_300':
            AdaBoostClassifier(
                n_estimators=300,
                random_state=42
            ),

        # ── XGBOOST ──────────────────────────────────────
        'XGBoost_100':
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective='multi:softmax',
                num_class=5,
                random_state=42,
                eval_metric='mlogloss'
            ),

        'XGBoost_300':
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                objective='multi:softmax',
                num_class=5,
                random_state=42,
                eval_metric='mlogloss'
            ),

    }

# ════════════════════════════════════════════════════════
# CONFUSION MATRIX PLOT
# ════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, name, classes):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True',
           title=f'Confusion Matrix — {name}')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]}',
                    ha='center', va='center',
                    color='white' if cm[i,j] > thresh else 'black', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'outputs/confusion_matrices/{name}.png', dpi=120)
    plt.close()

# ════════════════════════════════════════════════════════
# ROC CURVE PLOT (per gesture, paper Fig. 4 style)
# ════════════════════════════════════════════════════════
def plot_roc(clf, X_test, y_test, name, n_classes=5):
    y_bin  = label_binarize(y_test, classes=list(range(1, n_classes+1)))
    y_prob = clf.predict_proba(X_test)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'ROC Curves — {name}', fontsize=13)

    for i, ax in enumerate(axes.flat):
        gesture_idx = i
        if gesture_idx >= n_classes: break
        fpr, tpr, _ = roc_curve(y_bin[:, gesture_idx], y_prob[:, gesture_idx])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='red', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0,1],[0,1], 'k--', lw=0.8)
        ax.set(xlim=[0,0.15], ylim=[0.4,1.02],
               xlabel='1 - Specificity', ylabel='Sensitivity',
               title=f'Gesture {i+1}: {GESTURE_NAMES[i+1]}')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'outputs/roc_curves/{name}_ROC.png', dpi=120)
    plt.close()

# ════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ════════════════════════════════════════════════════════
def evaluate_all(X, y):
    classifiers = get_classifiers()

    # Scale features (important for SVM + ANN)
    scaler  = StandardScaler()

    # 70/30 split — stratified (paper protocol)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"\nTrain: {len(X_tr)} events | Test: {len(X_te)} events")
    print(f"{'─'*65}")
    print(f"{'Classifier':<28} {'Train Acc':>10} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>8}")
    print(f"{'─'*65}")

    results = []
    for name, clf in classifiers.items():
        # Fit
        clf.fit(X_tr_s, y_tr)

        # Accuracies
        train_acc = accuracy_score(y_tr, clf.predict(X_tr_s))
        test_acc  = accuracy_score(y_te, clf.predict(X_te_s))

        # 5-fold cross validation on full dataset
        X_s   = scaler.fit_transform(X)
        cv_sc = cross_val_score(clf, X_s, y, cv=5, scoring='accuracy')

        print(f"{name:<28} {train_acc*100:>9.1f}% {test_acc*100:>9.1f}% {cv_sc.mean()*100:>9.1f}% {cv_sc.std()*100:>7.1f}%")

        # Confusion matrix
        cm = confusion_matrix(y_te, clf.predict(X_te_s))
        plot_confusion_matrix(cm, name, [GESTURE_NAMES[i] for i in range(1,6)])

        # ROC curves
        try:
            plot_roc(clf, X_te_s, y_te, name)
        except Exception:
            pass

        results.append({
            'classifier': name,
            'train_accuracy': round(train_acc*100, 2),
            'test_accuracy':  round(test_acc*100,  2),
            'cv_mean':        round(cv_sc.mean()*100, 2),
            'cv_std':         round(cv_sc.std()*100,  2),
        })

    print(f"{'─'*65}")

    # Save results
    res_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
    res_df.to_csv('outputs/classifier_results.csv', index=False)

    print(f"\n{'═'*65}")
    print(f"TOP 5 CLASSIFIERS (by test accuracy):")
    print(f"{'═'*65}")
    print(res_df.head(5).to_string(index=False))

    print(f"\n✅ Results saved: outputs/classifier_results.csv")
    print(f"✅ Confusion matrices: outputs/confusion_matrices/")
    print(f"✅ ROC curves: outputs/roc_curves/")

    return res_df

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║      EMG CLASSIFIER — Arteaga et al. (2020)         ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    X, y, df = load_data()
    results  = evaluate_all(X, y)

    print("\nPer-gesture classification report (best classifier):")
    best_name = results.iloc[0]['classifier']
    classifiers = get_classifiers()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.3, random_state=87642, stratify=y)
    best_clf = classifiers[best_name]
    best_clf.fit(X_tr, y_tr)
    print(classification_report(y_te, best_clf.predict(X_te),
          target_names=[GESTURE_NAMES[i] for i in range(1,6)]))
    


    import joblib
scaler_rt = StandardScaler()
X_s = scaler_rt.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.3, random_state=42, stratify=y)
best_clf = get_classifiers()[results.iloc[0]['classifier']]
best_clf.fit(X_tr, y_tr)
joblib.dump(best_clf, 'best_model.pkl')
joblib.dump(scaler_rt,'scaler.pkl')
