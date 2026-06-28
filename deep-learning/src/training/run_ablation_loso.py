"""
run_ablation_loso.py - Direct Retrain Ablation under LOSO (GPU-ready)
=======================================================================
Channel importance is measured as the **mean accuracy drop** across subjects
when each channel is REMOVED and the model is RETRAINED from scratch.

This is the gold-standard approach:
  Importance(c) = mean_subjects[ Acc_baseline(s) - Acc_drop_c(s) ]

  - Baseline: DB-ATCNet trained on all 19 channels under LOSO-CV
  - Ablation: retrain with channel c removed; measure accuracy drop
  - No weight magnitude proxies - every number comes from a real retrained model

Usage:
  # Run ALL subjects (baseline + all ablations):
  python run_ablation_loso.py

  # Run specific subjects only (baseline + ablations for those subjects):
  python run_ablation_loso.py --subjects A B C
"""

import os
import sys

# Forces UTF-8 for Windows Terminal if possible
if sys.platform == "win32":
    import _locale
    _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

# Prevent HDF5 file-locking errors on Windows
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import models
import HALT_DataLoad

# =============================================================================
# 0.  REPRODUCIBILITY & GPU SETUP
# =============================================================================
tf.keras.utils.set_random_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[SUCCESS] GPU detected and configured: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"[WARN] GPU config error: {e}")
else:
    print("[WARN] No GPU found - falling back to CPU (training will be slow).")

# =============================================================================
# 1.  CONFIGURATION
# =============================================================================
ALL_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                'O1',  'O2',  'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                'Fz',  'Cz',  'Pz']
N_CHANNELS = len(ALL_CHANNELS)

DATA_PATH       = "./dataset"
RESULTS_BASE    = "./results/ablation_loso"
ABLATION_DIR    = os.path.join(RESULTS_BASE, "runs")
CHECKPOINT_FILE = os.path.join(RESULTS_BASE, "checkpoint.json")

EXCLUDE_SUBJECTS = ['H', 'I']
TARGET_CLASSES   = [2, 4]

TRAIN_CONF = {
    'batch_size': 16,
    'epochs':     300,
    'patience':   20,
    'lr':         0.0009,
}

MODEL_PARAMS = dict(
    n_classes=2, in_samples=600,
    eegn_F1=16,  eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
    drop1=0.35,  depth1=2, depth2=4, n_windows=5, attention='mha',
    tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
    drop2=0.1,   drop3=0.15, drop4=0.15, tcn_activation='elu'
)

CLASS_LABELS = ['Hand', 'Leg']

# =============================================================================
# 2.  VISUALISATION HELPERS
# =============================================================================
def plot_confusion(cf_matrix, labels, title, out_path):
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                  display_labels=labels)
    disp.plot(cmap='Blues', values_format='.2f', ax=plt.gca())
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_learning_curves(history, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylabel in zip(
            axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(history.history[metric],       label='Train')
        ax.plot(history.history[f'val_{metric}'], label='Val')
        ax.set_title(f'{title} - {ylabel}')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# =============================================================================
# 3.  CORE LOSO TRAINING FUNCTION
# =============================================================================
def _load_subject_binary(subj):
    import scipy.io as sio
    eeg_indices      = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    samples_per_trial = 600

    files = sorted([f for f in os.listdir(DATA_PATH)
                    if f.startswith(f"HaLTSubject{subj}") and f.endswith(".mat")])
    if not files:
        return None, None, None, None

    xs_all, ys_raw, sess_all = [], [], []
    for sess_idx, file in enumerate(files):
        mat = sio.loadmat(os.path.join(DATA_PATH, file),
                          squeeze_me=True, struct_as_record=False)
        o = mat['o']
        data, marker = o.data, o.marker
        for cls in TARGET_CLASSES:
            transitions = np.where((marker[1:] == cls) &
                                   (marker[:-1] != cls))[0]
            if marker[0] == cls:
                transitions = np.insert(transitions, 0, 0)
            for start_idx in transitions:
                search_end = min(start_idx + 50, data.shape[0])
                if search_end > start_idx:
                    sync_offset = np.argmax(data[start_idx:search_end, 21])
                    aligned = start_idx + sync_offset
                else:
                    aligned = start_idx
                end = aligned + samples_per_trial
                if end <= data.shape[0]:
                    xs_all.append(data[aligned:end, eeg_indices].T)
                    ys_raw.append(cls)
                    sess_all.append(sess_idx)

    if not xs_all:
        return None, None, None, None

    X         = np.array(xs_all)
    N_tr, N_ch, _ = X.shape
    X         = X.reshape(N_tr, 1, N_ch, -1)
    ys_raw    = np.array(ys_raw)
    sess_ids  = np.array(sess_all)

    unique_cls = np.unique(ys_raw)
    cls_map    = {c: i for i, c in enumerate(unique_cls)}
    y_int      = np.array([cls_map[c] for c in ys_raw])
    y_oh       = tf.keras.utils.to_categorical(y_int, num_classes=2)

    n_sessions = len(np.unique(sess_ids))
    print(f"  [INFO] Loaded Subject {subj}: {X.shape}, trials={len(y_int)}, sessions={n_sessions}")
    return X, y_int, y_oh, sess_ids

def train_loso_subject(subj, channel_indices, config_label, save_dir):
    n_chans = len(channel_indices)
    X_full, y_int, y_oh, session_ids = _load_subject_binary(subj)
    if X_full is None:
        print(f"  [WARN] Subject {subj}: no trials found, skipping.")
        return None, None

    X = X_full[:, :, channel_indices, :]
    n_sessions = len(np.unique(session_ids))
    if n_sessions < 2:
        print(f"  [INFO] Subject {subj} has only 1 session. Falling back to 80/20 Stratified Split.")
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y_int))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y_int, random_state=42)
        splits = [(train_idx, test_idx)]
        fold_count_label = "1 session"
    else:
        logo   = LeaveOneGroupOut()
        splits = list(logo.split(X, y_int, session_ids))
        fold_count_label = f"{n_sessions} sessions"

    os.makedirs(save_dir, exist_ok=True)
    fold_accs, fold_kappas, fold_times = [], [], []
    start_subj = time.time()

    print(f"\nSubject {subj} - DB_ATCNet - CV ({fold_count_label})")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        start_fold = time.time()
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te_oh, y_true = y_oh[train_idx], y_oh[test_idx], y_int[test_idx]

        for j in range(n_chans):
            sc = StandardScaler()
            X_tr[:, 0, j, :] = sc.fit_transform(X_tr[:, 0, j, :])
            X_te[:, 0, j, :] = sc.transform(X_te[:, 0, j, :])

        fold_dir     = os.path.join(save_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        weights_path = os.path.join(fold_dir, f"best.weights.h5")

        tf.keras.utils.set_random_seed(42)
        model = models.DB_ATCNet(in_chans=n_chans, **MODEL_PARAMS)
        model.compile(optimizer=Adam(learning_rate=TRAIN_CONF['lr']),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            ModelCheckpoint(weights_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=0),
            EarlyStopping(monitor='val_accuracy', patience=TRAIN_CONF['patience'], verbose=0),
        ]

        history = model.fit(X_tr, y_tr, validation_data=(X_te, y_te_oh),
                            epochs=TRAIN_CONF['epochs'], batch_size=TRAIN_CONF['batch_size'],
                            callbacks=callbacks, verbose=0)

        model.load_weights(weights_path)
        y_pred    = model.predict(X_te, verbose=0).argmax(axis=-1)
        acc       = accuracy_score(y_true, y_pred)
        kappa     = cohen_kappa_score(y_true, y_pred)
        fold_time = (time.time() - start_fold) / 60

        fold_accs.append(acc)
        fold_kappas.append(kappa)
        fold_times.append(fold_time)

        plot_confusion(confusion_matrix(y_true, y_pred, normalize='pred'), 
                       CLASS_LABELS, f"Subj {subj} Fold {fold}", os.path.join(fold_dir, 'confusion.png'))
        
        print(f"  Subject {subj} Fold {fold}: Acc={acc:.5f}  Kappa={kappa:.5f}  Time={fold_time:.2f}min")
        keras.backend.clear_session()

    avg_acc = float(np.mean(fold_accs))
    summary = (
        f"\n{'=' * 60}\n"
        f"Subject {subj} Summary (LOSO, {n_sessions} sessions)\n"
        f"  Per-session Acc:   {[f'{a:.4f}' for a in fold_accs]}\n"
        f"  Per-session Kappa: {[f'{k:.4f}' for k in fold_kappas]}\n"
        f"  Avg Acc:   {avg_acc:.5f} +/- {np.std(fold_accs):.5f}\n"
        f"  Avg Kappa: {np.mean(fold_kappas):.5f} +/- {np.std(fold_kappas):.5f}\n"
        f"  Time: {(time.time() - start_subj)/60:.2f} min\n"
        f"{'=' * 60}\n"
    )
    print(summary)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f: f.write(summary)
    return avg_acc, {"avg_acc": avg_acc, "fold_accs": fold_accs, "fold_kappas": fold_kappas}

# =============================================================================
# 4.  MAIN EXECUTION
# =============================================================================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f: return json.load(f)
    return {"baseline": {}, "ablations": {}}

def save_checkpoint(data):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f: json.dump(data, f, indent=2)

def run_study(subjects, step='all', target_ch=None, force=False):
    os.makedirs(ABLATION_DIR, exist_ok=True)
    checkpoint = load_checkpoint()

    if step in ('all', 'baseline'):
        print("\n" + "#"*70 + "\n  STEP 1: Baseline (All Channels)\n" + "#"*70)
        baseline_dir = os.path.join(ABLATION_DIR, "baseline_19ch")
        for subj in subjects:
            if not force and f"subj_{subj}" in checkpoint["baseline"]: continue
            acc, res = train_loso_subject(subj, list(range(N_CHANNELS)), "Baseline", os.path.join(baseline_dir, f"subject_{subj}"))
            if res:
                checkpoint["baseline"][f"subj_{subj}"] = res
                save_checkpoint(checkpoint)

    if step == 'baseline': return

    channels = [target_ch] if target_ch else ALL_CHANNELS
    for ch_idx, ch_name in enumerate(ALL_CHANNELS):
        if ch_name not in channels: continue
        label = f"drop_{ch_name}"
        print("\n" + "#"*70 + f"\n  ABLATING '{ch_name}'\n" + "#"*70)
        if label not in checkpoint["ablations"]: checkpoint["ablations"][label] = {}
        
        for subj in subjects:
            if not force and f"subj_{subj}" in checkpoint["ablations"][label]: continue
            keep = [i for i in range(N_CHANNELS) if i != ch_idx]
            acc, res = train_loso_subject(subj, keep, label, os.path.join(ABLATION_DIR, label, f"subject_{subj}"))
            if res:
                checkpoint["ablations"][label][f"subj_{subj}"] = res
                save_checkpoint(checkpoint)
                # Analyze after each subject so we can see progress
                analyze(checkpoint)
    
    print("\n" + "#"*70 + "\n  ABLATION STUDY COMPLETE\n" + "#"*70)
    analyze(checkpoint)

def analyze(checkpoint):
    baseline = checkpoint.get("baseline", {})
    ablations = checkpoint.get("ablations", {})
    if not baseline or not ablations: return
    
    subjs = sorted([s.replace("subj_", "") for s in baseline.keys()])
    rows = []
    for ch in ALL_CHANNELS:
        label = f"drop_{ch}"
        if label not in ablations: continue
        drops = []
        for s in subjs:
            k = f"subj_{s}"
            if k in baseline and k in ablations[label]:
                drops.append(baseline[k]["avg_acc"] - ablations[label][k]["avg_acc"])
        if drops:
            rows.append({"Channel": ch, "Mean_Drop": np.mean(drops), "Std_Drop": np.std(drops)})

    df = pd.DataFrame(rows).sort_values("Mean_Drop", ascending=False)
    df.insert(0, "Rank", range(1, len(df)+1))
    path = os.path.join(RESULTS_BASE, "channel_ranking_ablation.csv")
    df.to_csv(path, index=False)
    print(f"\n[DATA] Results saved to {path}\n", df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+')
    parser.add_argument('--step', choices=['all', 'baseline', 'ablate'], default='all')
    parser.add_argument('--channel')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    print("[DEBUG] Getting available subjects...")
    all_avail = HALT_DataLoad.get_available_subjects(DATA_PATH)
    print(f"[DEBUG] Found subjects: {all_avail}")
    if args.subjects:
        subjects = [s.upper() for s in args.subjects if s.upper() in all_avail and s.upper() not in EXCLUDE_SUBJECTS]
    else:
        subjects = [s for s in all_avail if s not in EXCLUDE_SUBJECTS]
    
    print(f"[DEBUG] Starting study for subjects: {subjects}")
    run_study(subjects, args.step, args.channel, args.force)
