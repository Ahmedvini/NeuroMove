#!/usr/bin/env python3
"""
analyze_mha_entropy.py — Standalone MHA Attention Entropy Analysis for DB-ATCNet

Loads trained Keras weights and extracts attention weights from each of the 5
MultiHeadAttention layers (one per sliding window).  Computes normalised Shannon
entropy and produces summary statistics + three publication-quality plots.

Model layer map (from model.layers inspection):
  [44] 'multi_head_attention'     — window 0
  [45] 'multi_head_attention_1'   — window 1
  [46] 'multi_head_attention_2'   — window 2
  [47] 'multi_head_attention_3'   — window 3
  [48] 'multi_head_attention_4'   — window 4

Each MHA: num_heads=2, key_dim=8
Input to each MHA comes from layer_normalization_{0..4} (indices 39..43).
Attention scores shape per call: (batch, N_HEADS=2, SEQ_LEN=6, SEQ_LEN=6)

Run:  python analyze_mha_entropy.py
      python analyze_mha_entropy.py --dry_run   (prints layer names and exits)
"""

import os
import sys
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir="
    "/home/ezzo/anaconda3/lib/python3.13/site-packages/nvidia/cuda_nvcc"
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# ── Hardcoded configuration ────────────────────────────────────────────────
RESULTS_DIR  = "results/"
DATA_DIR     = "HALT_data/"          # fallback; actual path resolved below
SAVE_DIR     = "results/mha_analysis/"
N_TRIALS     = 100
N_HEADS      = 2
N_WINDOWS    = 5
SEQ_LEN      = 6
FEAT_DIM     = 32                    # eegn_F1 * eegn_D = 16 * 2
RANDOM_SEED  = 42
REP_SUBJECT  = "subject_A"          # representative subject for heatmap
N_CDF        = 1000                  # per-subject subsample for CDF
# ────────────────────────────────────────────────────────────────────────────

# MHA layer names inside the Keras model (one per window)
MHA_LAYER_NAMES = [
    "multi_head_attention",
    "multi_head_attention_1",
    "multi_head_attention_2",
    "multi_head_attention_3",
    "multi_head_attention_4",
]
# Corresponding LayerNormalization layers that feed into each MHA
LN_LAYER_NAMES = [
    "layer_normalization",
    "layer_normalization_1",
    "layer_normalization_2",
    "layer_normalization_3",
    "layer_normalization_4",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: build the model with identical hyperparameters
# ═══════════════════════════════════════════════════════════════════════════
def build_model():
    """Instantiate DB-ATCNet with the exact training hyperparameters."""
    import models
    return models.DB_ATCNet(
        n_classes=2, in_chans=19, in_samples=600,
        eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7,
        eegn_dropout=0.3, drop1=0.35, depth1=2, depth2=4,
        n_windows=5, attention="mha",
        tcn_depth=2, tcn_kernelSize=4, tcn_filters=32,
        tcn_dropout=0.3, drop2=0.1, drop3=0.15, drop4=0.15,
        tcn_activation="elu",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1 — Directory scanning
# ═══════════════════════════════════════════════════════════════════════════
def scan_results(results_dir: str):
    """Discover subject folders and fold sub-folders containing weights.

    Returns nested dict: { subject_name: { fold_name: Path_to_weights } }
    """
    root = Path(results_dir)
    subjects = {}
    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("subject_"):
            continue
        folds = {}
        for fold_dir in sorted(subj_dir.iterdir()):
            if not fold_dir.is_dir():
                continue
            wpath = fold_dir / "best_model.weights.h5"
            if wpath.exists():
                folds[fold_dir.name] = wpath
        if folds:
            subjects[subj_dir.name] = folds
    return subjects


def print_discovered(subjects):
    print("\n╔══ Discovered structure ══════════════════════════════════╗")
    for subj, folds in subjects.items():
        fold_names = list(folds.keys())
        n = len(fold_names)
        unit = "fold" if n == 1 else "folds"
        print(f"  {subj}: {n} {unit} → [{', '.join(fold_names)}]")
    print("╚═════════════════════════════════════════════════════════╝\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3 — Extract attention weights via intermediate Keras model
# ═══════════════════════════════════════════════════════════════════════════
def extract_attention_weights(model, X_batch):
    """Run X_batch through model and return attention weights per window.

    Strategy: build a temporary tf.keras.Model that outputs the inputs to
    each MHA layer (the LayerNorm outputs).  Then call each MHA layer
    manually with return_attention_scores=True.

    Returns list of 5 arrays, each shape (batch, N_HEADS, SEQ_LEN, SEQ_LEN).
    """
    import tensorflow as tf

    # Build sub-model that outputs LayerNorm outputs (MHA inputs)
    ln_outputs = [model.get_layer(ln).output for ln in LN_LAYER_NAMES]
    extractor = tf.keras.Model(inputs=model.input, outputs=ln_outputs)

    ln_vals = extractor(X_batch, training=False)
    if not isinstance(ln_vals, (list, tuple)):
        ln_vals = [ln_vals]

    all_scores = []
    for w_idx in range(N_WINDOWS):
        mha_layer = model.get_layer(MHA_LAYER_NAMES[w_idx])
        ln_val = ln_vals[w_idx]
        _, scores = mha_layer(ln_val, ln_val,
                              return_attention_scores=True, training=False)
        all_scores.append(scores.numpy())  # (batch, heads, seq, seq)
    return all_scores


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5 — Normalised Shannon entropy
# ═══════════════════════════════════════════════════════════════════════════
def normalized_entropy(attn_row):
    """Normalised Shannon entropy of a 1-D attention distribution."""
    H = -np.sum(attn_row * np.log2(attn_row + 1e-9))
    H_max = np.log2(SEQ_LEN)
    return H / H_max


def compute_entropies(attn_weights):
    """Compute per-row normalised entropy for an attention weight array.

    attn_weights: (N_TRIALS, N_HEADS, SEQ_LEN, SEQ_LEN)
    Returns 1-D array of all entropy scalars.
    """
    ents = []
    for t in range(attn_weights.shape[0]):
        for h in range(attn_weights.shape[1]):
            for q in range(attn_weights.shape[2]):
                ents.append(normalized_entropy(attn_weights[t, h, q, :]))
    return np.array(ents)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Load one model, print layer names, and exit")
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow import keras

    # ------------------------------------------------------------------
    #  Dry-run mode
    # ------------------------------------------------------------------
    if args.dry_run:
        print("=== DRY RUN: printing model layer names ===\n")
        model = build_model()
        for i, layer in enumerate(model.layers):
            tag = ""
            if "multi_head" in type(layer).__name__.lower():
                tag = f"  ← MHA  heads={layer.num_heads} key_dim={layer._key_dim}"
            elif "layer_normalization" in layer.name:
                tag = "  ← LN (MHA input)"
            print(f"  [{i:3d}] {layer.name!r:55s} {type(layer).__name__}{tag}")
        print(f"\nTotal parameters: {model.count_params():,}")
        return

    # ------------------------------------------------------------------
    #  Step 1 — Scan directories
    # ------------------------------------------------------------------
    subjects = scan_results(RESULTS_DIR)
    if not subjects:
        sys.exit(f"ERROR: no subject folders found under {RESULTS_DIR}")
    print_discovered(subjects)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Resolve data path (mirrors HALT_main.py logic)
    halt_data_path = os.path.join(os.getcwd(), "HALT")
    if not os.path.isdir(halt_data_path):
        halt_data_path = DATA_DIR
    print(f"Data path: {halt_data_path}\n")

    # Import data loader
    sys.path.insert(0, os.getcwd())
    import HALT_DataLoad
    from HALT_DataLoad import load_halt_subject_by_session, standardize_data

    # ------------------------------------------------------------------
    #  Steps 2-4 — Load model, extract attention weights per subject/fold
    # ------------------------------------------------------------------
    # raw_weights[subject][fold][window] → (N_TRIALS, N_HEADS, SEQ, SEQ)
    raw_weights = {}
    # entropy_vals[subject][fold] → 1-D array of all entropy scalars
    entropy_vals = {}

    for subj_name, folds in subjects.items():
        subj_char = subj_name.replace("subject_", "")
        raw_weights[subj_name] = {}
        entropy_vals[subj_name] = {}

        # Load subject data with session info
        X_all, y_oh, y_labels, session_ids, n_ch = \
            load_halt_subject_by_session(subj_char, halt_data_path)
        if len(X_all) == 0:
            print(f"  ⚠ {subj_name}: no data, skipping")
            continue

        from sklearn.model_selection import LeaveOneGroupOut, train_test_split
        n_sessions = len(np.unique(session_ids))

        if n_sessions == 1:
            _, test_idx = train_test_split(
                np.arange(len(y_labels)), test_size=0.2,
                stratify=y_labels, random_state=42)
            splits = {"train_test_split": test_idx}
        else:
            logo = LeaveOneGroupOut()
            splits = {}
            for train_idx, test_idx in logo.split(X_all, y_labels, session_ids):
                held_session = np.unique(session_ids[test_idx])[0] + 1
                fold_name = f"session_{held_session}_heldout"
                splits[fold_name] = test_idx

        for fold_name, weight_path in folds.items():
            if fold_name not in splits:
                print(f"  ⚠ {subj_name}/{fold_name}: no matching test split, skipping")
                continue
            test_idx = splits[fold_name]

            # ── Step 2: build & load ──
            model = build_model()
            model.load_weights(str(weight_path))
            model.trainable = False
            n_params = model.count_params()
            print(f"  ✓ Loaded {subj_name}/{fold_name}  "
                  f"params={n_params:,}  test_trials={len(test_idx)}")

            # ── Step 4: sample trials ──
            np.random.seed(RANDOM_SEED)
            n_avail = len(test_idx)
            n_sample = min(N_TRIALS, n_avail)
            chosen = np.random.choice(test_idx, size=n_sample, replace=False)

            X_test = X_all[chosen]
            # Standardise using full train split
            train_idx_full = np.setdiff1d(np.arange(len(X_all)), splits[fold_name])
            X_train_std = X_all[train_idx_full].copy()
            X_test_std = X_test.copy()
            X_train_std, X_test_std = standardize_data(X_train_std, X_test_std, n_ch)

            # ── Step 3: extract attention weights ──
            try:
                window_scores = extract_attention_weights(model, X_test_std)
            except Exception as e:
                print(f"  ⚠ {subj_name}/{fold_name}: extraction failed ({e}), skipping")
                keras.backend.clear_session()
                continue

            fold_weights = {}
            ok = True
            for w_idx, scores in enumerate(window_scores):
                expected = (n_sample, N_HEADS, SEQ_LEN, SEQ_LEN)
                if scores.shape != expected:
                    print(f"  ⚠ {subj_name} {fold_name} window_{w_idx}: "
                          f"shape {scores.shape} != expected {expected}, skipping fold")
                    ok = False
                    break
                fold_weights[w_idx] = scores
                print(f"    {subj_name} {fold_name} window_{w_idx}: {scores.shape} ✓")

            if ok:
                raw_weights[subj_name][fold_name] = fold_weights
                # Compute entropies for this fold (all windows concatenated)
                all_ents = []
                for w_idx in range(N_WINDOWS):
                    all_ents.append(compute_entropies(fold_weights[w_idx]))
                entropy_vals[subj_name][fold_name] = np.concatenate(all_ents)

            keras.backend.clear_session()
            del model
            import gc; gc.collect()

    # ------------------------------------------------------------------
    #  Step 5 — Fold-normalised aggregation
    # ------------------------------------------------------------------
    fold_means = {}   # fold_means[subj][fold] → scalar
    subject_means = {}
    for subj in entropy_vals:
        fold_means[subj] = {}
        for fold in entropy_vals[subj]:
            fold_means[subj][fold] = float(np.mean(entropy_vals[subj][fold]))
        if fold_means[subj]:
            subject_means[subj] = float(np.mean(list(fold_means[subj].values())))

    # ------------------------------------------------------------------
    #  Step 6 — Summary table (per subject × window × head)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  ENTROPY SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Subject':<14} {'Folds':>5} {'Window':>6} {'Head':>4}  " \
             f"{'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'Flag':>4}"
    print(header)
    print("-" * 90)

    csv_rows = []
    all_global_ents = []

    for subj in sorted(entropy_vals.keys()):
        n_folds = len(entropy_vals[subj])
        for w_idx in range(N_WINDOWS):
            for h_idx in range(N_HEADS):
                # Collect per-fold mean for this (subj, window, head)
                per_fold = []
                for fold in entropy_vals[subj]:
                    w_data = raw_weights[subj][fold][w_idx]  # (N, H, S, S)
                    ents = []
                    for t in range(w_data.shape[0]):
                        for q in range(SEQ_LEN):
                            ents.append(normalized_entropy(w_data[t, h_idx, q, :]))
                    per_fold.append(np.mean(ents))

                vals = np.array(per_fold)
                mn, sd = vals.mean(), vals.std()
                mi, mx = vals.min(), vals.max()
                flag = "⚠" if mn < 0.90 else ""
                all_global_ents.extend(per_fold)

                row = [subj, n_folds, f"w{w_idx}", f"h{h_idx+1}",
                       f"{mn:.4f}", f"{sd:.4f}", f"{mi:.4f}", f"{mx:.4f}", flag]
                csv_rows.append(row)
                print(f"{subj:<14} {n_folds:>5} {f'w{w_idx}':>6} {f'h{h_idx+1}':>4}  "
                      f"{mn:>7.4f} {sd:>7.4f} {mi:>7.4f} {mx:>7.4f} {flag:>4}")

    all_global_ents = np.array(all_global_ents)
    print("-" * 90)
    print(f"Global mean entropy: {all_global_ents.mean():.4f} "
          f"± {all_global_ents.std():.4f} across all subjects/windows/heads/folds")
    print("=" * 90)

    # Save CSV
    csv_path = os.path.join(SAVE_DIR, "entropy_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Folds", "Window", "Head",
                         "Mean Entropy", "Std", "Min", "Max", "Flag"])
        writer.writerows(csv_rows)
    print(f"  → Saved {csv_path}")

    # ------------------------------------------------------------------
    #  Step 7 — Plot 1: Entropy boxplot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    box_data = []
    box_labels = []
    for subj in sorted(subject_means.keys()):
        vals = list(fold_means[subj].values())
        box_data.append(vals)
        box_labels.append(subj.replace("subject_", "S"))

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                    medianprops=dict(color="white", linewidth=2))
    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1.2, label="Hypothesis threshold")
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=1, label="Theoretical maximum (uniform)")

    for i, line in enumerate(bp["medians"]):
        x, y = line.get_xydata()[1]
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="black")

    ax.set_ylabel("Normalised Entropy")
    ax.set_xlabel("Subject")
    ax.set_title("MHA Normalised Attention Entropy per Subject (fold-normalised)")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    boxplot_path = os.path.join(SAVE_DIR, "mha_entropy_boxplot.png")
    fig.savefig(boxplot_path, dpi=300)
    plt.close(fig)
    print(f"  → Saved {boxplot_path}")

    # ------------------------------------------------------------------
    #  Step 8 — Plot 2: Attention heatmap grid for REP_SUBJECT
    # ------------------------------------------------------------------
    if REP_SUBJECT in raw_weights and raw_weights[REP_SUBJECT]:
        # Average across all trials and all folds
        avg_attn = np.zeros((N_WINDOWS, N_HEADS, SEQ_LEN, SEQ_LEN))
        n_folds_rep = 0
        for fold in raw_weights[REP_SUBJECT]:
            for w_idx in range(N_WINDOWS):
                avg_attn[w_idx] += raw_weights[REP_SUBJECT][fold][w_idx].mean(axis=0)
            n_folds_rep += 1
        avg_attn /= n_folds_rep

        fig, axes = plt.subplots(N_WINDOWS, N_HEADS, figsize=(7, 14))
        for w_idx in range(N_WINDOWS):
            for h_idx in range(N_HEADS):
                ax = axes[w_idx, h_idx]
                mat = avg_attn[w_idx, h_idx]
                im = ax.imshow(mat, vmin=0, vmax=0.35, cmap="YlOrRd", aspect="equal")
                # Annotate with mean entropy
                ent_val = np.mean([normalized_entropy(mat[q, :]) for q in range(SEQ_LEN)])
                ax.set_title(f"W{w_idx} H{h_idx+1}  H={ent_val:.3f}", fontsize=9)
                ax.set_xticks(range(SEQ_LEN))
                ax.set_yticks(range(SEQ_LEN))
                if w_idx == N_WINDOWS - 1:
                    ax.set_xlabel("Key")
                if h_idx == 0:
                    ax.set_ylabel("Query")

        fig.suptitle(
            f"MHA Attention Weights — {REP_SUBJECT}\n"
            f"(averaged across all trials and folds)    "
            f"Uniform baseline = {1/SEQ_LEN:.3f}",
            fontsize=11)
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.025, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(os.path.join(SAVE_DIR, "mha_attention_heatmaps.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Saved {SAVE_DIR}mha_attention_heatmaps.png")
    else:
        print(f"  ⚠ {REP_SUBJECT} not found in results — skipping heatmap plot")

    # ------------------------------------------------------------------
    #  Step 9 — Plot 3: Entropy CDF (equal-weighted across subjects)
    # ------------------------------------------------------------------
    np.random.seed(RANDOM_SEED)
    subsampled = []
    for subj in sorted(entropy_vals.keys()):
        pool = np.concatenate(list(entropy_vals[subj].values()))
        if len(pool) >= N_CDF:
            sub = np.random.choice(pool, size=N_CDF, replace=False)
        else:
            sub = np.random.choice(pool, size=N_CDF, replace=True)
        subsampled.append(sub)
    all_cdf = np.concatenate(subsampled)

    sorted_vals = np.sort(all_cdf)
    cdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    frac_above = np.mean(all_cdf > 0.95) * 100
    global_mean = all_cdf.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_vals, cdf_y, linewidth=2, color="#4C72B0")
    ax.axvline(x=0.95, color="red", linestyle="--", linewidth=1.2,
               label=f"{frac_above:.1f}% > 0.95")
    ax.axvline(x=1.0, color="grey", linestyle="--", linewidth=1)
    ax.axvline(x=global_mean, color="blue", linestyle="--", linewidth=1.2,
               label=f"Global mean = {global_mean:.3f}")
    ax.set_xlabel("Normalised Entropy")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("CDF of MHA Attention Entropy (equal-weighted across all subjects)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "mha_entropy_cdf.png"), dpi=300)
    plt.close(fig)
    print(f"  → Saved {SAVE_DIR}mha_entropy_cdf.png")

    print("\n✓ Analysis complete.\n")


if __name__ == "__main__":
    main()
