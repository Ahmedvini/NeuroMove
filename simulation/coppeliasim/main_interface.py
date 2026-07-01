"""
main_interface.py
=================
Drop-in integration with your existing main.py / train.py.

HOW TO USE
----------
1. After model.predict() in your evaluate() function, add:

        from main_interface import attach_structured_output
        attach_structured_output(
            y_pred_probs=model.predict(X_test),   # shape (N, 4) softmax
            y_true=y_test_onehot,                  # one-hot (N, 4)
            results_path=results_path,
            model_name=model_name,
            subject_id=1,
        )

2. For k-fold, use attach_structured_output_kfold() — automatically handles
   fold numbering and trial offsets.

3. Run standalone to verify the package on a synthetic signal:
        python main_interface.py

Output files will appear in results_path:
    structured_frames.json  ← all frames as { action, angles, positions }
    structured_frames.csv   ← flat CSV version (easy to load in Excel/pandas)
    structured_summary.json ← per-class statistics
"""

from __future__ import annotations
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

# ── bci_exo package imports ────────────────────────────────────────────────
from bci_exo import (
    BCIDecoder,
    stream_from_model,
    FrameSink,
    CLASS_LABELS,
)


# ---------------------------------------------------------------------------
# Primary integration hook — drop into evaluate()
# ---------------------------------------------------------------------------

def attach_structured_output(
    y_pred_probs: np.ndarray,
    y_true: np.ndarray,
    results_path: str,
    model_name: str = "DB_ATCNet",
    subject_id: Optional[int] = None,
    start_trial: int = 0,
    verbose: bool = True,
) -> FrameSink:
    """
    Convert model softmax outputs into structured frames and save to disk.

    Parameters
    ----------
    y_pred_probs : np.ndarray, shape (N, 4)
        Direct output of model.predict(X_test). Must be softmax probabilities.
    y_true       : np.ndarray, shape (N, 4) one-hot  OR  (N,) integer labels
        Ground-truth labels.
    results_path : str
        Directory where output files will be written (same as your results_path).
    model_name   : str
        e.g. 'DB_ATCNet', 'ATCNet', 'EEGNet'
    subject_id   : int, optional
    start_trial  : int
        Offset for trial IDs (useful when combining folds).
    verbose      : bool

    Returns
    -------
    FrameSink — contains all BCIFrame objects; call .frames to access them.
    """
    sink = FrameSink(os.path.join(results_path, "structured_frames.json"))

    for frame in stream_from_model(
        y_pred_probs,
        model_name=model_name,
        subject_id=subject_id,
        start_trial=start_trial,
    ):
        sink.add(frame)

    # Save JSON + CSV
    sink.save()
    csv_sink = FrameSink(os.path.join(results_path, "structured_frames.csv"))
    for f in sink.frames:
        csv_sink.add(f)
    csv_sink.save()

    # Save summary
    _save_summary(sink, y_true, results_path, verbose)

    return sink


# ---------------------------------------------------------------------------
# K-fold integration hook — drop into train_kfold()
# ---------------------------------------------------------------------------

def attach_structured_output_kfold(
    y_pred_probs: np.ndarray,
    y_true: np.ndarray,
    fold: int,
    results_path: str,
    model_name: str = "DB_ATCNet",
    subject_id: Optional[int] = None,
    trials_per_fold: Optional[int] = None,
) -> FrameSink:
    """
    Like attach_structured_output but scoped to one fold.
    Saves to results_path/fold_{fold}/structured_frames.json

    Parameters
    ----------
    fold : int
        Current fold number (1-based, matching your train_kfold loop).
    trials_per_fold : int, optional
        Number of trials per fold (used to calculate trial_id offset).
    """
    fold_dir = os.path.join(results_path, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    start_trial = ((fold - 1) * trials_per_fold) if trials_per_fold else 0

    return attach_structured_output(
        y_pred_probs=y_pred_probs,
        y_true=y_true,
        results_path=fold_dir,
        model_name=model_name,
        subject_id=subject_id,
        start_trial=start_trial,
    )


# ---------------------------------------------------------------------------
# Internal: summary writer
# ---------------------------------------------------------------------------

def _save_summary(
    sink: FrameSink,
    y_true: np.ndarray,
    results_path: str,
    verbose: bool,
) -> None:
    counts = sink.summary()
    frames = sink.frames

    # Accuracy per class
    if y_true.ndim == 2:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true.astype(int)

    y_pred_labels = np.array([f.class_index for f in frames])
    n = len(frames)

    overall_acc = float(np.mean(y_pred_labels == y_true_labels)) if n > 0 else 0.0

    per_class = {}
    for idx, name in CLASS_LABELS.items():
        mask = y_true_labels == idx
        if mask.sum() == 0:
            per_class[name] = {"n_true": 0, "n_predicted": 0, "accuracy": None}
        else:
            per_class[name] = {
                "n_true": int(mask.sum()),
                "n_predicted": int((y_pred_labels == idx).sum()),
                "accuracy": float(np.mean(y_pred_labels[mask] == idx)),
            }

    summary = {
        "total_trials": n,
        "overall_accuracy": round(overall_acc, 4),
        "predictions_per_class": counts,
        "per_class_stats": per_class,
        "avg_confidence": round(float(np.mean([f.confidence for f in frames])), 4),
    }

    out_path = Path(results_path) / "structured_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        print("\n── Structured Output Summary ──────────────────────────")
        print(f"  Total frames  : {n}")
        print(f"  Overall acc   : {overall_acc:.2%}")
        print(f"  Avg confidence: {summary['avg_confidence']:.2%}")
        print("  Per class:")
        for name, stats in per_class.items():
            if stats["accuracy"] is not None:
                print(f"    {name:<14}: acc={stats['accuracy']:.2%}  "
                      f"(true={stats['n_true']}, pred={stats['n_predicted']})")
        print(f"  Files saved → {results_path}/")
        print("───────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Standalone demo / smoke test
# ---------------------------------------------------------------------------

def _demo():
    """Run a quick end-to-end test with synthetic data (no GPU/dataset needed)."""
    print("=" * 60)
    print("  BCI Exoskeleton Interface — Demo")
    print("=" * 60)

    # Simulate model.predict() output: 50 trials, 4 classes
    rng = np.random.default_rng(42)
    raw_logits = rng.random((50, 4)).astype(np.float32)
    # softmax
    exp = np.exp(raw_logits - raw_logits.max(axis=1, keepdims=True))
    y_pred_probs = exp / exp.sum(axis=1, keepdims=True)

    # Simulate one-hot ground truth
    y_true_idx = rng.integers(0, 4, size=50)
    y_true_onehot = np.eye(4, dtype=np.float32)[y_true_idx]

    print(f"\nSynthetic data: {y_pred_probs.shape} softmax outputs")
    print("\nFirst 3 frames:\n")

    decoder = BCIDecoder(model_name="DB_ATCNet", subject_id=1)
    for i, frame in enumerate(decoder.stream(y_pred_probs)):
        if i >= 3:
            break
        print(frame.to_json())
        print()

    # Save to /tmp
    out_dir = "/tmp/bci_exo_demo"
    os.makedirs(out_dir, exist_ok=True)

    sink = attach_structured_output(
        y_pred_probs=y_pred_probs,
        y_true=y_true_onehot,
        results_path=out_dir,
        model_name="DB_ATCNet",
        subject_id=1,
    )

    print(f"\nSaved files in {out_dir}:")
    for f in Path(out_dir).iterdir():
        print(f"  {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    _demo()
