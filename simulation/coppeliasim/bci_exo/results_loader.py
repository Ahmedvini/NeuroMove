"""
results_loader.py
=================
Loads prediction results produced by your training pipeline (main.py)
from the BCI_2A_Results directory (or any results folder).

Supported formats:
    .npz  — numpy archive (kfold_results.npz, perf_allRuns.npz)
    .csv  — comma-separated predictions
    .json — JSON array of prediction records

The loader returns numpy arrays compatible with BCIDecoder.
"""

from __future__ import annotations
import os
import json
import csv
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


# ---------------------------------------------------------------------------
# NPZ loader (matches your train_kfold output format exactly)
# ---------------------------------------------------------------------------

def load_npz_results(npz_path: str) -> Dict[str, Any]:
    """
    Load a .npz results file produced by train_kfold() in main.py.

    Expected keys in kfold_results.npz:
        accuracies, kappas, avg_acc, std_acc, avg_kappa, std_kappa

    Expected keys in perf_allRuns.npz:
        acc, kappa

    Parameters
    ----------
    npz_path : str
        Path to the .npz file.

    Returns
    -------
    dict with all stored arrays/scalars.
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(path, allow_pickle=True)
    result = {key: data[key] for key in data.files}
    print(f"[results_loader] Loaded {path.name}: keys={list(result.keys())}")
    return result


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv_predictions(
    csv_path: str,
    prob_columns: Optional[List[str]] = None,
    label_column: str = "predicted_class",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load predictions from a CSV file.

    Expected CSV format (flexible):
    ┌──────────┬─────────────────┬────────────┬──────────────────────────────────────────────┐
    │ trial_id │ predicted_class │ true_class │ prob_feet prob_left prob_both_fists prob_right│
    └──────────┴─────────────────┴────────────┴──────────────────────────────────────────────┘

    If prob_columns are present → returns (prob_matrix [N,4], y_true or None)
    If only label_column present → returns (label_vector [N,], y_true or None)

    Parameters
    ----------
    csv_path     : str
        Path to CSV file.
    prob_columns : list of 4 column names (in class-index order), optional.
        e.g. ['prob_feet', 'prob_left', 'prob_both_fists', 'prob_right']
    label_column : str
        Column name for predicted class index.

    Returns
    -------
    predictions : np.ndarray — shape (N, 4) if probs available, else (N,)
    y_true      : np.ndarray or None — shape (N,) if true_class column exists
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    print(f"[results_loader] Loaded {path.name}: {len(rows)} rows, columns={headers}")

    # --- Ground truth ---
    y_true = None
    if "true_class" in headers:
        y_true = np.array([int(r["true_class"]) for r in rows])

    # --- Probabilities ---
    default_prob_cols = ["prob_feet", "prob_left", "prob_both_fists", "prob_right"]
    prob_cols = prob_columns or default_prob_cols

    if all(c in headers for c in prob_cols):
        prob_matrix = np.array(
            [[float(r[c]) for c in prob_cols] for r in rows], dtype=np.float32
        )
        return prob_matrix, y_true

    # --- Fall back to integer labels ---
    if label_column in headers:
        labels = np.array([int(r[label_column]) for r in rows])
        return labels, y_true

    raise ValueError(
        f"CSV has neither probability columns {prob_cols} "
        f"nor label column '{label_column}'. "
        f"Available columns: {headers}"
    )


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

def load_json_predictions(
    json_path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load predictions from a JSON file.

    Expected format — array of objects:
    [
      {
        "trial_id": 0,
        "predicted_class": 3,
        "true_class": 3,          ← optional
        "probabilities": [0.05, 0.10, 0.15, 0.70]   ← optional
      },
      ...
    ]

    Returns
    -------
    predictions : np.ndarray — shape (N, 4) if probabilities key exists, else (N,)
    y_true      : np.ndarray or None
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(path) as f:
        records = json.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"JSON file must contain a non-empty list: {json_path}")

    print(f"[results_loader] Loaded {path.name}: {len(records)} records")

    y_true = None
    if "true_class" in records[0]:
        y_true = np.array([int(r["true_class"]) for r in records])

    if "probabilities" in records[0]:
        prob_matrix = np.array(
            [r["probabilities"] for r in records], dtype=np.float32
        )
        return prob_matrix, y_true

    labels = np.array([int(r["predicted_class"]) for r in records])
    return labels, y_true


# ---------------------------------------------------------------------------
# Auto-detect loader
# ---------------------------------------------------------------------------

def load_results(file_path: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Auto-detect file format and load results.

    Supports: .csv, .json, .npz
    For .npz files, returns (acc_array, kappa_array) as a tuple.

    Parameters
    ----------
    file_path : str
        Path to results file.
    **kwargs  : passed to underlying loader (e.g. prob_columns for CSV).

    Returns
    -------
    (predictions, y_true) for CSV/JSON
    (acc_array, kappa_array) for NPZ
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return load_csv_predictions(file_path, **kwargs)
    elif ext == ".json":
        return load_json_predictions(file_path)
    elif ext == ".npz":
        data = load_npz_results(file_path)
        acc = data.get("accuracies", data.get("acc", np.array([])))
        kappa = data.get("kappas", data.get("kappa", np.array([])))
        return acc, kappa
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Use .csv, .json, or .npz."
        )


# ---------------------------------------------------------------------------
# Folder scanner — finds all result files in BCI_2A_Results/
# ---------------------------------------------------------------------------

def scan_results_folder(folder_path: str) -> Dict[str, List[str]]:
    """
    Scan a results folder and catalogue all prediction files.

    Returns
    -------
    dict with keys: 'npz', 'csv', 'json', each mapping to list of file paths.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Results folder not found: {folder_path}")

    catalogue: Dict[str, List[str]] = {"npz": [], "csv": [], "json": []}
    for f in sorted(folder.rglob("*")):
        ext = f.suffix.lower().lstrip(".")
        if ext in catalogue:
            catalogue[ext].append(str(f))

    total = sum(len(v) for v in catalogue.values())
    print(f"[results_loader] Scanned '{folder_path}': found {total} result files.")
    for ext, files in catalogue.items():
        if files:
            print(f"  .{ext}: {len(files)} file(s)")
    return catalogue
