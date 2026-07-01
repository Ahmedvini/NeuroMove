"""
stream.py
=========
Ties the results loader and decoder together into a frame stream.

Two modes:
    1. File mode   — reads a saved results file (CSV/JSON/NPZ) and emits frames
    2. Live mode   — receives a numpy prob_matrix in real-time (e.g. from your
                     model.predict() call during inference) and emits frames

Both modes yield BCIFrame objects which carry the full structured output:
    { action, angles, positions, confidence, probabilities, ... }
"""

from __future__ import annotations
import time
import json
import csv
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, List, Callable

from .structured_output import BCIFrame
from .decoder import BCIDecoder
from .results_loader import load_results, scan_results_folder


# ---------------------------------------------------------------------------
# Stream from file
# ---------------------------------------------------------------------------

def stream_from_file(
    file_path: str,
    model_name: str = "DB_ATCNet",
    subject_id: Optional[int] = None,
    delay_seconds: float = 0.0,
    on_frame: Optional[Callable[[BCIFrame], None]] = None,
) -> Iterator[BCIFrame]:
    """
    Stream BCIFrame objects from a saved prediction file.

    Parameters
    ----------
    file_path      : path to CSV or JSON predictions file
    model_name     : label for the model that produced these predictions
    subject_id     : Physionet subject number (1-109)
    delay_seconds  : simulate real-time by sleeping N seconds between frames
    on_frame       : optional callback called with each BCIFrame as it is emitted

    Yields
    ------
    BCIFrame
    """
    predictions, y_true = load_results(file_path)
    decoder = BCIDecoder(model_name=model_name, subject_id=subject_id)

    if predictions.ndim == 2 and predictions.shape[1] == 4:
        # Full probability matrix
        frames = decoder.decode_batch(predictions)
    else:
        # Integer label vector
        frames = decoder.decode_from_labels(predictions, y_true=y_true)

    for frame in frames:
        if on_frame is not None:
            on_frame(frame)
        yield frame
        if delay_seconds > 0:
            time.sleep(delay_seconds)


# ---------------------------------------------------------------------------
# Stream from live model output
# ---------------------------------------------------------------------------

def stream_from_model(
    prob_matrix: np.ndarray,
    model_name: str = "DB_ATCNet",
    subject_id: Optional[int] = None,
    start_trial: int = 0,
    on_frame: Optional[Callable[[BCIFrame], None]] = None,
) -> Iterator[BCIFrame]:
    """
    Stream BCIFrame objects from a live numpy probability matrix.

    Plug this directly into your evaluate() / train_kfold() loop:

        y_pred_probs = model.predict(X_test)
        for frame in stream_from_model(y_pred_probs, model_name='DB_ATCNet', subject_id=1):
            print(frame.to_json())

    Parameters
    ----------
    prob_matrix  : np.ndarray, shape (N, 4)
        Softmax output from model.predict()
    model_name   : str
    subject_id   : int, optional
    start_trial  : int, trial counter offset (useful across k-folds)
    on_frame     : optional callback

    Yields
    ------
    BCIFrame
    """
    decoder = BCIDecoder(
        model_name=model_name,
        subject_id=subject_id,
        start_trial=start_trial,
    )
    for frame in decoder.stream(prob_matrix):
        if on_frame is not None:
            on_frame(frame)
        yield frame


# ---------------------------------------------------------------------------
# Stream sink — collect frames + write to output file
# ---------------------------------------------------------------------------

class FrameSink:
    """
    Collects BCIFrame objects and saves them to JSON or CSV.

    Usage
    -----
        sink = FrameSink("results/subject1_frames.json")
        for frame in stream_from_model(y_pred_probs):
            sink.add(frame)
        sink.save()
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self._frames: List[BCIFrame] = []

    def add(self, frame: BCIFrame) -> None:
        self._frames.append(frame)

    def save(self) -> None:
        ext = self.output_path.suffix.lower()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if ext == ".json":
            self._save_json()
        elif ext == ".csv":
            self._save_csv()
        else:
            raise ValueError(f"Unsupported output format: '{ext}'. Use .json or .csv.")

        print(f"[FrameSink] Saved {len(self._frames)} frames → {self.output_path}")

    def _save_json(self) -> None:
        records = [f.to_dict() for f in self._frames]
        with open(self.output_path, "w") as fh:
            json.dump(records, fh, indent=2)

    def _save_csv(self) -> None:
        if not self._frames:
            return
        rows = [f.to_dict() for f in self._frames]

        # Flatten nested dicts (angles, positions, probabilities)
        flat_rows = []
        for r in rows:
            flat = {
                "trial_id": r["trial_id"],
                "subject_id": r["subject_id"],
                "model_name": r["model_name"],
                "action": r["action"],
                "class_index": r["class_index"],
                "confidence": r["confidence"],
            }
            for k, v in r["probabilities"].items():
                flat[f"prob_{k.lower().replace(' ', '_')}"] = v
            flat.update({
                "shoulder_deg": r["angles"]["shoulder_deg"],
                "elbow_deg":    r["angles"]["elbow_deg"],
                "wrist_deg":    r["angles"]["wrist_deg"],
                "grip_deg":     r["angles"]["grip_deg"],
                "pos_x":        r["positions"]["x"],
                "pos_y":        r["positions"]["y"],
                "pos_z":        r["positions"]["z"],
            })
            flat_rows.append(flat)

        headers = list(flat_rows[0].keys())
        with open(self.output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(flat_rows)

    @property
    def frames(self) -> List[BCIFrame]:
        return list(self._frames)

    def summary(self) -> dict:
        """Return per-class trial counts."""
        from collections import Counter
        counts = Counter(f.action for f in self._frames)
        return dict(counts)
