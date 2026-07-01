"""
decoder.py
==========
Converts raw model predictions (numpy softmax arrays) into BCIFrame objects.

This module is the bridge between your model output in main.py / train.py
and the structured { action, angles, positions } interface.

Typical usage after model.predict():
--------------------------------------
    from bci_exo.decoder import BCIDecoder

    decoder = BCIDecoder(model_name="DB_ATCNet", subject_id=1)

    y_pred_probs = model.predict(X_test)          # shape: (N, 4)
    frames = decoder.decode_batch(y_pred_probs)   # list of BCIFrame

    for frame in frames:
        print(frame.to_json())
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Iterator

from .structured_output import BCIFrame, CLASS_LABELS
from .kinematics import map_class_to_kinematics


class BCIDecoder:
    """
    Decodes model softmax output into structured BCIFrame objects.

    Parameters
    ----------
    model_name  : str
        Name of the model that produced predictions (e.g. 'DB_ATCNet').
    subject_id  : int, optional
        Subject number from Physionet dataset (1-109, excluding invalid ones).
    start_trial : int
        Starting trial index (default 0). Useful when streaming fold-by-fold.
    """

    def __init__(
        self,
        model_name: str = "DB_ATCNet",
        subject_id: Optional[int] = None,
        start_trial: int = 0,
    ):
        self.model_name = model_name
        self.subject_id = subject_id
        self._trial_counter = start_trial

    # ------------------------------------------------------------------
    # Core decode methods
    # ------------------------------------------------------------------

    def decode_single(self, prob_vector: np.ndarray) -> BCIFrame:
        """
        Decode one softmax output vector into a BCIFrame.

        Parameters
        ----------
        prob_vector : np.ndarray, shape (4,)
            Softmax probabilities for [Both Feet, Left Fist, Both Fists, Right Fist].

        Returns
        -------
        BCIFrame
        """
        if prob_vector.ndim != 1 or len(prob_vector) != 4:
            raise ValueError(
                f"Expected a 1-D vector of length 4, got shape {prob_vector.shape}."
            )

        class_index = int(np.argmax(prob_vector))
        confidence = float(prob_vector[class_index])
        action = CLASS_LABELS[class_index]
        angles, position = map_class_to_kinematics(class_index)

        frame = BCIFrame(
            trial_id=self._trial_counter,
            action=action,
            class_index=class_index,
            confidence=confidence,
            probabilities=prob_vector.tolist(),
            angles=angles,
            positions=position,
            subject_id=self.subject_id,
            model_name=self.model_name,
        )
        self._trial_counter += 1
        return frame

    def decode_batch(self, prob_matrix: np.ndarray) -> List[BCIFrame]:
        """
        Decode a batch of softmax outputs.

        Parameters
        ----------
        prob_matrix : np.ndarray, shape (N, 4)
            N softmax probability vectors.

        Returns
        -------
        List[BCIFrame]
        """
        if prob_matrix.ndim == 1:
            # single sample passed without batch dimension
            return [self.decode_single(prob_matrix)]

        if prob_matrix.ndim != 2 or prob_matrix.shape[1] != 4:
            raise ValueError(
                f"Expected shape (N, 4), got {prob_matrix.shape}."
            )

        return [self.decode_single(row) for row in prob_matrix]

    def stream(self, prob_matrix: np.ndarray) -> Iterator[BCIFrame]:
        """
        Generator — yields one BCIFrame at a time from a probability matrix.
        Useful for simulating real-time streaming from saved results.

        Example
        -------
            for frame in decoder.stream(y_pred_probs):
                send_to_exoskeleton(frame.angles)
        """
        for row in prob_matrix:
            yield self.decode_single(row)

    # ------------------------------------------------------------------
    # Convenience: decode from argmax labels (no probabilities available)
    # ------------------------------------------------------------------

    def decode_from_labels(
        self,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> List[BCIFrame]:
        """
        Decode integer class labels (argmax already applied).
        Used when you only have y_pred from the results files.

        Parameters
        ----------
        y_pred : np.ndarray, shape (N,)
            Predicted class indices (0-3).
        y_true : np.ndarray, shape (N,), optional
            Ground truth class indices. Stored in frame.probabilities as
            one-hot if provided.

        Returns
        -------
        List[BCIFrame]
        """
        frames = []
        for i, cls in enumerate(y_pred):
            # Build a pseudo-probability vector: 1.0 at predicted class
            probs = np.zeros(4, dtype=float)
            probs[int(cls)] = 1.0

            frame = self.decode_single(probs)

            # Tag ground truth if available
            if y_true is not None:
                frame.probabilities = {
                    "predicted": int(cls),
                    "ground_truth": int(y_true[i]),
                    "correct": bool(int(cls) == int(y_true[i])),
                }
            frames.append(frame)
        return frames

    def reset(self, start_trial: int = 0) -> None:
        """Reset the trial counter (e.g. between folds)."""
        self._trial_counter = start_trial
