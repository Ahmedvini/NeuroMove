"""
pipeline.py
===========
Milestone 5 automation controller for:
    inference -> mapping -> simulation

This module orchestrates the full execution flow and centralizes error
management so the pipeline can run without manual intervention.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np

from bci_exo.decoder import BCIDecoder
from bci_exo.mapper import ControlSignal, class_to_control_signal, map_to_control_signal
from bci_exo.structured_output import BCIFrame
from bci_exo.stream import stream_from_file


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Base exception for pipeline stage failures."""


class InferenceStageError(PipelineError):
    """Raised when inference stage fails."""


class MappingStageError(PipelineError):
    """Raised when mapping stage fails."""


class SimulationStageError(PipelineError):
    """Raised when simulation stage fails."""


@dataclass
class PipelineConfig:
    """Configuration for full automation pipeline execution."""

    model_name: str = "DB_ATCNet"
    subject_id: Optional[int] = None
    start_trial: int = 0

    confidence_threshold: float = 0.0
    continue_on_frame_error: bool = False
    dry_run: bool = False

    sim: Optional[Any] = None
    feedback: Optional["FeedbackConfig"] = None


@dataclass
class FeedbackConfig:
    """Optional closed-loop feedback settings for simulation stage."""

    enabled: bool = False
    max_cycles_per_signal: int = 3
    correction_gain: float = 0.5
    error_tolerance_deg: float = 3.0
    collect_samples: bool = True


@dataclass
class PipelineRunResult:
    """Execution summary for one pipeline run."""

    frames_inferred: int = 0
    frames_mapped: int = 0
    signals_applied: int = 0
    frames_skipped_low_confidence: int = 0
    frame_errors: int = 0
    feedback_enabled: bool = False
    feedback_iterations: int = 0
    feedback_samples: List[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class PipelineController:
    """
    End-to-end pipeline controller:
        inference -> mapping -> simulation
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.decoder = BCIDecoder(
            model_name=self.cfg.model_name,
            subject_id=self.cfg.subject_id,
            start_trial=self.cfg.start_trial,
        )
        self.sim_controller = None

    # ------------------------------------------------------------------
    # Inference stage
    # ------------------------------------------------------------------

    def infer_from_probabilities(self, prob_matrix: np.ndarray) -> List[BCIFrame]:
        """Inference stage from model softmax outputs (shape N x 4)."""
        try:
            arr = np.asarray(prob_matrix)
            frames = self.decoder.decode_batch(arr)
            log.info("Inference stage complete: %d frames", len(frames))
            return frames
        except Exception as exc:
            raise InferenceStageError(f"Inference from probabilities failed: {exc}") from exc

    def infer_from_labels(
        self,
        labels: Sequence[int],
        confidences: Optional[Sequence[float]] = None,
    ) -> List[ControlSignal]:
        """
        Inference shortcut when only predicted class IDs are available.

        This returns ControlSignal objects directly because labels do not carry
        enough data to build complete model probability vectors reliably.
        """
        try:
            signals: List[ControlSignal] = []
            for i, label in enumerate(labels):
                conf = 1.0 if confidences is None else float(confidences[i])
                sig = class_to_control_signal(
                    class_index=int(label),
                    confidence=conf,
                    trial_id=self.cfg.start_trial + i,
                )
                signals.append(sig)
            log.info("Inference stage complete from labels: %d signals", len(signals))
            return signals
        except Exception as exc:
            raise InferenceStageError(f"Inference from labels failed: {exc}") from exc

    def infer_from_results_file(self, file_path: str) -> List[BCIFrame]:
        """Inference stage from a saved CSV/JSON results file."""
        try:
            frames = list(
                stream_from_file(
                    file_path=file_path,
                    model_name=self.cfg.model_name,
                    subject_id=self.cfg.subject_id,
                )
            )
            log.info("Inference stage complete from file: %d frames", len(frames))
            return frames
        except Exception as exc:
            raise InferenceStageError(f"Inference from file failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Mapping stage
    # ------------------------------------------------------------------

    def map_frames(self, frames: Sequence[BCIFrame], result: PipelineRunResult) -> List[ControlSignal]:
        """Mapping stage from decoded BCIFrame objects to ControlSignal objects."""
        try:
            signals: List[ControlSignal] = []
            for frame in frames:
                if frame.confidence < self.cfg.confidence_threshold:
                    result.frames_skipped_low_confidence += 1
                    continue
                signals.append(map_to_control_signal(frame))
            log.info(
                "Mapping stage complete: %d mapped, %d skipped below confidence threshold",
                len(signals),
                result.frames_skipped_low_confidence,
            )
            return signals
        except Exception as exc:
            raise MappingStageError(f"Mapping failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Simulation stage
    # ------------------------------------------------------------------

    def run_simulation(self, signals: Sequence[ControlSignal], result: PipelineRunResult) -> None:
        """Simulation stage that streams mapped control signals to CoppeliaSim."""
        if self.cfg.dry_run:
            result.signals_applied = len(signals)
            log.info("Dry run enabled: skipped simulation stage (%d signals)", len(signals))
            return

        connected = False
        started = False

        try:
            if self.sim_controller is None:
                try:
                    from sim_controller import SimConfig, SimController
                except ModuleNotFoundError as exc:
                    raise SimulationStageError(
                        "Simulation dependencies are missing. Install coppeliasim_zmqremoteapi_client "
                        "or run with dry_run=True."
                    ) from exc

                sim_cfg = self.cfg.sim if isinstance(self.cfg.sim, SimConfig) else SimConfig()
                self.sim_controller = SimController(sim_cfg)

            self.sim_controller.connect()
            connected = True
            self.sim_controller.start_simulation()
            started = True

            feedback_cfg = self.cfg.feedback or FeedbackConfig(enabled=False)
            result.feedback_enabled = bool(feedback_cfg.enabled)

            if feedback_cfg.enabled:
                self._run_simulation_with_feedback(signals, result, feedback_cfg)
                log.info(
                    "Simulation feedback loop complete: %d commands applied, %d feedback iterations",
                    result.signals_applied,
                    result.feedback_iterations,
                )
                return

            for signal in signals:
                try:
                    self.sim_controller.apply_signal(signal)
                    result.signals_applied += 1
                except Exception as frame_exc:
                    result.frame_errors += 1
                    if not self.cfg.continue_on_frame_error:
                        raise SimulationStageError(
                            f"Simulation failed while applying trial {signal.trial_id}: {frame_exc}"
                        ) from frame_exc
                    log.error(
                        "Frame error at trial_id=%s (continuing): %s",
                        signal.trial_id,
                        frame_exc,
                    )

            log.info("Simulation stage complete: %d signals applied", result.signals_applied)

        except PipelineError:
            raise
        except Exception as exc:
            raise SimulationStageError(f"Simulation stage failed: {exc}") from exc
        finally:
            # Always attempt clean shutdown when any part of simulation was reached.
            if started:
                try:
                    self.sim_controller.stop_simulation()
                except Exception as stop_exc:
                    log.warning("Failed to stop simulation cleanly: %s", stop_exc)
            if connected:
                try:
                    self.sim_controller.disconnect()
                except Exception as disc_exc:
                    log.warning("Failed to disconnect cleanly: %s", disc_exc)

    def _run_simulation_with_feedback(
        self,
        signals: Sequence[ControlSignal],
        result: PipelineRunResult,
        feedback_cfg: FeedbackConfig,
    ) -> None:
        """
        Closed-loop variant:
            apply command -> observe state -> correct command (bounded) -> repeat.

        This stays compatible with SimController by sending objects that implement
        `to_flat_dict()` and expose a `.position` attribute.
        """

        class _PreparedSignal:
            def __init__(self, base: ControlSignal, sh_deg: float, el_deg: float, wr_deg: float):
                self.position = base.position
                self.trial_id = base.trial_id
                self._flat = {
                    "shoulder_pitch_deg": sh_deg,
                    "elbow_pitch_deg": el_deg,
                    "wrist_roll_deg": wr_deg,
                }

            def to_flat_dict(self) -> dict:
                return dict(self._flat)

        if self.sim_controller is None:
            raise SimulationStageError("Simulation controller is not initialised")

        # Joint names are aligned with sim_controller defaults.
        shoulder_name = "/exo_joint_shoulder"
        elbow_name = "/exo_joint_elbow"
        wrist_name = "/exo_joint_wrist"

        for signal in signals:
            base = signal.to_flat_dict()
            target_sh = float(base.get("shoulder_pitch_deg", 0.0))
            target_el = float(base.get("elbow_pitch_deg", 0.0))
            target_wr = float(base.get("wrist_roll_deg", 0.0))

            cmd_sh, cmd_el, cmd_wr = target_sh, target_el, target_wr

            for cycle in range(max(1, int(feedback_cfg.max_cycles_per_signal))):
                try:
                    prepared = _PreparedSignal(signal, cmd_sh, cmd_el, cmd_wr)
                    self.sim_controller.apply_signal(prepared)
                    result.signals_applied += 1
                    result.feedback_iterations += 1

                    joints_rad = self.sim_controller.get_joint_positions()
                    obs_sh = math.degrees(float(joints_rad.get(shoulder_name, 0.0)))
                    obs_el = math.degrees(float(joints_rad.get(elbow_name, 0.0)))
                    obs_wr = math.degrees(float(joints_rad.get(wrist_name, 0.0)))

                    err_sh = target_sh - obs_sh
                    err_el = target_el - obs_el
                    err_wr = target_wr - obs_wr
                    mae = (abs(err_sh) + abs(err_el) + abs(err_wr)) / 3.0

                    if feedback_cfg.collect_samples:
                        ee_pos = self.sim_controller.get_ee_position()
                        result.feedback_samples.append(
                            {
                                "trial_id": signal.trial_id,
                                "cycle": cycle,
                                "target": {
                                    "shoulder_pitch_deg": round(target_sh, 4),
                                    "elbow_pitch_deg": round(target_el, 4),
                                    "wrist_roll_deg": round(target_wr, 4),
                                },
                                "commanded": {
                                    "shoulder_pitch_deg": round(cmd_sh, 4),
                                    "elbow_pitch_deg": round(cmd_el, 4),
                                    "wrist_roll_deg": round(cmd_wr, 4),
                                },
                                "observed": {
                                    "shoulder_pitch_deg": round(obs_sh, 4),
                                    "elbow_pitch_deg": round(obs_el, 4),
                                    "wrist_roll_deg": round(obs_wr, 4),
                                },
                                "mean_abs_error_deg": round(mae, 4),
                                "ee_position": [round(float(v), 6) for v in ee_pos] if ee_pos else None,
                            }
                        )

                    if mae <= float(feedback_cfg.error_tolerance_deg):
                        break

                    gain = max(0.0, min(1.0, float(feedback_cfg.correction_gain)))
                    cmd_sh = cmd_sh + gain * err_sh
                    cmd_el = cmd_el + gain * err_el
                    cmd_wr = cmd_wr + gain * err_wr

                    # Clamp to conservative biomechanical ranges in degrees.
                    cmd_sh = max(-90.0, min(135.0, cmd_sh))
                    cmd_el = max(0.0, min(135.0, cmd_el))
                    cmd_wr = max(-80.0, min(80.0, cmd_wr))

                except Exception as frame_exc:
                    result.frame_errors += 1
                    if not self.cfg.continue_on_frame_error:
                        raise SimulationStageError(
                            f"Simulation feedback failed at trial {signal.trial_id}, cycle {cycle}: {frame_exc}"
                        ) from frame_exc
                    log.error(
                        "Feedback frame error at trial_id=%s cycle=%s (continuing): %s",
                        signal.trial_id,
                        cycle,
                        frame_exc,
                    )

    # ------------------------------------------------------------------
    # End-to-end runs
    # ------------------------------------------------------------------

    def run_from_probabilities(self, prob_matrix: np.ndarray) -> PipelineRunResult:
        """Run full pipeline from model softmax output matrix."""
        result = PipelineRunResult()
        t0 = time.monotonic()

        frames = self.infer_from_probabilities(prob_matrix)
        result.frames_inferred = len(frames)

        signals = self.map_frames(frames, result)
        result.frames_mapped = len(signals)

        self.run_simulation(signals, result)

        result.elapsed_seconds = round(time.monotonic() - t0, 4)
        return result

    def run_from_labels(
        self,
        labels: Sequence[int],
        confidences: Optional[Sequence[float]] = None,
    ) -> PipelineRunResult:
        """Run full pipeline from class labels (inference shortcut)."""
        result = PipelineRunResult()
        t0 = time.monotonic()

        signals = self.infer_from_labels(labels, confidences)
        result.frames_inferred = len(signals)
        result.frames_mapped = len(signals)

        self.run_simulation(signals, result)

        result.elapsed_seconds = round(time.monotonic() - t0, 4)
        return result

    def run_from_results_file(self, file_path: str) -> PipelineRunResult:
        """Run full pipeline from a saved predictions CSV/JSON file."""
        result = PipelineRunResult()
        t0 = time.monotonic()

        frames = self.infer_from_results_file(file_path)
        result.frames_inferred = len(frames)

        signals = self.map_frames(frames, result)
        result.frames_mapped = len(signals)

        self.run_simulation(signals, result)

        result.elapsed_seconds = round(time.monotonic() - t0, 4)
        return result


if __name__ == "__main__":
    # Minimal no-manual-intervention smoke run.
    rng = np.random.default_rng(7)
    logits = rng.random((20, 4), dtype=np.float32)
    probs = logits / logits.sum(axis=1, keepdims=True)

    cfg = PipelineConfig(
        model_name="DB_ATCNet",
        subject_id=1,
        confidence_threshold=0.0,
        continue_on_frame_error=False,
        dry_run=True,
    )

    pipeline = PipelineController(cfg)
    summary = pipeline.run_from_probabilities(probs)
    print("Pipeline summary:", summary)
