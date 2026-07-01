"""
api.py
======
Milestone 6: FastAPI layer for online execution.

Endpoints
---------
POST /run       Full pipeline: input -> inference -> mapping -> simulation
POST /infer     Inference only, returns raw predictions
POST /map       Mapping only, returns simulation-ready control values
POST /simulate  Simulation only, applies prepared control values
GET  /models    Lists available model names
"""

from __future__ import annotations

import base64
import logging
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bci_exo.decoder import BCIDecoder
from bci_exo.mapper import class_to_control_signal, map_to_control_signal
from bci_exo.results_loader import load_results
from pipeline import FeedbackConfig, PipelineConfig, PipelineController, PipelineError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="BCI Online Execution API",
    description="Run the BCI pipeline online (inference -> mapping -> simulation).",
    version="1.1.0",
)

AVAILABLE_MODELS = ["DB_ATCNet", "ATCNet", "EEGNet"]


# -----------------------------------------------------------------------------
# Request models
# -----------------------------------------------------------------------------

class SimulationOptions(BaseModel):
    enabled: bool = False
    host: str = "localhost"
    port: int = 23000
    loop_hz: float = 20.0
    ready_timeout: float = 8.0


class FeedbackOptions(BaseModel):
    enabled: bool = False
    max_cycles_per_signal: int = 3
    correction_gain: float = 0.5
    error_tolerance_deg: float = 3.0
    collect_samples: bool = True


class RunRequest(BaseModel):
    input_type: Literal["dataset", "image", "parameters"] = "parameters"

    dataset_path: Optional[str] = None
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

    model_name: str = "DB_ATCNet"
    subject_id: Optional[int] = None
    start_trial: int = 0
    confidence_threshold: float = 0.0
    continue_on_frame_error: bool = False

    simulation: SimulationOptions = Field(default_factory=SimulationOptions)
    feedback: FeedbackOptions = Field(default_factory=FeedbackOptions)
    async_mode: bool = False


class InferRequest(BaseModel):
    input_type: Literal["dataset", "image", "parameters"] = "parameters"
    dataset_path: Optional[str] = None
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MapRequest(BaseModel):
    model_name: str = "DB_ATCNet"
    subject_id: Optional[int] = None
    start_trial: int = 0

    probabilities: Optional[List[List[float]]] = None
    labels: Optional[List[int]] = None
    confidences: Optional[List[float]] = None


class ControlValue(BaseModel):
    shoulder_pitch_deg: float
    elbow_pitch_deg: float
    wrist_roll_deg: float
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0


class SimulateRequest(BaseModel):
    control_values: List[ControlValue] = Field(default_factory=list)
    continue_on_frame_error: bool = False
    simulation: SimulationOptions = Field(default_factory=lambda: SimulationOptions(enabled=True))


# -----------------------------------------------------------------------------
# Async job store (for /run async mode)
# -----------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_job(job_id: str, patch: Dict[str, Any]) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(patch)


def _get_job(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return dict(job)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _build_pipeline_config(req: RunRequest) -> PipelineConfig:
    sim_cfg: Any = None
    if req.simulation.enabled:
        try:
            from sim_controller import SimConfig
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Simulation was enabled but dependency is missing. "
                    "Install coppeliasim-zmqremoteapi-client or disable simulation."
                ),
            ) from exc

        sim_cfg = SimConfig(
            host=req.simulation.host,
            port=req.simulation.port,
            loop_hz=req.simulation.loop_hz,
            ready_timeout=req.simulation.ready_timeout,
        )

    return PipelineConfig(
        model_name=req.model_name,
        subject_id=req.subject_id,
        start_trial=req.start_trial,
        confidence_threshold=req.confidence_threshold,
        continue_on_frame_error=req.continue_on_frame_error,
        dry_run=(not req.simulation.enabled),
        sim=sim_cfg,
        feedback=FeedbackConfig(
            enabled=req.feedback.enabled,
            max_cycles_per_signal=req.feedback.max_cycles_per_signal,
            correction_gain=req.feedback.correction_gain,
            error_tolerance_deg=req.feedback.error_tolerance_deg,
            collect_samples=req.feedback.collect_samples,
        ),
    )


def _generate_probs_from_parameters(params: Dict[str, Any]) -> np.ndarray:
    num_samples = int(params.get("num_samples", 20))
    seed = int(params.get("seed", 7))

    if num_samples <= 0:
        raise ValueError("parameters.num_samples must be > 0")

    rng = np.random.default_rng(seed)
    logits = rng.random((num_samples, 4), dtype=np.float32)
    return logits / logits.sum(axis=1, keepdims=True)


def _ensure_image_payload(image_path: Optional[str], image_base64: Optional[str]) -> None:
    if image_path:
        if not Path(image_path).exists():
            raise ValueError(f"image_path does not exist: {image_path}")
        return

    if image_base64:
        try:
            base64.b64decode(image_base64, validate=True)
            return
        except Exception as exc:
            raise ValueError("image_base64 is not valid base64") from exc

    raise ValueError("Provide image_path or image_base64 for image input")


def _infer_raw(req: InferRequest) -> Dict[str, Any]:
    if req.input_type == "dataset":
        if not req.dataset_path:
            raise ValueError("dataset_path is required for input_type='dataset'")

        pred, aux = load_results(req.dataset_path)
        arr = np.asarray(pred)

        return {
            "status": "completed",
            "input_type": "dataset",
            "predictions": arr.tolist(),
            "shape": list(arr.shape),
            "auxiliary": np.asarray(aux).tolist() if aux is not None else None,
        }

    if req.input_type == "image":
        _ensure_image_payload(req.image_path, req.image_base64)

        class_index = req.parameters.get("class_index")
        if class_index is None:
            raise ValueError(
                "Image inference model is not implemented in this repository. "
                "Provide parameters.class_index (0..3)."
            )
        confidence = float(req.parameters.get("confidence", 1.0))
        confidence = max(0.0, min(1.0, confidence))

        probs = [0.0, 0.0, 0.0, 0.0]
        probs[int(class_index)] = confidence

        return {
            "status": "completed",
            "input_type": "image",
            "predictions": [probs],
            "shape": [1, 4],
            "note": "Class hint used because image model is not part of this repository.",
        }

    # parameters mode
    if "probabilities" in req.parameters:
        arr = np.asarray(req.parameters["probabilities"], dtype=np.float32)
        return {
            "status": "completed",
            "input_type": "parameters",
            "predictions": arr.tolist(),
            "shape": list(arr.shape),
        }

    probs = _generate_probs_from_parameters(req.parameters)
    return {
        "status": "completed",
        "input_type": "parameters",
        "predictions": probs.tolist(),
        "shape": list(probs.shape),
    }


def _map_output(req: MapRequest) -> Dict[str, Any]:
    decoder = BCIDecoder(
        model_name=req.model_name,
        subject_id=req.subject_id,
        start_trial=req.start_trial,
    )

    mapped = []

    if req.probabilities is not None:
        probs = np.asarray(req.probabilities, dtype=np.float32)
        frames = decoder.decode_batch(probs)
        mapped = [map_to_control_signal(frame).to_dict() for frame in frames]

    elif req.labels is not None:
        conf = req.confidences or [1.0] * len(req.labels)
        if len(conf) != len(req.labels):
            raise ValueError("confidences length must match labels length")
        for idx, label in enumerate(req.labels):
            signal = class_to_control_signal(
                class_index=int(label),
                confidence=float(conf[idx]),
                trial_id=req.start_trial + idx,
            )
            mapped.append(signal.to_dict())

    else:
        raise ValueError("Provide either probabilities or labels in map request")

    return {
        "status": "completed",
        "count": len(mapped),
        "mapped_control_values": mapped,
    }


def _run_pipeline(req: RunRequest) -> Dict[str, Any]:
    cfg = _build_pipeline_config(req)
    pipeline = PipelineController(cfg)

    if req.input_type == "dataset":
        if not req.dataset_path:
            raise ValueError("dataset_path is required for input_type='dataset'")
        summary = pipeline.run_from_results_file(req.dataset_path)

    elif req.input_type == "image":
        _ensure_image_payload(req.image_path, req.image_base64)
        class_index = req.parameters.get("class_index")
        if class_index is None:
            raise ValueError(
                "Image input is accepted, but image inference is not implemented. "
                "Provide parameters.class_index (0..3)."
            )
        confidence = float(req.parameters.get("confidence", 1.0))
        summary = pipeline.run_from_labels([int(class_index)], [confidence])

    elif req.input_type == "parameters":
        if "labels" in req.parameters:
            labels = req.parameters["labels"]
            confidences = req.parameters.get("confidences")
            summary = pipeline.run_from_labels(labels, confidences)
        elif "probabilities" in req.parameters:
            probs = np.asarray(req.parameters["probabilities"], dtype=np.float32)
            summary = pipeline.run_from_probabilities(probs)
        else:
            probs = _generate_probs_from_parameters(req.parameters)
            summary = pipeline.run_from_probabilities(probs)

    else:
        raise ValueError(f"Unsupported input_type: {req.input_type}")

    return {
        "status": "completed",
        "input_type": req.input_type,
        "simulation_enabled": req.simulation.enabled,
        "result": asdict(summary),
    }


def _run_async_job(job_id: str, req: RunRequest) -> None:
    _set_job(
        job_id,
        {
            "job_id": job_id,
            "status": "running",
            "started_at": _utc_now(),
            "finished_at": None,
            "result": None,
            "error": None,
        },
    )

    try:
        result = _run_pipeline(req)
        _set_job(
            job_id,
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "result": result,
            },
        )
    except Exception as exc:
        _set_job(
            job_id,
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": str(exc),
            },
        )
        log.exception("Async job failed: %s", job_id)


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "BCI Online Execution API",
        "version": "1.1.0",
        "endpoints": ["/run", "/infer", "/map", "/simulate", "/models"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "time_utc": _utc_now(),
    }


@app.get("/models")
def models() -> Dict[str, Any]:
    return {
        "status": "completed",
        "models": AVAILABLE_MODELS,
        "default": "DB_ATCNet",
    }


@app.post("/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    try:
        return _infer_raw(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.post("/map")
def map_endpoint(req: MapRequest) -> Dict[str, Any]:
    try:
        return _map_output(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.post("/run")
def run(req: RunRequest) -> Dict[str, Any]:
    if req.async_mode:
        job_id = str(uuid4())
        _set_job(
            job_id,
            {
                "job_id": job_id,
                "status": "queued",
                "queued_at": _utc_now(),
            },
        )

        thread = threading.Thread(target=_run_async_job, args=(job_id, req), daemon=True)
        thread.start()

        return {
            "status": "accepted",
            "job_id": job_id,
            "status_endpoint": f"/status/{job_id}",
            "result_endpoint": f"/result/{job_id}",
        }

    try:
        return _run_pipeline(req)
    except PipelineError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.post("/simulate")
def simulate(req: SimulateRequest) -> Dict[str, Any]:
    if not req.control_values:
        raise HTTPException(status_code=400, detail="control_values must not be empty")

    if not req.simulation.enabled:
        return {
            "status": "completed",
            "simulation_enabled": False,
            "signals_applied": len(req.control_values),
            "message": "Simulation disabled. Dry-run simulate endpoint completed.",
        }

    try:
        from sim_controller import SimConfig, SimController
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Missing dependency: coppeliasim-zmqremoteapi-client",
        ) from exc

    class _Position:
        def __init__(self, x: float, y: float, z: float):
            self.x = x
            self.y = y
            self.z = z

    class _PreparedSignal:
        def __init__(self, item: ControlValue):
            self._item = item
            self.position = _Position(item.pos_x, item.pos_y, item.pos_z)

        def to_flat_dict(self) -> Dict[str, float]:
            return {
                "shoulder_pitch_deg": self._item.shoulder_pitch_deg,
                "elbow_pitch_deg": self._item.elbow_pitch_deg,
                "wrist_roll_deg": self._item.wrist_roll_deg,
            }

    sim_cfg = SimConfig(
        host=req.simulation.host,
        port=req.simulation.port,
        loop_hz=req.simulation.loop_hz,
        ready_timeout=req.simulation.ready_timeout,
    )

    ctrl = SimController(sim_cfg)
    applied = 0

    try:
        ctrl.connect()
        ctrl.start_simulation()

        for item in req.control_values:
            try:
                ctrl.apply_signal(_PreparedSignal(item))
                applied += 1
            except Exception as frame_exc:
                if not req.continue_on_frame_error:
                    raise RuntimeError(f"Failed at control index {applied}: {frame_exc}") from frame_exc
                log.error("Simulation frame error at index %d: %s", applied, frame_exc)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}") from exc
    finally:
        try:
            ctrl.stop_simulation()
        except Exception:
            pass
        try:
            ctrl.disconnect()
        except Exception:
            pass

    return {
        "status": "completed",
        "simulation_enabled": True,
        "signals_applied": applied,
    }


@app.get("/status/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    job = _get_job(job_id)
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "queued_at": job.get("queued_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
    }


@app.get("/result/{job_id}")
def job_result(job_id: str) -> Dict[str, Any]:
    job = _get_job(job_id)

    status = job.get("status")
    if status in {"queued", "running"}:
        return {
            "job_id": job_id,
            "status": status,
            "message": "Result not ready. Poll /status/{job_id}.",
        }

    if status == "failed":
        raise HTTPException(status_code=500, detail=job.get("error") or "Job failed")

    return {
        "job_id": job_id,
        "status": status,
        "result": job.get("result"),
    }


if __name__ == "__main__":
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "uvicorn is required. Install with: pip install uvicorn"
        ) from exc

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
