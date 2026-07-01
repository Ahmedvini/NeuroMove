"""
mapper.py
=========
Converts BCI model outputs into simulation-compatible control signals.

Pipeline
--------
    BCIFrame (class_index, confidence)
        │
        ▼
    JointAngles  ──► EulerRotation  (yaw, pitch, roll)  per joint
        │
        ▼
    CartesianPosition ──► SimPosition (x, y, z) normalised to sim space
        │
        ▼
    ControlSignal  ── ready to feed into Unity / ROS / Gazebo / MuJoCo

Class → biomechanical mapping
------------------------------
    0  Both Feet   →  rest / home
    1  Left Fist   →  left-arm reach + grasp
    2  Both Fists  →  bilateral symmetric grasp
    3  Right Fist  →  right-arm reach + grasp

Coordinate conventions
-----------------------
    Anatomical  :  yaw   = internal/external rotation  (about vertical Z-axis)
                   pitch = flexion / extension          (about medial Y-axis)
                   roll  = pronation / supination       (about longitudinal X-axis)

    Simulation  :  right-handed coordinate system
                   X = lateral (positive = right)
                   Y = anterior (positive = forward)
                   Z = vertical (positive = up)

    Normalised  :  all values mapped to [−1, +1]  for sim engine input
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from .structured_output import CLASS_LABELS, BCIFrame, JointAngles, CartesianPosition
from .kinematics import map_class_to_kinematics, L1, L2


# ─────────────────────────────────────────────────────────────────────────────
# Simulation space constraints  (tune to match your engine's world scale)
# ─────────────────────────────────────────────────────────────────────────────

SIM_CONSTRAINTS = {
    # Joint rotation limits (degrees) used for normalisation to [−1, +1]
    "shoulder": {"yaw":   (-90.0,  90.0),
                 "pitch": (-45.0, 135.0),
                 "roll":  (-30.0,  30.0)},
    "elbow":    {"yaw":   (  0.0,   0.0),   # elbow has no yaw DoF
                 "pitch": (  0.0, 135.0),
                 "roll":  (  0.0,   0.0)},
    "wrist":    {"yaw":   (-45.0,  45.0),
                 "pitch": (-30.0,  30.0),
                 "roll":  (-80.0,  80.0)},
    "grip":     {"yaw":   (  0.0,   0.0),
                 "pitch": (  0.0,  90.0),
                 "roll":  (  0.0,   0.0)},

    # Cartesian workspace limits (metres) for normalisation
    "workspace": {
        "x": (-(L1 + L2), (L1 + L2)),   # ±0.55 m lateral
        "y": (        0.0, (L1 + L2)),   #  0 … 0.55 m forward
        "z": (-(L1 + L2),        0.0),   # −0.55 … 0 m vertical
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EulerRotation:
    """Euler angles (degrees) representing one joint's orientation."""
    yaw:   float = 0.0   # rotation about Z (vertical)
    pitch: float = 0.0   # rotation about Y (medial)
    roll:  float = 0.0   # rotation about X (longitudinal)

    def to_radians(self) -> "EulerRotation":
        return EulerRotation(
            yaw=math.radians(self.yaw),
            pitch=math.radians(self.pitch),
            roll=math.radians(self.roll),
        )

    def to_dict(self) -> dict:
        return {"yaw_deg": round(self.yaw, 4),
                "pitch_deg": round(self.pitch, 4),
                "roll_deg": round(self.roll, 4)}


@dataclass
class JointRotations:
    """Full arm: Euler rotation per joint."""
    shoulder: EulerRotation = field(default_factory=EulerRotation)
    elbow:    EulerRotation = field(default_factory=EulerRotation)
    wrist:    EulerRotation = field(default_factory=EulerRotation)
    grip:     EulerRotation = field(default_factory=EulerRotation)

    def to_dict(self) -> dict:
        return {
            "shoulder": self.shoulder.to_dict(),
            "elbow":    self.elbow.to_dict(),
            "wrist":    self.wrist.to_dict(),
            "grip":     self.grip.to_dict(),
        }


@dataclass
class NormalisedRotations:
    """Joint rotations scaled to [−1, +1] for simulation engine input."""
    shoulder: Dict[str, float] = field(default_factory=dict)
    elbow:    Dict[str, float] = field(default_factory=dict)
    wrist:    Dict[str, float] = field(default_factory=dict)
    grip:     Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimPosition:
    """Cartesian end-effector position in simulation space."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    # Normalised versions [-1, +1]
    x_norm: float = 0.0
    y_norm: float = 0.0
    z_norm: float = 0.0

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "x_norm": round(self.x_norm, 4),
            "y_norm": round(self.y_norm, 4),
            "z_norm": round(self.z_norm, 4),
        }


@dataclass
class ControlSignal:
    """
    Final simulation-compatible control signal for one BCI frame.

    All fields are ready to send to Unity / ROS / Gazebo / MuJoCo.

    Fields
    ------
    trial_id         : sequential frame number
    action           : decoded motor imagery label
    class_index      : 0-3
    confidence       : model softmax max value [0, 1]
    rotations        : per-joint Euler angles (degrees)
    rotations_norm   : per-joint normalised values [−1, +1]
    position         : end-effector Cartesian + normalised XYZ
    grasp_strength   : [0, 1]  — derived from grip angle
    is_active        : False when class is 'Both Feet' (rest/home)
    """
    trial_id:       int
    action:         str
    class_index:    int
    confidence:     float
    rotations:      JointRotations
    rotations_norm: NormalisedRotations
    position:       SimPosition
    grasp_strength: float
    is_active:      bool

    def to_dict(self) -> dict:
        return {
            "trial_id":       self.trial_id,
            "action":         self.action,
            "class_index":    self.class_index,
            "confidence":     round(self.confidence, 4),
            "is_active":      self.is_active,
            "grasp_strength": round(self.grasp_strength, 4),
            "rotations":      self.rotations.to_dict(),
            "rotations_norm": self.rotations_norm.to_dict(),
            "position":       self.position.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_flat_dict(self) -> dict:
        """Single-level dict — useful for CSV rows and ROS message fields."""
        d = self.to_dict()
        flat = {
            "trial_id":              d["trial_id"],
            "action":                d["action"],
            "class_index":           d["class_index"],
            "confidence":            d["confidence"],
            "is_active":             d["is_active"],
            "grasp_strength":        d["grasp_strength"],
        }
        for joint in ("shoulder", "elbow", "wrist", "grip"):
            for axis in ("yaw_deg", "pitch_deg", "roll_deg"):
                flat[f"{joint}_{axis}"] = d["rotations"][joint][axis]
        for joint in ("shoulder", "elbow", "wrist", "grip"):
            for axis in ("yaw", "pitch", "roll"):
                flat[f"{joint}_{axis}_norm"] = d["rotations_norm"][joint][axis]
        for ax in ("x", "y", "z", "x_norm", "y_norm", "z_norm"):
            flat[f"pos_{ax}"] = d["position"][ax]
        return flat


# ─────────────────────────────────────────────────────────────────────────────
# Helper: normalise a value from [lo, hi] → [−1, +1]
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(value: float, lo: float, hi: float) -> float:
    """Linear normalisation → [−1, +1].  Clamps outside-range values."""
    if math.isclose(hi, lo):
        return 0.0
    norm = 2.0 * (value - lo) / (hi - lo) - 1.0
    return round(max(-1.0, min(1.0, norm)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Core conversion: JointAngles → EulerRotations (per joint)
# ─────────────────────────────────────────────────────────────────────────────

def angles_to_euler(angles: JointAngles) -> JointRotations:
    """
    Convert JointAngles (1-DoF scalars) to full Euler rotations per joint.

    Biomechanical mapping
    ---------------------
    Shoulder:
        pitch = flexion (elbow swings forward/backward)  ← shoulder_deg
        yaw   = internal/external rotation               ← 0 (not modelled)
        roll  = abduction / adduction                    ← shoulder_deg * 0.3

    Elbow:
        pitch = flexion / extension                      ← elbow_deg
        yaw, roll = 0 (single DoF hinge joint)

    Wrist:
        roll  = pronation / supination                   ← wrist_deg
        pitch = flexion / extension                      ← wrist_deg * 0.4
        yaw   = radial / ulnar deviation                 ← 0

    Grip:
        pitch = finger flexion (MCP joint proxy)         ← grip_deg
        yaw, roll = 0
    """
    return JointRotations(
        shoulder=EulerRotation(
            yaw=0.0,
            pitch=round(angles.shoulder_deg, 4),
            roll=round(angles.shoulder_deg * 0.3, 4),
        ),
        elbow=EulerRotation(
            yaw=0.0,
            pitch=round(angles.elbow_deg, 4),
            roll=0.0,
        ),
        wrist=EulerRotation(
            yaw=0.0,
            pitch=round(angles.wrist_deg * 0.4, 4),
            roll=round(angles.wrist_deg, 4),
        ),
        grip=EulerRotation(
            yaw=0.0,
            pitch=round(angles.grip_deg, 4),
            roll=0.0,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Normalise joint rotations to [−1, +1]
# ─────────────────────────────────────────────────────────────────────────────

def normalise_rotations(rotations: JointRotations) -> NormalisedRotations:
    """Scale each Euler angle to [−1, +1] using SIM_CONSTRAINTS limits."""
    C = SIM_CONSTRAINTS

    def _norm_joint(joint_name: str, euler: EulerRotation) -> Dict[str, float]:
        lims = C[joint_name]
        return {
            "yaw":   _normalise(euler.yaw,   *lims["yaw"]),
            "pitch": _normalise(euler.pitch, *lims["pitch"]),
            "roll":  _normalise(euler.roll,  *lims["roll"]),
        }

    return NormalisedRotations(
        shoulder=_norm_joint("shoulder", rotations.shoulder),
        elbow=_norm_joint("elbow",    rotations.elbow),
        wrist=_norm_joint("wrist",    rotations.wrist),
        grip=_norm_joint("grip",      rotations.grip),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Normalise Cartesian position to [−1, +1]
# ─────────────────────────────────────────────────────────────────────────────

def normalise_position(pos: CartesianPosition) -> SimPosition:
    """Scale XYZ from anatomical metres to [−1, +1] simulation space."""
    W = SIM_CONSTRAINTS["workspace"]
    return SimPosition(
        x=pos.x,
        y=pos.y,
        z=pos.z,
        x_norm=_normalise(pos.x, *W["x"]),
        y_norm=_normalise(pos.y, *W["y"]),
        z_norm=_normalise(pos.z, *W["z"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grasp strength  [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def compute_grasp_strength(grip_deg: float, max_grip: float = 90.0) -> float:
    """
    Map grip angle [0°, 90°] → grasp strength [0.0, 1.0].
    Clamps values outside the range.
    """
    return round(max(0.0, min(1.0, grip_deg / max_grip)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main mapper function
# ─────────────────────────────────────────────────────────────────────────────

def map_to_control_signal(frame: BCIFrame) -> ControlSignal:
    """
    Convert a BCIFrame → ControlSignal (simulation-compatible).

    Parameters
    ----------
    frame : BCIFrame
        Output of BCIDecoder.decode_single() / decode_batch().

    Returns
    -------
    ControlSignal
        Contains rotations (yaw/pitch/roll per joint), normalised values,
        Cartesian position + normalised XYZ, and grasp strength.
    """
    # Step 1: joint angles → Euler rotations
    rotations = angles_to_euler(frame.angles)

    # Step 2: normalise rotations to [−1, +1]
    rotations_norm = normalise_rotations(rotations)

    # Step 3: Cartesian position → normalised sim position
    sim_pos = normalise_position(frame.positions)

    # Step 4: grasp strength scalar
    grasp = compute_grasp_strength(frame.angles.grip_deg)

    # Step 5: is_active flag — Both Feet (class 0) = rest state
    is_active = (frame.class_index != 0)

    return ControlSignal(
        trial_id=frame.trial_id,
        action=frame.action,
        class_index=frame.class_index,
        confidence=frame.confidence,
        rotations=rotations,
        rotations_norm=rotations_norm,
        position=sim_pos,
        grasp_strength=grasp,
        is_active=is_active,
    )


def map_batch(frames: List[BCIFrame]) -> List[ControlSignal]:
    """Map a list of BCIFrames → list of ControlSignals."""
    return [map_to_control_signal(f) for f in frames]


# ─────────────────────────────────────────────────────────────────────────────
# Direct mapping from class index (no BCIFrame needed)
# ─────────────────────────────────────────────────────────────────────────────

def class_to_control_signal(
    class_index: int,
    confidence: float = 1.0,
    trial_id: int = 0,
) -> ControlSignal:
    """
    Shortcut: go directly from class index → ControlSignal.
    Useful for testing and real-time inference without a full BCIFrame.

    Example
    -------
        sig = class_to_control_signal(3)   # Right Fist
        print(sig.rotations.elbow.pitch)   # 80.0°
        print(sig.position.y_norm)         # normalised forward reach
    """
    angles, position = map_class_to_kinematics(class_index)

    # Build a minimal BCIFrame-compatible object
    from .structured_output import BCIFrame
    dummy = BCIFrame(
        trial_id=trial_id,
        action=CLASS_LABELS[class_index],
        class_index=class_index,
        confidence=confidence,
        probabilities=[0.0, 0.0, 0.0, 0.0],
        angles=angles,
        positions=position,
    )
    dummy.probabilities[class_index] = confidence
    return map_to_control_signal(dummy)


# ─────────────────────────────────────────────────────────────────────────────
# Update simulation constraints (calibration)
# ─────────────────────────────────────────────────────────────────────────────

def update_sim_constraints(joint: str, axis: str, lo: float, hi: float) -> None:
    """
    Override the normalisation range for a joint axis.
    Use this to match your specific simulation engine's joint limits.

    Parameters
    ----------
    joint : str   — 'shoulder', 'elbow', 'wrist', 'grip', or 'workspace'
    axis  : str   — 'yaw', 'pitch', 'roll'  (or 'x','y','z' for workspace)
    lo, hi: float — new range in degrees (or metres for workspace)

    Example
    -------
        from bci_exo.mapper import update_sim_constraints
        update_sim_constraints('elbow', 'pitch', 0.0, 150.0)
    """
    if joint not in SIM_CONSTRAINTS:
        raise ValueError(f"Unknown joint '{joint}'. Choose from {list(SIM_CONSTRAINTS.keys())}")
    if axis not in SIM_CONSTRAINTS[joint]:
        raise ValueError(f"Unknown axis '{axis}' for joint '{joint}'.")
    SIM_CONSTRAINTS[joint][axis] = (lo, hi)
    print(f"[mapper] Updated {joint}.{axis} range → [{lo}, {hi}]")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────────────────────

def _demo():
    print("=" * 65)
    print("  mapper.py — BCI → Simulation Control Signal Demo")
    print("=" * 65)

    for idx, label in CLASS_LABELS.items():
        sig = class_to_control_signal(class_index=idx, confidence=0.90)
        print(f"\n{'─'*65}")
        print(f"  Class {idx}: {label}   (is_active={sig.is_active})")
        print(f"{'─'*65}")

        r = sig.rotations
        rn = sig.rotations_norm
        print(f"  Joint rotations (degrees):")
        print(f"    shoulder  yaw={r.shoulder.yaw:>8.2f}°  pitch={r.shoulder.pitch:>8.2f}°  roll={r.shoulder.roll:>8.2f}°")
        print(f"    elbow     yaw={r.elbow.yaw:>8.2f}°  pitch={r.elbow.pitch:>8.2f}°  roll={r.elbow.roll:>8.2f}°")
        print(f"    wrist     yaw={r.wrist.yaw:>8.2f}°  pitch={r.wrist.pitch:>8.2f}°  roll={r.wrist.roll:>8.2f}°")
        print(f"    grip      yaw={r.grip.yaw:>8.2f}°  pitch={r.grip.pitch:>8.2f}°  roll={r.grip.roll:>8.2f}°")

        print(f"\n  Normalised rotations [−1, +1]:")
        for jname, jdict in rn.to_dict().items():
            vals = "  ".join(f"{k}={v:+.3f}" for k, v in jdict.items())
            print(f"    {jname:<10} {vals}")

        p = sig.position
        print(f"\n  Position:  x={p.x:+.4f}m  y={p.y:+.4f}m  z={p.z:+.4f}m")
        print(f"  Sim-norm:  x={p.x_norm:+.4f}  y={p.y_norm:+.4f}  z={p.z_norm:+.4f}")
        print(f"  Grasp strength: {sig.grasp_strength:.2f}")

    print(f"\n{'='*65}")
    print("  Flat dict (one row — ready for CSV / ROS message):")
    print(f"{'='*65}")
    sig = class_to_control_signal(3)
    for k, v in sig.to_flat_dict().items():
        print(f"  {k:<35} {v}")


if __name__ == "__main__":
    _demo()
