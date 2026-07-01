"""
structured_output.py
====================
Core dataclasses defining the structured output contract:
    { action, angles, positions }

Class map from your DB_ATCNet / ATCNet pipeline:
    Index 0 → 'Both Feet'   (label 'F'  in Physionet, class 2 in event_id)
    Index 1 → 'Left Fist'   (label 'L'  in Physionet, class 3 in event_id)
    Index 2 → 'Both Fists'  (label 'LR' in Physionet, class 4 in event_id)
    Index 3 → 'Right Fist'  (label 'R'  in Physionet, class 5 in event_id)

These match the confusion matrix display_labels in your main.py:
    ['Both feet', 'Left fist', 'Both fists', 'Right fist']
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# ---------------------------------------------------------------------------
# Motor-imagery class definitions
# ---------------------------------------------------------------------------

# Maps argmax index → human-readable label
# Order is IDENTICAL to your draw_confusion_matrix display_labels
CLASS_LABELS = {
    0: "Both Feet",
    1: "Left Fist",
    2: "Both Fists",
    3: "Right Fist",
}

# Reverse lookup: label string → class index
LABEL_TO_INDEX = {v: k for k, v in CLASS_LABELS.items()}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class JointAngles:
    """
    Exoskeleton joint angles (degrees) for a single arm configuration.
    All angles are in degrees; sign convention: positive = flexion.

    Mapping rationale
    -----------------
    Left Fist  → shoulder adducts, elbow flexes, wrist neutral, grip closes
    Right Fist → shoulder abducts, elbow flexes, wrist supinates, grip closes
    Both Fists → bilateral shoulder neutral, both elbows flex (only right arm modelled here)
    Both Feet  → arms at rest / return to home position
    """
    shoulder_deg: float = 0.0   # shoulder flexion/abduction
    elbow_deg: float = 0.0      # elbow flexion
    wrist_deg: float = 0.0      # wrist pronation/supination
    grip_deg: float = 0.0       # finger grip (0=open, 90=closed)

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (f"JointAngles(shoulder={self.shoulder_deg}°, "
                f"elbow={self.elbow_deg}°, "
                f"wrist={self.wrist_deg}°, "
                f"grip={self.grip_deg}°)")


@dataclass
class CartesianPosition:
    """
    End-effector Cartesian position in metres, relative to shoulder origin.
    Forward kinematics approximation based on joint angles.
    """
    x: float = 0.0   # lateral  (positive = right)
    y: float = 0.0   # anterior (positive = forward)
    z: float = 0.0   # vertical (positive = up)

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return f"CartesianPosition(x={self.x:.3f}m, y={self.y:.3f}m, z={self.z:.3f}m)"


@dataclass
class BCIFrame:
    """
    One decoded BCI frame — the complete structured output unit.

    Fields
    ------
    trial_id     : sequential trial number (0-based)
    action       : decoded motor-imagery class label (string)
    class_index  : integer class index (0-3), argmax of model softmax output
    confidence   : max softmax probability [0.0 – 1.0]
    probabilities: full 4-class softmax vector [P(Feet), P(Left), P(LR), P(Right)]
    angles       : JointAngles dataclass
    positions    : CartesianPosition dataclass
    subject_id   : optional subject identifier from your dataset
    model_name   : which model produced this prediction (e.g. 'DB_ATCNet')
    """
    trial_id: int
    action: str
    class_index: int
    confidence: float
    probabilities: List[float]
    angles: JointAngles
    positions: CartesianPosition
    subject_id: Optional[int] = None
    model_name: str = "DB_ATCNet"

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "subject_id": self.subject_id,
            "model_name": self.model_name,
            "action": self.action,
            "class_index": self.class_index,
            "confidence": round(self.confidence, 4),
            "probabilities": {
                CLASS_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(self.probabilities)
            },
            "angles": self.angles.to_dict(),
            "positions": self.positions.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return (f"BCIFrame(trial={self.trial_id}, action='{self.action}', "
                f"conf={self.confidence:.2%}, {self.angles}, {self.positions})")
