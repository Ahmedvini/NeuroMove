"""
kinematics.py
=============
Maps decoded motor-imagery class → exoskeleton joint angles + Cartesian position.

The four classes from your pipeline:
    0: Both Feet   → arms return to rest (home position)
    1: Left Fist   → left-arm grasp motion
    2: Both Fists  → bilateral symmetric grasp
    3: Right Fist  → right-arm grasp motion

Joint angle conventions (degrees, positive = flexion/closing):
    shoulder_deg : −90 (full adduction) … +90 (full abduction)
    elbow_deg    :   0 (fully extended) … +135 (fully flexed)
    wrist_deg    : −45 (pronation)      … +45  (supination)
    grip_deg     :   0 (fully open)     … +90  (fully closed)

Forward kinematics uses a simplified 2-link planar model:
    Upper arm length L1 = 0.30 m
    Forearm length   L2 = 0.25 m
"""

from __future__ import annotations
import math
from typing import Tuple
from .structured_output import JointAngles, CartesianPosition, CLASS_LABELS

# ---------------------------------------------------------------------------
# Segment lengths (metres)  – adjust to match your exoskeleton spec
# ---------------------------------------------------------------------------
L1 = 0.30  # upper arm
L2 = 0.25  # forearm


# ---------------------------------------------------------------------------
# Nominal joint-angle targets per class
# Designed to match the biomechanical intent of each motor-imagery task
# ---------------------------------------------------------------------------
_ANGLE_MAP: dict[int, JointAngles] = {
    # Both Feet → exoskeleton arms return to neutral / home position
    0: JointAngles(
        shoulder_deg=0.0,
        elbow_deg=10.0,
        wrist_deg=0.0,
        grip_deg=5.0,
    ),
    # Left Fist → shoulder adducts across body, elbow flexes, grip closes
    1: JointAngles(
        shoulder_deg=-20.0,
        elbow_deg=75.0,
        wrist_deg=-10.0,
        grip_deg=80.0,
    ),
    # Both Fists → symmetric bilateral grasp (right-arm representation)
    2: JointAngles(
        shoulder_deg=5.0,
        elbow_deg=60.0,
        wrist_deg=0.0,
        grip_deg=75.0,
    ),
    # Right Fist → shoulder abducts, elbow flexes, wrist supinates, grip closes
    3: JointAngles(
        shoulder_deg=25.0,
        elbow_deg=80.0,
        wrist_deg=20.0,
        grip_deg=85.0,
    ),
}


def _forward_kinematics(angles: JointAngles) -> CartesianPosition:
    """
    2-link planar forward kinematics.

    Using shoulder and elbow angles only (wrist/grip do not displace the
    end-effector significantly in this simplified model).

        x =  L1*sin(shoulder) + L2*sin(shoulder + elbow)
        y =  L1*cos(shoulder) + L2*cos(shoulder + elbow)   (forward)
        z = -L1*(1-cos(shoulder))                           (vertical drop)

    All angles converted from degrees to radians internally.
    """
    sh = math.radians(angles.shoulder_deg)
    el = math.radians(angles.elbow_deg)

    x = round(L1 * math.sin(sh) + L2 * math.sin(sh + el), 4)
    y = round(L1 * math.cos(sh) + L2 * math.cos(sh + el), 4)
    z = round(-L1 * (1.0 - math.cos(sh)), 4)

    return CartesianPosition(x=x, y=y, z=z)


def map_class_to_kinematics(
    class_index: int,
) -> Tuple[JointAngles, CartesianPosition]:
    """
    Given a model output class index (0-3), return the target
    (JointAngles, CartesianPosition) for the exoskeleton.

    Parameters
    ----------
    class_index : int
        Argmax of the model's softmax output. Must be in {0, 1, 2, 3}.

    Returns
    -------
    angles   : JointAngles
    position : CartesianPosition
    """
    if class_index not in _ANGLE_MAP:
        raise ValueError(
            f"class_index={class_index} is out of range. "
            f"Expected one of {list(_ANGLE_MAP.keys())}."
        )
    angles = _ANGLE_MAP[class_index]
    position = _forward_kinematics(angles)
    return angles, position


def get_angle_map() -> dict[int, JointAngles]:
    """Return a copy of the full class → angle mapping (read-only view)."""
    return dict(_ANGLE_MAP)


def update_angle_map(class_index: int, angles: JointAngles) -> None:
    """
    Override the default joint-angle target for a given class.
    Use this to calibrate to your specific exoskeleton geometry.

    Example
    -------
    >>> from bci_exo.kinematics import update_angle_map
    >>> from bci_exo.structured_output import JointAngles
    >>> update_angle_map(3, JointAngles(shoulder_deg=30, elbow_deg=90, wrist_deg=15, grip_deg=85))
    """
    if class_index not in _ANGLE_MAP:
        raise ValueError(f"class_index must be in {list(_ANGLE_MAP.keys())}")
    _ANGLE_MAP[class_index] = angles
