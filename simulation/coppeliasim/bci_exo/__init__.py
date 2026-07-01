"""
bci_exo
=======
BCI → Exoskeleton structured output interface.

Public API
----------
    from bci_exo import BCIDecoder, BCIFrame, JointAngles, CartesianPosition
    from bci_exo import stream_from_model, stream_from_file, FrameSink
    from bci_exo import load_results, scan_results_folder
    from bci_exo import map_class_to_kinematics, update_angle_map
"""

from .structured_output import (
    BCIFrame,
    JointAngles,
    CartesianPosition,
    CLASS_LABELS,
    LABEL_TO_INDEX,
)
from .kinematics import map_class_to_kinematics, update_angle_map, get_angle_map
from .mapper import (
    ControlSignal,
    JointRotations,
    EulerRotation,
    SimPosition,
    NormalisedRotations,
    map_to_control_signal,
    map_batch,
    class_to_control_signal,
    update_sim_constraints,
)
from .decoder import BCIDecoder
from .stream import stream_from_model, stream_from_file, FrameSink
from .results_loader import load_results, scan_results_folder

__all__ = [
    "BCIFrame",
    "JointAngles",
    "CartesianPosition",
    "CLASS_LABELS",
    "LABEL_TO_INDEX",
    "map_class_to_kinematics",
    "update_angle_map",
    "get_angle_map",
    "BCIDecoder",
    "stream_from_model",
    "stream_from_file",
    "FrameSink",
    "load_results",
    "scan_results_folder",
]

__version__ = "1.0.0"