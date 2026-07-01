"""
sim_controller.py — Milestone 4: CoppeliaSim Integration (FINAL)
=================================================================
BCI-Online-Simulation Project

Connects to the scene built by build_scene.lua and drives the 3 joints
from BCI classifier output via mapper.py.

Custom sync: we do NOT use sim.setStepping(). CoppeliaSim runs freely.
             A wall-clock timer (CustomSynchroniser) paces our send loop.

Usage
-----
    python sim_controller.py          # runs the built-in demo
    
    # or import and use:
    from sim_controller import SimController, SimConfig
    ctrl = SimController()
    ctrl.connect()
    ctrl.start_simulation()
    ctrl.apply_signal(signal)         # signal from mapper.py
    ctrl.stop_simulation()
    ctrl.disconnect()
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ── project import ────────────────────────────────────────────────────────────
try:
    from bci_exo.mapper import ControlSignal, class_to_control_signal
except ModuleNotFoundError:
    from mapper import ControlSignal, class_to_control_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Raw integer constants — no sim.* attribute access (version-safe)
# ─────────────────────────────────────────────────────────────────────────────
SIM_STOPPED  = 0    # simulation_stopped
W            = -1   # handle_world


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    host: str  = "localhost"
    port: int  = 23000
    loop_hz: float = 20.0          # Hz — keep ≤ CoppeliaSim step rate (default 20 Hz)
    ready_timeout: float = 8.0     # seconds to wait for sim to start

    joint_names: list = field(default_factory=lambda: [
        "/exo_joint_shoulder",
        "/exo_joint_elbow",
        "/exo_joint_wrist",
    ])
    ee_name: str = "/exo_end_effector"


# ─────────────────────────────────────────────────────────────────────────────
# Custom wall-clock synchroniser
# ─────────────────────────────────────────────────────────────────────────────

class CustomSynchroniser:
    """Rate-limits the control loop to a fixed Hz using time.monotonic()."""

    def __init__(self, hz: float):
        self._period = 1.0 / hz
        self._next   = 0.0

    def reset(self):
        self._next = time.monotonic() + self._period

    def wait(self):
        gap = self._next - time.monotonic()
        if gap > 0:
            time.sleep(gap)
        self._next += self._period


# ─────────────────────────────────────────────────────────────────────────────
# SimController
# ─────────────────────────────────────────────────────────────────────────────

class SimController:

    def __init__(self, config: Optional[SimConfig] = None):
        self.cfg    = config or SimConfig()
        self._sim   = None
        self._jh    = {}              # joint name → handle
        self._ee_h  = None            # end-effector handle
        self._sync  = CustomSynchroniser(self.cfg.loop_hz)
        self._alive = False           # True while simulation is running

    # ── connect / disconnect ──────────────────────────────────────────────────

    def connect(self):
        log.info("Connecting to CoppeliaSim %s:%d …", self.cfg.host, self.cfg.port)
        client = RemoteAPIClient(host=self.cfg.host, port=self.cfg.port)
        self._sim = client.require("sim")
        log.info("Connected.")
        self._resolve_handles()

    def _resolve_handles(self):
        log.info("Resolving object handles …")
        for name in self.cfg.joint_names:
            try:
                h = self._sim.getObject(name)
                self._jh[name] = h
                log.info("  %-30s → %d", name, h)
            except Exception as exc:
                log.warning("  %-30s → missing (%s)", name, exc)

        if not self._jh:
            raise RuntimeError(
                "None of the configured joint handles could be resolved. "
                "Check scene object names in SimConfig.joint_names."
            )

        if self.cfg.ee_name:
            try:
                self._ee_h = self._sim.getObject(self.cfg.ee_name)
                log.info("  %-30s → %d", self.cfg.ee_name, self._ee_h)
            except Exception as exc:
                self._ee_h = None
                log.warning("  %-30s → missing (%s)", self.cfg.ee_name, exc)

    def disconnect(self):
        if self._alive:
            self.stop_simulation()
        self._sim = None
        log.info("Disconnected.")

    # ── simulation lifecycle ──────────────────────────────────────────────────

    def start_simulation(self):
        self._check_connected()
        log.info("Starting simulation …")
        self._sim.startSimulation()

        # Poll until truly running (startSimulation is non-blocking)
        deadline = time.monotonic() + self.cfg.ready_timeout
        while time.monotonic() < deadline:
            state    = self._sim.getSimulationState()
            sim_time = self._sim.getSimulationTime()
            if state != SIM_STOPPED and sim_time > 0:
                break
            time.sleep(0.02)
        else:
            raise TimeoutError("Simulation did not start within timeout.")

        self._alive = True
        self._sync.reset()
        log.info("Running at %.0f Hz (%.1f ms/frame).",
                 self.cfg.loop_hz, 1000 / self.cfg.loop_hz)

    def stop_simulation(self):
        self._check_connected()
        self._sim.stopSimulation()
        self._alive = False
        log.info("Simulation stopped.")

    # ── send one BCI control signal ───────────────────────────────────────────

    def apply_signal(self, signal: ControlSignal):
        """
        Send joint targets derived from a mapper.ControlSignal.
        Blocks until the next wall-clock sync slot.

        Joint mapping (from mapper.py flat dict keys → radians):
          exo_joint_shoulder → shoulder_pitch_deg  (flexion/extension)
          exo_joint_elbow    → elbow_pitch_deg     (flexion/extension)
          exo_joint_wrist    → wrist_roll_deg      (pronation/supination)
        """
        self._check_running()
        self._sync.wait()

        import math

        # Exact key mapping: joint scene-name → flat dict degree key
        # All values from mapper are in DEGREES — convert to radians for CoppeliaSim
        JOINT_KEY_MAP = {
            "/exo_joint_shoulder": "shoulder_pitch_deg",
            "/exo_joint_elbow":    "elbow_pitch_deg",
            "/exo_joint_wrist":    "wrist_roll_deg",
        }

        flat = signal.to_flat_dict()

        for full_name, handle in self._jh.items():
            key = JOINT_KEY_MAP.get(full_name)
            if key and key in flat:
                angle_deg = float(flat[key])
                angle_rad = math.radians(angle_deg)
                self._sim.setJointPosition(handle, angle_rad)  # kinematic mode
                log.debug("  %s → %.1f° = %.4f rad", full_name, angle_deg, angle_rad)

        # Move end-effector dummy to the predicted Cartesian position
        if self._ee_h is not None:
            p = signal.position
            self._sim.setObjectPosition(self._ee_h, W, [p.x, p.y, p.z])

    @staticmethod
    def _pick(d: dict, keyword: str):
        kw = keyword.lower()
        for k, v in d.items():
            if kw in k.lower():
                return v
        return None

    # ── readback ──────────────────────────────────────────────────────────────

    def get_joint_positions(self) -> dict:
        self._check_connected()
        return {name: self._sim.getJointPosition(h) for name, h in self._jh.items()}

    def get_sim_time(self) -> float:
        self._check_connected()
        return self._sim.getSimulationTime()

    def get_ee_position(self):
        if self._ee_h is None:
            return None
        return list(self._sim.getObjectPosition(self._ee_h, W))

    # ── guards ────────────────────────────────────────────────────────────────

    def _check_connected(self):
        if self._sim is None:
            raise RuntimeError("Not connected. Call connect() first.")

    def _check_running(self):
        self._check_connected()
        if not self._alive:
            raise RuntimeError("Simulation not running. Call start_simulation() first.")


# ─────────────────────────────────────────────────────────────────────────────
# Demo  (python sim_controller.py)
# ─────────────────────────────────────────────────────────────────────────────

def _demo():
    print("=" * 65)
    print("Milestone 4 — CoppeliaSim BCI Integration Demo")
    print("=" * 65)

    cfg  = SimConfig(loop_hz=20.0)
    ctrl = SimController(cfg)
    ctrl.connect()
    ctrl.start_simulation()

    # Cycle classes 1-3 (skip 0 = rest/all-zeros), 3s each so motion is visible
    # Class 1 = Left Fist, Class 2 = Both Fists, Class 3 = Right Fist
    active_classes    = [1, 2, 3, 1, 2, 3]      # repeat twice = 18 s total
    frames_per_class  = int(3.0 * cfg.loop_hz)   # 60 frames = 3 s

    CLASS_NAMES = {0: "Rest", 1: "Left Fist", 2: "Both Fists", 3: "Right Fist"}

    print(f"\nCycling active classes (1=LeftFist, 2=BothFists, 3=RightFist)")
    print(f"3 seconds per class, {len(active_classes)*3}s total\n")

    for bci_class in active_classes:
        signal = class_to_control_signal(bci_class)
        flat   = signal.to_flat_dict()
        sh_deg = flat.get("shoulder_pitch_deg", 0)
        el_deg = flat.get("elbow_pitch_deg",    0)
        wr_deg = flat.get("wrist_roll_deg",     0)
        print(f"  Sending class {bci_class} ({CLASS_NAMES[bci_class]}): "
              f"sh={sh_deg:.1f}deg  el={el_deg:.1f}deg  wr={wr_deg:.1f}deg")

        for _ in range(frames_per_class):
            ctrl.apply_signal(signal)

        # Read back actual joint positions after settling
        t  = ctrl.get_sim_time()
        jp = ctrl.get_joint_positions()
        vals = list(jp.values())
        import math
        print(f"    -> sim_t={t:.2f}s  actual: "
              f"sh={math.degrees(vals[0]):+.1f}deg  "
              f"el={math.degrees(vals[1]):+.1f}deg  "
              f"wr={math.degrees(vals[2]):+.1f}deg\n")

    ctrl.stop_simulation()
    ctrl.disconnect()
    print("\nDemo complete.")


if __name__ == "__main__":
    _demo()