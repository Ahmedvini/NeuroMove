# CoppeliaSim

Robotics / experimental-setup simulation (formerly V-REP).

## Layout

```
coppeliasim/
├── scenes/     # Simulation scenes (.ttt)
├── models/     # Reusable models (.ttm)
├── scripts/    # Lua child scripts + Python/ZMQ remote-API clients
└── results/    # Logs, recorded trajectories, exports
```

## Remote API (Python)

```bash
pip install coppeliasim-zmqremoteapi-client
```

```python
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()
```

## Notes
- `.ttt`/`.ttm` are binary; consider Git LFS for large scenes.
- Keep remote-API control logic in `scripts/` so scenes stay reusable.
