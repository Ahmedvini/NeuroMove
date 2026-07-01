# Simulation

Robotics simulation supporting the project.

```
simulation/
└── coppeliasim/    # Online BCI→exoskeleton simulation (FastAPI + Docker + CoppeliaSim, ZMQ)
```

`coppeliasim/` is a full online pipeline: it decodes motor-imagery predictions, maps them to
joint kinematics via the `bci_exo` package, and drives a 3-joint exoskeleton arm in CoppeliaSim
over the ZMQ remote API — exposed through a FastAPI service. See
[`coppeliasim/PROJECT_DOCUMENTATION.md`](coppeliasim/PROJECT_DOCUMENTATION.md) for the full guide
(milestones, API, Docker, troubleshooting) and [`coppeliasim/README.md`](coppeliasim/README.md)
for a quick start.

> **Physics / FEA note:** the biophysical and structural–thermal modelling (formerly under
> `simulation/comsol/`) now lives in the top-level
> [`FEA-Biomaterial-Selection-COMSOL-main/`](../FEA-Biomaterial-Selection-COMSOL-main/README.md)
> component.
