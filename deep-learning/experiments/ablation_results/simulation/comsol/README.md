# COMSOL Multiphysics

Biophysical / multiphysics simulations — e.g. light propagation in tissue
(fNIRS optics), hemodynamics, or electromagnetics (EEG forward modeling).

## Layout

```
comsol/
├── models/     # COMSOL model files (.mph)
├── exports/    # Exported meshes, fields, tables (.txt/.csv/.vtu)
├── scripts/    # Automation via COMSOL with MATLAB LiveLink / Java API
└── results/    # Figures, processed outputs
```

## Notes
- `.mph` files are binary and large — consider Git LFS for versioning, or keep
  only canonical models in git and large sweeps out (see root `.gitignore`).
- Document each model's physics interfaces, materials, and boundary conditions
  in a short note alongside the `.mph` file.
