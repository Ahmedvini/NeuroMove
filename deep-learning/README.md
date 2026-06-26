# Deep Learning

Neural-network models for EEG/fNIRS analysis (classification, regression,
representation learning).

## Layout

```
deep-learning/
├── src/
│   ├── models/        # Model architectures (nn.Module / keras.Model)
│   ├── datasets/      # Dataset & DataLoader definitions
│   ├── training/      # Training loops, schedulers, callbacks
│   ├── evaluation/    # Metrics, inference, plotting
│   └── utils/         # Config, logging, seeding, I/O helpers
├── configs/           # YAML/Hydra experiment configs
├── notebooks/         # Exploratory analysis & prototyping
├── checkpoints/       # Saved weights (git-ignored)
└── experiments/       # Run logs / outputs per experiment
```

## Usage

```bash
# Train
python -m src.training.train --config configs/baseline.yaml

# Evaluate
python -m src.evaluation.evaluate --checkpoint checkpoints/best.pt
```

## Conventions
- One architecture per file in `src/models/`.
- Every experiment is fully described by a config in `configs/`.
- Checkpoints and large artifacts stay out of git (see root `.gitignore`).
