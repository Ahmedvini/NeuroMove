# Cross-Modal Synthesis of fNIRS Hemodynamic Signals from EEG

The project's **core task**: learn a mapping from electrical brain activity
(**EEG**) to hemodynamic responses (**fNIRS**, i.e. ΔHbO / ΔHbR), enabling
synthesis of fNIRS-like signals where only EEG is available.

## Layout

```
cross-modal-synthesis/
├── src/
│   ├── models/        # Generative / regression architectures (e.g. GAN, U-Net, Transformer, CNN-LSTM)
│   ├── data/          # Paired EEG–fNIRS dataset loaders & alignment
│   ├── training/      # Training loops, losses (e.g. L1 + adversarial)
│   ├── evaluation/    # Reconstruction metrics (RMSE, PCC, SSIM), visualization
│   └── utils/         # Config, logging, signal helpers
├── configs/           # Experiment configurations
├── notebooks/         # Prototyping & qualitative analysis
├── checkpoints/       # Trained models (git-ignored)
└── results/           # Synthesized signals, plots, reports
```

## Problem definition

- **Input:** multi-channel EEG (time series), optionally band-decomposed.
- **Target:** co-registered fNIRS hemodynamic signals (ΔHbO, ΔHbR).
- **Key challenge:** temporal mismatch — EEG is fast (ms), the hemodynamic
  response is slow (~5–8 s lag). Models must capture the neurovascular coupling.

## Suggested approaches
- Sequence-to-sequence models (CNN-LSTM, Temporal Conv Nets, Transformers).
- Conditional generative models (cGAN / diffusion) for realistic HRF dynamics.
- Physics-informed priors based on the canonical hemodynamic response function.

## Usage

```bash
python -m src.training.train --config configs/eeg2fnirs_baseline.yaml
python -m src.evaluation.evaluate --checkpoint checkpoints/best.pt
```

## Evaluation metrics
- Temporal: RMSE, MAE, Pearson correlation (per channel).
- Structural: dynamic time warping distance, peak latency error.
