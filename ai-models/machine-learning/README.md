# Machine Learning

Classical machine-learning models and feature-engineering pipelines
(baselines and interpretable models).

## Layout

```
machine-learning/
├── src/
│   ├── preprocessing/   # Filtering, epoching, normalization
│   ├── features/        # Feature extraction (band power, connectivity, …)
│   ├── models/          # SVM, RF, XGBoost, etc. + train/predict
│   └── evaluation/      # Cross-validation, metrics, reports
├── notebooks/           # EDA & experiments
└── saved-models/        # Serialized models (.pkl/.joblib, git-ignored)
```

## Usage

```bash
python -m src.models.train --features bandpower --model xgboost
```

## Conventions
- Keep feature extraction reusable and separate from model code.
- Report results with proper (subject-wise) cross-validation.
