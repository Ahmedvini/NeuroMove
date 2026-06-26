# Data

Datasets are **not** tracked by git. This folder defines the expected layout.

```
data/
├── raw/         # Original, immutable data (never edit by hand)
├── interim/     # Intermediate, partially processed data
├── processed/   # Final, model-ready datasets
└── external/    # Third-party / public datasets
```

## Workflow
`raw → interim → processed` — keep `raw/` read-only; all transforms are
reproducible from scripts so the pipeline can be re-run.

## Common EEG/fNIRS formats
- EEG: `.edf`, `.bdf`, `.fif`, `.set`/`.fdt`
- fNIRS: `.snirf`, `.nirs`
- Tabular: `.csv`, `.parquet`

## Public datasets to consider
- Simultaneous EEG–fNIRS datasets (e.g. Shin et al. open hybrid BCI dataset).
- Document the exact source, version, and download steps here.
