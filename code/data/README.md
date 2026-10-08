# Official input data

Raw challenge data is not redistributed in this repository. Obtain the official dataset through the challenge provider and keep the original file contents and names. The supplied runner validates the SHA-256 of the two CSV inputs.

Place the following files beside this README:

```text
data/
├── ch2026_metrics_train.csv
├── ch2026_submission_sample.csv
└── ch2025_data_items/
    ├── ch2025_mACStatus.parquet
    ├── ch2025_mActivity.parquet
    ├── ch2025_mAmbience.parquet
    ├── ch2025_mBle.parquet
    ├── ch2025_mGps.parquet
    ├── ch2025_mLight.parquet
    ├── ch2025_mScreenStatus.parquet
    ├── ch2025_mUsageStats.parquet
    ├── ch2025_mWifi.parquet
    ├── ch2025_wHr.parquet
    ├── ch2025_wLight.parquet
    └── ch2025_wPedo.parquet
```

`artifacts/` and `submissions/` are generated during execution. Do not add participant data or generated predictions to Git. This directory is ignored except for this README.
