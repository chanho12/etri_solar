# SOLAR

**A Stepwise Sleep Prediction Framework via Long-Term Anchoring and Refinement**

Kyubin Kim*, Chanho Park*, Kyuho Lee*, Euntae Kim*<br>
Department of Computer Science, Korea University<br>
\* Equal contribution

[Project page](https://chanho12.github.io/etri_solar/) · [Paper](https://drive.google.com/file/d/18jN1ZuiLQeukAqOCAjvDY2MZCu7U_5IE/view?usp=sharing) · [Code](code/) · [Slides](docs/solar-slides.pdf) · [Model report](docs/model-report.pdf)

SOLAR predicts seven daily sleep-related probabilities from smartphone and wearable lifelog data. It separates long-term personalized anchoring, day-specific residual refinement, and target-specific history-aware refinement.

## Repository layout

```text
etri_solar/
├── index.html                  # GitHub Pages project page
├── assets/                     # Page styles, scripts, figures, local paper copy
├── code/                       # Raw-to-submission training implementation
│   ├── run.py
│   ├── solar_pipeline.py
│   ├── solar_core.py
│   ├── setup_conda.sh
│   ├── run_solar.sh
│   ├── environment.yml
│   ├── requirements.txt
│   └── data/README.md           # Required input layout; raw data is not included
└── docs/
    ├── MODEL_DESCRIPTION.md    # Detailed Korean technical documentation
    ├── model-report.pdf        # Supplied submission model report
    ├── solar-slides.pdf        # ICTC 2026 presentation
    └── solar-code.zip          # Standalone code and documentation bundle
```

## Reproduce the submission

Use a 64-bit Linux environment with Miniconda or Anaconda. Obtain the official challenge inputs separately and place them in [`code/data/`](code/data/README.md).

```bash
git clone https://github.com/chanho12/etri_solar.git
cd etri_solar/code
./setup_conda.sh
./run_solar.sh
```

The final output is `code/data/submissions/submission_final.csv`. See the [code instructions](code/README.md) and [model description](docs/MODEL_DESCRIPTION.md) for environment requirements, optional execution flags, and validation details. Raw participant data, trained checkpoints, generated features, and prediction CSVs are excluded from this repository.

## Reported results

Average test Log-Loss across seven targets; lower is better. These values are from the supplied paper, not a new training run.

| Method | Test Log-Loss ↓ |
|---|---:|
| CatBoost | 0.6053 |
| LightGBM | 0.6316 |
| XGBoost | 0.6985 |
| SCH CSM (CatBoost–LightGBM) | 0.6017 |
| **SOLAR** | **0.5560** |

## Website

The website uses plain HTML, CSS, and JavaScript. GitHub Pages serves `main` from the repository root; no build step is needed. To preview locally:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`. The layout is inspired by [Conversation Chronicles](https://conversation-chronicles.github.io/); the page implementation is original and the figures come from the SOLAR paper.

## Citation

```bibtex
@misc{kim2026solar,
  title  = {SOLAR: A Stepwise Sleep Prediction Framework
            via Long-Term Anchoring and Refinement},
  author = {Kim, Kyubin and Park, Chanho and
            Lee, Kyuho and Kim, Euntae},
  year   = {2026},
  url    = {https://chanho12.github.io/etri_solar/}
}
```

This is a project citation; use the publisher's bibliographic record when available.
