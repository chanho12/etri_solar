"""SOLAR end-to-end training pipeline.

The orchestration follows the three stages proposed in the accompanying paper:

1. Long-term personalized temporal anchor prediction.
2. Day-specific personalized residual refinement.
3. Target-specific history-aware refinement.

The implementation currently delegates stable model/feature primitives to the
reachability-pruned ``run_refactored`` core.  Keeping the orchestration here
makes the research structure explicit while the remaining legacy primitives
are migrated and regression-tested incrementally.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import solar_core as core


@dataclass(frozen=True)
class SolarConfig:
    """Frozen configuration of the submitted SOLAR model."""

    seed: int = 21040
    step1_profile: str = "transition_prior"
    step2_source_runs: tuple[tuple[str, int], ...] = (
        ("S01_K128", 128),
        ("S00_BASE_V7", 32),
        ("S01_K32", 32),
    )
    final_submission_name: str = (
        "submission_final.csv"
    )


def log(stage: str, message: str, started: float | None = None) -> None:
    suffix = ""
    if started is not None:
        suffix = f" ({time.perf_counter() - started:.1f}s)"
    print(f"[SOLAR][{stage}] {message}{suffix}", flush=True)


def validate_raw_inputs() -> None:
    """Validate that training starts from the frozen raw challenge inputs."""
    core._final_v1_validate_inputs()
    missing = [path for path in core.REQUIRED_INPUTS if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw input files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def step1_long_term_anchor(
    config: SolarConfig,
    *,
    include_mis_lstm: bool = False,
    reuse_base_artifacts: bool = False,
) -> None:
    """Step 1: learn a target-wise anchor from accumulated personal history.

    Multiple history, calendar, state-transition, sensor, and tabular sources
    are generated as OOF predictions.  Target-wise XGBoost/logistic/ridge
    stacking then produces the personalized temporal anchor used by Step 2.
    """
    started = time.perf_counter()
    log("STEP1-ANCHOR", "long-term personalized anchor training started")
    args = argparse.Namespace(
        reuse_step1=False,
        include_mis_lstm=include_mis_lstm,
        reuse_step1_base_artifacts=reuse_base_artifacts,
        stop_after_step1_base=False,
        personal_longterm_profile=config.step1_profile,
        restore_step1_targetwise_checkpoints=False,
        restore_step1_candidate_checkpoints=False,
    )
    core.pipeline_run_step1(args)
    log("STEP1-ANCHOR", "personalized temporal anchor ready", started)


def _build_v6_day_window_cache(*, rebuild: bool) -> None:
    """Build only the day/sleep-window feature cache, without V6 ablations."""
    core.restore_step2_features_defaults()
    cache = core._STEP2_FEATURES_PATHS_SAVED["feature_cache"]
    if cache.exists() and not rebuild:
        log("FEATURES", f"reusing day-window cache: {cache}")
        return

    started = time.perf_counter()
    log("FEATURES", "building day-specific numeric sensor windows")
    core.STEP2_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    schema, specs = core._step2_sensor_schema(
        core.DATA_DIR / "ch2025_data_items"
    )
    schema.to_csv(core.STEP2_FEATURES_PATHS["sensor_schema"], index=False)
    train, sample, _, _, _ = core._step2_load_anchor()
    meta = core._step2_meta(train, sample)
    core._step2_build_paper_features(
        meta,
        specs,
        check_only=False,
        rebuild_cache=True,
    )
    if not cache.exists():
        raise FileNotFoundError(f"V6 day-window cache was not created: {cache}")
    log("FEATURES", "day-specific numeric sensor windows ready", started)


def _build_v7_personal_state_cache(*, rebuild: bool) -> None:
    """Build nested-sensor and personal-state features used by refinement."""
    cache = core.V7_STATE_CACHE
    if cache.exists() and not rebuild:
        log("FEATURES", f"reusing personal-state cache: {cache}")
        return

    started = time.perf_counter()
    log("FEATURES", "building nested-sensor personal-state features")
    core.restore_step2_features_defaults()
    train, sample, _, _, _ = core._step2_load_anchor()
    meta = core._step2_meta(train, sample)
    core._sensor_helpers_build_state_features(
        meta,
        check_only=False,
        rebuild_cache=True,
    )
    if not cache.exists():
        raise FileNotFoundError(f"V7 personal-state cache was not created: {cache}")
    log("FEATURES", "nested-sensor personal-state features ready", started)


def step2_day_specific_residual(
    config: SolarConfig,
    *,
    rebuild_features: bool = True,
    resume: bool = False,
) -> None:
    """Step 2: refine the anchor using prediction-day sensor residuals.

    Only the three feature-selection runs used by the submitted E02 logit
    ensemble are trained.  Historical feature/model ablations are intentionally
    excluded from this final-model path.
    """
    started = time.perf_counter()
    log("STEP2-RESIDUAL", "day-specific residual refinement started")
    random.seed(config.seed)
    np.random.seed(config.seed)

    _build_v6_day_window_cache(rebuild=rebuild_features)
    _build_v7_personal_state_cache(rebuild=rebuild_features)
    core.restore_step2_features_defaults()

    context = core.load_fixed_context()
    core.write_leakage_checks(context)
    core.register_fixed_anchor(context)
    base_features = core.load_v7_base_features(
        context,
        reuse_cache=not rebuild_features,
    )

    for run_tag, top_k in config.step2_source_runs:
        log(
            "STEP2-RESIDUAL",
            f"training {run_tag}: top-{top_k} residual features",
        )
        core.fit_existing_scenario(
            context,
            run_tag,
            base_features,
            top_k,
            resume=resume,
        )

    source_tags = [run_tag for run_tag, _ in config.step2_source_runs]
    core.blend_predictions(
        context,
        source_tags,
        "logit",
        "E02_LOGIT_BLEND",
    )
    # The temperature OOF artifact is retained because the submitted final-v5
    # verification compares the raw and calibrated E02 anchors.
    core.temperature_calibration(context, "E02_LOGIT_BLEND")
    log("STEP2-RESIDUAL", "day-specific residual prediction ready", started)


def step3_history_aware_refinement(config: SolarConfig) -> Path:
    """Step 3: apply validated temporal priors and target-wise calibration.

    Q2/Q3/S2 receive two-scale same-subject temporal-neighbor refinement.  The
    submitted Q1, Q2, S4, and Q3 gates are then applied in their frozen order.
    """
    started = time.perf_counter()
    log("STEP3-HISTORY", "target-specific history-aware refinement started")
    core.final_v5_postprocess()
    core.final_v5_q1basis_postprocess()
    core.final_v5_combo_best_postprocess()
    final_path = Path(core.final_v5_q3joint_best_postprocess())

    expected = core.SUBMISSION_DIR / config.final_submission_name
    if final_path != expected:
        raise RuntimeError(
            f"Unexpected final path: expected {expected}, received {final_path}"
        )
    core._run_validate_submission(final_path)
    log("STEP3-HISTORY", f"final submission ready: {final_path}", started)
    return final_path


def train_solar_from_raw(
    config: SolarConfig,
    *,
    include_mis_lstm: bool = False,
    reuse_base_artifacts: bool = False,
    rebuild_features: bool = True,
    resume: bool = False,
) -> Path:
    """Train all three SOLAR stages from raw sensor inputs."""
    started = time.perf_counter()
    core._final_v1_seed_everything()
    validate_raw_inputs()
    step1_long_term_anchor(
        config,
        include_mis_lstm=include_mis_lstm,
        reuse_base_artifacts=reuse_base_artifacts,
    )
    step2_day_specific_residual(
        config,
        rebuild_features=rebuild_features,
        resume=resume,
    )
    final_path = step3_history_aware_refinement(config)
    log("DONE", f"sha256={core._run_sha256(final_path)}", started)
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the three-stage SOLAR submission model from raw sensors."
    )
    parser.add_argument(
        "--no-frozen",
        action="store_true",
        help="Accepted for compatibility; raw end-to-end training is the only mode.",
    )
    parser.add_argument("--include-mis-lstm", action="store_true")
    parser.add_argument("--reuse-step1-base-artifacts", action="store_true")
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_solar_from_raw(
        SolarConfig(),
        include_mis_lstm=args.include_mis_lstm,
        reuse_base_artifacts=args.reuse_step1_base_artifacts,
        rebuild_features=not args.reuse_feature_cache,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
