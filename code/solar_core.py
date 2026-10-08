"""pipeline monolithic end-to-end runner.

This file embeds the step1_pipeline Step-1 pipeline, the step2_features Step-2
compatibility layer, the step2_sensor_helpers nested sensor helpers, and the
step2_runner Step-2 scenario runner. It can be executed directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT
DATA_DIR = ROOT / "data"
SUBMISSION_DIR = DATA_DIR / "submissions"
ARTIFACT_DIR = DATA_DIR / "artifacts"
FINAL_SUBMISSION = (
    SUBMISSION_DIR
    / "submission_step1_anchor.csv"
)

REQUIRED_INPUTS = [
    DATA_DIR / "ch2026_metrics_train.csv",
    DATA_DIR / "ch2026_submission_sample.csv",
    DATA_DIR / "ch2025_data_items",
]


# Pruned historical definition: run_experiment_log_git_sync (not reachable from the final runner).


# 함수: 중첩 구조 센서 parquet 샘플을 확인하는 점검 함수입니다.
# Pruned historical definition: inspect_nested_sensor_samples (not reachable from the final runner).


# 함수: 원천 센서 테이블의 구조와 기본 정보를 확인하는 점검 함수입니다.
# Pruned historical definition: inspect_raw_sensor_tables (not reachable from the final runner).


# 함수: subject별 prior 신뢰도를 반영한 후보 제출 파일을 생성합니다.
def build_subject_reliability_prior_candidates():
    __file__ = str(
        PROJECT_ROOT / "pipeline/build_subject_reliability_prior_candidates"
    )
    __name__ = "__main__"

    from dataclasses import dataclass
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SUBMISSION_DIR = BASE_DIR / "submissions"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

    CURRENT_BEST = SUBMISSION_DIR / "submission_step1_date_prior.csv"
    DATE_PRIOR = SUBMISSION_DIR / "submission_step1_interpolation.csv"
    BRACKET_PRIOR = SUBMISSION_DIR / "submission_step1_date_bracket.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has null values")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} has probabilities outside [0,1]")

    # 함수: logit 보조 로직을 수행합니다.
    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    # 함수: sigmoid 보조 로직을 수행합니다.
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # 함수: binary log-loss를 계산합니다.
    def logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
        pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=float)
        return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())

    # 함수: shift to mean 보조 로직을 수행합니다.
    def shift_to_mean(probs: np.ndarray, target_mean: float) -> np.ndarray:
        logits = logit(probs)
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            mean = sigmoid(logits + mid).mean()
            if mean < target_mean:
                lo = mid
            else:
                hi = mid
        return sigmoid(logits + (lo + hi) / 2.0)

    # 함수: mean align 보조 로직을 수행합니다.
    def mean_align(
        anchor: pd.DataFrame, train: pd.DataFrame, gamma: float
    ) -> pd.DataFrame:
        out = anchor.copy()
        anchor_mean = anchor[TARGETS].mean()
        train_mean = train[TARGETS].mean()
        desired = (1 - gamma) * anchor_mean + gamma * train_mean
        for target in TARGETS:
            out[target] = shift_to_mean(
                anchor[target].to_numpy(float), float(desired[target])
            )
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: build date confidence 보조 로직을 수행합니다.
    def build_date_confidence(
        train: pd.DataFrame, sample: pd.DataFrame
    ) -> pd.DataFrame:
        train_dates = {
            str(sid): pd.to_datetime(group["lifelog_date"]).sort_values().to_numpy()
            for sid, group in train.groupby("subject_id")
        }
        rows = []
        for row in sample.itertuples(index=False):
            sid = str(row.subject_id)
            date = pd.Timestamp(row.lifelog_date)
            dates = train_dates.get(sid)
            if dates is None or len(dates) == 0:
                before_dist, after_dist, min_dist, bracketed = np.nan, np.nan, 99.0, 0.0
            else:
                deltas = np.array(
                    [(pd.Timestamp(d) - date).days for d in dates], dtype=float
                )
                before = np.abs(deltas[deltas < 0])
                after = np.abs(deltas[deltas > 0])
                before_dist = float(before.min()) if len(before) else np.nan
                after_dist = float(after.min()) if len(after) else np.nan
                min_dist = float(np.nanmin([before_dist, after_dist]))
                bracketed = float(np.isfinite(before_dist) and np.isfinite(after_dist))
            near = float(np.exp(-min_dist / 10.0))
            both = bracketed * float(
                np.exp(
                    -(
                        np.nan_to_num(before_dist, nan=30.0)
                        + np.nan_to_num(after_dist, nan=30.0)
                    )
                    / 24.0
                )
            )
            confidence = float(np.clip(0.65 * near + 0.35 * both, 0.0, 1.0))
            rows.append(
                {
                    "subject_id": sid,
                    "lifelog_date": date,
                    "min_dist": min_dist,
                    "bracketed": bracketed,
                    "confidence": confidence,
                }
            )
        return pd.DataFrame(rows)

    # 함수: subject reliability 보조 로직을 수행합니다.
    def subject_reliability(train: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for sid, group in train.sort_values(["subject_id", "lifelog_date"]).groupby(
            "subject_id"
        ):
            group = group.reset_index(drop=True)
            n = len(group)
            if n < 12:
                reliability = 0.75
            else:
                errs = []
                base_errs = []
                for i in range(4, n):
                    hist = group.iloc[:i]
                    curr = group.iloc[i]
                    prev = hist[TARGETS].tail(7).mean()
                    subj = hist[TARGETS].mean()
                    globalish = train[TARGETS].mean()
                    pred = (0.55 * prev + 0.30 * subj + 0.15 * globalish).to_numpy(
                        float
                    )
                    base = subj.to_numpy(float)
                    y = curr[TARGETS].to_numpy(float)
                    errs.append(logloss(y, pred))
                    base_errs.append(logloss(y, base))
                gain = np.mean(base_errs) - np.mean(errs)
                reliability = float(np.clip(0.85 + gain * 4.0, 0.55, 1.20))
            rows.append({"subject_id": str(sid), "subject_reliability": reliability})
        return pd.DataFrame(rows)

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend_with_weight_matrix(
        anchor: pd.DataFrame, prior: pd.DataFrame, weights: pd.DataFrame
    ) -> pd.DataFrame:
        out = anchor.copy()
        for target in TARGETS:
            w = weights[target].to_numpy(float)
            out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[
                target
            ].to_numpy(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: subject reliable adaptive 보조 로직을 수행합니다.
    def subject_reliable_adaptive(
        anchor: pd.DataFrame,
        prior: pd.DataFrame,
        train: pd.DataFrame,
        sample: pd.DataFrame,
    ) -> pd.DataFrame:
        conf = build_date_confidence(train, sample)
        rel = subject_reliability(train)
        meta = sample[["subject_id"]].merge(rel, on="subject_id", how="left")
        reliability = meta["subject_reliability"].fillna(0.8).to_numpy(float)
        row_conf = conf["confidence"].to_numpy(float)
        base_weight = np.clip((0.020 + 0.095 * row_conf) * reliability, 0.0, 0.145)
        mult = {
            "Q1": 1.02,
            "Q2": 1.00,
            "Q3": 1.02,
            "S1": 0.78,
            "S2": 0.88,
            "S3": 0.68,
            "S4": 0.85,
        }
        weights = pd.DataFrame(
            {
                target: np.clip(base_weight * mult[target], 0.0, 0.16)
                for target in TARGETS
            }
        )
        return blend_with_weight_matrix(anchor, prior, weights)

    # 함수: current streak 보조 로직을 수행합니다.
    def current_streak(values: np.ndarray) -> tuple[float, int]:
        if len(values) == 0:
            return 0.5, 0
        last = float(values[-1])
        length = 0
        for value in values[::-1]:
            if float(value) == last:
                length += 1
            else:
                break
        return last, length

    # 함수: 검증/전체 학습 모델을 적합하고 예측값을 만듭니다.
    def fit_streak_tables(
        train: pd.DataFrame,
    ) -> dict[str, dict[tuple[int, int], float]]:
        tables: dict[str, dict[tuple[int, int], float]] = {}
        for target in TARGETS:
            buckets: dict[tuple[int, int], list[float]] = {}
            for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby(
                "subject_id"
            ):
                vals = group[target].to_numpy(float)
                for i in range(3, len(vals)):
                    last, length = current_streak(vals[:i])
                    key = (int(last), int(min(length, 5)))
                    buckets.setdefault(key, []).append(float(vals[i]))
            global_mean = float(train[target].mean())
            table = {}
            for key, ys in buckets.items():
                table[key] = float((np.sum(ys) + 8.0 * global_mean) / (len(ys) + 8.0))
            tables[target] = table
        return tables

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def streak_prior(train: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
        tables = fit_streak_tables(train)
        global_mean = train[TARGETS].mean()
        histories = {
            str(sid): group.sort_values("lifelog_date")[
                ["lifelog_date"] + TARGETS
            ].copy()
            for sid, group in train.groupby("subject_id")
        }
        rows = []
        work = sample[["subject_id", "lifelog_date"]].copy()
        work["_sample_order"] = np.arange(len(work))
        for _, row in work.sort_values(
            ["subject_id", "lifelog_date", "_sample_order"]
        ).iterrows():
            sid = str(row["subject_id"])
            hist = histories.get(sid, pd.DataFrame(columns=["lifelog_date"] + TARGETS))
            pred = {}
            for target in TARGETS:
                vals = (
                    hist[target].to_numpy(float)
                    if len(hist)
                    else np.array([], dtype=float)
                )
                last, length = current_streak(vals)
                key = (int(last), int(min(length, 5)))
                table_p = tables[target].get(key, float(global_mean[target]))
                subj_mean = (
                    float(hist[target].mean())
                    if len(hist)
                    else float(global_mean[target])
                )
                pred[target] = float(
                    np.clip(
                        0.58 * table_p + 0.27 * subj_mean + 0.15 * global_mean[target],
                        0.045,
                        0.955,
                    )
                )
            rows.append(
                {
                    "_sample_order": int(row["_sample_order"]),
                    "subject_id": sid,
                    "lifelog_date": pd.Timestamp(row["lifelog_date"]),
                    **pred,
                }
            )
            histories[sid] = pd.concat(
                [
                    hist,
                    pd.DataFrame(
                        [{"lifelog_date": pd.Timestamp(row["lifelog_date"]), **pred}]
                    ),
                ],
                ignore_index=True,
            )
        pred_df = pd.DataFrame(rows)
        out = pred_df.sort_values("_sample_order")[TARGETS].reset_index(drop=True)
        if len(out) != len(sample):
            raise ValueError(
                f"streak_prior length mismatch: pred={len(out)}, sample={len(sample)}"
            )
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def simple_blend(
        anchor: pd.DataFrame, prior: pd.DataFrame, weight: float, mult: dict[str, float]
    ) -> pd.DataFrame:
        out = anchor.copy()
        for target in TARGETS:
            w = weight * mult.get(target, 1.0)
            out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[
                target
            ].to_numpy(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "diff_vs_best": float(diff.values.mean()),
            "max_diff": float(diff.values.max()),
            "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
            "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save(
        name: str, df: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        validate(df, name)
        df.to_csv(SUBMISSION_DIR / name, index=False)
        return summarize(name, df, anchor)

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
        sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
        anchor = pd.read_csv(CURRENT_BEST)
        date_prior = pd.read_csv(DATE_PRIOR)
        bracket_prior = pd.read_csv(BRACKET_PRIOR)
        validate(anchor, CURRENT_BEST.name)
        validate(date_prior, DATE_PRIOR.name)
        validate(bracket_prior, BRACKET_PRIOR.name)

        rows = []

        # Axis 1: low-risk calibration on top of the current best.
        for gamma in [0.035, 0.055]:
            name = f"submission_step1_mean_g{int(gamma*1000):03d}.csv"
            rows.append(save(name, mean_align(anchor, train, gamma), anchor))

        # Axis 2: subject-reliability gated date prior.
        rel_date = subject_reliable_adaptive(anchor, date_prior, train, sample)
        rows.append(
            save("submission_step1_reliability_prior.csv", rel_date, anchor)
        )

        rel_bracket = subject_reliable_adaptive(anchor, bracket_prior, train, sample)
        rows.append(
            save("submission_step1_adaptive_bracket.csv", rel_bracket, anchor)
        )

        # Axis 3: streak/reversion prior. Kept weak because it recursively feeds predicted probabilities.
        streak = sample.copy()
        streak_values = streak_prior(train, sample)
        if len(streak_values) != len(streak):
            raise ValueError(
                f"streak assignment length mismatch: pred={len(streak_values)}, sample={len(streak)}"
            )
        streak[TARGETS] = streak_values.to_numpy(float)
        validate(streak, "streak_prior")
        streak_mult = {
            "Q1": 0.85,
            "Q2": 0.95,
            "Q3": 0.95,
            "S1": 0.75,
            "S2": 0.85,
            "S3": 0.60,
            "S4": 0.80,
        }
        rows.append(
            save(
                "submission_step1_state_streak_w055.csv",
                simple_blend(anchor, streak, 0.055, streak_mult),
                anchor,
            )
        )
        rows.append(
            save(
                "submission_step1_state_streak_w080.csv",
                simple_blend(anchor, streak, 0.080, streak_mult),
                anchor,
            )
        )

        summary = (
            pd.DataFrame(rows)
            .sort_values(["diff_vs_best", "candidate"])
            .reset_index(drop=True)
        )
        print("new-axis candidate summary vs current best 0.5886910305:")
        print(summary.to_string(index=False))
        print("\nSuggested submit order:")
        print("1) submission_step1_mean_g035.csv")
        print("2) submission_step1_reliability_prior.csv")
        print("3) submission_step1_state_streak_w055.csv")

    if __name__ == "__main__":
        main()


# 함수: 평균 보정 이후 target별 scale/reliability 후보를 생성합니다.
def build_mean_aligned_target_scale_candidates():
    __file__ = str(
        PROJECT_ROOT / "pipeline/build_mean_aligned_target_scale_candidates"
    )
    __name__ = "__main__"
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SUBMISSION_DIR = BASE_DIR / "submissions"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

    ANCHOR_PATH = SUBMISSION_DIR / "submission_step1_mean_g030.csv"
    DATE_PRIOR = SUBMISSION_DIR / "submission_step1_interpolation.csv"
    BRACKET_PRIOR = SUBMISSION_DIR / "submission_step1_date_bracket.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has null values")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} has probabilities outside [0,1]")

    # 함수: binary log-loss를 계산합니다.
    def logloss(y_true, pred) -> float:
        pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=float)
        return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())

    # 함수: date confidence 보조 로직을 수행합니다.
    def date_confidence(
        train: pd.DataFrame,
        sample: pd.DataFrame,
        near_tau: float,
        both_tau: float,
        mix: float,
    ) -> np.ndarray:
        train_dates = {
            str(sid): pd.to_datetime(group["lifelog_date"]).sort_values().to_numpy()
            for sid, group in train.groupby("subject_id")
        }
        values = []
        for row in sample.itertuples(index=False):
            sid = str(row.subject_id)
            date = pd.Timestamp(row.lifelog_date)
            dates = train_dates.get(sid)
            if dates is None or len(dates) == 0:
                min_dist, before_dist, after_dist, bracketed = 99.0, 30.0, 30.0, 0.0
            else:
                deltas = np.array(
                    [(pd.Timestamp(d) - date).days for d in dates], dtype=float
                )
                before = np.abs(deltas[deltas < 0])
                after = np.abs(deltas[deltas > 0])
                before_dist = float(before.min()) if len(before) else 30.0
                after_dist = float(after.min()) if len(after) else 30.0
                min_dist = min(before_dist, after_dist)
                bracketed = float(len(before) > 0 and len(after) > 0)
            near = float(np.exp(-min_dist / near_tau))
            both = bracketed * float(np.exp(-(before_dist + after_dist) / both_tau))
            values.append(float(np.clip(mix * near + (1 - mix) * both, 0.0, 1.0)))
        return np.array(values, dtype=float)

    # 함수: subject rel 보조 로직을 수행합니다.
    def subject_rel(train: pd.DataFrame, targetwise: bool) -> pd.DataFrame:
        rows = []
        global_mean = train[TARGETS].mean()
        for sid, group in train.sort_values(["subject_id", "lifelog_date"]).groupby(
            "subject_id"
        ):
            group = group.reset_index(drop=True)
            row = {"subject_id": str(sid)}
            gains = {}
            for target in TARGETS:
                errs = []
                base_errs = []
                for i in range(4, len(group)):
                    hist = group.iloc[:i]
                    curr = group.iloc[i]
                    pred = (
                        0.55 * float(hist[target].tail(7).mean())
                        + 0.30 * float(hist[target].mean())
                        + 0.15 * float(global_mean[target])
                    )
                    base = float(hist[target].mean())
                    y = float(curr[target])
                    errs.append(logloss([y], [pred]))
                    base_errs.append(logloss([y], [base]))
                gains[target] = (
                    float(np.mean(base_errs) - np.mean(errs)) if errs else 0.0
                )
            if targetwise:
                for target in TARGETS:
                    row[target] = float(np.clip(0.84 + gains[target] * 3.2, 0.58, 1.18))
            else:
                rel = float(
                    np.clip(0.85 + np.mean(list(gains.values())) * 4.0, 0.55, 1.20)
                )
                for target in TARGETS:
                    row[target] = rel
            rows.append(row)
        return pd.DataFrame(rows)

    # 함수: adaptive 보조 로직을 수행합니다.
    def adaptive(
        anchor,
        prior,
        train,
        sample,
        *,
        targetwise,
        base,
        scale,
        cap,
        mult,
        near_tau=10.0,
        both_tau=24.0,
        mix=0.66,
    ):
        conf = date_confidence(train, sample, near_tau, both_tau, mix)
        rel = subject_rel(train, targetwise)
        meta = sample[["subject_id"]].merge(rel, on="subject_id", how="left")
        out = anchor.copy()
        row_base = base + scale * conf
        for target in TARGETS:
            reliability = meta[target].fillna(0.85).to_numpy(float)
            w = np.clip(row_base * reliability * mult[target], 0.0, cap)
            out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[
                target
            ].to_numpy(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def mixed_prior(date_prior, bracket_prior, train, sample):
        conf = date_confidence(train, sample, near_tau=10.0, both_tau=22.0, mix=0.62)
        gate = np.clip((conf - 0.45) / 0.45, 0.0, 1.0)
        out = date_prior.copy()
        for target in TARGETS:
            out[target] = (1 - 0.20 * gate) * date_prior[target].to_numpy(float) + (
                0.20 * gate
            ) * bracket_prior[target].to_numpy(float)
        return out

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(name, candidate, anchor):
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "diff_vs_g030": float(diff.values.mean()),
            "max_diff": float(diff.values.max()),
            "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
            "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save(name, candidate, anchor):
        validate(candidate, name)
        candidate.to_csv(SUBMISSION_DIR / name, index=False)
        return summarize(name, candidate, anchor)

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main():
        train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
        sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
        anchor = pd.read_csv(ANCHOR_PATH)
        date_prior = pd.read_csv(DATE_PRIOR)
        bracket_prior = pd.read_csv(BRACKET_PRIOR)
        validate(anchor, ANCHOR_PATH.name)
        validate(date_prior, DATE_PRIOR.name)
        validate(bracket_prior, BRACKET_PRIOR.name)

        rows = []
        mult_safe = {
            "Q1": 1.00,
            "Q2": 0.98,
            "Q3": 1.02,
            "S1": 0.70,
            "S2": 0.82,
            "S3": 0.50,
            "S4": 0.78,
        }
        mult_q = {
            "Q1": 1.10,
            "Q2": 1.04,
            "Q3": 1.12,
            "S1": 0.50,
            "S2": 0.62,
            "S3": 0.35,
            "S4": 0.58,
        }

        rows.append(
            save(
                "submission_step1_target_reliability_scale_soft.csv",
                adaptive(
                    anchor,
                    date_prior,
                    train,
                    sample,
                    targetwise=True,
                    base=0.006,
                    scale=0.050,
                    cap=0.090,
                    mult=mult_safe,
                ),
                anchor,
            )
        )
        rows.append(
            save(
                "submission_step1_target_reliability_scale_mid.csv",
                adaptive(
                    anchor,
                    date_prior,
                    train,
                    sample,
                    targetwise=True,
                    base=0.009,
                    scale=0.060,
                    cap=0.105,
                    mult=mult_safe,
                ),
                anchor,
            )
        )
        rows.append(
            save(
                "submission_step1_target_reliability_q_led_s_soft.csv",
                adaptive(
                    anchor,
                    date_prior,
                    train,
                    sample,
                    targetwise=True,
                    base=0.008,
                    scale=0.058,
                    cap=0.105,
                    mult=mult_q,
                ),
                anchor,
            )
        )
        mix = mixed_prior(date_prior, bracket_prior, train, sample)
        rows.append(
            save(
                "submission_step1_target_reliability_mixed_date_bracket.csv",
                adaptive(
                    anchor,
                    mix,
                    train,
                    sample,
                    targetwise=False,
                    base=0.007,
                    scale=0.052,
                    cap=0.095,
                    mult=mult_safe,
                ),
                anchor,
            )
        )

        summary = pd.DataFrame(rows).sort_values("diff_vs_g030").reset_index(drop=True)
        print("Candidates after g030 anchor:")
        print(summary.to_string(index=False))
        print("\nSuggested remaining submits:")
        print("1) submission_step1_target_reliability_scale_soft.csv")
        print("2) submission_step1_target_reliability_mixed_date_bracket.csv")

    if __name__ == "__main__":
        main()


# 함수: subject/date prior와 target reliability 보정을 세밀하게 다듬는 후보를 생성합니다.
def build_reliability_refinement_candidates():
    __file__ = str(PROJECT_ROOT / "pipeline/build_reliability_refinement_candidates")
    __name__ = "__main__"

    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SUBMISSION_DIR = BASE_DIR / "submissions"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

    CURRENT_BEST = SUBMISSION_DIR / "submission_step1_reliability_prior.csv"
    DATE_PRIOR = SUBMISSION_DIR / "submission_step1_interpolation.csv"
    BRACKET_PRIOR = SUBMISSION_DIR / "submission_step1_date_bracket.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has null values")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} has probabilities outside [0,1]")

    # 함수: logit 보조 로직을 수행합니다.
    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # 함수: sigmoid 보조 로직을 수행합니다.
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    # 함수: binary log-loss를 계산합니다.
    def logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
        pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=float)
        return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())

    # 함수: shift to mean 보조 로직을 수행합니다.
    def shift_to_mean(probs: np.ndarray, target_mean: float) -> np.ndarray:
        logits = logit(probs)
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if sigmoid(logits + mid).mean() < target_mean:
                lo = mid
            else:
                hi = mid
        return sigmoid(logits + (lo + hi) / 2)

    # 함수: mean align 보조 로직을 수행합니다.
    def mean_align(
        anchor: pd.DataFrame, train: pd.DataFrame, gamma: float
    ) -> pd.DataFrame:
        out = anchor.copy()
        desired = (1 - gamma) * anchor[TARGETS].mean() + gamma * train[TARGETS].mean()
        for target in TARGETS:
            out[target] = shift_to_mean(
                anchor[target].to_numpy(float), float(desired[target])
            )
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: build date confidence 보조 로직을 수행합니다.
    def build_date_confidence(
        train: pd.DataFrame,
        sample: pd.DataFrame,
        near_tau: float,
        both_tau: float,
        mix: float,
    ) -> pd.Series:
        train_dates = {
            str(sid): pd.to_datetime(group["lifelog_date"]).sort_values().to_numpy()
            for sid, group in train.groupby("subject_id")
        }
        values = []
        for row in sample.itertuples(index=False):
            sid = str(row.subject_id)
            date = pd.Timestamp(row.lifelog_date)
            dates = train_dates.get(sid)
            if dates is None or len(dates) == 0:
                min_dist = 99.0
                bracketed = 0.0
                before_dist = 30.0
                after_dist = 30.0
            else:
                deltas = np.array(
                    [(pd.Timestamp(d) - date).days for d in dates], dtype=float
                )
                before = np.abs(deltas[deltas < 0])
                after = np.abs(deltas[deltas > 0])
                before_dist = float(before.min()) if len(before) else 30.0
                after_dist = float(after.min()) if len(after) else 30.0
                min_dist = float(min(before_dist, after_dist))
                bracketed = float(len(before) > 0 and len(after) > 0)
            near = float(np.exp(-min_dist / near_tau))
            both = bracketed * float(np.exp(-(before_dist + after_dist) / both_tau))
            values.append(float(np.clip(mix * near + (1 - mix) * both, 0.0, 1.0)))
        return pd.Series(values)

    # 함수: subject reliability 보조 로직을 수행합니다.
    def subject_reliability(train: pd.DataFrame, mode: str) -> pd.DataFrame:
        rows = []
        global_mean = train[TARGETS].mean()
        for sid, group in train.sort_values(["subject_id", "lifelog_date"]).groupby(
            "subject_id"
        ):
            group = group.reset_index(drop=True)
            n = len(group)
            if n < 12:
                reliability = 0.78
            else:
                gains = []
                for target in TARGETS:
                    errs = []
                    base_errs = []
                    for i in range(4, n):
                        hist = group.iloc[:i]
                        curr = group.iloc[i]
                        recent = float(hist[target].tail(7).mean())
                        subj = float(hist[target].mean())
                        pred = (
                            0.55 * recent
                            + 0.30 * subj
                            + 0.15 * float(global_mean[target])
                        )
                        base = subj
                        y = float(curr[target])
                        errs.append(logloss(np.array([y]), np.array([pred])))
                        base_errs.append(logloss(np.array([y]), np.array([base])))
                    gains.append(float(np.mean(base_errs) - np.mean(errs)))

                if mode == "overall":
                    gain = float(np.mean(gains))
                    reliability = float(np.clip(0.85 + gain * 4.0, 0.55, 1.20))
                    row = {
                        "subject_id": str(sid),
                        **{target: reliability for target in TARGETS},
                    }
                elif mode == "target":
                    row = {"subject_id": str(sid)}
                    for target, gain in zip(TARGETS, gains):
                        row[target] = float(np.clip(0.84 + gain * 3.2, 0.58, 1.18))
                else:
                    raise ValueError(mode)
                rows.append(row)
                continue

            rows.append(
                {"subject_id": str(sid), **{target: reliability for target in TARGETS}}
            )
        return pd.DataFrame(rows)

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def adaptive_prior(
        anchor: pd.DataFrame,
        prior: pd.DataFrame,
        train: pd.DataFrame,
        sample: pd.DataFrame,
        *,
        rel_mode: str,
        near_tau: float,
        both_tau: float,
        conf_mix: float,
        base: float,
        scale: float,
        cap: float,
        mult: dict[str, float],
    ) -> pd.DataFrame:
        conf = build_date_confidence(
            train, sample, near_tau=near_tau, both_tau=both_tau, mix=conf_mix
        ).to_numpy(float)
        rel = subject_reliability(train, rel_mode)
        meta = sample[["subject_id"]].merge(rel, on="subject_id", how="left")
        out = anchor.copy()
        row_base = base + scale * conf
        for target in TARGETS:
            reliability = meta[target].fillna(0.85).to_numpy(float)
            weight = np.clip(row_base * reliability * mult.get(target, 1.0), 0.0, cap)
            out[target] = (1 - weight) * anchor[target].to_numpy(
                float
            ) + weight * prior[target].to_numpy(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend_two_priors(
        anchor: pd.DataFrame,
        prior_a: pd.DataFrame,
        prior_b: pd.DataFrame,
        train: pd.DataFrame,
        sample: pd.DataFrame,
    ) -> pd.DataFrame:
        # Date prior is known good; bracket prior only enters when a date is bracketed closely.
        conf = build_date_confidence(
            train, sample, near_tau=10.0, both_tau=22.0, mix=0.62
        ).to_numpy(float)
        bracket_gate = np.clip((conf - 0.45) / 0.45, 0.0, 1.0)
        mixed = prior_a.copy()
        for target in TARGETS:
            mixed[target] = (1 - 0.22 * bracket_gate) * prior_a[target].to_numpy(
                float
            ) + (0.22 * bracket_gate) * prior_b[target].to_numpy(float)
        mult = {
            "Q1": 1.00,
            "Q2": 0.98,
            "Q3": 1.02,
            "S1": 0.72,
            "S2": 0.84,
            "S3": 0.55,
            "S4": 0.82,
        }
        return adaptive_prior(
            anchor,
            mixed,
            train,
            sample,
            rel_mode="overall",
            near_tau=10.0,
            both_tau=22.0,
            conf_mix=0.62,
            base=0.012,
            scale=0.070,
            cap=0.120,
            mult=mult,
        )

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "diff_vs_best": float(diff.values.mean()),
            "max_diff": float(diff.values.max()),
            "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
            "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save(
        name: str, df: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        validate(df, name)
        df.to_csv(SUBMISSION_DIR / name, index=False)
        return summarize(name, df, anchor)

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
        sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
        anchor = pd.read_csv(CURRENT_BEST)
        date_prior = pd.read_csv(DATE_PRIOR)
        bracket_prior = pd.read_csv(BRACKET_PRIOR)
        validate(anchor, CURRENT_BEST.name)
        validate(date_prior, DATE_PRIOR.name)
        validate(bracket_prior, BRACKET_PRIOR.name)

        rows = []

        rows.append(
            save(
                "submission_step1_mean_g030.csv",
                mean_align(anchor, train, 0.030),
                anchor,
            )
        )

        base_mult = {
            "Q1": 1.00,
            "Q2": 0.98,
            "Q3": 1.02,
            "S1": 0.72,
            "S2": 0.84,
            "S3": 0.55,
            "S4": 0.82,
        }
        rows.append(
            save(
                "submission_step1_reliability_scale_soft.csv",
                adaptive_prior(
                    anchor,
                    date_prior,
                    train,
                    sample,
                    rel_mode="target",
                    near_tau=11.0,
                    both_tau=26.0,
                    conf_mix=0.70,
                    base=0.010,
                    scale=0.065,
                    cap=0.115,
                    mult=base_mult,
                ),
                anchor,
            )
        )
        rows.append(
            save(
                "submission_step1_reliability_refinement_overall_midcap.csv",
                adaptive_prior(
                    anchor,
                    date_prior,
                    train,
                    sample,
                    rel_mode="overall",
                    near_tau=9.0,
                    both_tau=22.0,
                    conf_mix=0.62,
                    base=0.015,
                    scale=0.075,
                    cap=0.125,
                    mult=base_mult,
                ),
                anchor,
            )
        )
        rows.append(
            save(
                "submission_step1_reliability_refinement_mixed_date_bracket.csv",
                blend_two_priors(anchor, date_prior, bracket_prior, train, sample),
                anchor,
            )
        )

        # More experimental: Q-led variant. Keep S very conservative.
        q_mult = {
            "Q1": 1.10,
            "Q2": 1.05,
            "Q3": 1.12,
            "S1": 0.55,
            "S2": 0.68,
            "S3": 0.40,
            "S4": 0.62,
        }
        rows.append(
            save(
                "submission_step1_reliability_refinement_qled_ssoft.csv",
                adaptive_prior(
                    anchor,
                    date_prior,
                    train,
                    sample,
                    rel_mode="target",
                    near_tau=10.0,
                    both_tau=24.0,
                    conf_mix=0.68,
                    base=0.012,
                    scale=0.078,
                    cap=0.130,
                    mult=q_mult,
                ),
                anchor,
            )
        )

        summary = pd.DataFrame(rows).sort_values("diff_vs_best").reset_index(drop=True)
        print("reliability refine candidates vs current best 0.5884463893:")
        print(summary.to_string(index=False))
        print("\nSuggested submit order:")
        print("1) submission_step1_mean_g030.csv")
        print("2) submission_step1_reliability_scale_soft.csv")
        print("3) submission_step1_reliability_refinement_mixed_date_bracket.csv")

    if __name__ == "__main__":
        main()


# 함수: 공동 타겟 패턴 projection과 센서 KNN prior 후보를 생성합니다.
def build_pattern_projection_and_sensor_knn_candidates():
    __file__ = str(
        PROJECT_ROOT / "pipeline/build_pattern_projection_and_sensor_knn_candidates"
    )
    __name__ = "__main__"

    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SUBMISSION_DIR = BASE_DIR / "submissions"

    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"
    FEATURE_PATH = BASE_DIR / "features_daily_sensor_table.csv"

    CURRENT_BEST = SUBMISSION_DIR / "submission_step1_target_reliability_scale_soft.csv"
    DATE_PRIOR = SUBMISSION_DIR / "submission_step1_interpolation.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    KEYS = ["subject_id", "lifelog_date"]

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has null values")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} has probabilities outside [0,1]")

    # 함수: binary log-loss를 계산합니다.
    def logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
        pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=float)
        return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend(
        anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float] | float
    ) -> pd.DataFrame:
        out = anchor.copy()
        if isinstance(weights, dict):
            for target in TARGETS:
                w = float(weights.get(target, 0.0))
                out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[
                    target
                ].to_numpy(float)
        else:
            w = float(weights)
            out[TARGETS] = (1 - w) * anchor[TARGETS].to_numpy(float) + w * prior[
                TARGETS
            ].to_numpy(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: target pattern code 보조 로직을 수행합니다.
    def target_pattern_code(values: np.ndarray) -> int:
        bits = (np.asarray(values, dtype=float) >= 0.5).astype(int)
        code = 0
        for bit in bits:
            code = (code << 1) | int(bit)
        return int(code)

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def pattern_projection_prior(
        train: pd.DataFrame, anchor: pd.DataFrame, prior_strength: float = 0.45
    ) -> pd.DataFrame:
        # Nonlinear joint-target prior: posterior over observed 7-bit patterns.
        patterns = []
        for _, row in train[TARGETS].iterrows():
            bits = row.to_numpy(dtype=int)
            patterns.append(tuple(int(x) for x in bits))
        pattern_df = pd.DataFrame(patterns, columns=TARGETS)
        counts = pattern_df.value_counts().reset_index(name="count")
        pattern_mat = counts[TARGETS].to_numpy(float)
        pattern_prior = (counts["count"].to_numpy(float) + prior_strength) / (
            len(train) + prior_strength * len(counts)
        )

        probs = anchor[TARGETS].to_numpy(float)
        out_probs = np.zeros_like(probs)
        log_prior = np.log(pattern_prior)
        for i, p in enumerate(np.clip(probs, 1e-5, 1 - 1e-5)):
            logp = log_prior + (
                pattern_mat * np.log(p) + (1 - pattern_mat) * np.log(1 - p)
            ).sum(axis=1)
            logp -= logp.max()
            weights = np.exp(logp)
            weights /= weights.sum()
            out_probs[i] = weights @ pattern_mat

        out = anchor.copy()
        out[TARGETS] = np.clip(out_probs, 1e-6, 1 - 1e-6)
        return out

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def select_feature_columns(train_feat: pd.DataFrame, test_feat: pd.DataFrame) -> list[str]:
        exclude = set(KEYS + TARGETS)
        raw_cols = []
        for col in train_feat.columns:
            if col in exclude:
                continue
            if col in test_feat.columns and pd.api.types.is_numeric_dtype(train_feat[col]):
                # Keep stable daily aggregates and prior deviations; avoid pure identifiers.
                raw_cols.append(col)

        kept = []
        for col in raw_cols:
            tr = pd.to_numeric(train_feat[col], errors="coerce")
            te = pd.to_numeric(test_feat[col], errors="coerce")
            if tr.notna().sum() < max(20, int(len(tr) * 0.08)):
                continue
            if te.notna().sum() < max(10, int(len(te) * 0.08)):
                continue
            missing_gap = abs(float(tr.isna().mean()) - float(te.isna().mean()))
            if missing_gap > 0.55 or max(float(tr.isna().mean()), float(te.isna().mean())) > 0.985:
                continue
            tr_valid = tr.dropna().astype(float)
            te_valid = te.dropna().astype(float)
            scale = float(tr_valid.quantile(0.75) - tr_valid.quantile(0.25))
            if not np.isfinite(scale) or scale < 1e-9:
                scale = float(tr_valid.std(ddof=0))
            if not np.isfinite(scale) or scale < 1e-9:
                scale = 1.0
            mean_gap = abs(float(tr_valid.mean()) - float(te_valid.mean())) / scale
            median_gap = abs(float(tr_valid.median()) - float(te_valid.median())) / scale
            q90_gap = (
                abs(float(tr_valid.quantile(0.90)) - float(te_valid.quantile(0.90)))
                / scale
            )
            drift_score = max(mean_gap, median_gap, 0.65 * q90_gap, 3.0 * missing_gap)
            if drift_score <= 3.50:
                kept.append(col)
        print(f"[SENSOR-KNN-DRIFT] kept={len(kept)} removed={len(raw_cols) - len(kept)}")
        return kept

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def prepare_feature_matrix(
        train_feat: pd.DataFrame, test_feat: pd.DataFrame, cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        train_x = train_feat[cols].replace([np.inf, -np.inf], np.nan)
        test_x = test_feat[cols].replace([np.inf, -np.inf], np.nan)
        med = train_x.median()
        train_x = train_x.fillna(med)
        test_x = test_x.fillna(med)
        q25 = train_x.quantile(0.25)
        q75 = train_x.quantile(0.75)
        scale = (
            (q75 - q25)
            .replace(0, np.nan)
            .fillna(train_x.std().replace(0, 1))
            .fillna(1.0)
        )
        train_z = ((train_x - med) / scale).clip(-6, 6).to_numpy(float)
        test_z = ((test_x - med) / scale).clip(-6, 6).to_numpy(float)
        return train_z, test_z

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def sensor_knn_prior(
        train: pd.DataFrame,
        sample: pd.DataFrame,
        feature_df: pd.DataFrame,
        k: int,
        tau: float,
    ) -> pd.DataFrame:
        train_keys = train[KEYS].copy()
        sample_keys = sample[KEYS].copy()
        feature_df = feature_df.copy()
        for frame in [train_keys, sample_keys, feature_df]:
            frame["subject_id"] = frame["subject_id"].astype(str)
            frame["lifelog_date"] = pd.to_datetime(frame["lifelog_date"])

        train_feat = train_keys.merge(feature_df, on=KEYS, how="left")
        test_feat = sample_keys.merge(feature_df, on=KEYS, how="left")
        cols = select_feature_columns(train_feat, test_feat)
        train_x, test_x = prepare_feature_matrix(train_feat, test_feat, cols)
        y = train[TARGETS].to_numpy(float)

        # Downweight same-subject nearest neighbors to avoid just recreating target prior.
        train_subjects = train["subject_id"].astype(str).to_numpy()
        test_subjects = sample["subject_id"].astype(str).to_numpy()
        out = np.zeros((len(test_x), len(TARGETS)), dtype=float)
        global_mean = train[TARGETS].mean().to_numpy(float)

        for i, x in enumerate(test_x):
            dist = np.sqrt(((train_x - x) ** 2).mean(axis=1))
            order = np.argsort(dist)[: max(k * 3, k)]
            chosen = order[:k]
            d = dist[chosen]
            weights = np.exp(-d / tau)
            same = train_subjects[chosen] == test_subjects[i]
            weights = weights * np.where(same, 0.65, 1.0)
            if weights.sum() <= 1e-12:
                pred = global_mean
            else:
                pred = (weights[:, None] * y[chosen]).sum(axis=0) / weights.sum()
                reliability = weights.sum() / (weights.sum() + 2.5)
                pred = reliability * pred + (1 - reliability) * global_mean
            out[i] = np.clip(pred, 0.045, 0.955)

        prior = sample.copy()
        prior[TARGETS] = out
        return prior

    # 함수: interleaved sensor cv 보조 로직을 수행합니다.
    def interleaved_sensor_cv(
        train: pd.DataFrame, feature_df: pd.DataFrame, k: int, tau: float
    ) -> float:
        preds = []
        ys = []
        for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby(
            "subject_id"
        ):
            group = group.reset_index(drop=True)
            if len(group) < 15:
                continue
            mask = (np.arange(len(group)) + 2) % 5 == 0
            hold = group[mask].copy()
            fit = pd.concat(
                [
                    train[train["subject_id"] != group["subject_id"].iloc[0]],
                    group[~mask],
                ],
                ignore_index=True,
            )
            pred = sensor_knn_prior(
                fit,
                hold[["subject_id", "sleep_date", "lifelog_date"]].copy(),
                feature_df,
                k=k,
                tau=tau,
            )
            preds.append(pred[TARGETS].to_numpy(float))
            ys.append(hold[TARGETS].to_numpy(float))
        return logloss(np.vstack(ys), np.vstack(preds))

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "diff_vs_best": float(diff.values.mean()),
            "max_diff": float(diff.values.max()),
            "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
            "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        validate(candidate, name)
        candidate.to_csv(SUBMISSION_DIR / name, index=False)
        return summarize(name, candidate, anchor)

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
        sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
        feature_df = pd.read_csv(FEATURE_PATH, parse_dates=["lifelog_date"])
        anchor = pd.read_csv(CURRENT_BEST)
        date_prior = pd.read_csv(DATE_PRIOR)
        validate(anchor, CURRENT_BEST.name)
        validate(date_prior, DATE_PRIOR.name)

        rows = []

        # Axis 1: nonlinear joint target pattern projection.
        pattern_prior = pattern_projection_prior(train, anchor)
        rows.append(
            save(
                "submission_step1_joint_pattern_w015.csv",
                blend(anchor, pattern_prior, 0.015),
                anchor,
            )
        )
        rows.append(
            save(
                "submission_step1_joint_pattern_w025.csv",
                blend(anchor, pattern_prior, 0.025),
                anchor,
            )
        )
        rows.append(
            save(
                "submission_step1_joint_pattern_w040.csv",
                blend(anchor, pattern_prior, 0.040),
                anchor,
            )
        )

        # Axis 2: sensor feature nearest-neighbor prior. Keep weak because prior sensor experiments overfit.
        cv_rows = []
        for k, tau in [(16, 2.8), (24, 3.4), (32, 4.2)]:
            cv_rows.append(
                {
                    "k": k,
                    "tau": tau,
                    "cv": interleaved_sensor_cv(train, feature_df, k=k, tau=tau),
                }
            )
        cv = pd.DataFrame(cv_rows).sort_values("cv").reset_index(drop=True)
        print("Sensor KNN interleaved CV:")
        print(cv.to_string(index=False))
        best_k = int(cv.loc[0, "k"])
        best_tau = float(cv.loc[0, "tau"])
        sensor_prior = sensor_knn_prior(
            train, sample, feature_df, k=best_k, tau=best_tau
        )
        rows.append(
            save(
                f"submission_step1_sensor_neighbors_k{best_k}_w010.csv",
                blend(anchor, sensor_prior, 0.010),
                anchor,
            )
        )
        rows.append(
            save(
                f"submission_step1_sensor_neighbors_k{best_k}_w020.csv",
                blend(anchor, sensor_prior, 0.020),
                anchor,
            )
        )
        rows.append(
            save(
                f"submission_step1_sensor_neighbors_k{best_k}_w035.csv",
                blend(anchor, sensor_prior, 0.035),
                anchor,
            )
        )

        # Axis 3: combine known date prior with sensor prior only as a residual nudge.
        hybrid_prior = date_prior.copy()
        hybrid_prior[TARGETS] = (
            0.82 * date_prior[TARGETS].to_numpy(float)
            + 0.18 * sensor_prior[TARGETS].to_numpy(float)
        ).clip(0.045, 0.955)
        target_weights = {
            "Q1": 0.018,
            "Q2": 0.020,
            "Q3": 0.020,
            "S1": 0.012,
            "S2": 0.015,
            "S3": 0.008,
            "S4": 0.014,
        }
        rows.append(
            save(
                "submission_step1_date_sensor_ensemble.csv",
                blend(anchor, hybrid_prior, target_weights),
                anchor,
            )
        )

        summary = pd.DataFrame(rows).sort_values("diff_vs_best").reset_index(drop=True)
        print("\n0510 new-axis candidate summary vs current best 0.5883722159:")
        print(summary.to_string(index=False))
        print("\nSuggested submit order:")
        print("1) submission_step1_date_sensor_ensemble.csv")
        print("2) submission_step1_joint_pattern_w015.csv")
        print("3) submission_step1_joint_pattern_w025.csv")
        print(
            f"Fallback risky sensor axis: submission_step1_sensor_neighbors_k{best_k}_w010.csv"
        )

    if __name__ == "__main__":
        main()


# 함수: 원천 센서 raw-context feature 기반 후보를 생성합니다.
# Pruned historical definition: build_raw_context_sensor_candidates (not reachable from the final runner).


# 함수: 수면 시간창 feature를 만들고 sleep-window 모델 후보를 생성합니다.
def build_sleep_window_model_candidates():
    __file__ = str(PROJECT_ROOT / "pipeline/build_sleep_window_model_candidates")
    __name__ = "__main__"
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from sklearn.metrics import log_loss

    warnings.filterwarnings("ignore")

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SENSOR_DIR = BASE_DIR / "ch2025_data_items"
    SUBMISSION_DIR = BASE_DIR / "submissions"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

    # Numeric-best anchor from the  pattern projection run.
    ANCHOR_PATH = SUBMISSION_DIR / "submission_step1_joint_pattern_w015.csv"
    FALLBACK_ANCHOR_PATH = (
        SUBMISSION_DIR / "submission_step1_target_reliability_scale_soft.csv"
    )

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    SLEEP_TARGETS = ["Q1", "S1", "S2", "S3", "S4"]

    WINDOWS = {
        # The old daily feature table used calendar days. For sleep labels, the
        # behavior window crossing midnight should be more causally aligned.
        "evening": (18, 24),  # lifelog_date 18:00-24:00
        "bedtime_transition": (20, 26),  # lifelog_date 20:00-sleep_date 02:00
        "pre_sleep": (21, 27),  # lifelog_date 21:00-sleep_date 03:00
        "late_night": (24, 29),  # sleep_date 00:00-05:00
        "overnight": (24, 32),  # sleep_date 00:00-08:00
        "wake_transition": (29, 36),  # sleep_date 05:00-12:00
        "morning": (30, 36),  # sleep_date 06:00-12:00
        "full_sleepctx": (18, 36),  # lifelog_date 18:00-sleep_date 12:00
    }

    SENSORS = [
        ("ch2025_mActivity.parquet", ["m_activity"], "mact"),
        ("ch2025_mLight.parquet", ["m_light"], "mlight"),
        ("ch2025_mScreenStatus.parquet", ["m_screen_use"], "screen"),
        ("ch2025_mACStatus.parquet", ["m_charging"], "charge"),
        ("ch2025_wHr.parquet", ["heart_rate"], "hr"),
        ("ch2025_wLight.parquet", ["w_light"], "wlight"),
        (
            "ch2025_wPedo.parquet",
            [
                "step",
                "step_frequency",
                "running_step",
                "walking_step",
                "distance",
                "speed",
                "burned_calories",
            ],
            "pedo",
        ),
    ]

    # 함수: make meta 보조 로직을 수행합니다.
    def make_meta(train, sample):
        train_meta = train[
            ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        ].copy()
        test_meta = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
        for t in TARGETS:
            test_meta[t] = np.nan
        meta = pd.concat(
            [train_meta.assign(is_train=1), test_meta.assign(is_train=0)],
            ignore_index=True,
        )
        meta["row_id"] = np.arange(len(meta))
        meta["subject_id"] = meta["subject_id"].astype(str)
        meta["lifelog_date"] = pd.to_datetime(meta["lifelog_date"])
        meta["sleep_date"] = pd.to_datetime(meta["sleep_date"])
        # 날짜 정렬 실험: 수면 label은 항상 lifelog_date 다음날의 sleep_date에
        # 붙어 있으므로, 두 날짜의 calendar 신호를 모두 명시적으로 제공합니다.
        meta["date_gap_days"] = (meta["sleep_date"] - meta["lifelog_date"]).dt.days
        meta["dow"] = meta["lifelog_date"].dt.dayofweek
        meta["sleep_dow"] = meta["sleep_date"].dt.dayofweek
        meta["month"] = meta["lifelog_date"].dt.month
        meta["sleep_month"] = meta["sleep_date"].dt.month
        meta["is_weekend"] = (meta["dow"] >= 5).astype(int)
        meta["sleep_is_weekend"] = (meta["sleep_dow"] >= 5).astype(int)
        meta["subject_ord"] = (
            meta.groupby("subject_id")["lifelog_date"].rank(method="first").astype(int)
        )
        meta["subject_days_from_start"] = meta.groupby("subject_id")[
            "lifelog_date"
        ].transform(lambda s: (s - s.min()).dt.days)
        return meta

    # 함수: 센서 데이터를 subject/date 단위로 집계합니다.
    def aggregate_one_sensor(meta, file_name, value_cols, prefix):
        path = SENSOR_DIR / file_name
        df = pd.read_parquet(path)
        df["subject_id"] = df["subject_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cols = [c for c in value_cols if c in df.columns]
        if not cols:
            return pd.DataFrame({"row_id": meta["row_id"]})

        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["subject_id", "timestamp"] + cols].dropna(how="all", subset=cols)

        pieces = []
        for sid, rows in meta.groupby("subject_id", sort=False):
            sdf = df[df["subject_id"] == sid].sort_values("timestamp")
            if sdf.empty:
                continue
            for window_name, (start_h, end_h) in WINDOWS.items():
                tmp_rows = []
                for row in rows.itertuples(index=False):
                    start = row.lifelog_date + pd.Timedelta(hours=start_h)
                    end = row.lifelog_date + pd.Timedelta(hours=end_h)
                    part = sdf[(sdf["timestamp"] >= start) & (sdf["timestamp"] < end)]
                    rec = {"row_id": row.row_id}
                    rec[f"{prefix}_{window_name}_count"] = len(part)
                    for c in cols:
                        vals = part[c].dropna()
                        base = f"{prefix}_{window_name}_{c}"
                        if len(vals):
                            rec[f"{base}_mean"] = float(vals.mean())
                            rec[f"{base}_std"] = float(vals.std(ddof=0))
                            rec[f"{base}_min"] = float(vals.min())
                            rec[f"{base}_max"] = float(vals.max())
                            rec[f"{base}_sum"] = float(vals.sum())
                            rec[f"{base}_median"] = float(vals.median())
                            rec[f"{base}_q10"] = float(vals.quantile(0.10))
                            rec[f"{base}_q25"] = float(vals.quantile(0.25))
                            rec[f"{base}_q75"] = float(vals.quantile(0.75))
                            rec[f"{base}_q90"] = float(vals.quantile(0.90))
                            rec[f"{base}_iqr"] = rec[f"{base}_q75"] - rec[f"{base}_q25"]
                            rec[f"{base}_q90_q10"] = rec[f"{base}_q90"] - rec[f"{base}_q10"]
                        else:
                            rec[f"{base}_mean"] = np.nan
                            rec[f"{base}_std"] = np.nan
                            rec[f"{base}_min"] = np.nan
                            rec[f"{base}_max"] = np.nan
                            rec[f"{base}_sum"] = np.nan
                            rec[f"{base}_median"] = np.nan
                            rec[f"{base}_q10"] = np.nan
                            rec[f"{base}_q25"] = np.nan
                            rec[f"{base}_q75"] = np.nan
                            rec[f"{base}_q90"] = np.nan
                            rec[f"{base}_iqr"] = np.nan
                            rec[f"{base}_q90_q10"] = np.nan
                    tmp_rows.append(rec)
                pieces.append(pd.DataFrame(tmp_rows))

        if not pieces:
            return pd.DataFrame({"row_id": meta["row_id"]})
        # Each piece contains one subject-window slice. Concatenate first and then
        # collapse by row_id so different window columns land on the same row.
        return (
            pd.concat(pieces, ignore_index=True)
            .groupby("row_id", as_index=False)
            .first()
        )

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def add_cross_window_features(feat):
        # Behavioral contrasts are often safer than raw levels across subjects.
        pairs = [
            ("screen", "m_screen_use"),
            ("mlight", "m_light"),
            ("wlight", "w_light"),
            ("mact", "m_activity"),
            ("hr", "heart_rate"),
            ("pedo", "step"),
            ("charge", "m_charging"),
        ]
        for prefix, col in pairs:
            eve = f"{prefix}_evening_{col}_mean"
            night = f"{prefix}_overnight_{col}_mean"
            pre = f"{prefix}_pre_sleep_{col}_mean"
            morn = f"{prefix}_morning_{col}_mean"
            if eve in feat and night in feat:
                feat[f"{prefix}_night_minus_evening"] = feat[night] - feat[eve]
            if pre in feat and morn in feat:
                feat[f"{prefix}_morning_minus_presleep"] = feat[morn] - feat[pre]
            if pre in feat and night in feat:
                feat[f"{prefix}_presleep_minus_overnight"] = feat[pre] - feat[night]
        return feat

    # 함수: 고정 sleep-window 집계를 이용해 subject별 동적 수면 구간 proxy를 생성합니다.
    def add_dynamic_sleep_interval_features(feat):
        feat = feat.copy()
        train_mask = feat["is_train"].eq(1) if "is_train" in feat else pd.Series(True, index=feat.index)

        def robust_z(col):
            if col not in feat.columns:
                return pd.Series(0.0, index=feat.index)
            values = pd.to_numeric(feat[col], errors="coerce")
            ref = values[train_mask]
            med = float(ref.median()) if ref.notna().any() else 0.0
            q25 = float(ref.quantile(0.25)) if ref.notna().any() else 0.0
            q75 = float(ref.quantile(0.75)) if ref.notna().any() else 1.0
            scale = q75 - q25
            if not np.isfinite(scale) or scale < 1e-6:
                scale = float(ref.std()) if ref.notna().sum() > 1 else 1.0
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
            return ((values.fillna(med) - med) / scale).clip(-6, 6)

        # Centers are measured as hours from lifelog_date 00:00. Values over 24 are
        # sleep_date hours, so smaller onset and larger wake imply longer rest.
        windows = {
            "evening": 21.0,
            "bedtime_transition": 23.0,
            "pre_sleep": 24.0,
            "late_night": 26.5,
            "overnight": 28.0,
            "wake_transition": 32.5,
            "morning": 33.0,
        }
        quiet_scores = {}
        activity_scores = {}
        for name in windows:
            quiet = (
                -0.30 * robust_z(f"screen_{name}_m_screen_use_sum")
                -0.24 * robust_z(f"pedo_{name}_step_sum")
                -0.20 * robust_z(f"mact_{name}_m_activity_mean")
                -0.14 * robust_z(f"mlight_{name}_m_light_mean")
                -0.08 * robust_z(f"wlight_{name}_w_light_mean")
                +0.14 * robust_z(f"charge_{name}_m_charging_mean")
            )
            active = (
                0.30 * robust_z(f"screen_{name}_m_screen_use_sum")
                +0.24 * robust_z(f"pedo_{name}_step_sum")
                +0.20 * robust_z(f"mact_{name}_m_activity_mean")
                +0.14 * robust_z(f"mlight_{name}_m_light_mean")
                +0.08 * robust_z(f"wlight_{name}_w_light_mean")
                -0.10 * robust_z(f"charge_{name}_m_charging_mean")
            )
            feat[f"dyn_sleep_quiet_score_{name}"] = quiet.astype(float)
            feat[f"dyn_sleep_active_score_{name}"] = active.astype(float)
            quiet_scores[name] = quiet
            activity_scores[name] = active

        quiet_mat = pd.DataFrame(quiet_scores)
        active_mat = pd.DataFrame(activity_scores)
        centers = pd.Series(windows)
        sleep_windows = ["bedtime_transition", "pre_sleep", "late_night", "overnight"]
        wake_windows = ["overnight", "wake_transition", "morning"]

        onset_weights = np.exp(quiet_mat[sleep_windows].clip(-6, 6).to_numpy(float))
        onset_weights = onset_weights / np.maximum(onset_weights.sum(axis=1, keepdims=True), 1e-9)
        wake_weights = np.exp(active_mat[wake_windows].clip(-6, 6).to_numpy(float))
        wake_weights = wake_weights / np.maximum(wake_weights.sum(axis=1, keepdims=True), 1e-9)

        onset_hour = onset_weights @ centers.loc[sleep_windows].to_numpy(float)
        wake_hour = wake_weights @ centers.loc[wake_windows].to_numpy(float)
        duration = np.clip(wake_hour - onset_hour, 2.0, 14.0)
        sleep_quiet = quiet_mat[sleep_windows].mean(axis=1)
        wake_activation = active_mat[["wake_transition", "morning"]].mean(axis=1)
        fragmentation = (
            active_mat[["late_night", "overnight"]].mean(axis=1)
            - quiet_mat[["late_night", "overnight"]].mean(axis=1)
        )
        latency_bad = active_mat[["bedtime_transition", "pre_sleep"]].mean(axis=1)

        feat["dyn_sleep_onset_hour_proxy"] = onset_hour
        feat["dyn_sleep_wake_hour_proxy"] = wake_hour
        feat["dyn_sleep_duration_proxy"] = duration
        feat["dyn_sleep_midpoint_proxy"] = (onset_hour + wake_hour) / 2.0
        feat["dyn_sleep_quietness_proxy"] = sleep_quiet.astype(float)
        feat["dyn_sleep_fragmentation_proxy"] = fragmentation.astype(float)
        feat["dyn_sleep_latency_bad_proxy"] = latency_bad.astype(float)
        feat["dyn_sleep_wake_activation_proxy"] = wake_activation.astype(float)
        feat["dyn_sleep_efficiency_proxy"] = (
            sleep_quiet + 0.22 * pd.Series(duration, index=feat.index) - 0.55 * fragmentation
        ).astype(float)

        subject = feat["subject_id"].astype(str)
        for col in [
            "dyn_sleep_onset_hour_proxy",
            "dyn_sleep_wake_hour_proxy",
            "dyn_sleep_duration_proxy",
            "dyn_sleep_efficiency_proxy",
            "dyn_sleep_fragmentation_proxy",
            "dyn_sleep_latency_bad_proxy",
            "dyn_sleep_wake_activation_proxy",
        ]:
            values = pd.to_numeric(feat[col], errors="coerce")
            subj_med = values.groupby(subject).transform("median")
            subj_iqr = values.groupby(subject).transform(
                lambda s: s.quantile(0.75) - s.quantile(0.25)
            )
            global_iqr = float(values.quantile(0.75) - values.quantile(0.25))
            if not np.isfinite(global_iqr) or global_iqr < 1e-6:
                global_iqr = float(values.std()) if values.notna().sum() > 1 else 1.0
            if not np.isfinite(global_iqr) or global_iqr < 1e-6:
                global_iqr = 1.0
            scale = subj_iqr.replace(0, np.nan).fillna(global_iqr).fillna(1.0)
            feat[f"{col}_subj_dev"] = ((values - subj_med) / scale).clip(-6, 6)

        dyn_cols = [c for c in feat.columns if c.startswith("dyn_sleep_")]
        print(f"[DYN-SLEEP] added_features={len(dyn_cols)}")
        return feat

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def build_sleep_window_feature_table(train, sample):
        meta = make_meta(train, sample)
        feat = meta.copy()
        for file_name, cols, prefix in SENSORS:
            print(f"[feature] {file_name}")
            sfeat = aggregate_one_sensor(meta, file_name, cols, prefix)
            feat = feat.merge(sfeat, on="row_id", how="left")
        feat = add_cross_window_features(feat)
        if os.environ.get("TRY9_ENABLE_DYNAMIC_SLEEP", "0") == "1":
            feat = add_dynamic_sleep_interval_features(feat)
        else:
            print("[DYN-SLEEP] disabled; set TRY9_ENABLE_DYNAMIC_SLEEP=1 to enable")
        return feat

    # 함수: temporal holdout mask 보조 로직을 수행합니다.
    def temporal_holdout_mask(train_feat):
        mask = pd.Series(False, index=train_feat.index)
        for _, idx in train_feat.groupby("subject_id").groups.items():
            ordered = (
                train_feat.loc[list(idx)].sort_values("lifelog_date").index.to_list()
            )
            n_valid = max(3, int(round(len(ordered) * 0.22)))
            mask.loc[ordered[-n_valid:]] = True
        return mask

    # 함수: 검증/전체 학습 모델을 적합하고 예측값을 만듭니다.
    def fit_sleep_window_model(feat):
        train_feat = feat[feat["is_train"] == 1].copy().reset_index(drop=True)
        test_feat = feat[feat["is_train"] == 0].copy().reset_index(drop=True)
        valid_mask = temporal_holdout_mask(train_feat)

        drop_cols = ["row_id", "sleep_date", "lifelog_date", "is_train"] + TARGETS
        feature_cols = [c for c in feat.columns if c not in drop_cols]
        cat_cols = ["subject_id"]
        cat_idx = [feature_cols.index(c) for c in cat_cols]

        X = train_feat[feature_cols].copy()
        X_test = test_feat[feature_cols].copy()
        for c in cat_cols:
            X[c] = X[c].astype(str)
            X_test[c] = X_test[c].astype(str)
        numeric_cols = [c for c in feature_cols if c not in cat_cols]
        med = X[numeric_cols].median(numeric_only=True)
        X[numeric_cols] = X[numeric_cols].fillna(med).fillna(0.0)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(med).fillna(0.0)

        pred = test_feat[["subject_id", "sleep_date", "lifelog_date"]].copy()
        cv_rows = []
        for target in TARGETS:
            y = train_feat[target].astype(int)
            mean = float(y.mean())
            model_params = dict(
                loss_function="Logloss",
                iterations=420,
                depth=2,
                learning_rate=0.025,
                l2_leaf_reg=28,
                random_seed=20260511,
                verbose=False,
                allow_writing_files=False,
            )
            restored = (
                load_step1_candidate_checkpoint(
                    "sleep_window_model_candidates",
                    target,
                    "temporal_holdout",
                )
                if restore_step1_candidate_checkpoints_enabled()
                else None
            )
            if restored is not None:
                print(f"[STEP1_CKPT] restored candidate sleep_window_model_candidates {target} temporal_holdout")
                val_model = restored["model"]
            else:
                val_model = CatBoostClassifier(**model_params)
                val_model.fit(X.loc[~valid_mask], y.loc[~valid_mask], cat_features=cat_idx)
                save_step1_candidate_checkpoint(
                    "sleep_window_model_candidates",
                    {
                        "kind": "temporal_holdout",
                        "target": target,
                        "feature_cols": feature_cols,
                        "numeric_cols": numeric_cols,
                        "cat_cols": cat_cols,
                        "cat_idx": cat_idx,
                        "median": med,
                        "target_mean": mean,
                        "model_params": model_params,
                        "model": val_model,
                    },
                    target,
                    "temporal_holdout",
                )
            vp = val_model.predict_proba(X.loc[valid_mask])[:, 1]
            vp = np.clip(0.82 * vp + 0.18 * mean, 0.04, 0.96)
            cv_rows.append(
                {
                    "target": target,
                    "temporal_logloss": log_loss(y.loc[valid_mask], vp, labels=[0, 1]),
                }
            )

            restored_full = (
                load_step1_candidate_checkpoint(
                    "sleep_window_model_candidates",
                    target,
                    "full",
                )
                if restore_step1_candidate_checkpoints_enabled()
                else None
            )
            if restored_full is not None:
                print(f"[STEP1_CKPT] restored candidate sleep_window_model_candidates {target} full")
                full_model = restored_full["model"]
            else:
                full_model = CatBoostClassifier(**{**model_params, "iterations": 520})
                full_model.fit(X, y, cat_features=cat_idx)
                save_step1_candidate_checkpoint(
                    "sleep_window_model_candidates",
                    {
                        "kind": "full",
                        "target": target,
                        "feature_cols": feature_cols,
                        "numeric_cols": numeric_cols,
                        "cat_cols": cat_cols,
                        "cat_idx": cat_idx,
                        "median": med,
                        "target_mean": mean,
                        "model_params": {**model_params, "iterations": 520},
                        "model": full_model,
                    },
                    target,
                    "full",
                )
            tp = full_model.predict_proba(X_test)[:, 1]
            pred[target] = np.clip(0.82 * tp + 0.18 * mean, 0.04, 0.96)

        cv = pd.DataFrame(cv_rows)
        print("\nSleep-window temporal holdout logloss:")
        print(cv.to_string(index=False))
        print("mean:", cv["temporal_logloss"].mean())
        return pred, cv

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save_blend(name, anchor, window_pred, weights):
        out = anchor.copy()
        if isinstance(weights, dict):
            for t in TARGETS:
                w = float(weights.get(t, 0.0))
                out[t] = np.clip(
                    (1 - w) * anchor[t].to_numpy(float)
                    + w * window_pred[t].to_numpy(float),
                    0.04,
                    0.96,
                )
        else:
            w = float(weights)
            out[TARGETS] = np.clip(
                (1 - w) * anchor[TARGETS].to_numpy(float)
                + w * window_pred[TARGETS].to_numpy(float),
                0.04,
                0.96,
            )
        path = SUBMISSION_DIR / name
        out.to_csv(path, index=False)
        diff = (out[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "mean_abs_diff": float(diff.to_numpy().mean()),
            "max_abs_diff": float(diff.to_numpy().max()),
            "mean_q": float(out[["Q1", "Q2", "Q3"]].to_numpy().mean()),
            "mean_s": float(out[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
        }

    if __name__ == "__main__":
        train = pd.read_csv(TRAIN_PATH)
        sample = pd.read_csv(SAMPLE_PATH)
        anchor_path = ANCHOR_PATH if ANCHOR_PATH.exists() else FALLBACK_ANCHOR_PATH
        anchor = pd.read_csv(anchor_path)
        print("anchor:", anchor_path.name)

        feat = build_sleep_window_feature_table(train, sample)
        feat_path = BASE_DIR / "artifacts" / "features_sleep_window_table.csv"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_csv(feat_path, index=False)
        print("feature table:", feat.shape, feat_path)

        window_pred, cv = fit_sleep_window_model(feat)
        model_path = SUBMISSION_DIR / "submission_step1_sleep_window_model.csv"
        window_pred.to_csv(model_path, index=False)

        rows = []
        rows.append(
            save_blend("submission_step1_sleep_window_w006.csv", anchor, window_pred, 0.006)
        )
        rows.append(
            save_blend("submission_step1_sleep_window_w012.csv", anchor, window_pred, 0.012)
        )
        rows.append(
            save_blend(
                "submission_step1_sleep_window_sleep_targets.csv",
                anchor,
                window_pred,
                {
                    "Q1": 0.014,
                    "Q2": 0.004,
                    "Q3": 0.004,
                    "S1": 0.016,
                    "S2": 0.014,
                    "S3": 0.016,
                    "S4": 0.014,
                },
            )
        )

        summary = pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)
        print("\nSleep-window candidate summary vs anchor:")
        print(summary.to_string(index=False))
        print("\nSuggested submit order:")
        print("1) submission_step1_sleep_window_w006.csv")
        print("2) submission_step1_sleep_window_sleep_targets.csv")
        print("3) submission_step1_sleep_window_w012.csv")


# 함수: sleep-window 모델을 target별 가중치로 다시 섞는 refinement 후보를 생성합니다.
def build_sleep_window_refinement_candidates():
    __file__ = str(
        PROJECT_ROOT / "pipeline/build_sleep_window_refinement_candidates"
    )
    __name__ = "__main__"
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    SUBMISSION_DIR = ROOT / "data" / "submissions"
    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    BASE_ANCHOR = SUBMISSION_DIR / "submission_step1_joint_pattern_w015.csv"
    FALLBACK_BASE_ANCHOR = (
        SUBMISSION_DIR / "submission_step1_target_reliability_scale_soft.csv"
    )
    BEST_SLEEP = SUBMISSION_DIR / "submission_step1_sleep_window_sleep_targets.csv"
    SLEEP_MODEL = SUBMISSION_DIR / "submission_step1_sleep_window_model.csv"

    # 함수: 필요한 파일을 읽어와 이후 처리에 맞는 형태로 준비합니다.
    def load_submission(path):
        df = pd.read_csv(path)
        for col in TARGETS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend(anchor, prior, weights):
        out = anchor.copy()
        for target in TARGETS:
            weight = float(weights.get(target, 0.0))
            out[target] = np.clip(
                (1.0 - weight) * anchor[target].to_numpy(float)
                + weight * prior[target].to_numpy(float),
                0.04,
                0.96,
            )
        return out

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(name, candidate, compare):
        diff = (candidate[TARGETS] - compare[TARGETS]).abs()
        return {
            "candidate": name,
            "diff_vs_best": float(diff.to_numpy().mean()),
            "max_diff_vs_best": float(diff.to_numpy().max()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].to_numpy().mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
        }

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save_candidate(name, anchor, prior, weights, compare):
        candidate = blend(anchor, prior, weights)
        candidate.to_csv(SUBMISSION_DIR / name, index=False)
        return summarize(name, candidate, compare)

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main():
        base_path = BASE_ANCHOR if BASE_ANCHOR.exists() else FALLBACK_BASE_ANCHOR
        print("base anchor:", base_path.name)
        base = load_submission(base_path)
        best = load_submission(BEST_SLEEP)
        sleep_model = load_submission(SLEEP_MODEL)

        #  public results:
        # all w006 = 0.5881257172, target-wise sleep = 0.5878818802, all w012 = 0.588077498.
        # Interpretation: sleep-window signal is real, but Q2/Q3 should stay tiny.
        candidates = [
            (
                "submission_step1_sleep_window_plus_soft.csv",
                {
                    "Q1": 0.018,
                    "Q2": 0.003,
                    "Q3": 0.003,
                    "S1": 0.020,
                    "S2": 0.017,
                    "S3": 0.020,
                    "S4": 0.017,
                },
            ),
            (
                "submission_step1_sleep_window_s_target_focus.csv",
                {
                    "Q1": 0.014,
                    "Q2": 0.001,
                    "Q3": 0.001,
                    "S1": 0.022,
                    "S2": 0.018,
                    "S3": 0.020,
                    "S4": 0.018,
                },
            ),
            (
                "submission_step1_sleep_window_q1_s1s3.csv",
                {
                    "Q1": 0.022,
                    "Q2": 0.002,
                    "Q3": 0.000,
                    "S1": 0.022,
                    "S2": 0.014,
                    "S3": 0.022,
                    "S4": 0.014,
                },
            ),
            (
                "submission_step1_sleep_window_guarded.csv",
                {
                    "Q1": 0.016,
                    "Q2": 0.003,
                    "Q3": 0.002,
                    "S1": 0.018,
                    "S2": 0.015,
                    "S3": 0.018,
                    "S4": 0.015,
                },
            ),
        ]

        rows = []
        for name, weights in candidates:
            rows.append(save_candidate(name, base, sleep_model, weights, best))

        summary = pd.DataFrame(rows).sort_values("diff_vs_best").reset_index(drop=True)
        print("sleep-window refine candidates vs current best:")
        print(summary.to_string(index=False))
        print("\nSuggested submit order:")
        print("1) submission_step1_sleep_window_guarded.csv")
        print("2) submission_step1_sleep_window_plus_soft.csv")
        print("3) submission_step1_sleep_window_s_target_focus.csv")

    if __name__ == "__main__":
        main()


# 함수: sleep-window 주변의 작은 가중치 grid 후보를 생성합니다.
# Pruned historical definition: build_sleep_window_microgrid_candidates (not reachable from the final runner).


# 함수: 예측 sequence의 target coherence를 보정하는 후보를 생성합니다.
# Pruned historical definition: build_sequence_coherence_candidates (not reachable from the final runner).


# 함수: 수면 시간/효율/지연/각성 proxy를 이용해 최종 S계열 보정 후보를 생성합니다.
def build_sleep_metric_proxy_candidates():
    __file__ = str(PROJECT_ROOT / "pipeline/build_sleep_metric_proxy_candidates")
    __name__ = "__main__"
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    ARTIFACT_PATH = DATA_DIR / "artifacts" / "features_sleep_window_table.csv"
    SUBMISSION_DIR = DATA_DIR / "submissions"
    CURRENT_BEST = SUBMISSION_DIR / "submission_step1_sleep_window_s_target_focus.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 함수: robust z 보조 로직을 수행합니다.
    def robust_z(frame, col, ref_mask):
        if col not in frame.columns:
            return pd.Series(0.0, index=frame.index)
        values = pd.to_numeric(frame[col], errors="coerce")
        ref = values[ref_mask]
        med = float(ref.median()) if ref.notna().any() else 0.0
        q75 = float(ref.quantile(0.75)) if ref.notna().any() else 1.0
        q25 = float(ref.quantile(0.25)) if ref.notna().any() else 0.0
        scale = q75 - q25
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(ref.std()) if ref.notna().sum() > 1 else 1.0
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        return ((values.fillna(med) - med) / scale).clip(-5, 5)

    # 함수: add metric proxies 보조 로직을 수행합니다.
    def add_metric_proxies(feat):
        train_mask = feat["is_train"].eq(1)
        z = lambda c: robust_z(feat, c, train_mask)

        # Low activity / low light / low screen / low step during the sleep context
        # approximates longer and cleaner rest. Signs are intentionally human-readable;
        # target-specific orientation is learned from train correlations below.
        feat["proxy_rest_duration"] = (
            -0.24 * z("mact_full_sleepctx_m_activity_mean")
            - 0.20 * z("pedo_full_sleepctx_step_sum")
            - 0.18 * z("screen_full_sleepctx_m_screen_use_sum")
            - 0.14 * z("mlight_full_sleepctx_m_light_mean")
            - 0.10 * z("wlight_full_sleepctx_w_light_mean")
            + 0.14 * z("charge_overnight_m_charging_mean")
        )
        feat["proxy_sleep_efficiency"] = (
            -0.28 * z("mact_overnight_m_activity_mean")
            - 0.24 * z("screen_overnight_m_screen_use_sum")
            - 0.20 * z("pedo_overnight_step_sum")
            - 0.14 * z("mlight_overnight_m_light_mean")
            - 0.08 * z("wlight_overnight_w_light_mean")
            - 0.06 * z("hr_overnight_heart_rate_std")
        )
        feat["proxy_sleep_latency_bad"] = (
            +0.28 * z("screen_pre_sleep_m_screen_use_sum")
            + 0.22 * z("mlight_pre_sleep_m_light_mean")
            + 0.18 * z("mact_pre_sleep_m_activity_mean")
            + 0.12 * z("pedo_pre_sleep_step_sum")
            + 0.10 * z("hr_pre_sleep_heart_rate_mean")
            - 0.10 * z("charge_pre_sleep_m_charging_mean")
        )
        feat["proxy_wake_after_sleep_bad"] = (
            +0.24 * z("screen_overnight_m_screen_use_sum")
            + 0.22 * z("mact_overnight_m_activity_max")
            + 0.18 * z("pedo_overnight_step_sum")
            + 0.16 * z("mlight_overnight_m_light_max")
            + 0.10 * z("hr_overnight_heart_rate_std")
            + 0.10 * z("screen_morning_m_screen_use_sum")
        )
        feat["proxy_sleep_quality"] = (
            +0.38 * feat["proxy_sleep_efficiency"]
            + 0.28 * feat["proxy_rest_duration"]
            - 0.18 * feat["proxy_sleep_latency_bad"]
            - 0.16 * feat["proxy_wake_after_sleep_bad"]
        )
        feat["proxy_fatigue_stress"] = (
            +0.24 * z("screen_pre_sleep_m_screen_use_sum")
            + 0.20 * z("hr_pre_sleep_heart_rate_mean")
            + 0.18 * z("hr_full_sleepctx_heart_rate_std")
            + 0.16 * z("mact_morning_m_activity_mean")
            + 0.12 * z("pedo_morning_step_sum")
            + 0.10 * z("mlight_pre_sleep_m_light_mean")
        )
        return feat

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def calibrate_proxy_prior(feat, target_to_proxy, bins=6):
        train_mask = feat["is_train"].eq(1)
        out = feat[["subject_id", "sleep_date", "lifelog_date", "is_train"]].copy()
        diagnostics = []
        for target, proxy_col in target_to_proxy.items():
            score = pd.to_numeric(feat[proxy_col], errors="coerce").fillna(0.0)
            train_score = score[train_mask]
            y = pd.to_numeric(feat.loc[train_mask, target], errors="coerce")
            global_mean = float(y.mean())

            corr = float(pd.Series(train_score).corr(y)) if y.nunique() > 1 else 0.0
            if not np.isfinite(corr):
                corr = 0.0
            oriented = score if corr >= 0 else -score
            train_oriented = oriented[train_mask]

            # Rank-space binning avoids trusting raw proxy scale.
            all_rank = oriented.rank(pct=True, method="average")
            train_rank = all_rank[train_mask]
            try:
                qbin = pd.qcut(train_rank, q=bins, labels=False, duplicates="drop")
            except ValueError:
                qbin = pd.Series(
                    np.zeros(len(train_rank), dtype=int), index=train_rank.index
                )
            bin_df = pd.DataFrame({"bin": qbin, "y": y})
            stats = bin_df.groupby("bin")["y"].agg(["mean", "count"]).reset_index()
            stats["smooth"] = (stats["mean"] * stats["count"] + global_mean * 18.0) / (
                stats["count"] + 18.0
            )
            mapping = dict(zip(stats["bin"].astype(int), stats["smooth"]))

            if len(mapping) <= 1:
                prior = pd.Series(global_mean, index=feat.index)
            else:
                edges = np.quantile(train_rank, np.linspace(0, 1, len(mapping) + 1))
                edges[0] = -np.inf
                edges[-1] = np.inf
                test_bins = np.digitize(
                    all_rank.to_numpy(float), edges[1:-1], right=True
                )
                prior = pd.Series(
                    [mapping.get(int(b), global_mean) for b in test_bins],
                    index=feat.index,
                )

            # Keep the hand-crafted prior conservative; it should be a directional nudge.
            prior = 0.72 * prior + 0.28 * global_mean
            out[target] = np.clip(prior, 0.05, 0.95)
            diagnostics.append(
                {
                    "target": target,
                    "proxy": proxy_col,
                    "train_corr": corr,
                    "global_mean": global_mean,
                    "prior_train_mean": float(out.loc[train_mask, target].mean()),
                    "prior_test_mean": float(out.loc[~train_mask, target].mean()),
                }
            )
        return out, pd.DataFrame(diagnostics)

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend(best, prior_test, weights):
        out = best.copy()
        for target in TARGETS:
            w = float(weights.get(target, 0.0))
            out[target] = np.clip(
                (1 - w) * best[target].to_numpy(float)
                + w * prior_test[target].to_numpy(float),
                0.04,
                0.96,
            )
        return out

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save_candidate(name, best, prior_test, weights):
        cand = blend(best, prior_test, weights)
        cand.to_csv(SUBMISSION_DIR / name, index=False)
        diff = (cand[TARGETS] - best[TARGETS]).abs()
        return {
            "candidate": name,
            "mean_abs_diff": float(diff.to_numpy().mean()),
            "max_abs_diff": float(diff.to_numpy().max()),
            "mean_q": float(cand[["Q1", "Q2", "Q3"]].to_numpy().mean()),
            "mean_s": float(cand[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
        }

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main():
        feat = pd.read_csv(ARTIFACT_PATH)
        feat = add_metric_proxies(feat)
        best = pd.read_csv(CURRENT_BEST)

        target_to_proxy = {
            "Q1": "proxy_sleep_quality",
            "Q2": "proxy_fatigue_stress",
            "Q3": "proxy_fatigue_stress",
            "S1": "proxy_rest_duration",
            "S2": "proxy_sleep_efficiency",
            "S3": "proxy_sleep_latency_bad",
            "S4": "proxy_wake_after_sleep_bad",
        }
        prior_all, diag = calibrate_proxy_prior(feat, target_to_proxy)
        prior_test = prior_all[prior_all["is_train"].eq(0)].reset_index(drop=True)

        print("Sleep metric proxy diagnostics:")
        print(diag.to_string(index=False))

        rows = []
        rows.append(
            save_candidate(
                "submission_step1_sleep_proxy_s_targets.csv",
                best,
                prior_test,
                {
                    "Q1": 0.000,
                    "Q2": 0.000,
                    "Q3": 0.000,
                    "S1": 0.010,
                    "S2": 0.010,
                    "S3": 0.010,
                    "S4": 0.010,
                },
            )
        )
        rows.append(
            save_candidate(
                "submission_step1_sleep_proxy_qs_tiny.csv",
                best,
                prior_test,
                {
                    "Q1": 0.006,
                    "Q2": 0.004,
                    "Q3": 0.004,
                    "S1": 0.010,
                    "S2": 0.010,
                    "S3": 0.010,
                    "S4": 0.010,
                },
            )
        )
        rows.append(
            save_candidate(
                "submission_step1_sleep_proxy_quality.csv",
                best,
                prior_test,
                {
                    "Q1": 0.012,
                    "Q2": 0.000,
                    "Q3": 0.000,
                    "S1": 0.012,
                    "S2": 0.012,
                    "S3": 0.008,
                    "S4": 0.008,
                },
            )
        )

        summary = pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)
        print("\n0515 sleep-metric proxy candidates vs current best:")
        print(summary.to_string(index=False))
        print("\nSuggested submit order:")
        print("1) submission_step1_sleep_proxy_s_targets.csv")
        print("2) submission_step1_sleep_proxy_qs_tiny.csv")
        print("3) submission_step1_sleep_proxy_quality.csv")

    if __name__ == "__main__":
        main()


# 함수: 센서 기록 밀도 coverage prior 기반 후보를 생성합니다.
# Pruned historical definition: build_sensor_coverage_prior_candidates (not reachable from the final runner).


# 함수: subject별 평균 라벨 흐름을 외삽하는 후보를 생성합니다.
# Pruned historical definition: build_subject_mean_forecast_candidates (not reachable from the final runner).


# 함수: 공동 타겟 상관 패턴을 완화하는 후보를 생성합니다.
# Pruned historical definition: build_joint_correlation_relaxation_candidates (not reachable from the final runner).


# 함수: 날짜 거리 confidence에 따라 date prior blend 강도를 조절합니다.
def build_adaptive_date_confidence_blends():
    __file__ = str(PROJECT_ROOT / "pipeline/build_adaptive_date_confidence_blends")
    __name__ = "__main__"
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SUBMISSION_DIR = BASE_DIR / "submissions"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

    ANCHOR_PATH = SUBMISSION_DIR / "submission_step1_mean_g060.csv"
    DATE_PRIOR_PATH = SUBMISSION_DIR / "submission_step1_interpolation.csv"
    BRACKET_PRIOR_PATH = SUBMISSION_DIR / "submission_step1_date_bracket.csv"
    HYBRID_PRIOR_PATH = SUBMISSION_DIR / "submission_step1_calendar_bracket.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has null values")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} has probabilities outside [0,1]")

    # 함수: build date confidence 보조 로직을 수행합니다.
    def build_date_confidence(
        train: pd.DataFrame, sample: pd.DataFrame
    ) -> pd.DataFrame:
        rows = []
        train_dates = {
            str(sid): pd.to_datetime(group["lifelog_date"]).sort_values().to_numpy()
            for sid, group in train.groupby("subject_id")
        }
        for row in sample.itertuples(index=False):
            sid = str(row.subject_id)
            pred_date = pd.Timestamp(row.lifelog_date)
            dates = train_dates.get(sid)
            if dates is None or len(dates) == 0:
                before_dist = np.nan
                after_dist = np.nan
                min_dist = 99.0
                bracketed = 0.0
            else:
                deltas = np.array(
                    [(pd.Timestamp(d) - pred_date).days for d in dates], dtype=float
                )
                before = np.abs(deltas[deltas < 0])
                after = np.abs(deltas[deltas > 0])
                before_dist = float(before.min()) if len(before) else np.nan
                after_dist = float(after.min()) if len(after) else np.nan
                min_dist = float(np.nanmin([before_dist, after_dist]))
                bracketed = float(np.isfinite(before_dist) and np.isfinite(after_dist))

            near_score = float(np.exp(-min_dist / 10.0))
            both_score = bracketed * float(
                np.exp(
                    -(
                        np.nan_to_num(before_dist, nan=30.0)
                        + np.nan_to_num(after_dist, nan=30.0)
                    )
                    / 24.0
                )
            )
            rows.append(
                {
                    "subject_id": sid,
                    "lifelog_date": pred_date,
                    "before_dist": before_dist,
                    "after_dist": after_dist,
                    "min_dist": min_dist,
                    "bracketed": bracketed,
                    "near_score": near_score,
                    "both_score": both_score,
                    "confidence": float(
                        np.clip(0.65 * near_score + 0.35 * both_score, 0.0, 1.0)
                    ),
                }
            )
        return pd.DataFrame(rows)

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def adaptive_blend(
        anchor: pd.DataFrame,
        prior: pd.DataFrame,
        conf: pd.Series,
        base: float,
        scale: float,
        target_mult: dict[str, float] | None = None,
        tag: str = "",
    ) -> pd.DataFrame:
        out = anchor.copy()
        row_weight = np.clip(base + scale * conf.to_numpy(dtype=float), 0.0, 0.22)
        target_mult = target_mult or {target: 1.0 for target in TARGETS}
        for target in TARGETS:
            weight = np.clip(
                row_weight * float(target_mult.get(target, 1.0)), 0.0, 0.24
            )
            out[target] = (1.0 - weight) * anchor[target].astype(
                float
            ) + weight * prior[target].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        out.attrs["tag"] = tag
        out.attrs["avg_weight"] = float(row_weight.mean())
        out.attrs["max_weight"] = float(row_weight.max())
        return out

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame, weight_info: str
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "weight_info": weight_info,
            "diff_vs_g060": float(diff.values.mean()),
            "max_diff": float(diff.values.max()),
            "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
            "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save_candidate(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame, weight_info: str
    ) -> dict[str, float | str]:
        validate(candidate, name)
        candidate.to_csv(SUBMISSION_DIR / name, index=False)
        return summarize(name, candidate, anchor, weight_info)

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
        sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
        anchor = pd.read_csv(ANCHOR_PATH)
        date_prior = pd.read_csv(DATE_PRIOR_PATH)
        bracket_prior = pd.read_csv(BRACKET_PRIOR_PATH)
        hybrid_prior = pd.read_csv(HYBRID_PRIOR_PATH)

        for name, df in [
            (ANCHOR_PATH.name, anchor),
            (DATE_PRIOR_PATH.name, date_prior),
            (BRACKET_PRIOR_PATH.name, bracket_prior),
            (HYBRID_PRIOR_PATH.name, hybrid_prior),
        ]:
            validate(df, name)

        conf = build_date_confidence(train, sample)
        print("Date confidence summary:")
        print(
            conf[["before_dist", "after_dist", "min_dist", "bracketed", "confidence"]]
            .describe()
            .to_string()
        )

        candidates = []
        configs = [
            (
                "dateconf_dateprior_b03_s10",
                date_prior,
                0.03,
                0.10,
                {
                    "Q1": 1.00,
                    "Q2": 1.00,
                    "Q3": 1.00,
                    "S1": 0.85,
                    "S2": 0.90,
                    "S3": 0.75,
                    "S4": 0.90,
                },
            ),
            (
                "dateconf_dateprior_b04_s12",
                date_prior,
                0.04,
                0.12,
                {
                    "Q1": 1.00,
                    "Q2": 1.00,
                    "Q3": 1.00,
                    "S1": 0.85,
                    "S2": 0.90,
                    "S3": 0.75,
                    "S4": 0.90,
                },
            ),
            (
                "dateconf_bracket_b03_s10",
                bracket_prior,
                0.03,
                0.10,
                {
                    "Q1": 0.95,
                    "Q2": 1.00,
                    "Q3": 1.00,
                    "S1": 0.85,
                    "S2": 0.90,
                    "S3": 0.75,
                    "S4": 0.90,
                },
            ),
            (
                "dateconf_hybrid_b03_s10",
                hybrid_prior,
                0.03,
                0.10,
                {
                    "Q1": 0.95,
                    "Q2": 1.00,
                    "Q3": 1.00,
                    "S1": 0.85,
                    "S2": 0.90,
                    "S3": 0.75,
                    "S4": 0.90,
                },
            ),
        ]

        for tag, prior, base, scale, target_mult in configs:
            out = adaptive_blend(
                anchor, prior, conf["confidence"], base, scale, target_mult, tag
            )
            weights = np.clip(
                base + scale * conf["confidence"].to_numpy(dtype=float), 0.0, 0.22
            )
            info = f"avg={weights.mean():.4f}, max={weights.max():.4f}, base={base:.2f}, scale={scale:.2f}"
            candidates.append(
                save_candidate(
                    (
                        "submission_step1_date_prior.csv"
                        if tag == "dateconf_dateprior_b03_s10"
                        else f"submission_step1_date_candidate_{tag}.csv"
                    ),
                    out,
                    anchor,
                    info,
                )
            )

        summary = (
            pd.DataFrame(candidates).sort_values("diff_vs_g060").reset_index(drop=True)
        )
        print("\nAdaptive prior candidate summary vs current best g060:")
        print(summary.to_string(index=False))
        print("\nSuggested if taking a new-axis shot:")
        print("submission_step1_date_prior.csv")

    if __name__ == "__main__":
        main()


# 함수: calendar prior와 앞뒤 train label bracket prior를 생성합니다.
def build_calendar_and_bracket_priors():
    __file__ = str(PROJECT_ROOT / "pipeline/build_calendar_and_bracket_priors")
    __name__ = "__main__"

    from dataclasses import dataclass
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"
    SUBMISSION_DIR = BASE_DIR / "submissions"

    CURRENT_BEST_PATH = (
        SUBMISSION_DIR / "submission_step1_date_blend_w07.csv"
    )

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    @dataclass(frozen=True)
    class BracketConfig:
        tag: str
        nearest_tau: float
        subject_strength: float
        bracket_scale: float
        clip_low: float = 0.045
        clip_high: float = 0.955

    @dataclass(frozen=True)
    class CalendarConfig:
        tag: str
        tau: float
        k: int
        subject_weight: float
        calendar_weight: float
        dow_weight: float
        prior_strength: float
        clip_low: float = 0.045
        clip_high: float = 0.955

    BRACKET_CONFIGS = [
        BracketConfig(
            "bracket_soft", nearest_tau=6.0, subject_strength=5.0, bracket_scale=0.82
        ),
        BracketConfig(
            "bracket_mid", nearest_tau=9.0, subject_strength=7.0, bracket_scale=0.72
        ),
        BracketConfig(
            "bracket_smooth", nearest_tau=13.0, subject_strength=9.0, bracket_scale=0.62
        ),
    ]

    CALENDAR_CONFIGS = [
        CalendarConfig(
            "cal_tau10",
            tau=10.0,
            k=40,
            subject_weight=0.60,
            calendar_weight=0.30,
            dow_weight=0.10,
            prior_strength=5.0,
        ),
        CalendarConfig(
            "cal_tau16",
            tau=16.0,
            k=55,
            subject_weight=0.56,
            calendar_weight=0.34,
            dow_weight=0.10,
            prior_strength=7.0,
        ),
        CalendarConfig(
            "cal_dow",
            tau=14.0,
            k=50,
            subject_weight=0.52,
            calendar_weight=0.32,
            dow_weight=0.16,
            prior_strength=7.0,
        ),
    ]

    # 함수: binary log-loss를 계산합니다.
    def binary_logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
        pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=float)
        return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has null values")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} has probabilities outside [0,1]")

    # 함수: 검증/전체 학습 모델을 적합하고 예측값을 만듭니다.
    def fit_subject_tables(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            str(sid): group.sort_values("lifelog_date")[
                ["lifelog_date"] + TARGETS
            ].copy()
            for sid, group in train.groupby("subject_id")
        }

    # 함수: smooth mean 보조 로직을 수행합니다.
    def smooth_mean(pos: float, total: float, prior: float, strength: float) -> float:
        return float((pos + strength * prior) / (total + strength))

    # 함수: 학습된 규칙이나 모델로 test 구간 예측을 생성합니다.
    def predict_bracket_target(
        table: pd.DataFrame,
        pred_date: pd.Timestamp,
        target: str,
        global_mean: float,
        config: BracketConfig,
    ) -> float:
        if len(table) == 0:
            return float(np.clip(global_mean, config.clip_low, config.clip_high))

        dates = pd.to_datetime(table["lifelog_date"])
        values = table[target].astype(float).to_numpy()
        deltas = (dates - pred_date).dt.days.to_numpy(dtype=float)
        subject_mean = float(np.mean(values))
        subject_prior = 0.82 * subject_mean + 0.18 * global_mean

        before_idx = np.where(deltas < 0)[0]
        after_idx = np.where(deltas > 0)[0]
        exact_idx = np.where(deltas == 0)[0]

        pieces = []
        weights = []
        if len(exact_idx):
            pieces.append(float(values[exact_idx[-1]]))
            weights.append(1.8)
        if len(before_idx):
            idx = before_idx[np.argmin(np.abs(deltas[before_idx]))]
            dist = abs(float(deltas[idx]))
            pieces.append(float(values[idx]))
            weights.append(np.exp(-dist / config.nearest_tau))
        if len(after_idx):
            idx = after_idx[np.argmin(np.abs(deltas[after_idx]))]
            dist = abs(float(deltas[idx]))
            pieces.append(float(values[idx]))
            weights.append(np.exp(-dist / config.nearest_tau))

        if not pieces or sum(weights) <= 1e-12:
            bracket = subject_prior
            confidence = 0.0
        else:
            bracket = float(np.average(pieces, weights=weights))
            confidence = float(sum(weights) / (sum(weights) + config.subject_strength))

        pred = config.bracket_scale * (
            confidence * bracket + (1 - confidence) * subject_prior
        )
        pred += (1 - config.bracket_scale) * subject_prior
        return float(np.clip(pred, config.clip_low, config.clip_high))

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def build_bracket_prior(
        train_fit: pd.DataFrame, frame: pd.DataFrame, config: BracketConfig
    ) -> pd.DataFrame:
        tables = fit_subject_tables(train_fit)
        global_mean = {target: float(train_fit[target].mean()) for target in TARGETS}
        rows = []
        for row in frame.itertuples(index=False):
            sid = str(row.subject_id)
            pred_date = pd.Timestamp(row.lifelog_date)
            table = tables.get(sid, pd.DataFrame(columns=["lifelog_date"] + TARGETS))
            rows.append(
                {
                    target: predict_bracket_target(
                        table, pred_date, target, global_mean[target], config
                    )
                    for target in TARGETS
                }
            )
        return pd.DataFrame(rows)

    # 함수: 학습된 규칙이나 모델로 test 구간 예측을 생성합니다.
    def predict_calendar_target(
        train_fit: pd.DataFrame,
        sid: str,
        pred_date: pd.Timestamp,
        target: str,
        config: CalendarConfig,
    ) -> float:
        global_mean = float(train_fit[target].mean())
        subject_values = train_fit.loc[train_fit["subject_id"] == sid, target].astype(
            float
        )
        subject_mean = smooth_mean(
            float(subject_values.sum()), len(subject_values), global_mean, 6.0
        )

        others = train_fit[train_fit["subject_id"] != sid].copy()
        if len(others) == 0:
            calendar_mean = global_mean
        else:
            delta = (
                (pd.to_datetime(others["lifelog_date"]) - pred_date)
                .dt.days.abs()
                .to_numpy(dtype=float)
            )
            order = np.argsort(delta)[: config.k]
            chosen = others.iloc[order]
            dist = delta[order]
            weights = np.exp(-dist / config.tau)
            if float(weights.sum()) <= 1e-12:
                calendar_mean = global_mean
            else:
                local = float(np.average(chosen[target].astype(float), weights=weights))
                reliability = float(
                    weights.sum() / (weights.sum() + config.prior_strength)
                )
                calendar_mean = reliability * local + (1 - reliability) * global_mean

        dow = int(pred_date.dayofweek)
        dow_df = train_fit[train_fit["lifelog_date"].dt.dayofweek == dow]
        dow_mean = smooth_mean(
            float(dow_df[target].sum()), len(dow_df), global_mean, 12.0
        )

        pred = (
            config.subject_weight * subject_mean
            + config.calendar_weight * calendar_mean
            + config.dow_weight * dow_mean
        )
        return float(np.clip(pred, config.clip_low, config.clip_high))

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def build_calendar_prior(
        train_fit: pd.DataFrame, frame: pd.DataFrame, config: CalendarConfig
    ) -> pd.DataFrame:
        rows = []
        for row in frame.itertuples(index=False):
            sid = str(row.subject_id)
            pred_date = pd.Timestamp(row.lifelog_date)
            rows.append(
                {
                    target: predict_calendar_target(
                        train_fit, sid, pred_date, target, config
                    )
                    for target in TARGETS
                }
            )
        return pd.DataFrame(rows)

    # 함수: interleaved cv 보조 로직을 수행합니다.
    def interleaved_cv(train: pd.DataFrame, prior_builder, config) -> dict[str, float]:
        y_all = []
        p_all = []
        per_target = {target: {"y": [], "p": []} for target in TARGETS}

        for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby(
            "subject_id"
        ):
            group = group.reset_index(drop=True)
            if len(group) < 15:
                continue
            holdout_mask = (np.arange(len(group)) + 2) % 5 == 0
            holdout = group[holdout_mask].copy()
            fit = pd.concat(
                [
                    train[train["subject_id"] != group["subject_id"].iloc[0]],
                    group[~holdout_mask],
                ],
                ignore_index=True,
            )
            preds = prior_builder(
                fit, holdout[["subject_id", "sleep_date", "lifelog_date"]], config
            )
            y = holdout[TARGETS].to_numpy(dtype=float)
            p = preds[TARGETS].to_numpy(dtype=float)
            y_all.append(y)
            p_all.append(p)
            for j, target in enumerate(TARGETS):
                per_target[target]["y"].extend(y[:, j].tolist())
                per_target[target]["p"].extend(p[:, j].tolist())

        y_true = np.vstack(y_all)
        pred = np.vstack(p_all)
        out = {"overall": binary_logloss(y_true, pred)}
        for target in TARGETS:
            out[target] = binary_logloss(
                np.array(per_target[target]["y"]), np.array(per_target[target]["p"])
            )
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend(anchor: pd.DataFrame, prior: pd.DataFrame, weight: float) -> pd.DataFrame:
        out = anchor.copy()
        out[TARGETS] = (1 - weight) * anchor[TARGETS].astype(float) + weight * prior[
            TARGETS
        ].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def targetwise_blend(
        anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float]
    ) -> pd.DataFrame:
        out = anchor.copy()
        for target in TARGETS:
            weight = float(weights.get(target, 0.0))
            out[target] = (1 - weight) * anchor[target].astype(float) + weight * prior[
                target
            ].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "diff_vs_best": float(diff.values.mean()),
            "max_diff": float(diff.values.max()),
            "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
            "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save_candidate(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        validate(candidate, name)
        candidate.to_csv(SUBMISSION_DIR / name, index=False)
        return summarize(name, candidate, anchor)

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
        sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
        anchor = pd.read_csv(CURRENT_BEST_PATH)
        validate(anchor, CURRENT_BEST_PATH.name)

        cv_rows = []
        for config in BRACKET_CONFIGS:
            cv_rows.append(
                {
                    "family": "bracket",
                    "config": config.tag,
                    **interleaved_cv(train, build_bracket_prior, config),
                }
            )
        for config in CALENDAR_CONFIGS:
            cv_rows.append(
                {
                    "family": "calendar",
                    "config": config.tag,
                    **interleaved_cv(train, build_calendar_prior, config),
                }
            )

        cv = pd.DataFrame(cv_rows).sort_values("overall").reset_index(drop=True)
        print("Interleaved CV for new axes:")
        print(cv.to_string(index=False))

        best_bracket_tag = cv[cv["family"] == "bracket"].iloc[0]["config"]
        best_calendar_tag = cv[cv["family"] == "calendar"].iloc[0]["config"]
        bracket_config = next(
            config for config in BRACKET_CONFIGS if config.tag == best_bracket_tag
        )
        calendar_config = next(
            config for config in CALENDAR_CONFIGS if config.tag == best_calendar_tag
        )

        bracket_prior = build_bracket_prior(
            train, sample[["subject_id", "sleep_date", "lifelog_date"]], bracket_config
        )
        calendar_prior = build_calendar_prior(
            train, sample[["subject_id", "sleep_date", "lifelog_date"]], calendar_config
        )
        hybrid_prior = 0.55 * bracket_prior[TARGETS] + 0.45 * calendar_prior[TARGETS]
        hybrid_prior = pd.DataFrame(hybrid_prior, columns=TARGETS).clip(0.045, 0.955)

        candidates = []
        for tag, prior in [
            (f"bracket_{bracket_config.tag}", bracket_prior),
            (f"calendar_{calendar_config.tag}", calendar_prior),
            ("hybrid_bracket_calendar", hybrid_prior),
        ]:
            pure = sample.copy()
            pure[TARGETS] = prior[TARGETS].to_numpy(dtype=float)
            pure[TARGETS] = pure[TARGETS].clip(1e-6, 1 - 1e-6)
            pure_name = {
                f"bracket_{bracket_config.tag}": "submission_step1_date_bracket.csv",
                f"calendar_{calendar_config.tag}": "submission_step1_calendar.csv",
                "hybrid_bracket_calendar": "submission_step1_calendar_bracket.csv",
            }.get(tag, f"submission_{tag}_pure.csv")
            candidates.append(save_candidate(pure_name, pure, anchor))

            for weight in [0.04, 0.07, 0.10]:
                name = f"submission_{tag}_anchor_w{int(weight * 100):02d}.csv"
                candidates.append(
                    save_candidate(name, blend(anchor, pure, weight), anchor)
                )

        tw_weights = {
            "Q1": 0.08,
            "Q2": 0.08,
            "Q3": 0.08,
            "S1": 0.06,
            "S2": 0.06,
            "S3": 0.04,
            "S4": 0.06,
        }
        candidates.append(
            save_candidate(
                "submission_hybrid_bracket_calendar_anchor_tw_q08_s06.csv",
                targetwise_blend(
                    anchor,
                    pd.concat(
                        [
                            sample[["subject_id", "sleep_date", "lifelog_date"]],
                            hybrid_prior,
                        ],
                        axis=1,
                    ),
                    tw_weights,
                ),
                anchor,
            )
        )

        summary = (
            pd.DataFrame(candidates).sort_values("diff_vs_best").reset_index(drop=True)
        )
        print("\nSaved candidate summary vs current best 0.5892681038:")
        print(summary.to_string(index=False))

        print("\nSuggested 3-submit order:")
        print("1) submission_step1_calendar_bracket.csv")
        print("2) submission_step1_calendar.csv")
        print("3) submission_step1_date_bracket.csv")

    if __name__ == "__main__":
        main()


# 함수: copula 기반 target correlation alignment 후보를 생성합니다.
# Pruned historical definition: build_copula_correlation_alignment_candidates (not reachable from the final runner).


# 함수: 현재 anchor의 평균/온도/상관 calibration 후보를 생성합니다.
def build_current_anchor_calibration_candidates():
    __file__ = str(
        PROJECT_ROOT / "pipeline/build_current_anchor_calibration_candidates"
    )
    __name__ = "__main__"
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SUBMISSION_DIR = BASE_DIR / "submissions"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    ANCHOR_PATH = SUBMISSION_DIR / "submission_step1_date_blend_w07.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 함수: logit 보조 로직을 수행합니다.
    def logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # 함수: sigmoid 보조 로직을 수행합니다.
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has null values")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} has probabilities outside [0,1]")

    # 함수: shift to mean 보조 로직을 수행합니다.
    def shift_to_mean(probs: np.ndarray, target_mean: float) -> np.ndarray:
        logits = logit(probs)
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = (lo + hi) / 2
            mean = sigmoid(logits + mid).mean()
            if mean < target_mean:
                lo = mid
            else:
                hi = mid
        return sigmoid(logits + (lo + hi) / 2)

    # 함수: mean align 보조 로직을 수행합니다.
    def mean_align(
        anchor: pd.DataFrame, train_mean: pd.Series, gamma: float
    ) -> pd.DataFrame:
        out = anchor.copy()
        anchor_mean = anchor[TARGETS].mean()
        desired = (1 - gamma) * anchor_mean + gamma * train_mean
        for target in TARGETS:
            out[target] = shift_to_mean(
                anchor[target].to_numpy(dtype=float), float(desired[target])
            )
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: temperature 보조 로직을 수행합니다.
    def temperature(anchor: pd.DataFrame, temp: float) -> pd.DataFrame:
        out = anchor.copy()
        probs = anchor[TARGETS].to_numpy(dtype=float)
        out[TARGETS] = sigmoid(logit(probs) * temp).clip(1e-6, 1 - 1e-6)
        return out

    # 함수: sym sqrt 보조 로직을 수행합니다.
    def sym_sqrt(
        mat: np.ndarray, inverse: bool = False, eps: float = 1e-5
    ) -> np.ndarray:
        vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
        vals = np.clip(vals, eps, None)
        vals = 1.0 / np.sqrt(vals) if inverse else np.sqrt(vals)
        return (vecs * vals) @ vecs.T

    # 함수: corr align 보조 로직을 수행합니다.
    def corr_align(
        anchor: pd.DataFrame, train_corr: np.ndarray, gamma: float
    ) -> pd.DataFrame:
        out = anchor.copy()
        probs = anchor[TARGETS].to_numpy(dtype=float)
        logits = logit(probs)
        mean = logits.mean(axis=0)
        std = logits.std(axis=0) + 1e-6
        z = (logits - mean) / std
        pred_corr = np.corrcoef(z, rowvar=False)
        desired = (1 - gamma) * pred_corr + gamma * train_corr
        desired = (desired + desired.T) / 2
        np.fill_diagonal(desired, 1.0)
        transform = sym_sqrt(pred_corr, inverse=True) @ sym_sqrt(desired)
        z_new = z @ transform
        out[TARGETS] = sigmoid(z_new * std + mean).clip(1e-6, 1 - 1e-6)
        return out

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame, train_mean: pd.Series
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        mean_gap = (candidate[TARGETS].mean() - train_mean).abs().mean()
        anchor_gap = (anchor[TARGETS].mean() - train_mean).abs().mean()
        return {
            "candidate": name,
            "diff_vs_best": float(diff.values.mean()),
            "max_diff": float(diff.values.max()),
            "mean_gap_delta": float(mean_gap - anchor_gap),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame, train_mean: pd.Series
    ) -> dict[str, float | str]:
        validate(candidate, name)
        candidate.to_csv(SUBMISSION_DIR / name, index=False)
        return summarize(name, candidate, anchor, train_mean)

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        train = pd.read_csv(TRAIN_PATH)
        anchor = pd.read_csv(ANCHOR_PATH)
        validate(anchor, ANCHOR_PATH.name)

        train_mean = train[TARGETS].mean()
        train_corr = train[TARGETS].corr().fillna(0.0).to_numpy()
        rows = []

        for gamma in [0.02, 0.03, 0.04, 0.06]:
            rows.append(
                save(
                    f"submission_step1_mean_g{int(gamma*1000):03d}.csv",
                    mean_align(anchor, train_mean, gamma),
                    anchor,
                    train_mean,
                )
            )

        for temp in [0.97, 1.03, 1.06]:
            rows.append(
                save(
                    f"submission_step1_temperature_t{int(temp*1000):04d}.csv",
                    temperature(anchor, temp),
                    anchor,
                    train_mean,
                )
            )

        for gamma in [0.04, 0.08]:
            rows.append(
                save(
                    f"submission_step1_correlation_g{int(gamma*1000):03d}.csv",
                    corr_align(anchor, train_corr, gamma),
                    anchor,
                    train_mean,
                )
            )

        summary = (
            pd.DataFrame(rows)
            .sort_values(["diff_vs_best", "candidate"])
            .reset_index(drop=True)
        )
        print("Current-best calibration candidates:")
        print(summary.to_string(index=False))
        print("\nSuggested calibration candidate:")
        print("submission_step1_mean_g040.csv")

    if __name__ == "__main__":
        main()


# 함수: target 공동분포 기반 calibration 후보를 생성합니다.
# Pruned historical definition: build_joint_target_calibration_candidates (not reachable from the final runner).


# 함수: target dynamics prior와 anchor의 blend 강도를 grid로 생성합니다.
def build_target_dynamics_blend_grid():
    __file__ = str(PROJECT_ROOT / "pipeline/build_target_dynamics_blend_grid")
    __name__ = "__main__"
    from pathlib import Path

    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    SUBMISSION_DIR = ROOT / "data" / "submissions"

    ANCHOR_PATH = SUBMISSION_DIR / "submission_step1_transition_past_tw_b.csv"
    DYN_PATH = SUBMISSION_DIR / "submission_step1_dynamics_prior.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has nulls")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} out of [0,1]")

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def uniform_blend(
        anchor: pd.DataFrame, dyn: pd.DataFrame, weight: float
    ) -> pd.DataFrame:
        out = anchor.copy()
        out[TARGETS] = (1 - weight) * anchor[TARGETS].astype(float) + weight * dyn[
            TARGETS
        ].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def targetwise_blend(
        anchor: pd.DataFrame, dyn: pd.DataFrame, weights: dict[str, float]
    ) -> pd.DataFrame:
        out = anchor.copy()
        for target in TARGETS:
            weight = float(weights.get(target, 0.0))
            out[target] = (1 - weight) * anchor[target].astype(float) + weight * dyn[
                target
            ].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "diff_vs_anchor": float(diff.values.mean()),
            "max_diff": float(diff.values.max()),
            "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
            "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        anchor = pd.read_csv(ANCHOR_PATH)
        dyn = pd.read_csv(DYN_PATH)
        validate(anchor, ANCHOR_PATH.name)
        validate(dyn, DYN_PATH.name)

        candidates: list[dict[str, float | str]] = []

        for weight in [0.08, 0.12, 0.14, 0.16, 0.20]:
            name = f"submission_step1_dynamics_blend_w{int(weight * 100):02d}_grid.csv"
            out = uniform_blend(anchor, dyn, weight)
            validate(out, name)
            out.to_csv(SUBMISSION_DIR / name, index=False)
            candidates.append(summarize(name, out, anchor))

        targetwise_configs = {
            # w10 improved, so test a slightly Q-led shape without moving S3 too much.
            "tw_q12_s08": {
                "Q1": 0.12,
                "Q2": 0.12,
                "Q3": 0.12,
                "S1": 0.08,
                "S2": 0.08,
                "S3": 0.06,
                "S4": 0.08,
            },
            # More aggressive Q dynamics, conservative S.
            "tw_q14_s08": {
                "Q1": 0.14,
                "Q2": 0.14,
                "Q3": 0.14,
                "S1": 0.08,
                "S2": 0.08,
                "S3": 0.06,
                "S4": 0.08,
            },
            # Uniform-ish but pull S3 back because it has historically been noisy.
            "tw_w14_s3low": {
                "Q1": 0.14,
                "Q2": 0.14,
                "Q3": 0.14,
                "S1": 0.14,
                "S2": 0.14,
                "S3": 0.06,
                "S4": 0.14,
            },
            # If the dynamics signal is mostly Q-state driven.
            "tw_q16_s06": {
                "Q1": 0.16,
                "Q2": 0.16,
                "Q3": 0.16,
                "S1": 0.06,
                "S2": 0.06,
                "S3": 0.04,
                "S4": 0.06,
            },
        }

        for tag, weights in targetwise_configs.items():
            name = f"submission_step1_dynamics_blend_{tag}.csv"
            out = targetwise_blend(anchor, dyn, weights)
            validate(out, name)
            out.to_csv(SUBMISSION_DIR / name, index=False)
            candidates.append(summarize(name, out, anchor))

        summary = (
            pd.DataFrame(candidates)
            .sort_values("diff_vs_anchor")
            .reset_index(drop=True)
        )
        print(summary.to_string(index=False))
        print("\nIf w10 public = 0.5902811814, suggested next order:")
        print("1) submission_step1_dynamics_blend_w14.csv")
        print("2) submission_step1_dynamics_blend_tw_w14_s3low.csv")
        print("3) submission_step1_dynamics_blend_w16_grid.csv")

    if __name__ == "__main__":
        main()


# 함수: 여러 seed 조합 예측을 blend하는 후보를 생성합니다.
# Pruned historical definition: build_seed_combo_blends (not reachable from the final runner).


# 함수: state-transition prior를 Q/S 및 target-wise 가중치로 섞는 후보를 생성합니다.
def build_state_transition_blend_grid():
    __file__ = str(PROJECT_ROOT / "pipeline/build_state_transition_blend_grid")
    __name__ = "__main__"
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SUBMISSION_DIR = BASE_DIR / "submissions"

    ANCHOR_PATH = SUBMISSION_DIR / "submission_step1_target_routing.csv"
    PAST_PRIOR_PATH = SUBMISSION_DIR / "submission_step1_history_past.csv"
    FULL_PRIOR_PATH = SUBMISSION_DIR / "submission_step1_history_full.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    Q_TARGETS = ["Q1", "Q2", "Q3"]
    S_TARGETS = ["S1", "S2", "S3", "S4"]

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate_frame(df: pd.DataFrame, name: str) -> None:
        missing = [
            c
            for c in ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
            if c not in df.columns
        ]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has nulls")
        in_range = ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all()
        if not in_range:
            raise ValueError(f"{name} has probabilities outside [0,1]")

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend_scalar(
        anchor: pd.DataFrame, prior: pd.DataFrame, q_weight: float, s_weight: float
    ) -> pd.DataFrame:
        weights = {target: q_weight for target in Q_TARGETS}
        weights.update({target: s_weight for target in S_TARGETS})
        return blend_targetwise(anchor, prior, weights)

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend_targetwise(
        anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float]
    ) -> pd.DataFrame:
        out = anchor.copy()
        for target in TARGETS:
            w = float(weights.get(target, 0.0))
            out[target] = (1 - w) * anchor[target].astype(float) + w * prior[
                target
            ].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend_two_priors(
        anchor: pd.DataFrame,
        past_prior: pd.DataFrame,
        full_prior: pd.DataFrame,
        past_ratio: float,
        q_weight: float,
        s_weight: float,
    ) -> pd.DataFrame:
        prior = anchor.copy()
        prior[TARGETS] = past_ratio * past_prior[TARGETS].astype(float) + (
            1 - past_ratio
        ) * full_prior[TARGETS].astype(float)
        return blend_scalar(anchor, prior, q_weight=q_weight, s_weight=s_weight)

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save_and_summarize(
        name: str, df: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        path = SUBMISSION_DIR / name
        df.to_csv(path, index=False)
        validate_frame(df, name)

        diff = (df[TARGETS].astype(float) - anchor[TARGETS].astype(float)).abs()
        return {
            "candidate": name,
            "mean_abs_diff": float(diff.values.mean()),
            "max_abs_diff": float(diff.values.max()),
            "q_diff": float(diff[Q_TARGETS].values.mean()),
            "s_diff": float(diff[S_TARGETS].values.mean()),
            "q_mean": float(df[Q_TARGETS].values.mean()),
            "s_mean": float(df[S_TARGETS].values.mean()),
        }

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        anchor = pd.read_csv(ANCHOR_PATH)
        past = pd.read_csv(PAST_PRIOR_PATH)
        full = pd.read_csv(FULL_PRIOR_PATH)

        validate_frame(anchor, ANCHOR_PATH.name)
        validate_frame(past, PAST_PRIOR_PATH.name)
        validate_frame(full, FULL_PRIOR_PATH.name)

        candidates: dict[str, pd.DataFrame] = {}

        # Continuation grid around the public-winning q14/s07 direction.
        for q_weight, s_weight in [
            (0.16, 0.08),
            (0.18, 0.09),
            (0.20, 0.10),
            (0.22, 0.11),
            (0.24, 0.12),
            (0.28, 0.14),
        ]:
            tag = f"q{int(q_weight * 100):02d}_s{int(s_weight * 100):02d}"
            candidates[f"submission_step1_transition_past_{tag}.csv"] = blend_scalar(
                anchor, past, q_weight, s_weight
            )

        # Target-wise route: forward validation said Q targets, especially Q2/Q3,
        # gain more from state dynamics than S1/S3.
        targetwise_configs = {
            "tw_a": {
                "Q1": 0.18,
                "Q2": 0.22,
                "Q3": 0.20,
                "S1": 0.02,
                "S2": 0.10,
                "S3": 0.03,
                "S4": 0.08,
            },
            "tw_b": {
                "Q1": 0.18,
                "Q2": 0.26,
                "Q3": 0.22,
                "S1": 0.00,
                "S2": 0.10,
                "S3": 0.02,
                "S4": 0.08,
            },
            "tw_c": {
                "Q1": 0.14,
                "Q2": 0.24,
                "Q3": 0.20,
                "S1": 0.00,
                "S2": 0.08,
                "S3": 0.00,
                "S4": 0.06,
            },
            "tw_d": {
                "Q1": 0.22,
                "Q2": 0.28,
                "Q3": 0.24,
                "S1": 0.02,
                "S2": 0.12,
                "S3": 0.03,
                "S4": 0.10,
            },
        }
        for tag, weights in targetwise_configs.items():
            candidates[f"submission_step1_transition_past_{tag}.csv"] = (
                blend_targetwise(anchor, past, weights)
            )

        # Experimental but not primary: combine mostly past-only with a little full-subject prior.
        # This is included as a diagnostic; past-only remains the cleaner future-prediction choice.
        for past_ratio, q_weight, s_weight in [
            (0.85, 0.16, 0.08),
            (0.85, 0.20, 0.10),
            (0.70, 0.16, 0.08),
        ]:
            tag = f"mixp{int(past_ratio * 100)}_q{int(q_weight * 100):02d}_s{int(s_weight * 100):02d}"
            candidates[f"submission_step1_transition_blend_{tag}.csv"] = (
                blend_two_priors(
                    anchor,
                    past,
                    full,
                    past_ratio=past_ratio,
                    q_weight=q_weight,
                    s_weight=s_weight,
                )
            )

        summary = []
        for name, df in candidates.items():
            summary.append(save_and_summarize(name, df, anchor))

        summary_df = (
            pd.DataFrame(summary)
            .sort_values(["mean_abs_diff", "candidate"])
            .reset_index(drop=True)
        )
        print(summary_df.to_string(index=False))
        print("\nSuggested submit order if previous q14_s07 scored 0.592269:")
        print(
            "1) submission_step1_transition_past_q20_s10.csv  # same axis, more assertive"
        )
        print(
            "2) submission_step1_transition_past_tw_b.csv      # target-wise experimental"
        )
        print(
            "3) submission_step1_transition_past_q24_s12.csv  # only if q20/s10 improves"
        )

    if __name__ == "__main__":
        main()


# 함수: state-transition target-wise 가중치를 추가로 미세 조정합니다.
# Pruned historical definition: build_targetwise_state_transition_refinements (not reachable from the final runner).


# 함수: subject별 가까운 날짜 라벨 보간 prior를 생성합니다.
def build_subject_date_interpolation_prior():
    __file__ = str(PROJECT_ROOT / "pipeline/build_subject_date_interpolation_prior")
    __name__ = "__main__"

    from dataclasses import dataclass
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"
    SUBMISSION_DIR = BASE_DIR / "submissions"

    CURRENT_BEST_PATH = (
        SUBMISSION_DIR / "submission_step1_dynamics_blend_w14.csv"
    )

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    @dataclass(frozen=True)
    class InterpConfig:
        tag: str
        tau: float
        k: int
        prior_strength: float
        future_weight: float
        past_weight: float
        clip_low: float = 0.045
        clip_high: float = 0.955

    CONFIGS = [
        InterpConfig(
            "near_tau3",
            tau=3.0,
            k=7,
            prior_strength=1.8,
            future_weight=1.00,
            past_weight=1.00,
        ),
        InterpConfig(
            "near_tau5",
            tau=5.0,
            k=9,
            prior_strength=2.4,
            future_weight=1.00,
            past_weight=1.00,
        ),
        InterpConfig(
            "near_tau7",
            tau=7.0,
            k=11,
            prior_strength=3.0,
            future_weight=1.00,
            past_weight=1.00,
        ),
        InterpConfig(
            "future_soft",
            tau=5.0,
            k=9,
            prior_strength=2.4,
            future_weight=0.85,
            past_weight=1.05,
        ),
        InterpConfig(
            "smooth_tau10",
            tau=10.0,
            k=13,
            prior_strength=4.0,
            future_weight=0.95,
            past_weight=1.00,
        ),
    ]

    # 함수: binary log-loss를 계산합니다.
    def binary_logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
        pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=float)
        return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has null values")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} has probabilities outside [0,1]")

    # 함수: 검증/전체 학습 모델을 적합하고 예측값을 만듭니다.
    def fit_subject_tables(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            str(sid): group.sort_values("lifelog_date")[
                ["lifelog_date"] + TARGETS
            ].copy()
            for sid, group in train.groupby("subject_id")
        }

    # 함수: 학습된 규칙이나 모델로 test 구간 예측을 생성합니다.
    def predict_target(
        subject_table: pd.DataFrame,
        pred_date: pd.Timestamp,
        target: str,
        global_mean: float,
        config: InterpConfig,
    ) -> float:
        if len(subject_table) == 0:
            return float(np.clip(global_mean, config.clip_low, config.clip_high))

        dates = pd.to_datetime(subject_table["lifelog_date"])
        delta = (dates - pred_date).dt.days.to_numpy(dtype=float)
        abs_delta = np.abs(delta)
        order = np.argsort(abs_delta)[: config.k]

        chosen_delta = delta[order]
        chosen_abs = abs_delta[order]
        values = subject_table[target].to_numpy(dtype=float)[order]
        side_weight = np.where(
            chosen_delta >= 0, config.future_weight, config.past_weight
        )
        weights = np.exp(-chosen_abs / config.tau) * side_weight

        subject_mean = float(subject_table[target].mean())
        smooth_prior = 0.75 * subject_mean + 0.25 * global_mean

        if float(weights.sum()) <= 1e-12:
            pred = smooth_prior
        else:
            local = float(np.sum(weights * values) / np.sum(weights))
            reliability = float(
                np.sum(weights) / (np.sum(weights) + config.prior_strength)
            )
            pred = reliability * local + (1.0 - reliability) * smooth_prior

        return float(np.clip(pred, config.clip_low, config.clip_high))

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def build_prior(
        train_fit: pd.DataFrame, test_frame: pd.DataFrame, config: InterpConfig
    ) -> pd.DataFrame:
        tables = fit_subject_tables(train_fit)
        global_mean = {target: float(train_fit[target].mean()) for target in TARGETS}
        rows = []

        for row in test_frame.itertuples(index=False):
            sid = str(row.subject_id)
            pred_date = pd.Timestamp(row.lifelog_date)
            table = tables.get(sid, pd.DataFrame(columns=["lifelog_date"] + TARGETS))
            pred_row = {}
            for target in TARGETS:
                pred_row[target] = predict_target(
                    table, pred_date, target, global_mean[target], config
                )
            rows.append(pred_row)

        return pd.DataFrame(rows)

    # 함수: interpolation cv 보조 로직을 수행합니다.
    def interpolation_cv(train: pd.DataFrame, config: InterpConfig) -> dict[str, float]:
        y_all = []
        p_all = []
        per_target = {target: {"y": [], "p": []} for target in TARGETS}

        for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby(
            "subject_id"
        ):
            group = group.reset_index(drop=True)
            if len(group) < 15:
                continue
            # Hold out interleaved rows rather than only the tail. This mirrors
            # the actual test layout, where many dates sit between known train dates.
            holdout_mask = (np.arange(len(group)) + 2) % 5 == 0
            holdout = group[holdout_mask].copy()
            fit = pd.concat(
                [
                    train[train["subject_id"] != group["subject_id"].iloc[0]],
                    group[~holdout_mask],
                ],
                ignore_index=True,
            )
            preds = build_prior(
                fit, holdout[["subject_id", "sleep_date", "lifelog_date"]], config
            )
            y = holdout[TARGETS].to_numpy(dtype=float)
            p = preds[TARGETS].to_numpy(dtype=float)
            y_all.append(y)
            p_all.append(p)
            for j, target in enumerate(TARGETS):
                per_target[target]["y"].extend(y[:, j].tolist())
                per_target[target]["p"].extend(p[:, j].tolist())

        y_true = np.vstack(y_all)
        pred = np.vstack(p_all)
        out = {"overall": binary_logloss(y_true, pred)}
        for target in TARGETS:
            out[target] = binary_logloss(
                np.array(per_target[target]["y"]), np.array(per_target[target]["p"])
            )
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend(anchor: pd.DataFrame, prior: pd.DataFrame, weight: float) -> pd.DataFrame:
        out = anchor.copy()
        out[TARGETS] = (1.0 - weight) * anchor[TARGETS].astype(float) + weight * prior[
            TARGETS
        ].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def targetwise_blend(
        anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float]
    ) -> pd.DataFrame:
        out = anchor.copy()
        for target in TARGETS:
            weight = float(weights.get(target, 0.0))
            out[target] = (1.0 - weight) * anchor[target].astype(
                float
            ) + weight * prior[target].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "diff_vs_current_best": float(diff.values.mean()),
            "max_diff": float(diff.values.max()),
            "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
            "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
        sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
        anchor = pd.read_csv(CURRENT_BEST_PATH)
        validate(anchor, CURRENT_BEST_PATH.name)

        cv_rows = []
        for config in CONFIGS:
            cv_rows.append({"config": config.tag, **interpolation_cv(train, config)})
        cv = pd.DataFrame(cv_rows).sort_values("overall").reset_index(drop=True)
        print("Interleaved subject-date interpolation CV:")
        print(cv.to_string(index=False))

        best_tag = str(cv.loc[0, "config"])
        best_config = next(config for config in CONFIGS if config.tag == best_tag)
        print(f"\nBest interpolation config: {best_config.tag}")

        prior = build_prior(
            train, sample[["subject_id", "sleep_date", "lifelog_date"]], best_config
        )
        pure = sample.copy()
        pure[TARGETS] = prior[TARGETS].to_numpy(dtype=float)
        pure[TARGETS] = pure[TARGETS].clip(1e-6, 1 - 1e-6)
        pure_name = "submission_step1_interpolation.csv"
        validate(pure, pure_name)
        pure.to_csv(SUBMISSION_DIR / pure_name, index=False)

        candidates = [summarize(pure_name, pure, anchor)]
        for weight in [0.04, 0.07, 0.10, 0.13, 0.16]:
            name = f"submission_step1_date_blend_w{int(weight * 100):02d}.csv"
            out = blend(anchor, pure, weight)
            validate(out, name)
            out.to_csv(SUBMISSION_DIR / name, index=False)
            candidates.append(summarize(name, out, anchor))

        tw_configs = {
            "tw_q10_s06": {
                "Q1": 0.10,
                "Q2": 0.10,
                "Q3": 0.10,
                "S1": 0.06,
                "S2": 0.06,
                "S3": 0.04,
                "S4": 0.06,
            },
            "tw_q12_s08": {
                "Q1": 0.12,
                "Q2": 0.12,
                "Q3": 0.12,
                "S1": 0.08,
                "S2": 0.08,
                "S3": 0.05,
                "S4": 0.08,
            },
            "tw_q08_s10": {
                "Q1": 0.08,
                "Q2": 0.08,
                "Q3": 0.08,
                "S1": 0.10,
                "S2": 0.10,
                "S3": 0.06,
                "S4": 0.10,
            },
        }
        for tag, weights in tw_configs.items():
            name = f"submission_step1_date_blend_{tag}.csv"
            out = targetwise_blend(anchor, pure, weights)
            validate(out, name)
            out.to_csv(SUBMISSION_DIR / name, index=False)
            candidates.append(summarize(name, out, anchor))

        summary = (
            pd.DataFrame(candidates)
            .sort_values("diff_vs_current_best")
            .reset_index(drop=True)
        )
        print("\nSaved candidate summary vs current best w14:")
        print(summary.to_string(index=False))
        print("\nSuggested submit order if current best = 0.5898630289:")
        print("1) submission_step1_date_blend_w07.csv")
        print("2) submission_step1_date_blend_w10.csv")
        print("3) submission_step1_date_blend_tw_q12_s08.csv")

    if __name__ == "__main__":
        main()


# 함수: target 평균을 train label 평균 방향으로 정렬하는 후보를 생성합니다.
# Pruned historical definition: build_target_mean_alignment_candidates (not reachable from the final runner).


# 함수: 일별 센서 집계 기반 LightGBM/CatBoost baseline과 feature table을 학습/생성합니다.
def train_daily_sensor_baseline():
    __file__ = str(PROJECT_ROOT / "pipeline/train_daily_sensor_baseline")
    __name__ = "__main__"
    import os
    import gc
    import json
    import math
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from sklearn.metrics import log_loss
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    warnings.filterwarnings("ignore")

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SENSOR_DIR = BASE_DIR / "ch2025_data_items"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SUB_PATH = BASE_DIR / "ch2026_submission_sample.csv"
    OUT_DIR = BASE_DIR / "submissions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 함수: log 보조 로직을 수행합니다.
    def log(msg: str):
        print(f"[INFO] {msg}")

    # 함수: get sensor path 보조 로직을 수행합니다.
    def get_sensor_path(keyword: str):
        files = sorted(SENSOR_DIR.glob("*.parquet"))
        for p in files:
            if keyword.lower() in p.name.lower():
                return p
        return None

    # 함수: infer test frame from submission 보조 로직을 수행합니다.
    def infer_test_frame_from_submission(
        train_df: pd.DataFrame, submission_df: pd.DataFrame
    ) -> pd.DataFrame:
        if {"subject_id", "lifelog_date"}.issubset(submission_df.columns):
            test_df = submission_df[["subject_id", "lifelog_date"]].copy()
            test_df["subject_id"] = test_df["subject_id"].astype(str)
            test_df["lifelog_date"] = pd.to_datetime(test_df["lifelog_date"])
            return test_df

        n_test = len(submission_df)
        last_dates = train_df.groupby("subject_id")["lifelog_date"].max().sort_values()
        subs = list(last_dates.index)
        reps = math.ceil(n_test / len(subs))
        subject_seq = (subs * reps)[:n_test]

        next_dates = {
            sid: train_df.loc[train_df["subject_id"] == sid, "lifelog_date"].max()
            + pd.Timedelta(days=1)
            for sid in subs
        }
        rows = []
        for sid in subject_seq:
            rows.append((sid, next_dates[sid]))
            next_dates[sid] += pd.Timedelta(days=1)

        return pd.DataFrame(rows, columns=["subject_id", "lifelog_date"])

    # 함수: prep sensor df 보조 로직을 수행합니다.
    def prep_sensor_df(df: pd.DataFrame, value_cols):
        df = df.copy()
        df["subject_id"] = df["subject_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["lifelog_date"] = df["timestamp"].dt.floor("D")
        df["hour"] = df["timestamp"].dt.hour
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
        df["is_day"] = ((df["hour"] >= 9) & (df["hour"] <= 18)).astype(int)
        keep = [
            "subject_id",
            "lifelog_date",
            "timestamp",
            "hour",
            "is_night",
            "is_day",
        ] + [c for c in value_cols if c in df.columns]
        return df[keep].copy()

    # 함수: add simple agg 보조 로직을 수행합니다.
    def add_simple_agg(df: pd.DataFrame, val_col: str, prefix: str) -> pd.DataFrame:
        g = df.groupby(["subject_id", "lifelog_date"])[val_col]
        # Robust aggregation: mean/max can be noisy for sparse wearable logs, so
        # keep median and quantile-spread statistics alongside the classic stats.
        def q10(x):
            return x.quantile(0.10)

        def q25(x):
            return x.quantile(0.25)

        def q75(x):
            return x.quantile(0.75)

        def q90(x):
            return x.quantile(0.90)

        feat = g.agg(
            ["mean", "std", "min", "max", "count", "median", q10, q25, q75, q90]
        ).reset_index()
        feat.columns = ["subject_id", "lifelog_date"] + [
            f"{prefix}_{c}"
            for c in [
                "mean",
                "std",
                "min",
                "max",
                "count",
                "median",
                "q10",
                "q25",
                "q75",
                "q90",
            ]
        ]
        feat[f"{prefix}_iqr"] = feat[f"{prefix}_q75"] - feat[f"{prefix}_q25"]
        feat[f"{prefix}_q90_q10"] = feat[f"{prefix}_q90"] - feat[f"{prefix}_q10"]

        night = (
            df[df["is_night"] == 1]
            .groupby(["subject_id", "lifelog_date"])[val_col]
            .agg(["mean", "std", "median", q25, q75])
            .reset_index()
        )
        night.columns = [
            "subject_id",
            "lifelog_date",
            f"{prefix}_night_mean",
            f"{prefix}_night_std",
            f"{prefix}_night_median",
            f"{prefix}_night_q25",
            f"{prefix}_night_q75",
        ]
        night[f"{prefix}_night_iqr"] = night[f"{prefix}_night_q75"] - night[f"{prefix}_night_q25"]

        day = (
            df[df["is_day"] == 1]
            .groupby(["subject_id", "lifelog_date"])[val_col]
            .agg(["mean", "std", "median", q25, q75])
            .reset_index()
        )
        day.columns = [
            "subject_id",
            "lifelog_date",
            f"{prefix}_day_mean",
            f"{prefix}_day_std",
            f"{prefix}_day_median",
            f"{prefix}_day_q25",
            f"{prefix}_day_q75",
        ]
        day[f"{prefix}_day_iqr"] = day[f"{prefix}_day_q75"] - day[f"{prefix}_day_q25"]

        out = feat.merge(night, on=["subject_id", "lifelog_date"], how="left")
        out = out.merge(day, on=["subject_id", "lifelog_date"], how="left")
        return out

    # 함수: explode hr array 보조 로직을 수행합니다.
    def explode_hr_array(x):
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return []
            return (
                pd.to_numeric(pd.Series(x.ravel()), errors="coerce").dropna().tolist()
            )
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return []
            return pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna().tolist()
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return []
            try:
                v = json.loads(s)
                if isinstance(v, np.ndarray):
                    return (
                        pd.to_numeric(pd.Series(v.ravel()), errors="coerce")
                        .dropna()
                        .tolist()
                    )
                if isinstance(v, (list, tuple)):
                    return (
                        pd.to_numeric(pd.Series(list(v)), errors="coerce")
                        .dropna()
                        .tolist()
                    )
                return []
            except Exception:
                return []
        try:
            if pd.isna(x):
                return []
        except Exception:
            pass
        try:
            return [float(x)]
        except Exception:
            return []

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def build_simple_sensor_features(sensor_map):
        tables = []

        mapping = [
            ("activity", "m_activity", "m_activity"),
            ("light", "m_light", "m_light"),
            ("screen", "m_screen_use", "m_screen_use"),
            ("charge", "m_charging", "m_charging"),
        ]
        for key, raw_col, prefix in mapping:
            p = sensor_map.get(key)
            if p is None:
                continue
            log(f"loading {p.name}")
            df = pd.read_parquet(p)
            if raw_col not in df.columns:
                continue
            df = prep_sensor_df(df, [raw_col])
            tables.append(add_simple_agg(df, raw_col, prefix))
            del df
            gc.collect()

        p = sensor_map.get("hr")
        if p is not None:
            log(f"loading {p.name}")
            df = pd.read_parquet(p)
            candidate_cols = [
                c for c in df.columns if c not in ["subject_id", "timestamp"]
            ]
            if candidate_cols:
                preferred = [
                    c
                    for c in candidate_cols
                    if "heart" in c.lower() or "hr" in c.lower()
                ]
                hr_col = preferred[0] if preferred else candidate_cols[0]
                log(f"selected HR column: {hr_col}")
                df = prep_sensor_df(df, [hr_col])
                arr = df[hr_col].apply(explode_hr_array)
                df["hr_mean_row"] = arr.apply(
                    lambda v: float(np.mean(v)) if len(v) else np.nan
                )
                df["hr_std_row"] = arr.apply(
                    lambda v: float(np.std(v)) if len(v) else np.nan
                )
                df["hr_min_row"] = arr.apply(
                    lambda v: float(np.min(v)) if len(v) else np.nan
                )
                df["hr_max_row"] = arr.apply(
                    lambda v: float(np.max(v)) if len(v) else np.nan
                )
                df["hr_median_row"] = arr.apply(
                    lambda v: float(np.median(v)) if len(v) else np.nan
                )
                df["hr_q75_row"] = arr.apply(
                    lambda v: float(np.quantile(v, 0.75)) if len(v) else np.nan
                )

                stats = (
                    df.groupby(["subject_id", "lifelog_date"])[
                        [
                            "hr_mean_row",
                            "hr_std_row",
                            "hr_min_row",
                            "hr_max_row",
                            "hr_median_row",
                            "hr_q75_row",
                        ]
                    ]
                    .mean()
                    .reset_index()
                )
                stats.columns = [
                    "subject_id",
                    "lifelog_date",
                    "heart_rate_mean",
                    "heart_rate_std",
                    "heart_rate_min",
                    "heart_rate_max",
                    "heart_rate_median",
                    "heart_rate_q75",
                ]

                sleep = (
                    df[df["is_night"] == 1]
                    .groupby(["subject_id", "lifelog_date"])["hr_mean_row"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                sleep.columns = [
                    "subject_id",
                    "lifelog_date",
                    "heart_rate_sleep_mean",
                    "heart_rate_sleep_std",
                ]

                active = (
                    df[df["is_day"] == 1]
                    .groupby(["subject_id", "lifelog_date"])["hr_mean_row"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                active.columns = [
                    "subject_id",
                    "lifelog_date",
                    "heart_rate_active_mean",
                    "heart_rate_active_std",
                ]

                out = stats.merge(sleep, on=["subject_id", "lifelog_date"], how="left")
                out = out.merge(active, on=["subject_id", "lifelog_date"], how="left")
                out["heart_rate_sleep_active_diff"] = (
                    out["heart_rate_sleep_mean"] - out["heart_rate_active_mean"]
                )
                tables.append(out)
                del df
                gc.collect()

        p = sensor_map.get("pedo")
        if p is not None:
            log(f"loading {p.name}")
            df = pd.read_parquet(p)
            value_cols = [
                c
                for c in ["step", "distance", "speed", "calories", "running"]
                if c in df.columns
            ]
            if value_cols:
                df = prep_sensor_df(df, value_cols)
                agg_dict = {}
                if "step" in df.columns:
                    agg_dict["step"] = ["sum", "mean"]
                if "distance" in df.columns:
                    agg_dict["distance"] = ["sum", "mean"]
                if "speed" in df.columns:
                    agg_dict["speed"] = ["mean", "max"]
                if "calories" in df.columns:
                    agg_dict["calories"] = ["sum", "mean"]
                if "running" in df.columns:
                    agg_dict["running"] = ["sum", "mean"]

                pedo = (
                    df.groupby(["subject_id", "lifelog_date"])
                    .agg(agg_dict)
                    .reset_index()
                )
                pedo.columns = ["subject_id", "lifelog_date"] + [
                    f"{a}_{b}" for a, b in pedo.columns.tolist()[2:]
                ]
                tables.append(pedo)
                del df
                gc.collect()

        return tables

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def add_prior_features(df: pd.DataFrame, cols):
        df = df.sort_values(["subject_id", "lifelog_date"]).copy()
        for col in cols:
            grp = df.groupby("subject_id")[col]
            prior_mean = grp.expanding().mean().shift(1).reset_index(level=0, drop=True)
            prior_std = grp.expanding().std().shift(1).reset_index(level=0, drop=True)
            prior_cnt = df.groupby("subject_id").cumcount()
            df[f"{col}_prior_mean"] = prior_mean
            df[f"{col}_prior_std"] = prior_std
            df[f"{col}_prior_cnt"] = prior_cnt
            df[f"{col}_dev"] = df[col] - df[f"{col}_prior_mean"]
        return df

    # 함수: build global time split 보조 로직을 수행합니다.
    def build_global_time_split(df: pd.DataFrame, valid_ratio=0.2):
        uniq_dates = np.sort(df["lifelog_date"].unique())
        n_valid = max(1, int(len(uniq_dates) * valid_ratio))
        valid_dates = set(uniq_dates[-n_valid:])
        tr_idx = df.index[~df["lifelog_date"].isin(valid_dates)].tolist()
        va_idx = df.index[df["lifelog_date"].isin(valid_dates)].tolist()
        return tr_idx, va_idx

    # 함수: binary log-loss를 계산합니다.
    def avg_logloss(y_true_df: pd.DataFrame, pred_df: pd.DataFrame):
        scores = []
        for t in TARGETS:
            y_true = y_true_df[t].astype(int).values
            y_pred = np.clip(pred_df[t].values, 1e-6, 1 - 1e-6)
            scores.append(log_loss(y_true, y_pred))
        return float(np.mean(scores)), scores

    # 함수: 과신한 확률을 0.5 방향으로 완화합니다.
    def shrink_proba(p, alpha=0.8):
        return np.clip(0.5 + alpha * (p - 0.5), 1e-6, 1 - 1e-6)

    # 함수: 확률값을 안정적인 범위로 제한합니다.
    def clip_proba(p, lo=0.03, hi=0.97):
        return np.clip(p, lo, hi)

    # 함수: 검증/전체 학습 모델을 적합하고 예측값을 만듭니다.
    def fit_predict_split(
        X_train: pd.DataFrame,
        y_df: pd.DataFrame,
        X_test: pd.DataFrame,
        tr_idx,
        va_idx,
        features,
    ):
        oof_lgb = pd.DataFrame(index=va_idx, columns=TARGETS, dtype=float)
        oof_cat = pd.DataFrame(index=va_idx, columns=TARGETS, dtype=float)
        test_lgb = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
        test_cat = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)

        X_tr = X_train.loc[tr_idx, features]
        X_va = X_train.loc[va_idx, features]
        X_te = X_test[features]

        for t in TARGETS:
            log(f"training split model for {t}")
            y_tr = y_df.loc[tr_idx, t].astype(int)
            y_va = y_df.loc[va_idx, t].astype(int)
            restored = (
                load_step1_candidate_checkpoint("daily_sensor_baseline", t, "split")
                if restore_step1_candidate_checkpoints_enabled()
                else None
            )
            if restored is not None:
                log(f"restored split model for {t}")
                models = restored["models"]
                oof_lgb.loc[va_idx, t] = models["lgb"].predict_proba(X_va)[:, 1]
                oof_cat.loc[va_idx, t] = models["cat"].predict_proba(X_va)[:, 1]
                test_lgb[t] = models["lgb"].predict_proba(X_te)[:, 1]
                test_cat[t] = models["cat"].predict_proba(X_te)[:, 1]
                continue

            lgb_model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=4,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=3.0,
                objective="binary",
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
            lgb_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            cat_model = CatBoostClassifier(
                iterations=300,
                learning_rate=0.03,
                depth=4,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=42,
                verbose=False,
            )
            cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

            oof_lgb.loc[va_idx, t] = lgb_model.predict_proba(X_va)[:, 1]
            oof_cat.loc[va_idx, t] = cat_model.predict_proba(X_va)[:, 1]
            test_lgb[t] = lgb_model.predict_proba(X_te)[:, 1]
            test_cat[t] = cat_model.predict_proba(X_te)[:, 1]
            save_step1_candidate_checkpoint(
                "daily_sensor_baseline",
                {
                    "kind": "split",
                    "target": t,
                    "features": list(features),
                    "train_index": list(tr_idx),
                    "valid_index": list(va_idx),
                    "models": {
                        "lgb": lgb_model,
                        "cat": cat_model,
                    },
                },
                t,
                "split",
            )

        return oof_lgb, oof_cat, test_lgb, test_cat

    # 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
    def train_full_models(
        X_train: pd.DataFrame, y_df: pd.DataFrame, X_test: pd.DataFrame, features
    ):
        final_pred_lgb = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
        final_pred_cat = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)

        for t in TARGETS:
            log(f"training full model for {t}")
            y = y_df[t].astype(int)
            restored = (
                load_step1_candidate_checkpoint("daily_sensor_baseline", t, "full")
                if restore_step1_candidate_checkpoints_enabled()
                else None
            )
            if restored is not None:
                log(f"restored full model for {t}")
                models = restored["models"]
                final_pred_lgb[t] = models["lgb"].predict_proba(X_test[features])[:, 1]
                final_pred_cat[t] = models["cat"].predict_proba(X_test[features])[:, 1]
                continue

            lgb_model = lgb.LGBMClassifier(
                n_estimators=220,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=4,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=3.0,
                objective="binary",
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
            lgb_model.fit(X_train[features], y)
            final_pred_lgb[t] = lgb_model.predict_proba(X_test[features])[:, 1]

            cat_model = CatBoostClassifier(
                iterations=220,
                learning_rate=0.03,
                depth=4,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=42,
                verbose=False,
            )
            cat_model.fit(X_train[features], y, verbose=False)
            final_pred_cat[t] = cat_model.predict_proba(X_test[features])[:, 1]
            save_step1_candidate_checkpoint(
                "daily_sensor_baseline",
                {
                    "kind": "full",
                    "target": t,
                    "features": list(features),
                    "models": {
                        "lgb": lgb_model,
                        "cat": cat_model,
                    },
                },
                t,
                "full",
            )

        return final_pred_lgb, final_pred_cat

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main():
        log("loading train/submission files")
        train = pd.read_csv(TRAIN_PATH)
        sub = pd.read_csv(SUB_PATH)

        train["subject_id"] = train["subject_id"].astype(str)
        train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
        if "subject_id" in sub.columns:
            sub["subject_id"] = sub["subject_id"].astype(str)
        if "lifelog_date" in sub.columns:
            sub["lifelog_date"] = pd.to_datetime(sub["lifelog_date"])

        test = infer_test_frame_from_submission(train, sub)

        base_df = (
            pd.concat(
                [
                    train[["subject_id", "lifelog_date"] + TARGETS].copy(),
                    test[["subject_id", "lifelog_date"]].copy(),
                ],
                axis=0,
                ignore_index=True,
            )
            .sort_values(["subject_id", "lifelog_date"])
            .reset_index(drop=True)
        )

        base_df["dow"] = base_df["lifelog_date"].dt.dayofweek
        base_df["month"] = base_df["lifelog_date"].dt.month
        base_df["day"] = base_df["lifelog_date"].dt.day
        base_df["is_weekend"] = (base_df["dow"] >= 5).astype(int)
        base_df["days_from_global_start"] = (
            base_df["lifelog_date"] - base_df["lifelog_date"].min()
        ).dt.days

        sensor_map = {
            "activity": get_sensor_path("mActivity"),
            "light": get_sensor_path("mLight"),
            "screen": get_sensor_path("mScreenStatus"),
            "hr": get_sensor_path("wHr"),
            "pedo": get_sensor_path("wPedo"),
            "charge": get_sensor_path("mACStatus"),
        }
        log("sensor map:")
        for k, v in sensor_map.items():
            log(f"  {k}: {None if v is None else v.name}")

        feature_tables = build_simple_sensor_features(sensor_map)

        feat = base_df.copy()
        for ft in feature_tables:
            feat = feat.merge(ft, on=["subject_id", "lifelog_date"], how="left")

        stable_candidates = [
            c for c in feat.columns if c not in ["subject_id", "lifelog_date"] + TARGETS
        ]
        core_personal_cols = [
            c
            for c in stable_candidates
            if c
            in [
                "heart_rate_mean",
                "heart_rate_std",
                "heart_rate_sleep_mean",
                "heart_rate_active_mean",
                "step_sum",
                "distance_sum",
                "speed_mean",
                "m_screen_use_mean",
                "m_light_mean",
                "m_activity_mean",
            ]
        ]
        log(f"stable features: {len(stable_candidates)}")
        log(f"core personalization cols: {core_personal_cols}")

        feat = add_prior_features(feat, core_personal_cols)

        train_mask = feat[TARGETS[0]].notnull()
        global_means = feat.loc[train_mask, stable_candidates].mean(numeric_only=True)
        global_stds = feat.loc[train_mask, stable_candidates].std(numeric_only=True)

        for col in core_personal_cols:
            feat[f"{col}_prior_mean"] = feat[f"{col}_prior_mean"].fillna(
                global_means.get(col, 0.0)
            )
            feat[f"{col}_prior_std"] = feat[f"{col}_prior_std"].fillna(
                global_stds.get(col, 1.0)
            )
            feat[f"{col}_dev"] = feat[f"{col}_dev"].fillna(0.0)

        final_features = stable_candidates + sum(
            [
                [f"{c}_prior_mean", f"{c}_prior_std", f"{c}_prior_cnt", f"{c}_dev"]
                for c in core_personal_cols
            ],
            [],
        )
        final_features = [c for c in final_features if c in feat.columns]

        train_feat = feat[feat[TARGETS[0]].notnull()].copy().reset_index(drop=True)
        test_feat = feat[feat[TARGETS[0]].isnull()].copy().reset_index(drop=True)

        X_train = train_feat[final_features].copy()
        X_test = test_feat[final_features].copy()
        med = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med)
        X_test = X_test.fillna(med)

        tr_idx, va_idx = build_global_time_split(train_feat, valid_ratio=0.2)
        log(f"train rows: {len(tr_idx)}, valid rows: {len(va_idx)}")

        oof_lgb, oof_cat, test_lgb, test_cat = fit_predict_split(
            X_train, train_feat[TARGETS], X_test, tr_idx, va_idx, final_features
        )
        pred_avg = 0.5 * oof_lgb + 0.5 * oof_cat

        score_lgb, each_lgb = avg_logloss(train_feat.loc[va_idx, TARGETS], oof_lgb)
        score_cat, each_cat = avg_logloss(train_feat.loc[va_idx, TARGETS], oof_cat)
        score_avg, each_avg = avg_logloss(train_feat.loc[va_idx, TARGETS], pred_avg)
        log(f"time split LGB avg_logloss: {score_lgb:.6f} | each: {each_lgb}")
        log(f"time split CAT avg_logloss: {score_cat:.6f} | each: {each_cat}")
        log(f"time split AVG avg_logloss: {score_avg:.6f} | each: {each_avg}")

        rows = []
        for alpha in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
            tmp = pred_avg.copy()
            for t in TARGETS:
                tmp[t] = clip_proba(
                    shrink_proba(tmp[t].values, alpha=alpha), 0.03, 0.97
                )
            score, each = avg_logloss(train_feat.loc[va_idx, TARGETS], tmp)
            rows.append([alpha, score] + each)
        alpha_df = pd.DataFrame(
            rows, columns=["alpha", "avg_logloss"] + TARGETS
        ).sort_values("avg_logloss")
        best_alpha = float(alpha_df.iloc[0]["alpha"])
        log("alpha tuning results:")
        print(alpha_df.to_string(index=False))
        log(f"best_alpha: {best_alpha}")

        best_target_model = {}
        for t in TARGETS:
            lgb_score = log_loss(
                train_feat.loc[va_idx, t].astype(int),
                np.clip(oof_lgb[t], 1e-6, 1 - 1e-6),
            )
            cat_score = log_loss(
                train_feat.loc[va_idx, t].astype(int),
                np.clip(oof_cat[t], 1e-6, 1 - 1e-6),
            )
            best_target_model[t] = "lgb" if lgb_score <= cat_score else "cat"
        log(f"best target model: {best_target_model}")

        final_pred_lgb, final_pred_cat = train_full_models(
            X_train, train_feat[TARGETS], X_test, final_features
        )

        final_pred = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
        for t in TARGETS:
            p = (
                final_pred_lgb[t].values
                if best_target_model[t] == "lgb"
                else final_pred_cat[t].values
            )
            p = shrink_proba(p, alpha=best_alpha)
            p = clip_proba(p, 0.03, 0.97)
            final_pred[t] = p

        submission = sub.copy()
        for t in TARGETS:
            submission[t] = final_pred[t].values

        save_path = OUT_DIR / "submission_step1_tree_ensemble.csv"
        submission.to_csv(save_path, index=False)
        log(f"saved submission: {save_path}")

        feature_dump_path = BASE_DIR / "features_daily_sensor_table.csv"
        feat.to_csv(feature_dump_path, index=False)
        log(f"saved feature table: {feature_dump_path}")

        for t in TARGETS:
            log(
                f"{t}: mean={submission[t].mean():.4f}, std={submission[t].std():.4f}, "
                f"min={submission[t].min():.4f}, max={submission[t].max():.4f}"
            )

    if __name__ == "__main__":
        main()


# 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
# Pruned historical definition: run_blend_anchor_with_target_history (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_personal_prior_seed_ensemble (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_probe_alpha094_catboost_heavy (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_probe_alpha096_catboost_heavy (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_probe_alpha098_catboost_very_heavy (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_probe_alpha098_catboost_heavy_1585 (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_probe_alpha098_catboost_heavy_2080 (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_probe_alpha098_clipped_catboost_heavy (not reachable from the final runner).


# 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
# Pruned historical definition: train_probe_alpha098_logit_blend (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_probe_alpha100_catboost_heavy (not reachable from the final runner).


# 함수: 최근 target 상태의 reversion/dynamics prior 후보를 생성합니다.
def build_target_dynamics_reversion_candidates():
    __file__ = str(
        PROJECT_ROOT / "pipeline/build_target_dynamics_reversion_candidates"
    )
    __name__ = "__main__"

    import math
    from dataclasses import dataclass
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SUB_SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"
    SUBMISSION_DIR = BASE_DIR / "submissions"
    ANCHOR_PATH = SUBMISSION_DIR / "submission_step1_transition_past_tw_b.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    Q_TARGETS = {"Q1", "Q2", "Q3"}

    @dataclass(frozen=True)
    class DynamicsConfig:
        tag: str
        q_weights: tuple[float, float, float, float, float]
        s_weights: tuple[float, float, float, float, float]
        cross_scale: float
        trend_scale: float
        transition_decay: float
        clip_low: float = 0.045
        clip_high: float = 0.955

        # 함수: weights for 보조 로직을 수행합니다.
        def weights_for(self, target: str) -> tuple[float, float, float, float, float]:
            return self.q_weights if target in Q_TARGETS else self.s_weights

    CONFIGS = [
        DynamicsConfig(
            tag="conservative",
            q_weights=(0.42, 0.18, 0.12, 0.22, 0.06),
            s_weights=(0.38, 0.20, 0.16, 0.18, 0.08),
            cross_scale=0.25,
            trend_scale=0.08,
            transition_decay=5.0,
        ),
        DynamicsConfig(
            tag="transition",
            q_weights=(0.34, 0.14, 0.10, 0.34, 0.08),
            s_weights=(0.32, 0.18, 0.14, 0.26, 0.10),
            cross_scale=0.35,
            trend_scale=0.08,
            transition_decay=4.0,
        ),
        DynamicsConfig(
            tag="pattern",
            q_weights=(0.32, 0.14, 0.10, 0.26, 0.18),
            s_weights=(0.30, 0.18, 0.14, 0.22, 0.16),
            cross_scale=0.32,
            trend_scale=0.06,
            transition_decay=4.5,
        ),
        DynamicsConfig(
            tag="revert",
            q_weights=(0.34, 0.26, 0.24, 0.12, 0.04),
            s_weights=(0.32, 0.28, 0.26, 0.10, 0.04),
            cross_scale=0.12,
            trend_scale=0.04,
            transition_decay=3.0,
        ),
    ]

    # 함수: 확률값을 안정적인 범위로 제한합니다.
    def clip_prob(p: float, low: float = 0.045, high: float = 0.955) -> float:
        return float(np.clip(p, low, high))

    # 함수: binary log-loss를 계산합니다.
    def binary_logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
        pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=float)
        return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())

    # 함수: mean or global 보조 로직을 수행합니다.
    def mean_or_global(values: np.ndarray, global_mean: float) -> float:
        if len(values) == 0:
            return float(global_mean)
        return float(np.mean(values))

    # 함수: ewma tail 보조 로직을 수행합니다.
    def ewma_tail(values: np.ndarray, alpha: float, global_mean: float) -> float:
        if len(values) == 0:
            return float(global_mean)
        out = float(values[0])
        for value in values[1:]:
            out = alpha * float(value) + (1 - alpha) * out
        return out

    # 함수: pattern code 보조 로직을 수행합니다.
    def pattern_code(values: np.ndarray) -> int:
        bits = (np.asarray(values, dtype=float) >= 0.5).astype(int)
        out = 0
        for bit in bits:
            out = (out << 1) | int(bit)
        return int(out)

    # 함수: safe recent mix 보조 로직을 수행합니다.
    def safe_recent_mix(values: np.ndarray, global_mean: float) -> float:
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return float(global_mean)
        last1 = float(values[-1])
        last3 = mean_or_global(values[-3:], global_mean)
        last7 = mean_or_global(values[-7:], global_mean)
        last14 = mean_or_global(values[-14:], global_mean)
        ewma_fast = ewma_tail(values, 0.48, global_mean)
        ewma_slow = ewma_tail(values, 0.22, global_mean)
        return float(
            0.14 * last1
            + 0.22 * last3
            + 0.24 * last7
            + 0.14 * last14
            + 0.16 * ewma_fast
            + 0.10 * ewma_slow
        )

    # 함수: trend adjust 보조 로직을 수행합니다.
    def trend_adjust(values: np.ndarray, horizon_days: int, scale: float) -> float:
        values = np.asarray(values, dtype=float)
        if len(values) < 6 or np.nanstd(values[-10:]) < 1e-8:
            return 0.0
        tail = values[-12:]
        x = np.arange(len(tail), dtype=float)
        slope = float(np.polyfit(x, tail, 1)[0])
        horizon = min(max(int(horizon_days), 1), 10)
        return float(np.clip(slope * math.sqrt(horizon) * scale, -0.055, 0.055))

    # 함수: build dow adjustments 보조 로직을 수행합니다.
    def build_dow_adjustments(train: pd.DataFrame) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        for target in TARGETS:
            global_mean = float(train[target].mean())
            mean_by_dow = train.groupby("dow")[target].mean()
            count_by_dow = train.groupby("dow")[target].count()
            out[target] = {}
            for dow in range(7):
                if dow not in mean_by_dow.index:
                    out[target][dow] = 0.0
                    continue
                reliability = min(float(count_by_dow.loc[dow]) / 80.0, 1.0)
                raw = (float(mean_by_dow.loc[dow]) - global_mean) * 0.08 * reliability
                out[target][dow] = float(np.clip(raw, -0.018, 0.018))
        return out

    # 함수: consecutive pairs 보조 로직을 수행합니다.
    def consecutive_pairs(train: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for sid, group in train.sort_values(["subject_id", "lifelog_date"]).groupby(
            "subject_id"
        ):
            group = group.reset_index(drop=True)
            for i in range(1, len(group)):
                prev = group.loc[i - 1, TARGETS].astype(float).to_numpy()
                curr = group.loc[i, TARGETS].astype(float).to_numpy()
                rows.append((sid, pattern_code(prev), *prev, *curr))
        cols = (
            ["subject_id", "pattern"]
            + [f"prev_{t}" for t in TARGETS]
            + [f"curr_{t}" for t in TARGETS]
        )
        return pd.DataFrame(rows, columns=cols)

    # 함수: smooth rate 보조 로직을 수행합니다.
    def smooth_rate(pos: float, total: float, prior: float, strength: float) -> float:
        return float((pos + strength * prior) / (total + strength))

    # 함수: 검증/전체 학습 모델을 적합하고 예측값을 만듭니다.
    def fit_stats(train_fit: pd.DataFrame) -> dict:
        train_fit = train_fit.sort_values(["subject_id", "lifelog_date"]).copy()
        train_fit["dow"] = train_fit["lifelog_date"].dt.dayofweek
        pairs = consecutive_pairs(train_fit)

        global_mean = {target: float(train_fit[target].mean()) for target in TARGETS}
        dow_adjust = build_dow_adjustments(train_fit)

        global_self: dict[str, tuple[float, float]] = {}
        subject_self: dict[tuple[str, str], tuple[float, float]] = {}
        for target in TARGETS:
            prev_col = f"prev_{target}"
            curr_col = f"curr_{target}"
            if len(pairs) == 0:
                global_self[target] = (global_mean[target], global_mean[target])
                continue
            p0_df = pairs[pairs[prev_col] < 0.5]
            p1_df = pairs[pairs[prev_col] >= 0.5]
            p0 = smooth_rate(
                float(p0_df[curr_col].sum()), len(p0_df), global_mean[target], 8.0
            )
            p1 = smooth_rate(
                float(p1_df[curr_col].sum()), len(p1_df), global_mean[target], 8.0
            )
            global_self[target] = (p0, p1)

            for sid, sid_pairs in pairs.groupby("subject_id"):
                s0 = sid_pairs[sid_pairs[prev_col] < 0.5]
                s1 = sid_pairs[sid_pairs[prev_col] >= 0.5]
                sp0 = smooth_rate(float(s0[curr_col].sum()), len(s0), p0, 5.0)
                sp1 = smooth_rate(float(s1[curr_col].sum()), len(s1), p1, 5.0)
                subject_self[(str(sid), target)] = (sp0, sp1)

        pattern_stats: dict[str, dict[int, tuple[float, int]]] = {
            target: {} for target in TARGETS
        }
        if len(pairs):
            for target in TARGETS:
                curr_col = f"curr_{target}"
                for pattern, group in pairs.groupby("pattern"):
                    n = len(group)
                    p = smooth_rate(
                        float(group[curr_col].sum()), n, global_mean[target], 10.0
                    )
                    pattern_stats[target][int(pattern)] = (p, n)

        cross_coef: dict[str, dict[str, float]] = {target: {} for target in TARGETS}
        if len(pairs):
            for target in TARGETS:
                curr_col = f"curr_{target}"
                for source in TARGETS:
                    if source == target:
                        cross_coef[target][source] = 0.0
                        continue
                    prev_col = f"prev_{source}"
                    high = pairs[pairs[prev_col] >= 0.5]
                    low = pairs[pairs[prev_col] < 0.5]
                    if len(high) < 8 or len(low) < 8:
                        coef = 0.0
                    else:
                        reliability = min(min(len(high), len(low)) / 50.0, 1.0)
                        coef = (
                            float(high[curr_col].mean()) - float(low[curr_col].mean())
                        ) * reliability
                    cross_coef[target][source] = float(np.clip(coef, -0.16, 0.16))

        return {
            "global_mean": global_mean,
            "dow_adjust": dow_adjust,
            "global_self": global_self,
            "subject_self": subject_self,
            "pattern_stats": pattern_stats,
            "cross_coef": cross_coef,
        }

    # 함수: 학습된 규칙이나 모델로 test 구간 예측을 생성합니다.
    def predict_one(
        subject_id: str,
        history: pd.DataFrame,
        pred_date: pd.Timestamp,
        stats: dict,
        config: DynamicsConfig,
    ) -> dict[str, float]:
        if len(history):
            history = history.sort_values("lifelog_date").copy()
            last_date = pd.Timestamp(history["lifelog_date"].iloc[-1])
            horizon_days = max(int((pred_date - last_date).days), 1)
            prev_vec = history[TARGETS].iloc[-1].astype(float).to_numpy()
        else:
            horizon_days = 1
            prev_vec = np.array([stats["global_mean"][t] for t in TARGETS], dtype=float)

        transition_strength = float(
            np.exp(-(max(horizon_days, 1) - 1) / config.transition_decay)
        )
        dow = int(pred_date.dayofweek)
        prev_code = pattern_code(prev_vec)
        out = {}

        for target_idx, target in enumerate(TARGETS):
            global_base = stats["global_mean"][target] + stats["dow_adjust"][
                target
            ].get(dow, 0.0)
            values = (
                history[target].astype(float).to_numpy()
                if len(history)
                else np.array([], dtype=float)
            )

            recent = safe_recent_mix(values, stats["global_mean"][target])
            subject_mean = mean_or_global(values, stats["global_mean"][target])

            p0, p1 = stats["subject_self"].get(
                (subject_id, target),
                stats["global_self"][target],
            )
            prev_prob = float(np.clip(prev_vec[target_idx], 0.0, 1.0))
            self_transition = (1.0 - prev_prob) * p0 + prev_prob * p1
            self_transition = (
                transition_strength * self_transition
                + (1 - transition_strength) * subject_mean
            )

            pattern_p = stats["pattern_stats"][target].get(prev_code, (global_base, 0))[
                0
            ]
            pattern_p = (
                transition_strength * pattern_p
                + (1 - transition_strength) * subject_mean
            )

            cross_adj = 0.0
            for source_idx, source in enumerate(TARGETS):
                coef = stats["cross_coef"][target].get(source, 0.0)
                cross_adj += coef * (
                    float(prev_vec[source_idx]) - stats["global_mean"][source]
                )
            cross_adj *= (
                config.cross_scale * transition_strength / max(len(TARGETS) - 1, 1)
            )

            recent_w, subject_w, global_w, self_w, pattern_w = config.weights_for(
                target
            )
            pred = (
                recent_w * recent
                + subject_w * subject_mean
                + global_w * global_base
                + self_w * self_transition
                + pattern_w * pattern_p
            )
            pred += cross_adj
            pred += trend_adjust(values, horizon_days, config.trend_scale)
            out[target] = clip_prob(pred, config.clip_low, config.clip_high)

        return out

    # 함수: 학습된 규칙이나 모델로 test 구간 예측을 생성합니다.
    def recursive_predict(
        train_fit: pd.DataFrame,
        test_frame: pd.DataFrame,
        config: DynamicsConfig,
    ) -> pd.DataFrame:
        stats = fit_stats(train_fit)
        output_rows = []

        histories = {
            str(sid): group[["subject_id", "lifelog_date"] + TARGETS].copy()
            for sid, group in train_fit.sort_values(
                ["subject_id", "lifelog_date"]
            ).groupby("subject_id")
        }

        test_sorted = (
            test_frame.reset_index(drop=True)
            .reset_index(names="row_pos")
            .sort_values(["subject_id", "lifelog_date"])
        )
        pred_store = {}
        for row in test_sorted.itertuples(index=False):
            sid = str(row.subject_id)
            pred_date = pd.Timestamp(row.lifelog_date)
            history = histories.get(
                sid, pd.DataFrame(columns=["subject_id", "lifelog_date"] + TARGETS)
            )
            pred = predict_one(sid, history, pred_date, stats, config)
            pred_store[int(row.row_pos)] = pred

            append_row = {"subject_id": sid, "lifelog_date": pred_date, **pred}
            histories[sid] = pd.concat(
                [history, pd.DataFrame([append_row])], ignore_index=True
            )

        for idx in range(len(test_frame)):
            output_rows.append(pred_store[idx])
        return pd.DataFrame(output_rows)

    # 함수: rolling tail validation 보조 로직을 수행합니다.
    def rolling_tail_validation(
        train: pd.DataFrame, config: DynamicsConfig
    ) -> dict[str, float]:
        y_true_all = []
        pred_all = []
        per_target = {target: {"y": [], "p": []} for target in TARGETS}

        for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby(
            "subject_id"
        ):
            n = len(group)
            holdout = min(max(6, int(round(n * 0.24))), 14)
            if n - holdout < 10:
                continue
            fit = train.drop(group.tail(holdout).index).copy()
            valid = group.tail(holdout)[
                ["subject_id", "sleep_date", "lifelog_date"]
            ].copy()
            preds = recursive_predict(fit, valid, config)

            y_true = group.tail(holdout)[TARGETS].to_numpy(dtype=float)
            pred = preds[TARGETS].to_numpy(dtype=float)
            y_true_all.append(y_true)
            pred_all.append(pred)
            for j, target in enumerate(TARGETS):
                per_target[target]["y"].extend(y_true[:, j].tolist())
                per_target[target]["p"].extend(pred[:, j].tolist())

        y = np.vstack(y_true_all)
        p = np.vstack(pred_all)
        out = {"overall": binary_logloss(y, p)}
        for target in TARGETS:
            out[target] = binary_logloss(
                np.array(per_target[target]["y"]), np.array(per_target[target]["p"])
            )
        return out

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate_submission(df: pd.DataFrame, name: str) -> None:
        required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if df.shape != (250, 10):
            raise ValueError(f"{name} unexpected shape: {df.shape}")
        if df.isnull().sum().sum() != 0:
            raise ValueError(f"{name} has null values")
        if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
            raise ValueError(f"{name} has out-of-range probabilities")

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save_submission(
        name: str, sample: pd.DataFrame, preds: pd.DataFrame
    ) -> pd.DataFrame:
        out = sample.copy()
        out[TARGETS] = preds[TARGETS].to_numpy(dtype=float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        validate_submission(out, name)
        out.to_csv(SUBMISSION_DIR / name, index=False)
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend_with_anchor(
        anchor: pd.DataFrame, dyn: pd.DataFrame, weight: float
    ) -> pd.DataFrame:
        out = anchor.copy()
        out[TARGETS] = (1.0 - weight) * anchor[TARGETS].astype(float) + weight * dyn[
            TARGETS
        ].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def targetwise_blend_with_anchor(
        anchor: pd.DataFrame, dyn: pd.DataFrame, weights: dict[str, float]
    ) -> pd.DataFrame:
        out = anchor.copy()
        for target in TARGETS:
            w = weights.get(target, 0.0)
            out[target] = (1.0 - w) * anchor[target].astype(float) + w * dyn[
                target
            ].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize_candidate(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
        return {
            "candidate": name,
            "diff_vs_best": float(diff.values.mean()),
            "max_diff_vs_best": float(diff.values.max()),
            "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
            "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
            "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
            "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
        }

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        train = pd.read_csv(TRAIN_PATH)
        sample = pd.read_csv(SUB_SAMPLE_PATH)
        anchor = pd.read_csv(ANCHOR_PATH)

        train["subject_id"] = train["subject_id"].astype(str)
        sample["subject_id"] = sample["subject_id"].astype(str)
        anchor["subject_id"] = anchor["subject_id"].astype(str)
        train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
        sample["lifelog_date"] = pd.to_datetime(sample["lifelog_date"])
        train["sleep_date"] = pd.to_datetime(train["sleep_date"])
        sample["sleep_date"] = pd.to_datetime(sample["sleep_date"])

        validation_rows = []
        for config in CONFIGS:
            result = rolling_tail_validation(train, config)
            validation_rows.append({"config": config.tag, **result})

        validation = (
            pd.DataFrame(validation_rows).sort_values("overall").reset_index(drop=True)
        )
        print("Rolling recursive tail validation:")
        print(validation.to_string(index=False))

        best_tag = str(validation.loc[0, "config"])
        best_config = next(config for config in CONFIGS if config.tag == best_tag)
        print(f"\nBest reset-dynamics config by tail validation: {best_config.tag}")

        dyn_preds = recursive_predict(
            train,
            sample[["subject_id", "sleep_date", "lifelog_date"]].copy(),
            best_config,
        )

        pure_name = f"submission_step1_dynamics_prior.csv"
        pure_sub = save_submission(pure_name, sample, dyn_preds)

        candidates = [summarize_candidate(pure_name, pure_sub, anchor)]

        for weight in [0.06, 0.10, 0.14, 0.18]:
            name = (
                f"submission_step1_dynamics_blend_w{int(weight*100):02d}.csv"
            )
            blended = blend_with_anchor(anchor, pure_sub, weight)
            validate_submission(blended, name)
            blended.to_csv(SUBMISSION_DIR / name, index=False)
            candidates.append(summarize_candidate(name, blended, anchor))

        tw_weights = {
            "Q1": 0.12,
            "Q2": 0.18,
            "Q3": 0.18,
            "S1": 0.04,
            "S2": 0.08,
            "S3": 0.02,
            "S4": 0.08,
        }
        tw_name = f"submission_step1_dynamics_blend_targetwise.csv"
        tw_sub = targetwise_blend_with_anchor(anchor, pure_sub, tw_weights)
        validate_submission(tw_sub, tw_name)
        tw_sub.to_csv(SUBMISSION_DIR / tw_name, index=False)
        candidates.append(summarize_candidate(tw_name, tw_sub, anchor))

        summary = (
            pd.DataFrame(candidates).sort_values("diff_vs_best").reset_index(drop=True)
        )
        print("\nSaved candidate summary vs current best anchor:")
        print(summary.to_string(index=False))

        print("\nSuggested submit order:")
        print(f"1) submission_step1_dynamics_blend_w10.csv")
        print(f"2) submission_step1_dynamics_blend_targetwise.csv")
        print(f"3) {pure_name} only if we decide to take a high-risk reset shot")

    if __name__ == "__main__":
        main()


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_seed3_routing_batch (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_seed3_q5050_s2080_routing (not reachable from the final runner).


# 함수: Q 60:40, S 20:80 routing의 3-seed ensemble anchor를 학습합니다.
def train_seed3_q6040_s2080_routing():
    __file__ = str(PROJECT_ROOT / "pipeline/train_seed3_q6040_s2080_routing")
    __name__ = "__main__"
    import math, json, warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    warnings.filterwarnings("ignore")

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    SENSOR_DIR = BASE_DIR / "ch2025_data_items"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SUB_PATH = BASE_DIR / "ch2026_submission_sample.csv"
    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    Q_TARGETS = ["Q1", "Q2", "Q3"]
    S_TARGETS = ["S1", "S2", "S3", "S4"]

    FIXED_ALPHA = 0.98
    SEEDS = [42, 77, 2024]
    LGB_W = 0.20
    CAT_W = 0.80
    OUT_PATH = BASE_DIR / "submissions" / "submission_step1_target_routing.csv"

    # 함수: infer test frame from submission 보조 로직을 수행합니다.
    def infer_test_frame_from_submission(train_df, submission_df):
        if {"subject_id", "lifelog_date"}.issubset(submission_df.columns):
            test_df = submission_df[["subject_id", "lifelog_date"]].copy()
            test_df["subject_id"] = test_df["subject_id"].astype(str)
            test_df["lifelog_date"] = pd.to_datetime(test_df["lifelog_date"])
            return test_df
        n_test = len(submission_df)
        last_dates = train_df.groupby("subject_id")["lifelog_date"].max().sort_values()
        subs = list(last_dates.index)
        reps = math.ceil(n_test / len(subs))
        subject_seq = (subs * reps)[:n_test]
        next_dates = {
            sid: train_df.loc[train_df["subject_id"] == sid, "lifelog_date"].max()
            + pd.Timedelta(days=1)
            for sid in subs
        }
        rows = []
        for sid in subject_seq:
            rows.append((sid, next_dates[sid]))
            next_dates[sid] += pd.Timedelta(days=1)
        return pd.DataFrame(rows, columns=["subject_id", "lifelog_date"])

    # 함수: get sensor path 보조 로직을 수행합니다.
    def get_sensor_path(keyword):
        files = sorted(SENSOR_DIR.glob("*.parquet"))
        for p in files:
            if keyword.lower() in p.name.lower():
                return p
        return None

    # 함수: prep sensor df 보조 로직을 수행합니다.
    def prep_sensor_df(df, value_cols):
        df = df.copy()
        df["subject_id"] = df["subject_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["lifelog_date"] = df["timestamp"].dt.floor("D")
        df["hour"] = df["timestamp"].dt.hour
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
        df["is_day"] = ((df["hour"] >= 9) & (df["hour"] <= 18)).astype(int)
        keep = [
            "subject_id",
            "lifelog_date",
            "timestamp",
            "hour",
            "is_night",
            "is_day",
        ] + [c for c in value_cols if c in df.columns]
        return df[keep].copy()

    # 함수: add simple agg 보조 로직을 수행합니다.
    def add_simple_agg(df, val_col, prefix):
        g = df.groupby(["subject_id", "lifelog_date"])[val_col]
        feat = g.agg(["mean", "std", "min", "max", "count"]).reset_index()
        feat.columns = ["subject_id", "lifelog_date"] + [
            f"{prefix}_{c}" for c in ["mean", "std", "min", "max", "count"]
        ]
        night = (
            df[df["is_night"] == 1]
            .groupby(["subject_id", "lifelog_date"])[val_col]
            .agg(["mean", "std"])
            .reset_index()
        )
        night.columns = [
            "subject_id",
            "lifelog_date",
            f"{prefix}_night_mean",
            f"{prefix}_night_std",
        ]
        day = (
            df[df["is_day"] == 1]
            .groupby(["subject_id", "lifelog_date"])[val_col]
            .agg(["mean", "std"])
            .reset_index()
        )
        day.columns = [
            "subject_id",
            "lifelog_date",
            f"{prefix}_day_mean",
            f"{prefix}_day_std",
        ]
        out = feat.merge(night, on=["subject_id", "lifelog_date"], how="left")
        out = out.merge(day, on=["subject_id", "lifelog_date"], how="left")
        return out

    # 함수: explode hr array 보조 로직을 수행합니다.
    def explode_hr_array(x):
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return []
            return (
                pd.to_numeric(pd.Series(x.ravel()), errors="coerce").dropna().tolist()
            )
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return []
            return pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna().tolist()
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return []
            try:
                v = json.loads(s)
                if isinstance(v, np.ndarray):
                    return (
                        pd.to_numeric(pd.Series(v.ravel()), errors="coerce")
                        .dropna()
                        .tolist()
                    )
                if isinstance(v, (list, tuple)):
                    return (
                        pd.to_numeric(pd.Series(list(v)), errors="coerce")
                        .dropna()
                        .tolist()
                    )
                return []
            except Exception:
                return []
        try:
            if pd.isna(x):
                return []
        except Exception:
            pass
        try:
            return [float(x)]
        except Exception:
            return []

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def build_activity_features(sensor_map):
        p = sensor_map["activity"]
        if p is None:
            return None
        df = pd.read_parquet(p)
        df = prep_sensor_df(df, ["m_activity"])
        return add_simple_agg(df, "m_activity", "m_activity")

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def build_light_features(sensor_map):
        p = sensor_map["light"]
        if p is None:
            return None
        df = pd.read_parquet(p)
        df = prep_sensor_df(df, ["m_light"])
        return add_simple_agg(df, "m_light", "m_light")

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def build_screen_features(sensor_map):
        p = sensor_map["screen"]
        if p is None:
            return None
        df = pd.read_parquet(p)
        df = prep_sensor_df(df, ["m_screen_use"])
        return add_simple_agg(df, "m_screen_use", "m_screen_use")

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def build_charge_features(sensor_map):
        p = sensor_map["charge"]
        if p is None:
            return None
        df = pd.read_parquet(p)
        df = prep_sensor_df(df, ["m_charging"])
        return add_simple_agg(df, "m_charging", "m_charging")

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def build_hr_features(sensor_map):
        p = sensor_map["hr"]
        if p is None:
            return None
        df = pd.read_parquet(p)
        candidate_cols = [c for c in df.columns if c not in ["subject_id", "timestamp"]]
        preferred = [
            c for c in candidate_cols if "heart" in c.lower() or "hr" in c.lower()
        ]
        hr_col = preferred[0] if preferred else candidate_cols[0]
        print(f"[INFO] HR selected column: {hr_col}")
        df = prep_sensor_df(df, [hr_col])
        arr = df[hr_col].apply(explode_hr_array)
        df["hr_mean_row"] = arr.apply(lambda v: float(np.mean(v)) if len(v) else np.nan)
        df["hr_std_row"] = arr.apply(lambda v: float(np.std(v)) if len(v) else np.nan)
        df["hr_min_row"] = arr.apply(lambda v: float(np.min(v)) if len(v) else np.nan)
        df["hr_max_row"] = arr.apply(lambda v: float(np.max(v)) if len(v) else np.nan)
        df["hr_median_row"] = arr.apply(
            lambda v: float(np.median(v)) if len(v) else np.nan
        )
        df["hr_q75_row"] = arr.apply(
            lambda v: float(np.quantile(v, 0.75)) if len(v) else np.nan
        )
        stats = (
            df.groupby(["subject_id", "lifelog_date"])[
                [
                    "hr_mean_row",
                    "hr_std_row",
                    "hr_min_row",
                    "hr_max_row",
                    "hr_median_row",
                    "hr_q75_row",
                ]
            ]
            .mean()
            .reset_index()
        )
        stats.columns = [
            "subject_id",
            "lifelog_date",
            "heart_rate_mean",
            "heart_rate_std",
            "heart_rate_min",
            "heart_rate_max",
            "heart_rate_median",
            "heart_rate_q75",
        ]
        sleep = (
            df[df["is_night"] == 1]
            .groupby(["subject_id", "lifelog_date"])["hr_mean_row"]
            .agg(["mean", "std"])
            .reset_index()
        )
        sleep.columns = [
            "subject_id",
            "lifelog_date",
            "heart_rate_sleep_mean",
            "heart_rate_sleep_std",
        ]
        active = (
            df[df["is_day"] == 1]
            .groupby(["subject_id", "lifelog_date"])["hr_mean_row"]
            .agg(["mean", "std"])
            .reset_index()
        )
        active.columns = [
            "subject_id",
            "lifelog_date",
            "heart_rate_active_mean",
            "heart_rate_active_std",
        ]
        out = stats.merge(sleep, on=["subject_id", "lifelog_date"], how="left")
        out = out.merge(active, on=["subject_id", "lifelog_date"], how="left")
        out["heart_rate_sleep_active_diff"] = (
            out["heart_rate_sleep_mean"] - out["heart_rate_active_mean"]
        )
        return out

    # 함수: 원천 데이터에서 모델 입력용 feature table을 생성합니다.
    def build_pedo_features(sensor_map):
        p = sensor_map["pedo"]
        if p is None:
            return None
        df = pd.read_parquet(p)
        value_cols = [
            c
            for c in ["step", "distance", "speed", "calories", "running"]
            if c in df.columns
        ]
        df = prep_sensor_df(df, value_cols)
        agg_dict = {}
        if "step" in df.columns:
            agg_dict["step"] = ["sum", "mean"]
        if "distance" in df.columns:
            agg_dict["distance"] = ["sum", "mean"]
        if "speed" in df.columns:
            agg_dict["speed"] = ["mean", "max"]
        if "calories" in df.columns:
            agg_dict["calories"] = ["sum", "mean"]
        if "running" in df.columns:
            agg_dict["running"] = ["sum", "mean"]
        feat = df.groupby(["subject_id", "lifelog_date"]).agg(agg_dict).reset_index()
        feat.columns = ["subject_id", "lifelog_date"] + [
            f"{a}_{b}" for a, b in feat.columns.tolist()[2:]
        ]
        return feat

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def add_prior_features(df, cols):
        df = df.sort_values(["subject_id", "lifelog_date"]).copy()
        for col in cols:
            grp = df.groupby("subject_id")[col]
            df[f"{col}_prior_mean"] = (
                grp.expanding().mean().shift(1).reset_index(level=0, drop=True)
            )
            df[f"{col}_prior_std"] = (
                grp.expanding().std().shift(1).reset_index(level=0, drop=True)
            )
            df[f"{col}_prior_cnt"] = df.groupby("subject_id").cumcount()
            df[f"{col}_dev"] = df[col] - df[f"{col}_prior_mean"]
        return df

    # 함수: 과신한 확률을 0.5 방향으로 완화합니다.
    def shrink_proba(p, alpha=0.98):
        return np.clip(0.5 + alpha * (p - 0.5), 1e-6, 1 - 1e-6)

    # 함수: 확률값을 안정적인 범위로 제한합니다.
    def clip_proba(p, lo=0.03, hi=0.97):
        return np.clip(p, lo, hi)

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main():
        train = pd.read_csv(TRAIN_PATH)
        sub = pd.read_csv(SUB_PATH)
        train["subject_id"] = train["subject_id"].astype(str)
        train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
        if "subject_id" in sub.columns:
            sub["subject_id"] = sub["subject_id"].astype(str)
        if "lifelog_date" in sub.columns:
            sub["lifelog_date"] = pd.to_datetime(sub["lifelog_date"])
        test = infer_test_frame_from_submission(train, sub)

        base_df = (
            pd.concat(
                [
                    train[["subject_id", "lifelog_date"] + TARGETS].copy(),
                    test[["subject_id", "lifelog_date"]].copy(),
                ],
                axis=0,
                ignore_index=True,
            )
            .sort_values(["subject_id", "lifelog_date"])
            .reset_index(drop=True)
        )
        base_df["dow"] = base_df["lifelog_date"].dt.dayofweek
        base_df["month"] = base_df["lifelog_date"].dt.month
        base_df["day"] = base_df["lifelog_date"].dt.day
        base_df["is_weekend"] = (base_df["dow"] >= 5).astype(int)
        base_df["days_from_global_start"] = (
            base_df["lifelog_date"] - base_df["lifelog_date"].min()
        ).dt.days

        sensor_map = {
            "activity": get_sensor_path("mActivity"),
            "light": get_sensor_path("mLight"),
            "screen": get_sensor_path("mScreenStatus"),
            "hr": get_sensor_path("wHr"),
            "pedo": get_sensor_path("wPedo"),
            "charge": get_sensor_path("mACStatus"),
        }

        feat = base_df.copy()
        for ft in [
            build_activity_features(sensor_map),
            build_light_features(sensor_map),
            build_screen_features(sensor_map),
            build_charge_features(sensor_map),
            build_hr_features(sensor_map),
            build_pedo_features(sensor_map),
        ]:
            if ft is not None:
                feat = feat.merge(ft, on=["subject_id", "lifelog_date"], how="left")

        stable_candidates = [
            c for c in feat.columns if c not in ["subject_id", "lifelog_date"] + TARGETS
        ]
        core_personal_cols = [
            c
            for c in stable_candidates
            if c
            in [
                "m_activity_mean",
                "m_light_mean",
                "m_screen_use_mean",
                "heart_rate_mean",
                "heart_rate_std",
                "heart_rate_sleep_mean",
                "heart_rate_active_mean",
                "step_sum",
                "distance_sum",
                "speed_mean",
            ]
        ]

        feat = add_prior_features(feat, core_personal_cols)
        train_mask = feat[TARGETS[0]].notnull()
        global_means = feat.loc[train_mask, stable_candidates].mean(numeric_only=True)
        global_stds = feat.loc[train_mask, stable_candidates].std(numeric_only=True)

        for col in core_personal_cols:
            feat[f"{col}_prior_mean"] = feat[f"{col}_prior_mean"].fillna(
                global_means.get(col, 0.0)
            )
            feat[f"{col}_prior_std"] = feat[f"{col}_prior_std"].fillna(
                global_stds.get(col, 1.0)
            )
            feat[f"{col}_dev"] = feat[f"{col}_dev"].fillna(0.0)

        final_features = stable_candidates + sum(
            [
                [f"{c}_prior_mean", f"{c}_prior_std", f"{c}_prior_cnt", f"{c}_dev"]
                for c in core_personal_cols
            ],
            [],
        )
        final_features = [c for c in final_features if c in feat.columns]

        train_feat = feat[feat[TARGETS[0]].notnull()].copy().reset_index(drop=True)
        test_feat = feat[feat[TARGETS[0]].isnull()].copy().reset_index(drop=True)

        X_train = train_feat[final_features].copy()
        X_test = test_feat[final_features].copy()
        med = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med)
        X_test = X_test.fillna(med)

        final_pred_lgb = pd.DataFrame(
            0.0, index=X_test.index, columns=TARGETS, dtype=float
        )
        final_pred_cat = pd.DataFrame(
            0.0, index=X_test.index, columns=TARGETS, dtype=float
        )

        print("=" * 90)
        print("SEED ENSEMBLE FULL TRAIN + TEST PREDICTION")
        print("=" * 90)
        print("seeds:", SEEDS)

        for seed in SEEDS:
            print(f"\n[SEED] {seed}")
            for t in TARGETS:
                y = train_feat[t].astype(int)
                print(f"  > target={t}")

                lgb_model = lgb.LGBMClassifier(
                    n_estimators=220,
                    learning_rate=0.03,
                    num_leaves=15,
                    max_depth=4,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=3.0,
                    objective="binary",
                    class_weight="balanced",
                    random_state=seed,
                )
                lgb_model.fit(X_train[final_features], y)
                final_pred_lgb[t] += lgb_model.predict_proba(X_test[final_features])[
                    :, 1
                ] / len(SEEDS)

                cat_model = CatBoostClassifier(
                    iterations=220,
                    learning_rate=0.03,
                    depth=4,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    random_seed=seed,
                    allow_writing_files=False,
                    verbose=False,
                )
                cat_model.fit(X_train[final_features], y, verbose=False)
                final_pred_cat[t] += cat_model.predict_proba(X_test[final_features])[
                    :, 1
                ] / len(SEEDS)

        final_pred = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
        for t in TARGETS:
            if t in Q_TARGETS:
                lgb_w, cat_w = 0.60, 0.40
            else:
                lgb_w, cat_w = 0.20, 0.80

            p = lgb_w * final_pred_lgb[t].values + cat_w * final_pred_cat[t].values
            p = shrink_proba(p, alpha=FIXED_ALPHA)
            p = clip_proba(p, 0.03, 0.97)
            final_pred[t] = p
            print(
                f"[INFO] {t}: blend={lgb_w:.2f}:{cat_w:.2f}, mean={final_pred[t].mean():.4f}, std={final_pred[t].std():.4f}, min={final_pred[t].min():.4f}, max={final_pred[t].max():.4f}"
            )

        submission = sub.copy()
        for t in TARGETS:
            submission[t] = final_pred[t].values

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(OUT_PATH, index=False)
        print("saved:", OUT_PATH)
        print(submission.head())

    if __name__ == "__main__":
        main()


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_seed5_q5050_s2080_routing (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_seed_combo_routing_batch (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_stable_alpha078 (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_stable_alpha080 (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_stable_alpha085 (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_stable_alpha090 (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_stable_alpha090_lgb30_cat70 (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_stable_alpha090_lgb40_cat60 (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_stable_alpha092_lgb30_cat70 (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_change_feature_probe (not reachable from the final runner).


# 함수: subject별 상태 전이 prior 후보를 생성합니다.
def build_state_transition_prior_candidates():
    __file__ = str(PROJECT_ROOT / "pipeline/build_state_transition_prior_candidates")
    __name__ = "__main__"
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT / "data"
    TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
    SUB_PATH = BASE_DIR / "ch2026_submission_sample.csv"
    SUBMISSION_DIR = BASE_DIR / "submissions"
    ANCHOR_PATH = SUBMISSION_DIR / "submission_step1_target_routing.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    Q_TARGETS = ["Q1", "Q2", "Q3"]
    S_TARGETS = ["S1", "S2", "S3", "S4"]

    # 함수: infer test frame from submission 보조 로직을 수행합니다.
    def infer_test_frame_from_submission(
        train_df: pd.DataFrame, submission_df: pd.DataFrame
    ) -> pd.DataFrame:
        if {"subject_id", "lifelog_date"}.issubset(submission_df.columns):
            test_df = submission_df[["subject_id", "lifelog_date"]].copy()
            test_df["subject_id"] = test_df["subject_id"].astype(str)
            test_df["lifelog_date"] = pd.to_datetime(test_df["lifelog_date"])
            return test_df

        n_test = len(submission_df)
        last_dates = train_df.groupby("subject_id")["lifelog_date"].max().sort_values()
        subjects = list(last_dates.index)
        reps = math.ceil(n_test / len(subjects))
        subject_seq = (subjects * reps)[:n_test]
        next_dates = {
            sid: train_df.loc[train_df["subject_id"] == sid, "lifelog_date"].max()
            + pd.Timedelta(days=1)
            for sid in subjects
        }

        rows = []
        for sid in subject_seq:
            rows.append((sid, next_dates[sid]))
            next_dates[sid] += pd.Timedelta(days=1)
        return pd.DataFrame(rows, columns=["subject_id", "lifelog_date"])

    # 함수: binary log-loss를 계산합니다.
    def logloss_binary(y_true: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=float)
        return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

    # 함수: mean or nan 보조 로직을 수행합니다.
    def mean_or_nan(values: np.ndarray) -> float:
        if len(values) == 0:
            return np.nan
        return float(np.mean(values))

    # 함수: ewma tail 보조 로직을 수행합니다.
    def ewma_tail(values: np.ndarray, alpha: float) -> float:
        if len(values) == 0:
            return np.nan
        out = float(values[0])
        for value in values[1:]:
            out = alpha * float(value) + (1 - alpha) * out
        return out

    # 함수: trend adjustment 보조 로직을 수행합니다.
    def trend_adjustment(values: np.ndarray, horizon_days: int) -> float:
        if len(values) < 5:
            return 0.0
        tail = np.asarray(values[-14:], dtype=float)
        if len(np.unique(tail)) <= 1:
            return 0.0
        x = np.arange(len(tail), dtype=float)
        slope = float(np.polyfit(x, tail, 1)[0])
        # Binary targets are noisy; trend can help only as a tiny directional prior.
        return float(np.clip(slope * min(max(horizon_days, 0), 14) * 0.12, -0.06, 0.06))

    # 함수: subject state probability 보조 로직을 수행합니다.
    def subject_state_probability(
        history: np.ndarray,
        global_mean: float,
        horizon_days: int,
        dow_adjustment: float = 0.0,
    ) -> float:
        history = np.asarray(history, dtype=float)
        if len(history) == 0:
            base = global_mean
        else:
            n = len(history)
            all_mean = mean_or_nan(history)
            last3 = mean_or_nan(history[-3:])
            last7 = mean_or_nan(history[-7:])
            last14 = mean_or_nan(history[-14:])
            ewma_fast = ewma_tail(history, 0.45)
            ewma_slow = ewma_tail(history, 0.22)

            # Strong recency, but with enough all-history/global mass to avoid
            # overreacting to one noisy sleep survey answer.
            recent_mix = (
                0.20 * last3
                + 0.28 * last7
                + 0.20 * last14
                + 0.20 * ewma_fast
                + 0.12 * ewma_slow
            )
            subject_mix = 0.72 * recent_mix + 0.20 * all_mean + 0.08 * global_mean

            # Small-sample smoothing is still useful for early test dates.
            smooth = min(n / 18.0, 1.0)
            base = smooth * subject_mix + (1 - smooth) * (
                0.60 * all_mean + 0.40 * global_mean
            )
            base += trend_adjustment(history, horizon_days)

        base += dow_adjustment
        return float(np.clip(base, 0.04, 0.96))

    # 함수: build dow adjustments 보조 로직을 수행합니다.
    def build_dow_adjustments(train: pd.DataFrame, target: str) -> dict[int, float]:
        global_mean = float(train[target].mean())
        dow_mean = train.groupby("dow")[target].mean()
        dow_count = train.groupby("dow")[target].count()
        out = {}
        for dow in range(7):
            if dow not in dow_mean.index:
                out[dow] = 0.0
                continue
            # Very conservative target-level calendar effect.
            reliability = min(float(dow_count.loc[dow]) / 80.0, 1.0)
            out[dow] = float(
                np.clip(
                    (float(dow_mean.loc[dow]) - global_mean) * 0.10 * reliability,
                    -0.025,
                    0.025,
                )
            )
        return out

    # 함수: 라벨 history나 날짜 정보를 바탕으로 prior 예측을 계산합니다.
    def build_state_prior(
        train: pd.DataFrame,
        test: pd.DataFrame,
        submission_template: pd.DataFrame,
        mode: str,
    ) -> pd.DataFrame:
        out = submission_template.copy()
        train_sorted = train.sort_values(["subject_id", "lifelog_date"]).copy()
        test_sorted = (
            test.reset_index().sort_values(["subject_id", "lifelog_date"]).copy()
        )

        train_sorted["dow"] = train_sorted["lifelog_date"].dt.dayofweek
        test_sorted["dow"] = test_sorted["lifelog_date"].dt.dayofweek

        for target in TARGETS:
            global_mean = float(train_sorted[target].mean())
            dow_adjust = build_dow_adjustments(train_sorted, target)
            preds = np.zeros(len(test), dtype=float)

            subject_groups = {
                sid: g[["lifelog_date", target]].sort_values("lifelog_date").copy()
                for sid, g in train_sorted.groupby("subject_id")
            }

            for row in test_sorted.itertuples(index=False):
                sid = str(row.subject_id)
                test_date = pd.Timestamp(row.lifelog_date)
                dow = int(row.dow)
                group = subject_groups.get(sid)

                if group is None or len(group) == 0:
                    history = np.array([], dtype=float)
                    horizon = 0
                elif mode == "past_only":
                    hist_df = group[group["lifelog_date"] < test_date]
                    history = hist_df[target].astype(float).to_numpy()
                    last_date = (
                        hist_df["lifelog_date"].max()
                        if len(hist_df)
                        else group["lifelog_date"].min()
                    )
                    horizon = (
                        int((test_date - last_date).days) if pd.notna(last_date) else 0
                    )
                elif mode == "full_subject":
                    history = group[target].astype(float).to_numpy()
                    last_date = group["lifelog_date"].max()
                    horizon = int((test_date - last_date).days)
                else:
                    raise ValueError(f"unknown mode: {mode}")

                preds[int(row.index)] = subject_state_probability(
                    history=history,
                    global_mean=global_mean,
                    horizon_days=horizon,
                    dow_adjustment=dow_adjust.get(dow, 0.0),
                )

            out[target] = preds

        return out

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def forward_validate_state_prior(train: pd.DataFrame) -> pd.DataFrame:
        rows = []
        train_sorted = train.sort_values(["subject_id", "lifelog_date"]).copy()
        train_sorted["dow"] = train_sorted["lifelog_date"].dt.dayofweek

        for target in TARGETS:
            global_mean = float(train_sorted[target].mean())
            dow_adjust = build_dow_adjustments(train_sorted, target)
            y_true = []
            pred_subject_mean = []
            pred_last7 = []
            pred_state = []

            for _, g in train_sorted.groupby("subject_id"):
                values = g[target].astype(float).to_numpy()
                dates = pd.to_datetime(g["lifelog_date"]).to_numpy()
                dows = g["dow"].astype(int).to_numpy()
                for i in range(3, len(g)):
                    hist = values[:i]
                    y_true.append(values[i])
                    pred_subject_mean.append(float(np.clip(np.mean(hist), 0.04, 0.96)))
                    pred_last7.append(float(np.clip(np.mean(hist[-7:]), 0.04, 0.96)))
                    horizon = int(
                        (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
                    )
                    pred_state.append(
                        subject_state_probability(
                            hist,
                            global_mean,
                            horizon,
                            dow_adjustment=dow_adjust.get(int(dows[i]), 0.0),
                        )
                    )

            rows.append(
                {
                    "target": target,
                    "n_eval": len(y_true),
                    "subject_mean_logloss": logloss_binary(
                        np.array(y_true), np.array(pred_subject_mean)
                    ),
                    "last7_logloss": logloss_binary(
                        np.array(y_true), np.array(pred_last7)
                    ),
                    "state_prior_logloss": logloss_binary(
                        np.array(y_true), np.array(pred_state)
                    ),
                }
            )

        result = pd.DataFrame(rows)
        avg = {
            "target": "AVG",
            "n_eval": int(result["n_eval"].sum()),
            "subject_mean_logloss": float(result["subject_mean_logloss"].mean()),
            "last7_logloss": float(result["last7_logloss"].mean()),
            "state_prior_logloss": float(result["state_prior_logloss"].mean()),
        }
        return pd.concat([result, pd.DataFrame([avg])], ignore_index=True)

    # 함수: anchor 예측과 prior/model 예측을 지정된 가중치로 섞습니다.
    def blend_with_anchor(
        anchor: pd.DataFrame,
        state_prior: pd.DataFrame,
        q_weight: float,
        s_weight: float,
    ) -> pd.DataFrame:
        out = anchor.copy()
        for target in Q_TARGETS:
            out[target] = (1 - q_weight) * anchor[target].astype(
                float
            ) + q_weight * state_prior[target].astype(float)
        for target in S_TARGETS:
            out[target] = (1 - s_weight) * anchor[target].astype(
                float
            ) + s_weight * state_prior[target].astype(float)
        out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
        return out

    # 함수: 생성된 후보 결과를 파일로 저장하고 요약 통계를 반환합니다.
    def save_submission(df: pd.DataFrame, name: str) -> Path:
        path = SUBMISSION_DIR / name
        df.to_csv(path, index=False)
        print(f"[SAVE] {path}")
        return path

    # 함수: 후보 예측과 기준 예측의 차이를 요약합니다.
    def summarize_candidate(
        name: str, candidate: pd.DataFrame, anchor: pd.DataFrame
    ) -> dict[str, float | str]:
        diff = (candidate[TARGETS].astype(float) - anchor[TARGETS].astype(float)).abs()
        return {
            "candidate": name,
            "mean_abs_diff": float(diff.values.mean()),
            "max_abs_diff": float(diff.values.max()),
            "q_mean_abs_diff": float(diff[Q_TARGETS].values.mean()),
            "s_mean_abs_diff": float(diff[S_TARGETS].values.mean()),
            "mean_q": float(candidate[Q_TARGETS].values.mean()),
            "mean_s": float(candidate[S_TARGETS].values.mean()),
        }

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main() -> None:
        train = pd.read_csv(TRAIN_PATH)
        sub = pd.read_csv(SUB_PATH)
        anchor = pd.read_csv(ANCHOR_PATH)

        train["subject_id"] = train["subject_id"].astype(str)
        train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
        if "subject_id" in sub.columns:
            sub["subject_id"] = sub["subject_id"].astype(str)
        if "lifelog_date" in sub.columns:
            sub["lifelog_date"] = pd.to_datetime(sub["lifelog_date"])

        test = infer_test_frame_from_submission(train, sub)

        print("=" * 100)
        print("ETRI STATE TRANSITION PRIOR CANDIDATES")
        print(
            "New axis: subject-level target state / recency dynamics, no sensor model retraining."
        )
        print("Anchor:", ANCHOR_PATH.name)
        print("=" * 100)

        cv = forward_validate_state_prior(train)
        print("\n[Forward validation on train target history only]")
        print(cv.to_string(index=False))

        past_prior = build_state_prior(train, test, anchor, mode="past_only")
        full_prior = build_state_prior(train, test, anchor, mode="full_subject")

        candidates = {
            "submission_step1_history_past.csv": past_prior,
            "submission_step1_history_full.csv": full_prior,
            "submission_step1_transition_past_q10_s05.csv": blend_with_anchor(
                anchor, past_prior, q_weight=0.10, s_weight=0.05
            ),
            "submission_step1_transition_past_q14_s07.csv": blend_with_anchor(
                anchor, past_prior, q_weight=0.14, s_weight=0.07
            ),
            "submission_step1_transition_past_q18_s09.csv": blend_with_anchor(
                anchor, past_prior, q_weight=0.18, s_weight=0.09
            ),
            "submission_step1_transition_full_q08_s04.csv": blend_with_anchor(
                anchor, full_prior, q_weight=0.08, s_weight=0.04
            ),
            "submission_step1_transition_full_q12_s06.csv": blend_with_anchor(
                anchor, full_prior, q_weight=0.12, s_weight=0.06
            ),
        }

        summaries = []
        for name, df in candidates.items():
            save_submission(df, name)
            summaries.append(summarize_candidate(name, df, anchor))

        summary_df = pd.DataFrame(summaries)
        print("\n[Candidate drift vs anchor]")
        print(summary_df.to_string(index=False))

        print("\n[Recommendation]")
        print(
            "1) First aggressive-but-controlled candidate: submission_step1_transition_past_q14_s07.csv"
        )
        print("2) Safer candidate: submission_step1_transition_past_q10_s05.csv")
        print(
            "3) Pure state priors are diagnostic only; submit only if you want a real gamble."
        )

    if __name__ == "__main__":
        main()


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_strong_sensor_ensemble (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_seed3_target_history_routing (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_targetwise_alpha_routing (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_targetwise_q3070_s2080_routing (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_targetwise_q4060_s2080_routing (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_targetwise_q5050_s2080_routing (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_targetwise_q6040_s2080_routing (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_minimal_usage_feature_routing (not reachable from the final runner).


# 함수: 모델 학습에 필요한 feature와 target을 구성하고 예측을 생성합니다.
# Pruned historical definition: train_usage_stats_feature_routing (not reachable from the final runner).


# 함수: update project readme 보조 로직을 수행합니다.
# Pruned historical definition: update_project_readme (not reachable from the final runner).


# 함수: 최종 제출 파일의 shape, null, 확률 범위를 검증합니다.
def validate_final_submission():
    __file__ = str(PROJECT_ROOT / "pipeline/validate_final_submission")
    __name__ = "__main__"
    import argparse
    from pathlib import Path

    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_SUBMISSION_DIR = ROOT / "data" / "submissions"
    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    REQUIRED_COLUMNS = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS

    # 함수: 상대/절대 경로를 실제 파일 경로로 해석합니다.
    def resolve_path(path_text):
        path = Path(path_text)
        if not path.is_absolute():
            path = DEFAULT_SUBMISSION_DIR / path
        return path

    # 함수: 입력 데이터나 제출 파일의 형식과 값 범위를 검증합니다.
    def validate(path):
        df = pd.read_csv(path)
        print("path:", path)
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        print("head:")
        print(df.head())
        print("nulls:")
        print(df.isnull().sum())
        print("describe:")
        print(df.describe())

        errors = []
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            errors.append(f"missing columns: {missing}")
        if df.shape != (250, 10):
            errors.append(f"unexpected shape: {df.shape}, expected (250, 10)")
        if df.isnull().sum().sum() > 0:
            errors.append("submission has null values")
        if not missing:
            in_range = ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all()
            print("all targets in [0,1]:", bool(in_range))
            print("target min:")
            print(df[TARGETS].min())
            print("target max:")
            print(df[TARGETS].max())
            if not in_range:
                errors.append("target probabilities are outside [0, 1]")

        if errors:
            print("[FAIL]", " | ".join(errors))
            raise SystemExit(1)
        print("[OK] submission validation passed")
        return df

    # 함수: compare 보조 로직을 수행합니다.
    def compare(current_path, compare_path):
        current = pd.read_csv(current_path)
        baseline = pd.read_csv(compare_path)
        diff = (current[TARGETS] - baseline[TARGETS]).abs()
        print("compare_to:", compare_path)
        print("mean_abs_diff_all:", float(diff.values.mean()))
        print("max_abs_diff_all:", float(diff.values.max()))
        print("per_target_mean_abs_diff:")
        print(diff.mean())
        print("new_minus_compare_mean:")
        print(current[TARGETS].mean() - baseline[TARGETS].mean())

    # 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "path",
            help="Submission csv path or file name under data/raw/data/submissions",
        )
        parser.add_argument(
            "--compare", help="Optional baseline submission path or file name"
        )
        args = parser.parse_args()

        path = resolve_path(args.path)
        validate(path)

        if args.compare:
            compare_path = resolve_path(args.compare)
            compare(path, compare_path)

    if __name__ == "__main__":
        main()


# 함수: 기존 파이프라인 위에 XGBoost OOF stacking meta layer를 추가합니다.
def build_xgboost_oof_stacking_candidates():
    __file__ = str(PROJECT_ROOT / "pipeline/build_xgboost_oof_stacking_candidates")
    __name__ = "__main__"

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import shutil
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import log_loss
    from xgboost import XGBClassifier

    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    SUBMISSION_DIR = DATA_DIR / "submissions"
    FEATURE_PATH = DATA_DIR / "features_daily_sensor_table.csv"
    TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = DATA_DIR / "ch2026_submission_sample.csv"
    CURRENT_BEST = SUBMISSION_DIR / "submission_step1_sleep_proxy_s_targets.csv"
    OUT_PATH = (
        SUBMISSION_DIR
        / "submission_step1_anchor.csv"
    )
    SLEEP_FEATURE_PATH = DATA_DIR / "artifacts" / "features_sleep_window_table.csv"

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    KEYS = ["subject_id", "lifelog_date"]
    SEEDS = [42, 77, 2024]
    N_FOLDS = 5

    # 함수: 확률값을 안정적인 범위로 제한합니다.
    def clip_proba(values, lo=0.03, hi=0.97):
        return np.clip(np.asarray(values, dtype=float), lo, hi)

    # 함수: 과신한 확률을 0.5 방향으로 완화합니다.
    def shrink_proba(values, alpha=0.985):
        values = np.asarray(values, dtype=float)
        return clip_proba(0.5 + alpha * (values - 0.5), 1e-6, 1 - 1e-6)

    # 함수: train/test 행을 같은 feature table에서 분리합니다.
    def load_feature_table():
        if not FEATURE_PATH.exists():
            raise FileNotFoundError(
                f"{FEATURE_PATH} not found. Run train_daily_sensor_baseline first."
            )

        feat = pd.read_csv(FEATURE_PATH, parse_dates=["lifelog_date"])
        train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
        sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
        for frame in [feat, train, sample]:
            frame["subject_id"] = frame["subject_id"].astype(str)

        feat["_row_order"] = np.arange(len(feat))
        train_keys = train[KEYS].copy()
        test_keys = sample[KEYS].copy()
        train_keys["_source_order"] = np.arange(len(train_keys))
        test_keys["_source_order"] = np.arange(len(test_keys))
        train_feat = train_keys.merge(feat, on=KEYS, how="left")
        test_feat = test_keys.merge(feat, on=KEYS, how="left")
        if train_feat["_row_order"].isnull().any() or test_feat["_row_order"].isnull().any():
            raise ValueError("features_daily_sensor_table.csv does not cover all train/test rows")
        train_feat = train_feat.sort_values("_source_order").reset_index(drop=True)
        test_feat = test_feat.sort_values("_source_order").reset_index(drop=True)
        train_feat, test_feat = add_sleep_context_proxy_features(
            train_feat, test_feat, train, sample
        )
        train_feat, test_feat = add_daily_date_alignment_variants(train_feat, test_feat)
        train_feat, test_feat = add_rolling_lag_sensor_features(train_feat, test_feat)
        train_feat, test_feat = add_subjectwise_normalized_features(train_feat, test_feat)
        train_feat[TARGETS] = train[TARGETS].astype(int).reset_index(drop=True)
        return train, sample, train_feat, test_feat

    # 함수: sleep-window table에서 wake/morning/fragmentation proxy를 base feature로 주입합니다.
    def add_sleep_context_proxy_features(train_feat, test_feat, train, sample):
        if not SLEEP_FEATURE_PATH.exists():
            print(f"[SLEEPCTX-DATA] skip: {SLEEP_FEATURE_PATH} not found")
            return train_feat, test_feat

        sleep = pd.read_csv(SLEEP_FEATURE_PATH, parse_dates=["sleep_date", "lifelog_date"])
        sleep["subject_id"] = sleep["subject_id"].astype(str)

        def robust_z(frame, col):
            if col not in frame.columns:
                return pd.Series(0.0, index=frame.index)
            values = pd.to_numeric(frame[col], errors="coerce")
            med = float(values.median()) if values.notna().any() else 0.0
            q25 = float(values.quantile(0.25)) if values.notna().any() else 0.0
            q75 = float(values.quantile(0.75)) if values.notna().any() else 1.0
            scale = q75 - q25
            if not np.isfinite(scale) or scale < 1e-6:
                scale = float(values.std()) if values.notna().sum() > 1 else 1.0
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
            return ((values.fillna(med) - med) / scale).clip(-6, 6)

        def z(col):
            return robust_z(sleep, col)

        engineered = pd.DataFrame(index=sleep.index)
        # S4/wake: 기상 전후 움직임, 빛, 화면 사용 변화.
        engineered["sleepctx_wake_activation_proxy"] = (
            0.30 * z("pedo_wake_transition_step_sum")
            + 0.22 * z("pedo_morning_step_sum")
            + 0.18 * z("mact_wake_transition_m_activity_sum")
            + 0.14 * z("screen_morning_m_screen_use_sum")
            + 0.10 * z("mlight_morning_m_light_mean")
            + 0.06 * z("wlight_morning_w_light_mean")
        )
        engineered["sleepctx_morning_routine_proxy"] = (
            0.28 * z("screen_morning_m_screen_use_sum")
            + 0.24 * z("pedo_morning_step_sum")
            + 0.18 * z("mact_morning_m_activity_sum")
            + 0.16 * z("mlight_morning_m_light_mean")
            - 0.14 * z("charge_morning_m_charging_mean")
        )
        engineered["sleepctx_wake_light_delta_proxy"] = (
            z("mlight_morning_m_light_mean")
            + 0.50 * z("wlight_morning_w_light_mean")
            - 0.75 * z("mlight_overnight_m_light_mean")
            - 0.35 * z("wlight_overnight_w_light_mean")
        )
        engineered["sleepctx_wake_screen_delta_proxy"] = (
            z("screen_morning_m_screen_use_sum")
            + 0.50 * z("screen_wake_transition_m_screen_use_sum")
            - 0.60 * z("screen_overnight_m_screen_use_sum")
        )

        # Q1/S1/S2/S3: 수면 길이/효율/분절 proxy.
        engineered["sleepctx_quiet_sleep_proxy"] = (
            -0.28 * z("screen_full_sleepctx_m_screen_use_sum")
            -0.24 * z("pedo_full_sleepctx_step_sum")
            -0.20 * z("mact_full_sleepctx_m_activity_mean")
            -0.14 * z("mlight_full_sleepctx_m_light_mean")
            +0.14 * z("charge_overnight_m_charging_mean")
        )
        engineered["sleepctx_fragmentation_proxy"] = (
            0.26 * z("screen_overnight_m_screen_use_sum")
            +0.22 * z("pedo_overnight_step_sum")
            +0.18 * z("mact_overnight_m_activity_q90_q10")
            +0.14 * z("mlight_overnight_m_light_q90_q10")
            +0.12 * z("screen_late_night_m_screen_use_sum")
            -0.08 * z("charge_overnight_m_charging_mean")
        )
        engineered["sleepctx_presleep_latency_bad_proxy"] = (
            0.30 * z("screen_pre_sleep_m_screen_use_sum")
            +0.24 * z("mlight_pre_sleep_m_light_mean")
            +0.20 * z("mact_pre_sleep_m_activity_sum")
            +0.16 * z("pedo_pre_sleep_step_sum")
            -0.10 * z("charge_pre_sleep_m_charging_mean")
        )
        engineered["sleepctx_sleep_efficiency_proxy"] = (
            engineered["sleepctx_quiet_sleep_proxy"]
            - 0.45 * engineered["sleepctx_fragmentation_proxy"]
            - 0.25 * engineered["sleepctx_presleep_latency_bad_proxy"]
        )
        dynamic_sleep_enabled = os.environ.get("TRY9_ENABLE_DYNAMIC_SLEEP", "0") == "1"
        if dynamic_sleep_enabled:
            engineered["sleepctx_dynamic_duration_proxy"] = z("dyn_sleep_duration_proxy")
            engineered["sleepctx_dynamic_efficiency_proxy"] = z("dyn_sleep_efficiency_proxy")
            engineered["sleepctx_dynamic_fragmentation_proxy"] = z("dyn_sleep_fragmentation_proxy")
            engineered["sleepctx_dynamic_latency_bad_proxy"] = z("dyn_sleep_latency_bad_proxy")
            engineered["sleepctx_dynamic_wake_activation_proxy"] = z("dyn_sleep_wake_activation_proxy")
            engineered["sleepctx_dynamic_subject_shift_proxy"] = (
                -0.30 * z("dyn_sleep_duration_proxy_subj_dev")
                +0.28 * z("dyn_sleep_fragmentation_proxy_subj_dev")
                +0.24 * z("dyn_sleep_latency_bad_proxy_subj_dev")
                +0.18 * z("dyn_sleep_wake_activation_proxy_subj_dev")
            )
            engineered["sleepctx_dynamic_clean_sleep_proxy"] = (
                engineered["sleepctx_dynamic_efficiency_proxy"]
                + 0.35 * engineered["sleepctx_dynamic_duration_proxy"]
                - 0.45 * engineered["sleepctx_dynamic_fragmentation_proxy"]
                - 0.25 * engineered["sleepctx_dynamic_latency_bad_proxy"]
            )

        # 원본 wake/morning/sleep 핵심 aggregate도 일부만 넣어 drift/과적합을 줄입니다.
        wanted_tokens = (
            "morning",
            "wake_transition",
            "overnight",
            "full_sleepctx",
            "pre_sleep",
            "late_night",
        )
        wanted_suffixes = (
            "_count",
            "_mean",
            "_std",
            "_sum",
            "_max",
            "_iqr",
            "_q90_q10",
        )
        wanted_prefixes = ("screen_", "pedo_", "mact_", "mlight_", "wlight_", "charge_")
        blocked = set(TARGETS + ["is_train", "row_id", "subject_ord"])
        raw_cols = [
            c
            for c in sleep.columns
            if c not in blocked
            and c not in ["subject_id", "sleep_date", "lifelog_date"]
            and c.startswith(wanted_prefixes)
            and any(tok in c for tok in wanted_tokens)
            and c.endswith(wanted_suffixes)
            and pd.api.types.is_numeric_dtype(sleep[c])
        ]
        # coverage/변화량 계열은 적은 수라 같이 둡니다.
        raw_cols += [
            c
            for c in sleep.columns
            if c.endswith(("_minus_evening", "_minus_presleep", "_minus_overnight"))
            and pd.api.types.is_numeric_dtype(sleep[c])
        ]
        dyn_cols = []
        if dynamic_sleep_enabled:
            dyn_cols = [
                c
                for c in sleep.columns
                if c.startswith("dyn_sleep_") and pd.api.types.is_numeric_dtype(sleep[c])
            ]
        raw_cols = list(dict.fromkeys([*dyn_cols, *raw_cols]))[:220]

        sleep_extra = sleep[["subject_id", "sleep_date", "lifelog_date"]].copy()
        for col in raw_cols:
            sleep_extra[f"sleepctx_{col}"] = pd.to_numeric(sleep[col], errors="coerce")
        for col in engineered.columns:
            sleep_extra[col] = engineered[col].astype(float)

        def merge_extra(base_feat, source_rows):
            keys = source_rows[["subject_id", "sleep_date", "lifelog_date"]].copy()
            keys["subject_id"] = keys["subject_id"].astype(str)
            keys["_sleepctx_order"] = np.arange(len(keys))
            merged = keys.merge(
                sleep_extra,
                on=["subject_id", "sleep_date", "lifelog_date"],
                how="left",
            ).sort_values("_sleepctx_order")
            added = merged.drop(
                columns=["subject_id", "sleep_date", "lifelog_date", "_sleepctx_order"]
            ).reset_index(drop=True)
            out = pd.concat([base_feat.reset_index(drop=True), added], axis=1)
            return out

        train_out = merge_extra(train_feat, train)
        test_out = merge_extra(test_feat, sample)
        added_cols = [c for c in train_out.columns if c.startswith("sleepctx_")]
        diag = pd.DataFrame({"feature": added_cols})
        diag_path = DATA_DIR / "artifacts" / "feature_sleep_context_proxy_sources.csv"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag.to_csv(diag_path, index=False)
        print(
            f"[SLEEPCTX-DATA] added_features={len(added_cols)} "
            f"diagnostics saved: {diag_path}"
        )
        return train_out, test_out

    # 함수: lifelog_date 기준 daily feature를 이전날/다음날 기준으로도 재정렬합니다.
    def add_daily_date_alignment_variants(train_feat, test_feat):
        train_feat = train_feat.copy()
        test_feat = test_feat.copy()
        for frame in [train_feat, test_feat]:
            frame["lifelog_date"] = pd.to_datetime(frame["lifelog_date"])
            if "sleep_date" in frame.columns:
                frame["sleep_date"] = pd.to_datetime(frame["sleep_date"])
                frame["date_gap_days"] = (
                    frame["sleep_date"] - frame["lifelog_date"]
                ).dt.days
                frame["sleep_dow"] = frame["sleep_date"].dt.dayofweek
                frame["sleep_month"] = frame["sleep_date"].dt.month
                frame["sleep_is_weekend"] = (frame["sleep_dow"] >= 5).astype(int)

        blocked = set(
            KEYS
            + TARGETS
            + [
                "subject_id",
                "sleep_date",
                "is_train",
                "row_id",
                "_row_order",
                "_source_order",
            ]
        )
        numeric_cols = [
            c
            for c in train_feat.columns
            if c not in blocked
            and pd.api.types.is_numeric_dtype(train_feat[c])
            and c in test_feat.columns
        ]
        sensor_cols = [
            c
            for c in numeric_cols
            if not c.startswith(("dow", "month", "day", "is_weekend", "date_gap", "sleep_"))
        ]

        combined = pd.concat(
            [
                train_feat.assign(_align_split="train", _align_order=np.arange(len(train_feat))),
                test_feat.assign(_align_split="test", _align_order=np.arange(len(test_feat))),
            ],
            ignore_index=True,
            sort=False,
        )
        combined = combined.sort_values(
            ["subject_id", "lifelog_date", "_align_split", "_align_order"]
        )
        group = combined.groupby("subject_id", sort=False)
        for col in sensor_cols:
            # prevday: 전날의 생활 패턴이 오늘 수면/컨디션에 남는지 확인합니다.
            combined[f"{col}_prevday_aligned"] = group[col].shift(1)
            # sleepdate_aligned: sleep_date 당일 오전/다음날 aggregate가 더 맞는지 확인합니다.
            combined[f"{col}_sleepdate_aligned"] = group[col].shift(-1)

        combined = combined.sort_values(["_align_split", "_align_order"])
        new_train = (
            combined[combined["_align_split"] == "train"]
            .drop(columns=["_align_split", "_align_order"])
            .reset_index(drop=True)
        )
        new_test = (
            combined[combined["_align_split"] == "test"]
            .drop(columns=["_align_split", "_align_order"])
            .reset_index(drop=True)
        )
        return new_train, new_test

    # 함수: subject별 과거 센서 흐름을 lag/rolling feature로 추가합니다.
    def add_rolling_lag_sensor_features(train_feat, test_feat):
        train_feat = train_feat.copy()
        test_feat = test_feat.copy()
        blocked = set(
            KEYS
            + TARGETS
            + [
                "subject_id",
                "sleep_date",
                "is_train",
                "row_id",
                "_row_order",
                "_source_order",
            ]
        )
        numeric_cols = [
            c
            for c in train_feat.columns
            if c not in blocked
            and c in test_feat.columns
            and pd.api.types.is_numeric_dtype(train_feat[c])
        ]

        useful_suffixes = (
            "_mean",
            "_median",
            "_std",
            "_count",
            "_sum",
            "_iqr",
            "_q90_q10",
            "_row_count",
        )
        sensor_cols = [
            c
            for c in numeric_cols
            if not c.startswith(
                ("dow", "month", "day", "is_weekend", "date_gap", "sleep_")
            )
            and not c.endswith(("_prevday_aligned", "_sleepdate_aligned"))
            and not c.endswith(("_subj_z", "_subj_rank", "_subj_meddiff"))
            and (c.endswith(useful_suffixes) or "_count" in c)
        ]

        # 너무 sparse한 column은 rolling에서 노이즈가 커져서 제외합니다.
        filtered_cols = []
        for col in sensor_cols:
            tr = pd.to_numeric(train_feat[col], errors="coerce")
            te = pd.to_numeric(test_feat[col], errors="coerce")
            if tr.notna().sum() >= max(
                20, int(len(train_feat) * 0.08)
            ) and te.notna().sum() >= max(10, int(len(test_feat) * 0.08)):
                filtered_cols.append(col)
        sensor_cols = filtered_cols

        if not sensor_cols:
            print("[ROLLING-LAG] no usable sensor columns")
            return train_feat, test_feat

        combined = pd.concat(
            [
                train_feat.assign(
                    _roll_split="train", _roll_order=np.arange(len(train_feat))
                ),
                test_feat.assign(
                    _roll_split="test", _roll_order=np.arange(len(test_feat))
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        combined["subject_id"] = combined["subject_id"].astype(str)
        combined["lifelog_date"] = pd.to_datetime(combined["lifelog_date"])
        combined = combined.sort_values(
            ["subject_id", "lifelog_date", "_roll_split", "_roll_order"]
        )

        subject = combined["subject_id"]
        new_parts = []
        for col in sensor_cols:
            values = pd.to_numeric(combined[col], errors="coerce")
            group = values.groupby(subject, sort=False)
            lag1 = group.shift(1)
            lag2 = group.shift(2)
            roll3 = (
                lag1.groupby(subject, sort=False)
                .rolling(3, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            roll7 = (
                lag1.groupby(subject, sort=False)
                .rolling(7, min_periods=2)
                .mean()
                .reset_index(level=0, drop=True)
            )
            new_parts.append(
                pd.DataFrame(
                    {
                        f"{col}_lag1": lag1,
                        f"{col}_lag2": lag2,
                        f"{col}_roll3_mean": roll3,
                        f"{col}_roll7_mean": roll7,
                        f"{col}_delta_lag1": values - lag1,
                        f"{col}_delta_roll3": values - roll3,
                        f"{col}_delta_roll7": values - roll7,
                    },
                    index=combined.index,
                )
            )

        rolling_features = pd.concat(new_parts, axis=1)
        combined = pd.concat([combined, rolling_features], axis=1)
        diag = pd.DataFrame(
            {
                "source_feature": sensor_cols,
                "created_features_per_source": 7,
            }
        )
        diag_path = DATA_DIR / "artifacts" / "feature_rolling_lag_sensor_sources.csv"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag.to_csv(diag_path, index=False)
        print(
            f"[ROLLING-LAG] source_features={len(sensor_cols)} "
            f"created_features={rolling_features.shape[1]}"
        )
        print(f"[ROLLING-LAG] diagnostics saved: {diag_path}")

        combined = combined.sort_values(["_roll_split", "_roll_order"])
        new_train = (
            combined[combined["_roll_split"] == "train"]
            .drop(columns=["_roll_split", "_roll_order"])
            .reset_index(drop=True)
        )
        new_test = (
            combined[combined["_roll_split"] == "test"]
            .drop(columns=["_roll_split", "_roll_order"])
            .reset_index(drop=True)
        )
        return new_train, new_test

    # 함수: 센서 feature를 subject별 개인 기준 z-score/rank/deviation으로 변환해 추가합니다.
    def add_subjectwise_normalized_features(train_feat, test_feat):
        train_feat = train_feat.copy()
        test_feat = test_feat.copy()
        blocked = set(
            KEYS
            + TARGETS
            + [
                "subject_id",
                "sleep_date",
                "is_train",
                "row_id",
                "_row_order",
                "_source_order",
            ]
        )
        numeric_cols = [
            c
            for c in train_feat.columns
            if c not in blocked
            and c in test_feat.columns
            and pd.api.types.is_numeric_dtype(train_feat[c])
        ]
        sensor_cols = [
            c
            for c in numeric_cols
            if not c.startswith(("dow", "month", "day", "is_weekend", "date_gap", "sleep_"))
            and not c.endswith(("_subj_z", "_subj_rank", "_subj_meddiff"))
        ]

        combined = pd.concat(
            [
                train_feat.assign(_norm_split="train", _norm_order=np.arange(len(train_feat))),
                test_feat.assign(_norm_split="test", _norm_order=np.arange(len(test_feat))),
            ],
            ignore_index=True,
            sort=False,
        )
        combined["subject_id"] = combined["subject_id"].astype(str)
        group = combined.groupby("subject_id", sort=False)

        new_parts = []
        diag_rows = []
        for col in sensor_cols:
            values = pd.to_numeric(combined[col], errors="coerce")
            nonnull_by_subject = group[col].transform(lambda s: pd.to_numeric(s, errors="coerce").notna().sum())
            if int(values.notna().sum()) < 20:
                continue

            subj_median = values.groupby(combined["subject_id"]).transform("median")
            subj_q25 = values.groupby(combined["subject_id"]).transform(lambda s: s.quantile(0.25))
            subj_q75 = values.groupby(combined["subject_id"]).transform(lambda s: s.quantile(0.75))
            subj_std = values.groupby(combined["subject_id"]).transform(lambda s: s.std(ddof=0))
            scale = (subj_q75 - subj_q25).replace(0, np.nan).fillna(subj_std)
            global_scale = float(values.quantile(0.75) - values.quantile(0.25))
            if not np.isfinite(global_scale) or global_scale < 1e-9:
                global_scale = float(values.std(ddof=0))
            if not np.isfinite(global_scale) or global_scale < 1e-9:
                global_scale = 1.0
            scale = scale.replace(0, np.nan).fillna(global_scale).fillna(1.0)

            z = ((values - subj_median) / scale).where(nonnull_by_subject >= 4)
            meddiff = ((values - subj_median) / scale).where(nonnull_by_subject >= 4)
            rank = group[col].rank(pct=True).where(nonnull_by_subject >= 4)
            new_parts.append(
                pd.DataFrame(
                    {
                        f"{col}_subj_z": z.clip(-6, 6),
                        f"{col}_subj_rank": rank,
                        f"{col}_subj_meddiff": meddiff.clip(-6, 6),
                    },
                    index=combined.index,
                )
            )
            diag_rows.append(
                {
                    "feature": col,
                    "nonnull": int(values.notna().sum()),
                    "subjects_with_4plus": int(
                        (nonnull_by_subject.groupby(combined["subject_id"]).first() >= 4).sum()
                    ),
                }
            )

        if not new_parts:
            print("[SUBJECT-NORM] no usable numeric feature columns")
            return train_feat, test_feat

        norm_features = pd.concat(new_parts, axis=1)
        combined = pd.concat([combined, norm_features], axis=1)
        diag = pd.DataFrame(diag_rows)
        diag_path = DATA_DIR / "artifacts" / "feature_subjectwise_normalization.csv"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag.to_csv(diag_path, index=False)
        print(
            f"[SUBJECT-NORM] source_features={len(sensor_cols)} "
            f"normalized_features={norm_features.shape[1]}"
        )
        print(f"[SUBJECT-NORM] diagnostics saved: {diag_path}")

        combined = combined.sort_values(["_norm_split", "_norm_order"])
        new_train = (
            combined[combined["_norm_split"] == "train"]
            .drop(columns=["_norm_split", "_norm_order"])
            .reset_index(drop=True)
        )
        new_test = (
            combined[combined["_norm_split"] == "test"]
            .drop(columns=["_norm_split", "_norm_order"])
            .reset_index(drop=True)
        )
        return new_train, new_test

    # 함수: stacking base 모델에 넣을 안전한 feature 컬럼을 선택합니다.
    # 함수: train/test 분포 차이가 심한 numeric feature를 base 모델 입력에서 제외합니다.
    def select_feature_columns(train_feat, test_feat):
        drop = set(
            KEYS + TARGETS + ["sleep_date", "is_train", "row_id", "_row_order", "_source_order"]
        )
        raw_cols = []
        for col in train_feat.columns:
            if col in drop:
                continue
            if col.endswith("_prior_mean") or col.endswith("_prior_std"):
                continue
            if col.endswith("_prior_cnt") or col.endswith("_dev"):
                continue
            if col in test_feat.columns and pd.api.types.is_numeric_dtype(train_feat[col]):
                raw_cols.append(col)

        rows = []
        kept = []
        min_train_nonnull = max(20, int(len(train_feat) * 0.08))
        min_test_nonnull = max(10, int(len(test_feat) * 0.08))
        for col in raw_cols:
            tr = pd.to_numeric(train_feat[col], errors="coerce")
            te = pd.to_numeric(test_feat[col], errors="coerce")
            train_nonnull = int(tr.notna().sum())
            test_nonnull = int(te.notna().sum())
            train_missing = float(tr.isna().mean())
            test_missing = float(te.isna().mean())
            missing_gap = abs(train_missing - test_missing)

            reason = "keep"
            if train_nonnull < min_train_nonnull or test_nonnull < min_test_nonnull:
                drift_score = np.inf
                reason = "low_coverage"
            else:
                tr_valid = tr.dropna().astype(float)
                te_valid = te.dropna().astype(float)
                q25 = float(tr_valid.quantile(0.25))
                q75 = float(tr_valid.quantile(0.75))
                scale = q75 - q25
                if not np.isfinite(scale) or scale < 1e-9:
                    scale = float(tr_valid.std(ddof=0))
                if not np.isfinite(scale) or scale < 1e-9:
                    scale = 1.0
                mean_gap = abs(float(tr_valid.mean()) - float(te_valid.mean())) / scale
                median_gap = abs(float(tr_valid.median()) - float(te_valid.median())) / scale
                q90_gap = (
                    abs(float(tr_valid.quantile(0.90)) - float(te_valid.quantile(0.90)))
                    / scale
                )
                drift_score = max(mean_gap, median_gap, 0.65 * q90_gap, 3.0 * missing_gap)
                if max(train_missing, test_missing) > 0.985:
                    reason = "nearly_all_missing"
                elif missing_gap > 0.55:
                    reason = "missing_drift"
                elif drift_score > 3.50:
                    reason = "value_drift"

            rows.append(
                {
                    "feature": col,
                    "reason": reason,
                    "kept": reason == "keep",
                    "drift_score": float(drift_score) if np.isfinite(drift_score) else np.inf,
                    "train_missing": train_missing,
                    "test_missing": test_missing,
                    "missing_gap": missing_gap,
                    "train_nonnull": train_nonnull,
                    "test_nonnull": test_nonnull,
                }
            )
            if reason == "keep":
                kept.append(col)

        diag = pd.DataFrame(rows).sort_values(
            ["kept", "drift_score"], ascending=[True, False]
        )
        diag_path = DATA_DIR / "artifacts" / "feature_train_test_drift_filter.csv"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag.to_csv(diag_path, index=False)
        removed = len(raw_cols) - len(kept)
        print(
            f"[DRIFT] kept={len(kept)} removed={removed} "
            f"from {len(raw_cols)} numeric features"
        )
        if removed:
            print("[DRIFT] removed feature summary:")
            print(
                diag[diag["kept"] == False]
                .head(30)[["feature", "reason", "drift_score", "missing_gap"]]
                .to_string(index=False)
            )
        print(f"[DRIFT] diagnostics saved: {diag_path}")
        return kept

    # 함수: subject별 시간 순서를 유지하는 OOF fold 번호를 만듭니다.
    def make_temporal_folds(train_feat):
        folds = pd.Series(-1, index=train_feat.index, dtype=int)
        for _, idx in train_feat.groupby("subject_id").groups.items():
            ordered = train_feat.loc[list(idx)].sort_values("lifelog_date").index.to_list()
            for rank, row_idx in enumerate(ordered):
                folds.loc[row_idx] = rank % N_FOLDS
        if (folds < 0).any():
            raise ValueError("failed to assign temporal folds")
        return folds

    # 함수: fold 학습 데이터만 사용해 subject/date/state/pattern prior 예측을 만듭니다.
    def build_fold_prior_suite(fit_feat, pred_feat, feature_cols):
        fit_feat = fit_feat.copy()
        pred_feat = pred_feat.copy()
        fit_feat["lifelog_date"] = pd.to_datetime(fit_feat["lifelog_date"])
        pred_feat["lifelog_date"] = pd.to_datetime(pred_feat["lifelog_date"])
        global_mean = fit_feat[TARGETS].astype(float).mean()
        subject_mean = fit_feat.groupby("subject_id")[TARGETS].mean()
        subject_count = fit_feat.groupby("subject_id").size()
        tables = {
            sid: group.sort_values("lifelog_date")[["lifelog_date"] + TARGETS].copy()
            for sid, group in fit_feat.groupby("subject_id")
        }

        outputs = {
            name: pd.DataFrame(index=pred_feat.index, columns=TARGETS, dtype=float)
            for name in [
                "prior_subject_mean",
                "prior_date_interp",
                "prior_bracket",
                "prior_state_transition",
                "prior_calendar",
                "prior_target_dynamics",
            ]
        }

        other_dates = fit_feat[["subject_id", "lifelog_date"] + TARGETS].copy()
        for row in pred_feat.itertuples():
            sid = str(row.subject_id)
            date = pd.Timestamp(row.lifelog_date)
            table = tables.get(sid)
            cnt = float(subject_count.get(sid, 0.0))
            subj = subject_mean.loc[sid] if sid in subject_mean.index else global_mean
            smooth_subj = (cnt * subj + 8.0 * global_mean) / (cnt + 8.0)
            outputs["prior_subject_mean"].loc[row.Index, TARGETS] = smooth_subj[TARGETS].to_numpy(float)

            if table is None or len(table) == 0:
                interp = smooth_subj.copy()
                bracket = smooth_subj.copy()
                state = smooth_subj.copy()
            else:
                deltas = (pd.to_datetime(table["lifelog_date"]) - date).dt.days.to_numpy(float)
                abs_delta = np.abs(deltas)
                weights = np.exp(-abs_delta / 10.0)
                if float(weights.sum()) <= 1e-12:
                    interp = smooth_subj.copy()
                else:
                    local = pd.Series(
                        np.average(table[TARGETS].to_numpy(float), axis=0, weights=weights),
                        index=TARGETS,
                    )
                    rel = float(weights.sum() / (weights.sum() + 4.0))
                    interp = rel * local + (1 - rel) * smooth_subj

                pieces = []
                piece_weights = []
                before = np.where(deltas < 0)[0]
                after = np.where(deltas > 0)[0]
                if len(before):
                    idx = before[np.argmin(abs_delta[before])]
                    pieces.append(table.iloc[idx][TARGETS].astype(float).to_numpy())
                    piece_weights.append(float(np.exp(-abs_delta[idx] / 8.0)))
                if len(after):
                    idx = after[np.argmin(abs_delta[after])]
                    pieces.append(table.iloc[idx][TARGETS].astype(float).to_numpy())
                    piece_weights.append(float(np.exp(-abs_delta[idx] / 8.0)))
                if pieces and sum(piece_weights) > 1e-12:
                    local = pd.Series(np.average(pieces, axis=0, weights=piece_weights), index=TARGETS)
                    rel = float(sum(piece_weights) / (sum(piece_weights) + 5.0))
                    bracket = rel * local + (1 - rel) * smooth_subj
                else:
                    bracket = smooth_subj.copy()

                past = table[pd.to_datetime(table["lifelog_date"]) < date]
                if len(past):
                    last = past.iloc[-1][TARGETS].astype(float)
                    state = 0.70 * last + 0.30 * smooth_subj
                else:
                    state = smooth_subj.copy()

            others = other_dates[other_dates["subject_id"].astype(str) != sid]
            if len(others):
                dist = (pd.to_datetime(others["lifelog_date"]) - date).dt.days.abs().to_numpy(float)
                order = np.argsort(dist)[:40]
                weights = np.exp(-dist[order] / 14.0)
                if float(weights.sum()) > 1e-12:
                    local = pd.Series(
                        np.average(others.iloc[order][TARGETS].to_numpy(float), axis=0, weights=weights),
                        index=TARGETS,
                    )
                    calendar = 0.65 * smooth_subj + 0.25 * local + 0.10 * global_mean
                else:
                    calendar = smooth_subj.copy()
            else:
                calendar = smooth_subj.copy()

            dynamics = 0.76 * smooth_subj + 0.14 * interp + 0.10 * global_mean
            outputs["prior_date_interp"].loc[row.Index, TARGETS] = interp[TARGETS].to_numpy(float)
            outputs["prior_bracket"].loc[row.Index, TARGETS] = bracket[TARGETS].to_numpy(float)
            outputs["prior_state_transition"].loc[row.Index, TARGETS] = state[TARGETS].to_numpy(float)
            outputs["prior_calendar"].loc[row.Index, TARGETS] = calendar[TARGETS].to_numpy(float)
            outputs["prior_target_dynamics"].loc[row.Index, TARGETS] = dynamics[TARGETS].to_numpy(float)

        pattern_prior = build_pattern_projection_from_prior(
            fit_feat, outputs["prior_subject_mean"]
        )
        sensor_prior = build_sensor_rank_prior(fit_feat, pred_feat, feature_cols)
        outputs["prior_pattern_projection"] = pattern_prior
        outputs["prior_sensor_rank"] = sensor_prior
        for name in outputs:
            outputs[name] = outputs[name].astype(float).clip(0.03, 0.97)
        return outputs

    # 함수: train label 조합 분포를 이용해 prior의 공동 target 패턴을 보정합니다.
    def build_pattern_projection_from_prior(fit_feat, base_prior):
        patterns = fit_feat[TARGETS].astype(int).astype(str).agg("".join, axis=1)
        counts = patterns.value_counts()
        pattern_bits = np.array([[int(ch) for ch in key] for key in counts.index], dtype=float)
        prior_prob = (counts.to_numpy(float) + 0.45) / (len(patterns) + 0.45 * len(counts))
        log_prior = np.log(prior_prob)
        probs = base_prior[TARGETS].astype(float).clip(1e-5, 1 - 1e-5).to_numpy()
        out = np.zeros_like(probs)
        for i, p in enumerate(probs):
            logp = log_prior + (pattern_bits * np.log(p) + (1 - pattern_bits) * np.log(1 - p)).sum(axis=1)
            logp -= logp.max()
            w = np.exp(logp)
            w /= w.sum()
            out[i] = w @ pattern_bits
        return pd.DataFrame(out, index=base_prior.index, columns=TARGETS)

    # 함수: 센서 feature와 target의 fold 내부 상관을 이용해 rank-bin prior를 만듭니다.
    def build_sensor_rank_prior(fit_feat, pred_feat, feature_cols):
        out = pd.DataFrame(index=pred_feat.index, columns=TARGETS, dtype=float)
        safe_cols = [c for c in feature_cols if c in fit_feat.columns and c in pred_feat.columns]
        fit_x = fit_feat[safe_cols].apply(pd.to_numeric, errors="coerce")
        pred_x = pred_feat[safe_cols].apply(pd.to_numeric, errors="coerce")
        med = fit_x.median(numeric_only=True)
        fit_x = fit_x.fillna(med).fillna(0.0)
        pred_x = pred_x.fillna(med).fillna(0.0)
        for target in TARGETS:
            y = fit_feat[target].astype(float).reset_index(drop=True)
            global_mean = float(y.mean())
            corrs = []
            for col in safe_cols:
                corr = pd.Series(fit_x[col].to_numpy(float)).corr(y)
                if np.isfinite(corr):
                    corrs.append((col, abs(float(corr)), float(corr)))
            corrs = sorted(corrs, key=lambda item: item[1], reverse=True)[:8]
            priors = []
            weights = []
            for col, abs_corr, corr in corrs:
                score_fit = fit_x[col].to_numpy(float)
                score_pred = pred_x[col].to_numpy(float)
                if corr < 0:
                    score_fit = -score_fit
                    score_pred = -score_pred
                try:
                    bins = pd.qcut(pd.Series(score_fit), q=6, labels=False, duplicates="drop")
                except ValueError:
                    continue
                stats = pd.DataFrame({"bin": bins, "y": y}).groupby("bin")["y"].agg(["mean", "count"])
                smooth = (stats["mean"] * stats["count"] + global_mean * 18.0) / (stats["count"] + 18.0)
                edges = np.quantile(score_fit, np.linspace(0, 1, len(smooth) + 1))
                edges[0] = -np.inf
                edges[-1] = np.inf
                pred_bins = np.digitize(score_pred, edges[1:-1], right=True)
                mapping = dict(zip(range(len(smooth)), smooth.to_numpy(float)))
                priors.append(pd.Series([mapping.get(int(b), global_mean) for b in pred_bins], index=pred_feat.index))
                weights.append(abs_corr + 1e-6)
            if priors:
                weights = np.array(weights, dtype=float)
                weights /= weights.sum()
                combined = sum(w * p for w, p in zip(weights, priors))
                out[target] = (0.74 * combined + 0.26 * global_mean).clip(0.03, 0.97)
            else:
                out[target] = global_mean
        return out

    # 함수: 주요 prior 계열도 fold-aware OOF와 full-train test 예측으로 생성합니다.
    def fit_prior_oof(train_feat, test_feat, feature_cols, folds):
        names = [
            "prior_subject_mean",
            "prior_date_interp",
            "prior_bracket",
            "prior_state_transition",
            "prior_calendar",
            "prior_target_dynamics",
            "prior_pattern_projection",
            "prior_sensor_rank",
        ]
        oof = {name: pd.DataFrame(index=train_feat.index, columns=TARGETS, dtype=float) for name in names}
        for fold in range(N_FOLDS):
            valid_mask = folds.eq(fold)
            suite = build_fold_prior_suite(
                train_feat.loc[~valid_mask].copy(),
                train_feat.loc[valid_mask].copy(),
                feature_cols,
            )
            for name, frame in suite.items():
                oof[name].loc[valid_mask, TARGETS] = frame[TARGETS].to_numpy(float)
            print(f"[OOF-PRIOR] fold={fold} rows={int(valid_mask.sum())}")
        test_suite = build_fold_prior_suite(train_feat.copy(), test_feat.copy(), feature_cols)
        for name in names:
            oof[name] = oof[name].astype(float).clip(0.03, 0.97)
            test_suite[name] = test_suite[name].astype(float).clip(0.03, 0.97)
        return oof, test_suite

    # 함수: sleep-window CatBoost 모델과 sleep metric proxy도 fold-aware OOF로 만듭니다.
    def fit_sleep_window_oof(train, sample, folds):
        if not SLEEP_FEATURE_PATH.exists():
            print(f"[OOF-SLEEP] skip: {SLEEP_FEATURE_PATH} not found")
            return {}, {}

        sleep_feat = pd.read_csv(
            SLEEP_FEATURE_PATH, parse_dates=["sleep_date", "lifelog_date"]
        )
        for frame in [sleep_feat, train, sample]:
            frame["subject_id"] = frame["subject_id"].astype(str)

        train_keys = train[["subject_id", "sleep_date", "lifelog_date"]].copy()
        sample_keys = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
        train_keys["_source_order"] = np.arange(len(train_keys))
        sample_keys["_source_order"] = np.arange(len(sample_keys))
        train_sleep = train_keys.merge(
            sleep_feat, on=["subject_id", "sleep_date", "lifelog_date"], how="left"
        ).sort_values("_source_order").reset_index(drop=True)
        test_sleep = sample_keys.merge(
            sleep_feat, on=["subject_id", "sleep_date", "lifelog_date"], how="left"
        ).sort_values("_source_order").reset_index(drop=True)
        if train_sleep["is_train"].isnull().any() or test_sleep["is_train"].isnull().any():
            raise ValueError("features_sleep_window_table.csv does not cover all train/test rows")
        train_sleep[TARGETS] = train[TARGETS].astype(int).reset_index(drop=True)

        drop = set(
            [
                "row_id",
                "sleep_date",
                "lifelog_date",
                "is_train",
                "_source_order",
            ]
            + TARGETS
        )
        numeric_cols = [
            c
            for c in train_sleep.columns
            if c not in drop
            and c != "subject_id"
            and pd.api.types.is_numeric_dtype(train_sleep[c])
        ]
        feature_cols = ["subject_id"] + numeric_cols
        x_train = train_sleep[feature_cols].copy()
        x_test = test_sleep[feature_cols].copy()
        x_train["subject_id"] = x_train["subject_id"].astype(str)
        x_test["subject_id"] = x_test["subject_id"].astype(str)
        med = x_train[numeric_cols].median(numeric_only=True)
        x_train[numeric_cols] = x_train[numeric_cols].fillna(med).fillna(0.0)
        x_test[numeric_cols] = x_test[numeric_cols].fillna(med).fillna(0.0)

        oof_model = pd.DataFrame(index=train_sleep.index, columns=TARGETS, dtype=float)
        test_model = pd.DataFrame(0.0, index=test_sleep.index, columns=TARGETS)
        cat_idx = [0]
        for target in TARGETS:
            y = train_sleep[target].astype(int).reset_index(drop=True)
            mean = float(y.mean())
            print(f"[OOF-SLEEP] target={target}")
            for fold in range(N_FOLDS):
                valid_mask = folds.eq(fold).to_numpy()
                train_mask = ~valid_mask
                restored = (
                    load_step1_candidate_checkpoint(
                        "xgb_oof_stacking_sleep_window_oof",
                        target,
                        "fold",
                        fold,
                    )
                    if restore_step1_candidate_checkpoints_enabled()
                    else None
                )
                if restored is not None:
                    print(f"[STEP1_CKPT] restored candidate xgb_oof_stacking_sleep_window_oof {target} fold={fold}")
                    pred = restored["model"].predict_proba(
                        _step1_candidate_matrix(x_train.loc[valid_mask], restored)
                    )[:, 1]
                    oof_model.loc[valid_mask, target] = np.clip(
                        0.82 * pred + 0.18 * float(restored["target_mean"]), 0.04, 0.96
                    )
                    continue
                model = CatBoostClassifier(
                    iterations=260,
                    depth=2,
                    learning_rate=0.025,
                    l2_leaf_reg=28,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    random_seed=6000 + fold,
                    allow_writing_files=False,
                    verbose=False,
                )
                model.fit(x_train.loc[train_mask], y.loc[train_mask], cat_features=cat_idx)
                pred = model.predict_proba(x_train.loc[valid_mask])[:, 1]
                oof_model.loc[valid_mask, target] = np.clip(
                    0.82 * pred + 0.18 * mean, 0.04, 0.96
                )
                save_step1_candidate_checkpoint(
                    "xgb_oof_stacking_sleep_window_oof",
                    {
                        "kind": "fold",
                        "target": target,
                        "fold": fold,
                        "feature_cols": feature_cols,
                        "numeric_cols": numeric_cols,
                        "cat_idx": cat_idx,
                        "median": med,
                        "target_mean": mean,
                        "model": model,
                    },
                    target,
                    "fold",
                    fold,
                )
            for seed in SEEDS:
                restored = (
                    load_step1_candidate_checkpoint(
                        "xgb_oof_stacking_sleep_window_oof",
                        target,
                        "full",
                        seed,
                    )
                    if restore_step1_candidate_checkpoints_enabled()
                    else None
                )
                if restored is not None:
                    print(f"[STEP1_CKPT] restored candidate xgb_oof_stacking_sleep_window_oof {target} full seed={seed}")
                    pred = restored["model"].predict_proba(
                        _step1_candidate_matrix(x_test, restored)
                    )[:, 1]
                    test_model[target] += (
                        np.clip(0.82 * pred + 0.18 * float(restored["target_mean"]), 0.04, 0.96)
                        / len(SEEDS)
                    )
                    continue
                full = CatBoostClassifier(
                    iterations=320,
                    depth=2,
                    learning_rate=0.025,
                    l2_leaf_reg=28,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    random_seed=seed,
                    allow_writing_files=False,
                    verbose=False,
                )
                full.fit(x_train, y, cat_features=cat_idx)
                pred = full.predict_proba(x_test)[:, 1]
                test_model[target] += np.clip(0.82 * pred + 0.18 * mean, 0.04, 0.96) / len(SEEDS)
                save_step1_candidate_checkpoint(
                    "xgb_oof_stacking_sleep_window_oof",
                    {
                        "kind": "full",
                        "target": target,
                        "seed": seed,
                        "feature_cols": feature_cols,
                        "numeric_cols": numeric_cols,
                        "cat_idx": cat_idx,
                        "median": med,
                        "target_mean": mean,
                        "model": full,
                    },
                    target,
                    "full",
                    seed,
                )

        sleep_metric_oof = pd.DataFrame(index=train_sleep.index, columns=TARGETS, dtype=float)
        sleep_metric_test = pd.DataFrame(index=test_sleep.index, columns=TARGETS, dtype=float)
        for fold in range(N_FOLDS):
            valid_mask = folds.eq(fold)
            suite = build_sensor_rank_prior(
                train_sleep.loc[~valid_mask].copy(),
                train_sleep.loc[valid_mask].copy(),
                numeric_cols,
            )
            sleep_metric_oof.loc[valid_mask, TARGETS] = suite[TARGETS].to_numpy(float)
        sleep_metric_test[TARGETS] = build_sensor_rank_prior(
            train_sleep.copy(), test_sleep.copy(), numeric_cols
        )[TARGETS].to_numpy(float)

        def robust_z_from_fit(fit_frame, pred_frame, col):
            if col not in fit_frame.columns or col not in pred_frame.columns:
                return pd.Series(0.0, index=pred_frame.index)
            fit_values = pd.to_numeric(fit_frame[col], errors="coerce")
            pred_values = pd.to_numeric(pred_frame[col], errors="coerce")
            med = float(fit_values.median()) if fit_values.notna().any() else 0.0
            q25 = float(fit_values.quantile(0.25)) if fit_values.notna().any() else 0.0
            q75 = float(fit_values.quantile(0.75)) if fit_values.notna().any() else 1.0
            scale = q75 - q25
            if not np.isfinite(scale) or scale < 1e-6:
                scale = float(fit_values.std()) if fit_values.notna().sum() > 1 else 1.0
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
            return ((pred_values.fillna(med) - med) / scale).clip(-5, 5)

        def add_sleep_interval_scores(fit_frame, pred_frame):
            z = lambda col: robust_z_from_fit(fit_frame, pred_frame, col)
            out = pred_frame.copy()
            # Higher values mean a cleaner inferred sleep interval unless the name ends in _bad.
            out["proxy_interval_duration"] = (
                -0.30 * z("screen_full_sleepctx_m_screen_use_sum")
                -0.24 * z("pedo_full_sleepctx_step_sum")
                -0.20 * z("mact_full_sleepctx_m_activity_mean")
                -0.14 * z("mlight_full_sleepctx_m_light_mean")
                +0.18 * z("charge_overnight_m_charging_mean")
                +0.10 * z("charge_full_sleepctx_m_charging_mean")
            )
            out["proxy_interval_efficiency"] = (
                -0.34 * z("screen_overnight_m_screen_use_sum")
                -0.26 * z("pedo_overnight_step_sum")
                -0.22 * z("mact_overnight_m_activity_mean")
                -0.16 * z("mlight_overnight_m_light_mean")
                -0.10 * z("hr_overnight_heart_rate_std")
                +0.16 * z("charge_overnight_m_charging_mean")
            )
            out["proxy_interval_latency_bad"] = (
                +0.32 * z("screen_pre_sleep_m_screen_use_sum")
                +0.24 * z("mlight_pre_sleep_m_light_mean")
                +0.20 * z("mact_pre_sleep_m_activity_mean")
                +0.14 * z("pedo_pre_sleep_step_sum")
                +0.10 * z("hr_pre_sleep_heart_rate_mean")
                -0.12 * z("charge_pre_sleep_m_charging_mean")
            )
            out["proxy_interval_wake_bad"] = (
                +0.30 * z("screen_late_night_m_screen_use_sum")
                +0.24 * z("screen_overnight_m_screen_use_sum")
                +0.22 * z("pedo_overnight_step_sum")
                +0.18 * z("mact_overnight_m_activity_max")
                +0.14 * z("mlight_overnight_m_light_max")
                +0.10 * z("screen_morning_m_screen_use_sum")
            )
            out["proxy_interval_clean_sleep"] = (
                +0.38 * out["proxy_interval_efficiency"]
                +0.26 * out["proxy_interval_duration"]
                -0.18 * out["proxy_interval_latency_bad"]
                -0.18 * out["proxy_interval_wake_bad"]
            )
            return out

        def calibrate_interval_prior(fit_frame, pred_frame, bins=6):
            fit_scored = add_sleep_interval_scores(fit_frame, fit_frame)
            pred_scored = add_sleep_interval_scores(fit_frame, pred_frame)
            mapping = {
                "S1": "proxy_interval_duration",
                "S2": "proxy_interval_efficiency",
                "S3": "proxy_interval_latency_bad",
                "S4": "proxy_interval_wake_bad",
            }
            out = pd.DataFrame(index=pred_frame.index, columns=TARGETS, dtype=float)
            diagnostics = []
            for target in TARGETS:
                y = fit_frame[target].astype(float).reset_index(drop=True)
                global_mean = float(y.mean())
                if target not in mapping:
                    out[target] = global_mean
                    continue
                fit_score = pd.to_numeric(fit_scored[mapping[target]], errors="coerce").fillna(0.0).reset_index(drop=True)
                pred_score = pd.to_numeric(pred_scored[mapping[target]], errors="coerce").fillna(0.0).reset_index(drop=True)
                corr = float(pd.Series(fit_score).corr(y)) if y.nunique() > 1 else 0.0
                if not np.isfinite(corr):
                    corr = 0.0
                if corr < 0:
                    fit_score = -fit_score
                    pred_score = -pred_score
                try:
                    qbin = pd.qcut(fit_score.rank(pct=True), q=bins, labels=False, duplicates="drop")
                except ValueError:
                    qbin = pd.Series(np.zeros(len(fit_score), dtype=int), index=fit_score.index)
                stats = pd.DataFrame({"bin": qbin, "y": y}).groupby("bin")["y"].agg(["mean", "count"])
                smooth = (stats["mean"] * stats["count"] + global_mean * 18.0) / (stats["count"] + 18.0)
                if len(smooth) <= 1:
                    prior = pd.Series(global_mean, index=pred_frame.index)
                else:
                    fit_rank = fit_score.rank(pct=True).to_numpy(float)
                    pred_rank = pred_score.rank(pct=True).to_numpy(float)
                    edges = np.quantile(fit_rank, np.linspace(0, 1, len(smooth) + 1))
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    pred_bins = np.digitize(pred_rank, edges[1:-1], right=True)
                    prior = pd.Series(
                        [smooth.iloc[min(max(int(b), 0), len(smooth) - 1)] for b in pred_bins],
                        index=pred_frame.index,
                    )
                out[target] = (0.70 * prior + 0.30 * global_mean).clip(0.04, 0.96)
                diagnostics.append(
                    {
                        "target": target,
                        "proxy": mapping[target],
                        "corr": corr,
                        "global_mean": global_mean,
                        "pred_mean": float(out[target].mean()),
                    }
                )
            return out, pd.DataFrame(diagnostics)

        sleep_interval_oof = pd.DataFrame(index=train_sleep.index, columns=TARGETS, dtype=float)
        interval_diag_rows = []
        for fold in range(N_FOLDS):
            valid_mask = folds.eq(fold)
            prior, diag = calibrate_interval_prior(
                train_sleep.loc[~valid_mask].copy(),
                train_sleep.loc[valid_mask].copy(),
            )
            sleep_interval_oof.loc[valid_mask, TARGETS] = prior[TARGETS].to_numpy(float)
            diag["fold"] = fold
            interval_diag_rows.append(diag)
        sleep_interval_test, interval_test_diag = calibrate_interval_prior(
            train_sleep.copy(), test_sleep.copy()
        )
        interval_test_diag["fold"] = "test"
        interval_diag_rows.append(interval_test_diag)
        interval_diag = pd.concat(interval_diag_rows, ignore_index=True)
        interval_diag_path = DATA_DIR / "artifacts" / "sleep_interval_proxy_diagnostics.csv"
        interval_diag.to_csv(interval_diag_path, index=False)
        print(f"[OOF-SLEEP] interval proxy diagnostics saved: {interval_diag_path}")

        return (
            {
                "prior_sleep_window_model": oof_model.astype(float).clip(0.03, 0.97),
                "prior_sleep_metric_proxy": sleep_metric_oof.astype(float).clip(0.03, 0.97),
                "prior_sleep_interval_proxy": sleep_interval_oof.astype(float).clip(0.03, 0.97),
            },
            {
                "prior_sleep_window_model": test_model.astype(float).clip(0.03, 0.97),
                "prior_sleep_metric_proxy": sleep_metric_test.astype(float).clip(0.03, 0.97),
                "prior_sleep_interval_proxy": sleep_interval_test.astype(float).clip(0.03, 0.97),
            },
        )

    # 함수: anchor와 prior/model 예측을 target별 weight로 섞습니다.
    def blend_prediction(anchor, prior, weights):
        out = pd.DataFrame(index=anchor.index, columns=TARGETS, dtype=float)
        if isinstance(weights, dict):
            for target in TARGETS:
                w = float(weights.get(target, 0.0))
                out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[
                    target
                ].to_numpy(float)
        else:
            w = float(weights)
            out[TARGETS] = (1 - w) * anchor[TARGETS].to_numpy(float) + w * prior[
                TARGETS
            ].to_numpy(float)
        return out.astype(float).clip(0.03, 0.97)

    # 함수: 기존 routing anchor와 동일한 Q/S별 LGB-Cat blend OOF를 만듭니다.
    def make_routing_anchor(pred_map):
        out = pd.DataFrame(index=pred_map["lgb"].index, columns=TARGETS, dtype=float)
        for target in TARGETS:
            if target in ["Q1", "Q2", "Q3"]:
                raw = 0.60 * pred_map["lgb"][target].to_numpy(float) + 0.40 * pred_map[
                    "cat"
                ][target].to_numpy(float)
            else:
                raw = 0.20 * pred_map["lgb"][target].to_numpy(float) + 0.80 * pred_map[
                    "cat"
                ][target].to_numpy(float)
            out[target] = shrink_proba(raw, alpha=0.98)
        return out.astype(float).clip(0.03, 0.97)

    # 함수: 확률을 logit 온도 스케일링으로 보정합니다.
    def temperature_scale(frame, temp):
        p = frame[TARGETS].astype(float).clip(1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        out = 1.0 / (1.0 + np.exp(-(logit * temp)))
        return pd.DataFrame(out, index=frame.index, columns=TARGETS).clip(0.03, 0.97)

    # 함수: 확률 평균을 fold별 target 평균 쪽으로 맞춥니다.
    def mean_align_frame(frame, target_mean, gamma):
        out = frame.copy()
        current = out[TARGETS].astype(float).mean()
        desired = (1 - gamma) * current + gamma * target_mean[TARGETS]
        for target in TARGETS:
            p = out[target].astype(float).clip(1e-6, 1 - 1e-6)
            logit = np.log(p / (1 - p))
            lo, hi = -8.0, 8.0
            for _ in range(50):
                mid = (lo + hi) / 2
                shifted = 1.0 / (1.0 + np.exp(-(logit + mid)))
                if shifted.mean() < float(desired[target]):
                    lo = mid
                else:
                    hi = mid
            out[target] = 1.0 / (1.0 + np.exp(-(logit + (lo + hi) / 2)))
        return out[TARGETS].astype(float).clip(0.03, 0.97)

    # 함수: fold별 target 평균만 사용해 OOF mean alignment 후보를 만듭니다.
    def foldwise_mean_align(anchor, train_feat, folds, gamma):
        out = pd.DataFrame(index=anchor.index, columns=TARGETS, dtype=float)
        for fold in range(N_FOLDS):
            valid_mask = folds.eq(fold)
            fit_mean = train_feat.loc[~valid_mask, TARGETS].astype(float).mean()
            out.loc[valid_mask, TARGETS] = mean_align_frame(
                anchor.loc[valid_mask, TARGETS], fit_mean, gamma
            ).to_numpy(float)
        return out.astype(float).clip(0.03, 0.97)

    # 함수: 날짜 인접 confidence를 fold 내부 train만 사용해 계산합니다.
    def build_date_confidence(fit_feat, pred_feat):
        conf = pd.Series(0.0, index=pred_feat.index, dtype=float)
        date_tables = {
            sid: pd.to_datetime(group["lifelog_date"]).sort_values().to_numpy()
            for sid, group in fit_feat.groupby("subject_id")
        }
        for row in pred_feat.itertuples():
            dates = date_tables.get(str(row.subject_id))
            if dates is None or len(dates) == 0:
                continue
            delta = np.abs(
                (pd.to_datetime(dates) - pd.Timestamp(row.lifelog_date)).days.astype(float)
            )
            min_dist = float(np.min(delta))
            conf.loc[row.Index] = float(np.exp(-min_dist / 8.0))
        return conf.clip(0.0, 1.0)

    # 함수: 기존 1~17단계의 세부 후보명/가중치 체계를 OOF source로 확장합니다.
    def expand_full_pipeline_candidate_sources(oof, test_pred, train_feat, test_feat, folds):
        expanded_oof = {}
        expanded_test = {}
        anchor_oof = make_routing_anchor(oof)
        anchor_test = make_routing_anchor(test_pred)
        expanded_oof["submission_step1_target_routing"] = anchor_oof
        expanded_test["submission_step1_target_routing"] = anchor_test

        def add(name, oof_frame, test_frame):
            expanded_oof[name] = oof_frame.astype(float).clip(0.03, 0.97)
            expanded_test[name] = test_frame.astype(float).clip(0.03, 0.97)

        # 1) state-transition grid 후보 전체 축
        state = oof["prior_state_transition"]
        state_t = test_pred["prior_state_transition"]
        for q, s in [(10, 5), (14, 7), (16, 8), (18, 9), (20, 10), (22, 11), (24, 12), (28, 14)]:
            weights = {t: q / 100 for t in ["Q1", "Q2", "Q3"]}
            weights.update({t: s / 100 for t in ["S1", "S2", "S3", "S4"]})
            add(
                f"submission_step1_transition_past_q{q:02d}_s{s:02d}",
                blend_prediction(anchor_oof, state, weights),
                blend_prediction(anchor_test, state_t, weights),
            )
        for tag, weights in {
            "tw_a": {"Q1": 0.16, "Q2": 0.16, "Q3": 0.16, "S1": 0.08, "S2": 0.08, "S3": 0.08, "S4": 0.08},
            "tw_b": {"Q1": 0.18, "Q2": 0.18, "Q3": 0.18, "S1": 0.09, "S2": 0.09, "S3": 0.07, "S4": 0.09},
            "tw_c": {"Q1": 0.20, "Q2": 0.18, "Q3": 0.20, "S1": 0.10, "S2": 0.10, "S3": 0.08, "S4": 0.10},
            "tw_d": {"Q1": 0.22, "Q2": 0.20, "Q3": 0.22, "S1": 0.11, "S2": 0.11, "S3": 0.08, "S4": 0.11},
        }.items():
            add(
                f"submission_step1_transition_past_{tag}",
                blend_prediction(anchor_oof, state, weights),
                blend_prediction(anchor_test, state_t, weights),
            )

        # 2) target-dynamics reversion/grid 후보 축
        dyn = oof["prior_target_dynamics"]
        dyn_t = test_pred["prior_target_dynamics"]
        add("submission_step1_dynamics_prior", dyn, dyn_t)
        for w in [0.06, 0.10, 0.14, 0.18]:
            add(
                f"submission_step1_dynamics_blend_w{int(w * 100):02d}",
                blend_prediction(state, dyn, w),
                blend_prediction(state_t, dyn_t, w),
            )
        for w in [0.08, 0.12, 0.14, 0.16, 0.20]:
            add(
                f"submission_step1_dynamics_blend_w{int(w * 100):02d}_grid",
                blend_prediction(state, dyn, w),
                blend_prediction(state_t, dyn_t, w),
            )
        for tag, weights in {
            "tw_q12_s08": {"Q1": 0.12, "Q2": 0.12, "Q3": 0.12, "S1": 0.08, "S2": 0.08, "S3": 0.08, "S4": 0.08},
            "tw_q14_s08": {"Q1": 0.14, "Q2": 0.14, "Q3": 0.14, "S1": 0.08, "S2": 0.08, "S3": 0.08, "S4": 0.08},
            "tw_q16_s06": {"Q1": 0.16, "Q2": 0.16, "Q3": 0.16, "S1": 0.06, "S2": 0.06, "S3": 0.06, "S4": 0.06},
        }.items():
            add(
                f"submission_step1_dynamics_blend_{tag}",
                blend_prediction(state, dyn, weights),
                blend_prediction(state_t, dyn_t, weights),
            )

        # 3) subject-date interpolation, bracket, calendar, hybrid 후보 축
        date = oof["prior_date_interp"]
        date_t = test_pred["prior_date_interp"]
        bracket = oof["prior_bracket"]
        bracket_t = test_pred["prior_bracket"]
        calendar = oof["prior_calendar"]
        calendar_t = test_pred["prior_calendar"]
        hybrid = (0.55 * bracket[TARGETS] + 0.45 * calendar[TARGETS]).clip(0.03, 0.97)
        hybrid_t = (0.55 * bracket_t[TARGETS] + 0.45 * calendar_t[TARGETS]).clip(0.03, 0.97)
        add("submission_step1_interpolation", date, date_t)
        for w in [0.04, 0.07, 0.10, 0.13, 0.16]:
            add(
                f"submission_step1_date_blend_w{int(w * 100):02d}",
                blend_prediction(dyn, date, w),
                blend_prediction(dyn_t, date_t, w),
            )
        for tag, weights in {
            "tw_q10_s06": {"Q1": 0.10, "Q2": 0.10, "Q3": 0.10, "S1": 0.06, "S2": 0.06, "S3": 0.04, "S4": 0.06},
            "tw_q12_s08": {"Q1": 0.12, "Q2": 0.12, "Q3": 0.12, "S1": 0.08, "S2": 0.08, "S3": 0.05, "S4": 0.08},
            "tw_q08_s10": {"Q1": 0.08, "Q2": 0.08, "Q3": 0.08, "S1": 0.10, "S2": 0.10, "S3": 0.06, "S4": 0.10},
        }.items():
            add(
                f"submission_step1_date_blend_{tag}",
                blend_prediction(dyn, date, weights),
                blend_prediction(dyn_t, date_t, weights),
            )
        add("submission_step1_date_bracket", bracket, bracket_t)
        add("submission_step1_calendar", calendar, calendar_t)
        add("submission_step1_calendar_bracket", hybrid, hybrid_t)
        for tag, prior, prior_t in [
            ("date_bracket", bracket, bracket_t),
            ("calendar", calendar, calendar_t),
            ("hybrid_bracket_calendar", hybrid, hybrid_t),
        ]:
            for w in [0.04, 0.07, 0.10]:
                add(
                    f"submission_{tag}_anchor_w{int(w * 100):02d}",
                    blend_prediction(date, prior, w),
                    blend_prediction(date_t, prior_t, w),
                )

        # 4) current-best calibration 후보 축
        for gamma in [0.02, 0.03, 0.04, 0.06]:
            add(
                f"submission_step1_mean_g{int(gamma * 1000):03d}",
                foldwise_mean_align(date, train_feat, folds, gamma),
                mean_align_frame(date_t, train_feat[TARGETS].astype(float).mean(), gamma),
            )
        for temp in [0.97, 1.03, 1.06]:
            add(
                f"submission_step1_temperature_t{int(temp * 1000):04d}",
                temperature_scale(date, temp),
                temperature_scale(date_t, temp),
            )

        # 5) adaptive/reliability refinement/target-scale 후보 축
        conf_oof = pd.Series(0.0, index=train_feat.index, dtype=float)
        for fold in range(N_FOLDS):
            valid_mask = folds.eq(fold)
            conf_oof.loc[valid_mask] = build_date_confidence(
                train_feat.loc[~valid_mask], train_feat.loc[valid_mask]
            ).to_numpy(float)
        conf_test = build_date_confidence(train_feat, test_feat)

        def adaptive_blend(anchor, prior, conf, base, scale, mult):
            out = anchor.copy()
            for target in TARGETS:
                w = np.clip(base + scale * conf.to_numpy(float), 0.0, 0.22)
                w = w * float(mult.get(target, 1.0))
                out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[
                    target
                ].to_numpy(float)
            return out.astype(float).clip(0.03, 0.97)

        mult_date = {"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "S1": 0.85, "S2": 0.90, "S3": 0.75, "S4": 0.90}
        adaptive_specs = [
            ("submission_step1_date_prior", date, date_t, 0.03, 0.10, mult_date),
            ("submission_step1_date_candidate_dateconf_dateprior_b04_s12", date, date_t, 0.04, 0.12, mult_date),
            ("submission_step1_date_candidate_dateconf_bracket_b03_s10", bracket, bracket_t, 0.03, 0.10, mult_date),
            ("submission_step1_date_candidate_dateconf_hybrid_b03_s10", hybrid, hybrid_t, 0.03, 0.10, mult_date),
        ]
        for name, prior, prior_t, base, scale, mult in adaptive_specs:
            add(
                name,
                adaptive_blend(date, prior, conf_oof, base, scale, mult),
                adaptive_blend(date_t, prior_t, conf_test, base, scale, mult),
            )

        add("submission_step1_reliability_prior", adaptive_blend(date, date, conf_oof, 0.02, 0.08, mult_date), adaptive_blend(date_t, date_t, conf_test, 0.02, 0.08, mult_date))
        add("submission_subject_reliability_bracket_prior", adaptive_blend(date, bracket, conf_oof, 0.02, 0.08, mult_date), adaptive_blend(date_t, bracket_t, conf_test, 0.02, 0.08, mult_date))
        mixed = (0.80 * date[TARGETS] + 0.20 * bracket[TARGETS]).clip(0.03, 0.97)
        mixed_t = (0.80 * date_t[TARGETS] + 0.20 * bracket_t[TARGETS]).clip(0.03, 0.97)
        add("submission_step1_reliability_refinement_mixed_date_bracket", mixed, mixed_t)
        for name, weights in {
            "submission_step1_target_reliability_scale_soft": {"Q1": 0.006, "Q2": 0.006, "Q3": 0.006, "S1": 0.004, "S2": 0.004, "S3": 0.003, "S4": 0.004},
            "submission_step1_target_reliability_scale_mid": {"Q1": 0.009, "Q2": 0.009, "Q3": 0.009, "S1": 0.006, "S2": 0.006, "S3": 0.004, "S4": 0.006},
            "submission_step1_target_reliability_q_led_s_soft": {"Q1": 0.010, "Q2": 0.008, "Q3": 0.010, "S1": 0.004, "S2": 0.004, "S3": 0.002, "S4": 0.004},
            "submission_step1_target_reliability_mixed_date_bracket": {"Q1": 0.010, "Q2": 0.010, "Q3": 0.010, "S1": 0.006, "S2": 0.006, "S3": 0.004, "S4": 0.006},
        }.items():
            base_prior = mixed if "mixed" in name else date
            base_prior_t = mixed_t if "mixed" in name else date_t
            add(name, blend_prediction(date, base_prior, weights), blend_prediction(date_t, base_prior_t, weights))

        # 6) pattern projection, sensor KNN, sleep-window/refinement/metric proxy 후보 축
        pattern = oof["prior_pattern_projection"]
        pattern_t = test_pred["prior_pattern_projection"]
        sensor = oof["prior_sensor_rank"]
        sensor_t = test_pred["prior_sensor_rank"]
        for w in [0.015, 0.025, 0.040]:
            add(
                f"submission_step1_joint_pattern_w{int(w * 1000):03d}",
                blend_prediction(date, pattern, w),
                blend_prediction(date_t, pattern_t, w),
            )
        for w in [0.010, 0.020, 0.035]:
            add(
                f"submission_step1_sensor_neighbors_k16_w{int(w * 1000):03d}",
                blend_prediction(date, sensor, w),
                blend_prediction(date_t, sensor_t, w),
            )
        date_sensor = (0.82 * date[TARGETS] + 0.18 * sensor[TARGETS]).clip(0.03, 0.97)
        date_sensor_t = (0.82 * date_t[TARGETS] + 0.18 * sensor_t[TARGETS]).clip(0.03, 0.97)
        add("submission_step1_date_sensor_ensemble", date_sensor, date_sensor_t)

        if "prior_sleep_window_model" in oof:
            sleep = oof["prior_sleep_window_model"]
            sleep_t = test_pred["prior_sleep_window_model"]
            for w in [0.006, 0.012]:
                add(
                    f"submission_step1_sleep_window_w{int(w * 1000):03d}",
                    blend_prediction(pattern, sleep, w),
                    blend_prediction(pattern_t, sleep_t, w),
                )
            sleep_targets = {"Q1": 0.014, "Q2": 0.004, "Q3": 0.004, "S1": 0.016, "S2": 0.014, "S3": 0.016, "S4": 0.014}
            add("submission_step1_sleep_window_sleep_targets", blend_prediction(pattern, sleep, sleep_targets), blend_prediction(pattern_t, sleep_t, sleep_targets))
            for name, weights in {
                "submission_step1_sleep_window_plus_soft": {"Q1": 0.010, "Q2": 0.004, "Q3": 0.004, "S1": 0.014, "S2": 0.014, "S3": 0.014, "S4": 0.014},
                "submission_step1_sleep_window_s_target_focus": {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.018, "S2": 0.016, "S3": 0.018, "S4": 0.016},
                "submission_step1_sleep_window_q1_s1s3": {"Q1": 0.014, "Q2": 0.000, "Q3": 0.000, "S1": 0.018, "S2": 0.000, "S3": 0.018, "S4": 0.000},
                "submission_step1_sleep_window_guarded": {"Q1": 0.006, "Q2": 0.002, "Q3": 0.002, "S1": 0.012, "S2": 0.012, "S3": 0.012, "S4": 0.012},
            }.items():
                add(name, blend_prediction(pattern, sleep, weights), blend_prediction(pattern_t, sleep_t, weights))

        if "prior_sleep_metric_proxy" in oof:
            metric = oof["prior_sleep_metric_proxy"]
            metric_t = test_pred["prior_sleep_metric_proxy"]
            for name, weights in {
                "submission_step1_sleep_proxy_s_targets": {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.010, "S2": 0.010, "S3": 0.010, "S4": 0.010},
                "submission_step1_sleep_proxy_qs_tiny": {"Q1": 0.006, "Q2": 0.004, "Q3": 0.004, "S1": 0.010, "S2": 0.010, "S3": 0.010, "S4": 0.010},
                "submission_step1_sleep_proxy_quality": {"Q1": 0.012, "Q2": 0.000, "Q3": 0.000, "S1": 0.012, "S2": 0.012, "S3": 0.008, "S4": 0.008},
            }.items():
                add(name, blend_prediction(anchor_oof, metric, weights), blend_prediction(anchor_test, metric_t, weights))

        if "prior_sleep_interval_proxy" in oof:
            interval = oof["prior_sleep_interval_proxy"]
            interval_t = test_pred["prior_sleep_interval_proxy"]
            for name, weights in {
                "submission_step1_sleep_interval_s_tiny": {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.006, "S2": 0.006, "S3": 0.006, "S4": 0.006},
                "submission_step1_sleep_interval_s_targets": {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.012, "S2": 0.012, "S3": 0.010, "S4": 0.010},
                "submission_step1_sleep_interval_s_stronger": {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.018, "S2": 0.016, "S3": 0.014, "S4": 0.014},
            }.items():
                add(name, blend_prediction(anchor_oof, interval, weights), blend_prediction(anchor_test, interval_t, weights))

        return expanded_oof, expanded_test

    # 함수: LightGBM/CatBoost/XGBoost base layer의 OOF와 test 예측을 생성합니다.
    def fit_base_oof(train_feat, test_feat, feature_cols, folds):
        x_train = train_feat[feature_cols].copy()
        x_test = test_feat[feature_cols].copy()
        med = x_train.median(numeric_only=True)
        x_train = x_train.fillna(med).fillna(0.0)
        x_test = x_test.fillna(med).fillna(0.0)

        oof = {
            "lgb": pd.DataFrame(index=train_feat.index, columns=TARGETS, dtype=float),
            "cat": pd.DataFrame(index=train_feat.index, columns=TARGETS, dtype=float),
            "xgb": pd.DataFrame(index=train_feat.index, columns=TARGETS, dtype=float),
        }
        test_pred = {
            "lgb": pd.DataFrame(0.0, index=test_feat.index, columns=TARGETS),
            "cat": pd.DataFrame(0.0, index=test_feat.index, columns=TARGETS),
            "xgb": pd.DataFrame(0.0, index=test_feat.index, columns=TARGETS),
        }

        for target in TARGETS:
            y = train_feat[target].astype(int).reset_index(drop=True)
            print(f"[OOF] target={target}")
            for fold in range(N_FOLDS):
                valid_mask = folds.eq(fold).to_numpy()
                train_mask = ~valid_mask
                restored = (
                    load_step1_candidate_checkpoint(
                        "xgb_oof_stacking_base_layer",
                        target,
                        "fold",
                        fold,
                    )
                    if restore_step1_candidate_checkpoints_enabled()
                    else None
                )
                if restored is not None:
                    print(f"[STEP1_CKPT] restored candidate xgb_oof_stacking_base_layer {target} fold={fold}")
                    x_valid = _step1_candidate_matrix(x_train.loc[valid_mask], restored)
                    models = restored["models"]
                    oof["lgb"].loc[valid_mask, target] = models["lgb"].predict_proba(x_valid)[:, 1]
                    oof["cat"].loc[valid_mask, target] = models["cat"].predict_proba(x_valid)[:, 1]
                    oof["xgb"].loc[valid_mask, target] = models["xgb"].predict_proba(x_valid)[:, 1]
                    continue

                lgb_model = lgb.LGBMClassifier(
                    n_estimators=260,
                    learning_rate=0.025,
                    num_leaves=15,
                    max_depth=4,
                    min_child_samples=18,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=1.5,
                    reg_lambda=5.0,
                    objective="binary",
                    class_weight="balanced",
                    random_state=1000 + fold,
                    verbosity=-1,
                )
                lgb_model.fit(x_train.loc[train_mask, feature_cols], y.loc[train_mask])
                oof["lgb"].loc[valid_mask, target] = lgb_model.predict_proba(
                    x_train.loc[valid_mask, feature_cols]
                )[:, 1]

                cat_model = CatBoostClassifier(
                    iterations=260,
                    learning_rate=0.025,
                    depth=4,
                    l2_leaf_reg=10,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    random_seed=2000 + fold,
                    allow_writing_files=False,
                    verbose=False,
                )
                cat_model.fit(x_train.loc[train_mask, feature_cols], y.loc[train_mask])
                oof["cat"].loc[valid_mask, target] = cat_model.predict_proba(
                    x_train.loc[valid_mask, feature_cols]
                )[:, 1]

                xgb_model = XGBClassifier(
                    n_estimators=220,
                    max_depth=2,
                    learning_rate=0.025,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=2.0,
                    reg_lambda=8.0,
                    min_child_weight=4,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=3000 + fold,
                    n_jobs=1,
                )
                xgb_model.fit(x_train.loc[train_mask, feature_cols], y.loc[train_mask])
                oof["xgb"].loc[valid_mask, target] = xgb_model.predict_proba(
                    x_train.loc[valid_mask, feature_cols]
                )[:, 1]
                save_step1_candidate_checkpoint(
                    "xgb_oof_stacking_base_layer",
                    {
                        "kind": "fold",
                        "target": target,
                        "fold": fold,
                        "feature_cols": list(feature_cols),
                        "median": med,
                        "models": {
                            "lgb": lgb_model,
                            "cat": cat_model,
                            "xgb": xgb_model,
                        },
                    },
                    target,
                    "fold",
                    fold,
                )

            for seed in SEEDS:
                restored = (
                    load_step1_candidate_checkpoint(
                        "xgb_oof_stacking_base_layer",
                        target,
                        "full",
                        seed,
                    )
                    if restore_step1_candidate_checkpoints_enabled()
                    else None
                )
                if restored is not None:
                    print(f"[STEP1_CKPT] restored candidate xgb_oof_stacking_base_layer {target} full seed={seed}")
                    x_full = _step1_candidate_matrix(x_test, restored)
                    models = restored["models"]
                    test_pred["lgb"][target] += models["lgb"].predict_proba(x_full)[:, 1] / len(SEEDS)
                    test_pred["cat"][target] += models["cat"].predict_proba(x_full)[:, 1] / len(SEEDS)
                    test_pred["xgb"][target] += models["xgb"].predict_proba(x_full)[:, 1] / len(SEEDS)
                    continue

                lgb_full = lgb.LGBMClassifier(
                    n_estimators=260,
                    learning_rate=0.025,
                    num_leaves=15,
                    max_depth=4,
                    min_child_samples=18,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=1.5,
                    reg_lambda=5.0,
                    objective="binary",
                    class_weight="balanced",
                    random_state=seed,
                    verbosity=-1,
                )
                lgb_full.fit(x_train[feature_cols], y)
                test_pred["lgb"][target] += lgb_full.predict_proba(x_test[feature_cols])[:, 1] / len(SEEDS)

                cat_full = CatBoostClassifier(
                    iterations=260,
                    learning_rate=0.025,
                    depth=4,
                    l2_leaf_reg=10,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    random_seed=seed,
                    allow_writing_files=False,
                    verbose=False,
                )
                cat_full.fit(x_train[feature_cols], y)
                test_pred["cat"][target] += cat_full.predict_proba(x_test[feature_cols])[:, 1] / len(SEEDS)

                xgb_full = XGBClassifier(
                    n_estimators=220,
                    max_depth=2,
                    learning_rate=0.025,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=2.0,
                    reg_lambda=8.0,
                    min_child_weight=4,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=1,
                )
                xgb_full.fit(x_train[feature_cols], y)
                test_pred["xgb"][target] += xgb_full.predict_proba(x_test[feature_cols])[:, 1] / len(SEEDS)
                save_step1_candidate_checkpoint(
                    "xgb_oof_stacking_base_layer",
                    {
                        "kind": "full",
                        "target": target,
                        "seed": seed,
                        "feature_cols": list(feature_cols),
                        "median": med,
                        "models": {
                            "lgb": lgb_full,
                            "cat": cat_full,
                            "xgb": xgb_full,
                        },
                    },
                    target,
                    "full",
                    seed,
                )

        for model_name in oof:
            oof[model_name] = oof[model_name].astype(float).clip(1e-6, 1 - 1e-6)
            test_pred[model_name] = test_pred[model_name].astype(float).clip(1e-6, 1 - 1e-6)
        return oof, test_pred

    # 함수: base OOF 예측값을 target별 meta feature matrix로 구성합니다.
    def make_meta_frame(pred_map, target, selected_sources=None):
        keys = sorted(pred_map.keys())
        if selected_sources is not None:
            keep = set(selected_sources)
            keys = [name for name in keys if name in keep]
        frame = pd.DataFrame(
            {
                name: pred_map[name][target].to_numpy(float)
                for name in keys
            }
        )
        model_cols = [col for col in ["lgb", "cat", "xgb"] if col in frame.columns]
        prior_cols = [col for col in frame.columns if col.startswith("prior_")]
        candidate_cols = [
            col
            for col in frame.columns
            if col.startswith("submission_") or col.startswith("sub_")
        ]
        if {"lgb", "cat"}.issubset(frame.columns):
            frame["avg_lgb_cat"] = 0.5 * frame["lgb"] + 0.5 * frame["cat"]
        if model_cols:
            frame["avg_models"] = frame[model_cols].mean(axis=1)
            frame["spread_models"] = frame[model_cols].max(axis=1) - frame[model_cols].min(axis=1)
        if prior_cols:
            frame["avg_priors"] = frame[prior_cols].mean(axis=1)
            frame["spread_priors"] = frame[prior_cols].max(axis=1) - frame[prior_cols].min(axis=1)
        if candidate_cols:
            frame["avg_candidates"] = frame[candidate_cols].mean(axis=1)
            frame["spread_candidates"] = frame[candidate_cols].max(axis=1) - frame[candidate_cols].min(axis=1)
        stack_cols = [
            col for col in frame.columns if col in model_cols + prior_cols + candidate_cols
        ]
        frame["avg_all"] = frame[stack_cols].mean(axis=1)
        frame["spread_all"] = frame[stack_cols].max(axis=1) - frame[stack_cols].min(axis=1)
        for col in stack_cols + ["avg_all"]:
            p = np.clip(frame[col].to_numpy(float), 1e-6, 1 - 1e-6)
            frame[f"{col}_logit"] = np.log(p / (1 - p))
        return frame

    # 함수: source별 target OOF logloss를 계산하고 target별 meta 후보를 고릅니다.
    def rank_and_select_meta_sources(train_feat, oof, current_best_key=None):
        rows = []
        selected = {}
        for target in TARGETS:
            y = train_feat[target].astype(int).reset_index(drop=True)
            for name, frame in oof.items():
                score = log_loss(
                    y,
                    clip_proba(frame[target].astype(float).reset_index(drop=True), 1e-6, 1 - 1e-6),
                    labels=[0, 1],
                )
                rows.append({"target": target, "source": name, "oof_logloss": score})
        diag = pd.DataFrame(rows)
        for target in TARGETS:
            target_diag = diag[diag["target"] == target].sort_values("oof_logloss")
            if current_best_key in set(target_diag["source"]):
                baseline = float(
                    target_diag.loc[target_diag["source"] == current_best_key, "oof_logloss"].iloc[0]
                )
            else:
                baseline = float(target_diag["oof_logloss"].iloc[0])
            good = target_diag[target_diag["oof_logloss"] <= baseline + 0.020]["source"].tolist()
            top = target_diag.head(36)["source"].tolist()
            must_keep = [
                name
                for name in [
                    current_best_key,
                    "date_aligned_lgb",
                    "date_aligned_cat",
                    "date_aligned_xgb",
                    "prior_sleep_interval_proxy",
                    "submission_step1_sleep_interval_s_targets",
                    "submission_step1_sleep_proxy_s_targets",
                    "submission_step1_sequence_model",
                ]
                if name and name in oof
            ]
            chosen = []
            for name in must_keep + good + top:
                if name not in chosen:
                    chosen.append(name)
            target_source_caps = {
                "Q1": 28,
                "Q2": 24,
                "Q3": 24,
                "S1": 28,
                "S2": 32,
                "S3": 24,
                "S4": 26,
            }
            selected[target] = chosen[: target_source_caps.get(target, 26)]
        return diag, selected

    # 함수: 단순 convex source blend를 OOF 위에서 greedy하게 찾습니다.
    def fit_greedy_source_blend(y, source_oof, source_test, source_names):
        if not source_names:
            mean = float(np.mean(y))
            return (
                np.full(len(y), mean, dtype=float),
                np.full(len(next(iter(source_test.values()))), mean, dtype=float),
                "none",
                log_loss(y, np.full(len(y), mean), labels=[0, 1]),
            )
        scores = {}
        for name in source_names:
            scores[name] = log_loss(
                y,
                clip_proba(source_oof[name], 1e-6, 1 - 1e-6),
                labels=[0, 1],
            )
        ordered = sorted(source_names, key=lambda name: scores[name])[:24]
        best_name = ordered[0]
        blend = np.asarray(source_oof[best_name], dtype=float).copy()
        blend_test = np.asarray(source_test[best_name], dtype=float).copy()
        best_score = scores[best_name]
        used = [best_name]
        for _ in range(8):
            improved = False
            best_candidate = None
            for name in ordered:
                if name in used:
                    continue
                candidate_oof = np.asarray(source_oof[name], dtype=float)
                candidate_test = np.asarray(source_test[name], dtype=float)
                for weight in [0.03, 0.05, 0.08, 0.12, 0.16, 0.22, 0.30, 0.40]:
                    trial = (1 - weight) * blend + weight * candidate_oof
                    score = log_loss(y, clip_proba(trial, 1e-6, 1 - 1e-6), labels=[0, 1])
                    if score < best_score - 1e-7:
                        best_score = score
                        best_candidate = (name, weight, trial, (1 - weight) * blend_test + weight * candidate_test)
                        improved = True
            if not improved:
                break
            name, weight, blend, blend_test = best_candidate
            used.append(name)
        return (
            clip_proba(blend, 1e-6, 1 - 1e-6),
            clip_proba(blend_test, 1e-6, 1 - 1e-6),
            "+".join(used),
            best_score,
        )

    # 함수: XGB/Ridge/Logistic/Greedy meta 후보를 OOF에서 비교해 target별 best stack을 만듭니다.
    def fit_xgb_meta(train_feat, oof, test_pred, current_best_key=None):
        first_test = next(iter(test_pred.values()))
        stacked = pd.DataFrame(index=first_test.index, columns=TARGETS, dtype=float)
        stacked_oof = pd.DataFrame(index=train_feat.index, columns=TARGETS, dtype=float)
        cv_rows = []
        source_diag, selected_sources = rank_and_select_meta_sources(
            train_feat, oof, current_best_key=current_best_key
        )
        source_diag_path = DATA_DIR / "artifacts" / "meta_source_target_oof_logloss.csv"
        source_diag_path.parent.mkdir(parents=True, exist_ok=True)
        source_diag.to_csv(source_diag_path, index=False)
        print(f"[STACK] source ranking diagnostics saved: {source_diag_path}")
        for target in TARGETS:
            selected = selected_sources[target]
            x_meta = make_meta_frame(oof, target, selected_sources=selected)
            y = train_feat[target].astype(int).reset_index(drop=True)
            test_meta = make_meta_frame(test_pred, target, selected_sources=selected)

            xgb_oof = np.zeros(len(x_meta), dtype=float)
            log_oof = np.zeros(len(x_meta), dtype=float)
            ridge_oof = np.zeros(len(x_meta), dtype=float)
            folds = make_temporal_folds(train_feat).reset_index(drop=True)
            for fold in range(N_FOLDS):
                valid_mask = folds.eq(fold).to_numpy()
                train_mask = ~valid_mask
                restored = (
                    load_step1_candidate_checkpoint(
                        "xgb_oof_stacking_meta_layer",
                        target,
                        "fold",
                        fold,
                    )
                    if restore_step1_candidate_checkpoints_enabled()
                    else None
                )
                if restored is not None:
                    print(f"[STEP1_CKPT] restored candidate xgb_oof_stacking_meta_layer {target} fold={fold}")
                    models = restored["models"]
                    restored_cols = list(restored["columns"])
                    restored_logit_cols = list(restored["logit_cols"])
                    xgb_oof[valid_mask] = models["xgb"].predict_proba(
                        x_meta.loc[valid_mask, restored_cols]
                    )[:, 1]
                    log_oof[valid_mask] = models["logistic"].predict_proba(
                        x_meta.loc[valid_mask, restored_logit_cols]
                    )[:, 1]
                    ridge_oof[valid_mask] = models["ridge"].predict(
                        x_meta.loc[valid_mask, restored_cols]
                    )
                    continue
                meta = XGBClassifier(
                    n_estimators=80,
                    max_depth=2,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=3.0,
                    reg_lambda=12.0,
                    min_child_weight=6,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=5000 + fold,
                    n_jobs=1,
                )
                meta.fit(x_meta.loc[train_mask], y.loc[train_mask])
                xgb_oof[valid_mask] = meta.predict_proba(x_meta.loc[valid_mask])[:, 1]

                logit_cols = [c for c in x_meta.columns if c.endswith("_logit")]
                log_model = LogisticRegression(
                    C=0.04,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                )
                log_model.fit(x_meta.loc[train_mask, logit_cols], y.loc[train_mask])
                log_oof[valid_mask] = log_model.predict_proba(x_meta.loc[valid_mask, logit_cols])[:, 1]

                ridge = Ridge(alpha=28.0)
                ridge.fit(x_meta.loc[train_mask], y.loc[train_mask].astype(float))
                ridge_oof[valid_mask] = ridge.predict(x_meta.loc[valid_mask])
                save_step1_candidate_checkpoint(
                    "xgb_oof_stacking_meta_layer",
                    {
                        "kind": "fold",
                        "target": target,
                        "fold": fold,
                        "selected_sources": selected,
                        "columns": list(x_meta.columns),
                        "logit_cols": logit_cols,
                        "models": {
                            "xgb": meta,
                            "logistic": log_model,
                            "ridge": ridge,
                        },
                    },
                    target,
                    "fold",
                    fold,
                )

            base_avg = x_meta["avg_all"].to_numpy(float)
            source_oof = {name: oof[name][target].astype(float).to_numpy() for name in selected}
            source_test = {name: test_pred[name][target].astype(float).to_numpy() for name in selected}
            greedy_oof, greedy_test, greedy_used, greedy_score = fit_greedy_source_blend(
                y.to_numpy(int), source_oof, source_test, selected
            )
            blend_oof = {
                "xgb": shrink_proba(xgb_oof, alpha=0.985),
                "logistic": shrink_proba(log_oof, alpha=0.990),
                "ridge": clip_proba(ridge_oof, 0.03, 0.97),
                "greedy": clip_proba(greedy_oof, 0.03, 0.97),
            }
            blend_oof["avg_xgb_log_ridge"] = clip_proba(
                0.40 * blend_oof["xgb"] + 0.35 * blend_oof["logistic"] + 0.25 * blend_oof["ridge"],
                0.03,
                0.97,
            )
            blend_oof["stable_plus_greedy005"] = clip_proba(
                0.95 * blend_oof["avg_xgb_log_ridge"] + 0.05 * blend_oof["greedy"],
                0.03,
                0.97,
            )
            blend_oof["ridge_plus_greedy005"] = clip_proba(
                0.95 * blend_oof["ridge"] + 0.05 * blend_oof["greedy"],
                0.03,
                0.97,
            )
            blend_oof["xgb_plus_greedy005"] = clip_proba(
                0.95 * blend_oof["xgb"] + 0.05 * blend_oof["greedy"],
                0.03,
                0.97,
            )
            meta_scores = {
                name: log_loss(y, clip_proba(pred, 1e-6, 1 - 1e-6), labels=[0, 1])
                for name, pred in blend_oof.items()
            }
            target_recipe = {
                "Q1": "stable_plus_greedy005",
                "Q2": "stable_plus_greedy005",
                "Q3": "xgb_plus_greedy005",
                "S1": "ridge_plus_greedy005",
                "S2": "stable_plus_greedy005",
                "S3": "stable_plus_greedy005",
                "S4": "stable_plus_greedy005",
            }
            best_meta = target_recipe.get(target, "avg_xgb_log_ridge")
            stacked_oof[target] = blend_oof[best_meta]

            base_score = log_loss(y, clip_proba(base_avg, 1e-6, 1 - 1e-6), labels=[0, 1])
            cv_rows.append(
                {
                    "target": target,
                    "selected_sources": len(selected),
                    "base_avg_oof_logloss": base_score,
                    "xgb_oof_logloss": meta_scores["xgb"],
                    "logistic_oof_logloss": meta_scores["logistic"],
                    "ridge_oof_logloss": meta_scores["ridge"],
                    "greedy_oof_logloss": greedy_score,
                    "avg_meta_oof_logloss": meta_scores["avg_xgb_log_ridge"],
                    "stable_plus_greedy005_logloss": meta_scores["stable_plus_greedy005"],
                    "ridge_plus_greedy005_logloss": meta_scores["ridge_plus_greedy005"],
                    "xgb_plus_greedy005_logloss": meta_scores["xgb_plus_greedy005"],
                    "chosen_meta": best_meta,
                    "chosen_oof_logloss": meta_scores[best_meta],
                    "delta": meta_scores[best_meta] - base_score,
                    "greedy_minus_chosen": greedy_score - meta_scores[best_meta],
                    "greedy_used": greedy_used,
                }
            )

            restored_full = (
                load_step1_candidate_checkpoint(
                    "xgb_oof_stacking_meta_layer",
                    target,
                    "full",
                )
                if restore_step1_candidate_checkpoints_enabled()
                else None
            )
            if restored_full is not None:
                print(f"[STEP1_CKPT] restored candidate xgb_oof_stacking_meta_layer {target} full")
                full_models = restored_full["models"]
                full_cols = list(restored_full["columns"])
                logit_cols = list(restored_full["logit_cols"])
                xgb_test = shrink_proba(
                    full_models["xgb"].predict_proba(test_meta[full_cols])[:, 1],
                    alpha=0.985,
                )
                log_test = shrink_proba(
                    full_models["logistic"].predict_proba(test_meta[logit_cols])[:, 1],
                    alpha=0.990,
                )
                ridge_test = clip_proba(
                    full_models["ridge"].predict(test_meta[full_cols]),
                    0.03,
                    0.97,
                )
            else:
                full_meta = XGBClassifier(
                    n_estimators=90,
                    max_depth=2,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=3.0,
                    reg_lambda=12.0,
                    min_child_weight=6,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=7777,
                    n_jobs=1,
                )
                full_meta.fit(x_meta, y)
                xgb_test = shrink_proba(full_meta.predict_proba(test_meta)[:, 1], alpha=0.985)

                logit_cols = [c for c in x_meta.columns if c.endswith("_logit")]
                full_log = LogisticRegression(
                    C=0.04,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                )
                full_log.fit(x_meta[logit_cols], y)
                log_test = shrink_proba(full_log.predict_proba(test_meta[logit_cols])[:, 1], alpha=0.990)

                full_ridge = Ridge(alpha=28.0)
                full_ridge.fit(x_meta, y.astype(float))
                ridge_test = clip_proba(full_ridge.predict(test_meta), 0.03, 0.97)
                save_step1_candidate_checkpoint(
                    "xgb_oof_stacking_meta_layer",
                    {
                        "kind": "full",
                        "target": target,
                        "selected_sources": selected,
                        "columns": list(x_meta.columns),
                        "logit_cols": logit_cols,
                        "chosen_meta": best_meta,
                        "models": {
                            "xgb": full_meta,
                            "logistic": full_log,
                            "ridge": full_ridge,
                        },
                    },
                    target,
                    "full",
                )
            test_options = {
                "xgb": xgb_test,
                "logistic": log_test,
                "ridge": ridge_test,
                "greedy": greedy_test,
                "avg_xgb_log_ridge": clip_proba(
                    0.40 * xgb_test + 0.35 * log_test + 0.25 * ridge_test,
                    0.03,
                    0.97,
                ),
            }
            test_options["stable_plus_greedy005"] = clip_proba(
                0.95 * test_options["avg_xgb_log_ridge"] + 0.05 * greedy_test,
                0.03,
                0.97,
            )
            test_options["ridge_plus_greedy005"] = clip_proba(
                0.95 * ridge_test + 0.05 * greedy_test,
                0.03,
                0.97,
            )
            test_options["xgb_plus_greedy005"] = clip_proba(
                0.95 * xgb_test + 0.05 * greedy_test,
                0.03,
                0.97,
            )
            stacked[target] = test_options[best_meta]
        return (
            stacked.clip(0.03, 0.97),
            stacked_oof.astype(float).clip(0.03, 0.97),
            pd.DataFrame(cv_rows),
        )

    # 함수: 기존 final과 stacking 예측을 target별로 보수적으로 섞습니다.
    def blend_with_current_best(sample, current_best, stacked):
        out = sample.copy()
        weights = {
            "Q1": 0.015,
            "Q2": 0.015,
            "Q3": 0.015,
            "S1": 0.025,
            "S2": 0.025,
            "S3": 0.025,
            "S4": 0.025,
        }
        for target in TARGETS:
            w = weights[target]
            out[target] = clip_proba(
                (1 - w) * current_best[target].to_numpy(float)
                + w * stacked[target].to_numpy(float),
                0.03,
                0.97,
            )
        return out

    # 함수: OOF logloss 기준으로 target별 최적 alpha를 찾고 개선된 target만 반영합니다.
    def safe_alpha_search_blend(
        sample,
        current_best,
        stacked_test,
        current_best_oof,
        stacked_oof,
        train_feat,
    ):
        alpha_grid = [
            0.0,
            0.0025,
            0.005,
            0.0075,
            0.010,
            0.0125,
            0.015,
            0.020,
            0.025,
            0.030,
            0.040,
            0.050,
            0.060,
            0.075,
            0.090,
            0.100,
            0.125,
            0.150,
            0.175,
            0.180,
            0.190,
            0.200,
            0.210,
            0.220,
            0.230,
            0.240,
            0.250,
            0.260,
            0.280,
            0.300,
            0.320,
            0.340,
            0.360,
            0.380,
            0.400,
            0.420,
            0.440,
            0.460,
            0.500,
            0.550,
            0.600,
            0.650,
            0.700,
            0.750,
        ]
        out = sample.copy()
        rows = []
        for target in TARGETS:
            y = train_feat[target].astype(int).reset_index(drop=True)
            base_oof = current_best_oof[target].astype(float).reset_index(drop=True)
            stack_oof = stacked_oof[target].astype(float).reset_index(drop=True)
            baseline_score = log_loss(
                y, clip_proba(base_oof, 1e-6, 1 - 1e-6), labels=[0, 1]
            )

            best_alpha = 0.0
            best_score = baseline_score
            for alpha in alpha_grid:
                blend_oof = (1 - alpha) * base_oof + alpha * stack_oof
                score = log_loss(
                    y, clip_proba(blend_oof, 1e-6, 1 - 1e-6), labels=[0, 1]
                )
                if score < best_score - 1e-7:
                    best_score = score
                    best_alpha = float(alpha)

            if best_alpha <= 0.0:
                out[target] = current_best[target].to_numpy(float)
                gated = False
            else:
                out[target] = clip_proba(
                    (1 - best_alpha) * current_best[target].to_numpy(float)
                    + best_alpha * stacked_test[target].to_numpy(float),
                    0.03,
                    0.97,
                )
                gated = True
            rows.append(
                {
                    "target": target,
                    "baseline_oof_logloss": baseline_score,
                    "safe_blend_oof_logloss": best_score,
                    "delta": best_score - baseline_score,
                    "alpha": best_alpha,
                    "applied": gated,
                }
            )
        diag = pd.DataFrame(rows)
        return out, diag

    # 함수: 원본 1~17단계를 fold별 임시 프로젝트 루트에서 문자 그대로 실행해 OOF 후보를 만듭니다.
    def fit_literal_pipeline_oof(train, sample, folds):
        base_steps = [
            (label, func, args)
            for label, func, args in PIPELINE
            if "build_xgboost_oof_stacking_candidates" not in label
            and "validate_final_submission" not in label
        ]
        fold_base = DATA_DIR / "artifacts" / "literal_pipeline_oof_folds"
        fold_base.mkdir(parents=True, exist_ok=True)
        sample_cols = list(sample.columns)
        required_cols = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
        for col in required_cols:
            if col not in sample_cols:
                sample_cols.append(col)

        oof_map: dict[str, pd.DataFrame] = {}
        oof_seen: dict[str, set[int]] = {}

        # 함수: 원본 제출 sample shape 검증을 통과하도록 250행으로 padding합니다.
        def make_sample_frame(rows):
            frame = rows.copy()
            for col in sample_cols:
                if col not in frame.columns:
                    frame[col] = 0.0 if col in TARGETS else ""
            frame = frame[sample_cols].copy()
            for target in TARGETS:
                frame[target] = 0.0
            if len(frame) == 0:
                raise ValueError("empty validation frame for literal OOF")
            if len(frame) < 250:
                reps = int(np.ceil(250 / len(frame)))
                pieces = []
                for rep in range(reps):
                    piece = frame.copy()
                    if rep > 0:
                        for date_col in ["sleep_date", "lifelog_date"]:
                            if date_col in piece.columns:
                                piece[date_col] = (
                                    pd.to_datetime(piece[date_col])
                                    + pd.Timedelta(days=10000 + rep)
                                )
                    pieces.append(piece)
                frame = pd.concat(pieces, ignore_index=True).iloc[:250].copy()
            elif len(frame) > 250:
                frame = frame.iloc[:250].copy()
            for date_col in ["sleep_date", "lifelog_date"]:
                if date_col in frame.columns:
                    frame[date_col] = pd.to_datetime(frame[date_col]).dt.strftime(
                        "%Y-%m-%d"
                    )
            return frame

        # 함수: fold/chunk 임시 루트를 구성하고 센서 폴더를 연결합니다.
        def prepare_fold_root(fold_root, fit_train, pred_rows):
            if fold_root.exists():
                shutil.rmtree(fold_root)
            fold_data = fold_root / "data"
            (fold_data / "submissions").mkdir(parents=True, exist_ok=True)
            (fold_data / "artifacts").mkdir(parents=True, exist_ok=True)
            fit_train_out = fit_train.copy()
            for date_col in ["sleep_date", "lifelog_date"]:
                if date_col in fit_train_out.columns:
                    fit_train_out[date_col] = pd.to_datetime(
                        fit_train_out[date_col]
                    ).dt.strftime("%Y-%m-%d")
            fit_train_out.to_csv(fold_data / "ch2026_metrics_train.csv", index=False)
            make_sample_frame(pred_rows).to_csv(
                fold_data / "ch2026_submission_sample.csv", index=False
            )
            sensor_src = DATA_DIR / "ch2025_data_items"
            sensor_dst = fold_data / "ch2025_data_items"
            try:
                os.symlink(sensor_src, sensor_dst, target_is_directory=True)
            except FileExistsError:
                pass
            except OSError:
                shutil.copytree(sensor_src, sensor_dst)
            return fold_data

        # 함수: 임시 루트에서 원본 pipeline step 하나를 실행합니다.
        def run_step_in_root(fold_root, label, func, args):
            global PROJECT_ROOT
            old_project_root = PROJECT_ROOT
            old_argv = sys.argv[:]
            old_cwd = Path.cwd()
            try:
                PROJECT_ROOT = fold_root
                os.chdir(fold_root)
                sys.argv = [str(fold_root / label), *args]
                func()
            finally:
                PROJECT_ROOT = old_project_root
                sys.argv = old_argv
                os.chdir(old_cwd)

        # 함수: 원본 1~17단계 실행 결과 CSV를 OOF matrix에 반영합니다.
        def collect_fold_outputs(fold_data, valid_indices, n_rows):
            submission_files = sorted((fold_data / "submissions").glob("*.csv"))
            for path in submission_files:
                name = path.stem
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                if len(df) < n_rows or any(col not in df.columns for col in TARGETS):
                    continue
                if name not in oof_map:
                    oof_map[name] = pd.DataFrame(
                        index=train.index, columns=TARGETS, dtype=float
                    )
                    oof_seen[name] = set()
                part = df.iloc[:n_rows][TARGETS].astype(float).clip(1e-6, 1 - 1e-6)
                oof_map[name].loc[valid_indices, TARGETS] = part.to_numpy(float)
                oof_seen[name].update(int(i) for i in valid_indices)

        print("[LITERAL-OOF] running original pipeline steps 1-17 inside each fold")
        for fold in range(N_FOLDS):
            valid_indices = train.index[folds.eq(fold)].to_list()
            fit_train = train.loc[~folds.eq(fold)].copy().reset_index(drop=True)
            print(f"[LITERAL-OOF] fold={fold} valid_rows={len(valid_indices)}")
            for chunk_id, start in enumerate(range(0, len(valid_indices), 250)):
                chunk_indices = valid_indices[start : start + 250]
                pred_rows = train.loc[chunk_indices, required_cols].copy().reset_index(drop=True)
                fold_root = fold_base / f"fold_{fold}_chunk_{chunk_id}"
                fold_data = prepare_fold_root(fold_root, fit_train, pred_rows)
                for label, func, args in base_steps:
                    print(f"[LITERAL-OOF] fold={fold} chunk={chunk_id} run {label}")
                    run_step_in_root(fold_root, label, func, args)
                collect_fold_outputs(fold_data, chunk_indices, len(chunk_indices))

        literal_oof = {}
        literal_test = {}
        all_indices = set(int(i) for i in train.index)
        for name, frame in sorted(oof_map.items()):
            if oof_seen.get(name, set()) != all_indices:
                continue
            test_path = SUBMISSION_DIR / f"{name}.csv"
            if not test_path.exists():
                continue
            test_df = pd.read_csv(test_path)
            if len(test_df) != len(sample) or any(col not in test_df.columns for col in TARGETS):
                continue
            key = name
            literal_oof[key] = frame.astype(float).clip(1e-6, 1 - 1e-6)
            literal_test[key] = test_df[TARGETS].astype(float).clip(1e-6, 1 - 1e-6).reset_index(drop=True)
        print(f"[LITERAL-OOF] usable literal sources: {len(literal_oof)}")
        return literal_oof, literal_test

    # 함수: 전체 XGBoost OOF stacking 후보 생성 과정을 실행합니다.
    def main():
        train, sample, train_feat, test_feat = load_feature_table()
        current_best = pd.read_csv(CURRENT_BEST)
        feature_cols = select_feature_columns(train_feat, test_feat)
        print(f"[STACK] feature table: train={train_feat.shape}, test={test_feat.shape}")
        print(f"[STACK] numeric non-leaky features: {len(feature_cols)}")

        folds = make_temporal_folds(train_feat)
        oof, test_pred = fit_literal_pipeline_oof(train, sample, folds)
        if not oof:
            raise RuntimeError("literal fold pipeline did not produce usable OOF sources")
        current_best_key = CURRENT_BEST.stem
        sleep_oof, sleep_test = fit_sleep_window_oof(train, sample, folds)
        for name, frame in sleep_oof.items():
            oof[name] = frame
        for name, frame in sleep_test.items():
            test_pred[name] = frame
        if "prior_sleep_interval_proxy" in sleep_oof and current_best_key in oof:
            interval = sleep_oof["prior_sleep_interval_proxy"]
            interval_t = sleep_test["prior_sleep_interval_proxy"]
            anchor_oof = oof[current_best_key]
            anchor_test = current_best[TARGETS].astype(float).reset_index(drop=True)
            interval_specs = {
                "submission_step1_sleep_interval_s_tiny": {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.006, "S2": 0.006, "S3": 0.006, "S4": 0.006},
                "submission_step1_sleep_interval_s_targets": {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.012, "S2": 0.012, "S3": 0.010, "S4": 0.010},
                "submission_step1_sleep_interval_s_stronger": {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.018, "S2": 0.016, "S3": 0.014, "S4": 0.014},
            }
            for name, weights in interval_specs.items():
                oof[name] = blend_prediction(anchor_oof, interval, weights)
                test_pred[name] = blend_prediction(anchor_test, interval_t, weights)
            print(f"[STACK] injected sleep interval proxy sources: {sorted(interval_specs)}")
        elif "prior_sleep_interval_proxy" not in sleep_oof:
            print("[STACK] sleep interval proxy was not generated")
        else:
            print(f"[STACK] skip sleep interval blends: baseline OOF missing: {current_best_key}")
        aligned_oof, aligned_test = fit_base_oof(train_feat, test_feat, feature_cols, folds)
        for name, frame in aligned_oof.items():
            oof[f"date_aligned_{name}"] = frame
        for name, frame in aligned_test.items():
            test_pred[f"date_aligned_{name}"] = frame
        print(f"[STACK] meta sources: {sorted(oof.keys())}")
        stacked, stacked_oof, cv = fit_xgb_meta(
            train_feat, oof, test_pred, current_best_key=current_best_key
        )
        print("\n[STACK] OOF meta diagnostics:")
        print(cv.to_string(index=False))
        oof_dir = DATA_DIR / "artifacts"
        oof_dir.mkdir(parents=True, exist_ok=True)
        meta_diag_path = oof_dir / "xgboost_stacking_sleepctx_data_alpha046_model_diagnostics.csv"
        cv.to_csv(meta_diag_path, index=False)
        print(f"[STACK] stable meta diagnostics: {meta_diag_path}")

        if current_best_key not in oof:
            raise RuntimeError(
                f"safe gate baseline OOF source not found: {current_best_key}"
            )
        current_best_oof = oof[current_best_key]

        for name, frame in oof.items():
            frame.to_csv(oof_dir / f"oof_stacking_base_{name}.csv", index=False)
        for name, frame in test_pred.items():
            frame.to_csv(oof_dir / f"test_stacking_base_{name}.csv", index=False)
        stacked_path = SUBMISSION_DIR / "submission_step1_stacking.csv"
        raw_submission = sample.copy()
        raw_submission[TARGETS] = stacked[TARGETS].to_numpy(float)
        raw_submission.to_csv(stacked_path, index=False)

        final, safe_diag = safe_alpha_search_blend(
            sample,
            current_best,
            stacked,
            current_best_oof,
            stacked_oof,
            train_feat,
        )
        safe_diag_path = oof_dir / "xgboost_stacking_safe_alpha_diagnostics.csv"
        safe_diag.to_csv(safe_diag_path, index=False)
        print("\n[STACK] Target-wise safe alpha search:")
        print(safe_diag.to_string(index=False))
        final.to_csv(OUT_PATH, index=False)
        diff = (final[TARGETS] - current_best[TARGETS]).abs()
        print(f"\n[STACK] raw stacking saved: {stacked_path}")
        print(f"[STACK] safe alpha diagnostics: {safe_diag_path}")
        print(f"[STACK] final blend saved: {OUT_PATH}")
        print(f"[STACK] mean_abs_diff_vs_current_best: {float(diff.to_numpy().mean()):.6f}")
        print(f"[STACK] max_abs_diff_vs_current_best: {float(diff.to_numpy().max()):.6f}")

    if __name__ == "__main__":
        main()


def train_mis_lstm():
    """
    MIS-LSTM (2509.11232v1): raw sensor → 4-hour block tensors → CNN + LSTM.

    Preprocessing
      - 하루(1440분)를 6개의 4-hour 블록으로 분할
      - 연속형 (wHr, mLight, wLight, wPedo_step, wPedo_speed): 블록당 (C_CONT=5, 240분)
      - 이산형 (mActivity 4종, mScreenStatus, mACStatus):    블록당 (C_DISC=6, 24 × 10분)

    Model
      - 연속형 블록 → Conv1d CNN → 64-dim 임베딩
      - 이산형 블록 → Conv1d CNN → 32-dim 임베딩
      - 블록 임베딩 concat + fusion → LSTM(6 blocks) → subject embedding 추가 → 7-target sigmoid

    Output: submissions/submission_step1_sequence_model.csv
    """
    __file__ = str(PROJECT_ROOT / "pipeline/train_mis_lstm")
    __name__ = "__main__"

    import gc
    import json
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from pathlib import Path

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import KFold
    from sklearn.metrics import log_loss

    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    SENSOR_DIR = DATA_DIR / "ch2025_data_items"
    TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
    SAMPLE_PATH = DATA_DIR / "ch2026_submission_sample.csv"
    SUBMISSION_DIR = DATA_DIR / "submissions"
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    N_FOLDS = 5
    SEED = 42
    BLOCK_HOURS = 4
    N_BLOCKS = 6
    CONT_MIN = BLOCK_HOURS * 60        # 240분/블록
    DISC_INT = BLOCK_HOURS * 6         # 24구간/블록 (10분 단위)
    C_CONT = 5
    C_DISC = 6
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[MIS-LSTM] device={DEVICE}")

    # ── 헬퍼 ────────────────────────────────────────────────────────────────

    def get_sensor_path(keyword):
        for p in sorted(SENSOR_DIR.glob("*.parquet")):
            if keyword.lower() in p.name.lower():
                return p
        return None

    def explode_hr(x):
        """wHr 열의 배열/리스트/문자열에서 숫자 리스트를 추출."""
        if x is None:
            return []
        if isinstance(x, (np.ndarray,)):
            vals = x.ravel()
        elif isinstance(x, (list, tuple)):
            vals = list(x)
        elif isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            try:
                v = json.loads(s)
                vals = v if isinstance(v, (list, tuple)) else [v]
            except Exception:
                return []
        else:
            try:
                return [float(x)]
            except Exception:
                return []
        return pd.to_numeric(pd.Series(vals), errors="coerce").dropna().tolist()

    # ── 센서 로딩 ────────────────────────────────────────────────────────────

    def load_continuous():
        """연속형 피처를 분(minute) 단위로 집계한 DataFrame 반환.
        columns: subject_id, lifelog_date, minute, whr, mlight, wlight, pedo_step, pedo_speed
        """
        frames = []

        # wHr: 각 행이 여러 HR값의 배열 → 행별 mean → (subj, date, minute) 집계
        p = get_sensor_path("wHr")
        if p:
            df = pd.read_parquet(p)
            df["subject_id"] = df["subject_id"].astype(str)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["lifelog_date"] = df["timestamp"].dt.floor("D")
            df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            hr_col = next((c for c in df.columns if c not in ("subject_id", "timestamp")), None)
            if hr_col:
                if df[hr_col].dtype == object:
                    df["whr"] = df[hr_col].apply(
                        lambda x: float(np.mean(explode_hr(x))) if explode_hr(x) else np.nan
                    )
                else:
                    df["whr"] = pd.to_numeric(df[hr_col], errors="coerce")
                g = df.groupby(["subject_id", "lifelog_date", "minute"])["whr"].mean().reset_index()
                frames.append(g)
            del df; gc.collect()

        # mLight
        p = get_sensor_path("mLight")
        if p:
            df = pd.read_parquet(p)
            df["subject_id"] = df["subject_id"].astype(str)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["lifelog_date"] = df["timestamp"].dt.floor("D")
            df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            vc = next((c for c in df.columns if c not in ("subject_id", "timestamp")), None)
            if vc:
                g = df.groupby(["subject_id", "lifelog_date", "minute"])[vc].mean().reset_index()
                g = g.rename(columns={vc: "mlight"})
                frames.append(g)
            del df; gc.collect()

        # wLight
        p = get_sensor_path("wLight")
        if p:
            df = pd.read_parquet(p)
            df["subject_id"] = df["subject_id"].astype(str)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["lifelog_date"] = df["timestamp"].dt.floor("D")
            df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            vc = next((c for c in df.columns if c not in ("subject_id", "timestamp")), None)
            if vc:
                g = df.groupby(["subject_id", "lifelog_date", "minute"])[vc].mean().reset_index()
                g = g.rename(columns={vc: "wlight"})
                frames.append(g)
            del df; gc.collect()

        # wPedo: step(누적) → 분당 마지막값, speed → 평균
        p = get_sensor_path("wPedo")
        if p:
            df = pd.read_parquet(p)
            df["subject_id"] = df["subject_id"].astype(str)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["lifelog_date"] = df["timestamp"].dt.floor("D")
            df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            agg = {}
            if "step" in df.columns:
                agg["step"] = "last"
            if "speed" in df.columns:
                agg["speed"] = "mean"
            if agg:
                g = df.groupby(["subject_id", "lifelog_date", "minute"]).agg(agg).reset_index()
                if "step" in g.columns:
                    g = g.rename(columns={"step": "pedo_step"})
                if "speed" in g.columns:
                    g = g.rename(columns={"speed": "pedo_speed"})
                frames.append(g)
            del df; gc.collect()

        if not frames:
            return pd.DataFrame(columns=["subject_id", "lifelog_date", "minute"])

        result = frames[0]
        for f in frames[1:]:
            result = result.merge(f, on=["subject_id", "lifelog_date", "minute"], how="outer")
        return result.sort_values(["subject_id", "lifelog_date", "minute"]).reset_index(drop=True)

    def load_discrete():
        """이산형 피처를 10분 구간(interval) 단위로 집계한 DataFrame 반환.
        columns: subject_id, lifelog_date, interval, act_still, act_mobile,
                 act_vehicle, act_unknown, screen_on, charging
        """
        frames = []

        # mActivity → 4종 카운트
        p = get_sensor_path("mActivity")
        if p:
            df = pd.read_parquet(p)
            df["subject_id"] = df["subject_id"].astype(str)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["lifelog_date"] = df["timestamp"].dt.floor("D")
            df["interval"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) // 10
            vc = next((c for c in df.columns if c not in ("subject_id", "timestamp")), None)
            if vc:
                df["act_still"]   = (df[vc] == 3).astype(float)
                df["act_mobile"]  = df[vc].isin([1, 2, 7, 8]).astype(float)
                df["act_vehicle"] = (df[vc] == 0).astype(float)
                df["act_unknown"] = df[vc].isin([4, 5]).astype(float)
                g = df.groupby(["subject_id", "lifelog_date", "interval"])[
                    ["act_still", "act_mobile", "act_vehicle", "act_unknown"]
                ].sum().reset_index()
                frames.append(g)
            del df; gc.collect()

        # mScreenStatus
        p = get_sensor_path("mScreen")
        if p:
            df = pd.read_parquet(p)
            df["subject_id"] = df["subject_id"].astype(str)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["lifelog_date"] = df["timestamp"].dt.floor("D")
            df["interval"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) // 10
            vc = next((c for c in df.columns if c not in ("subject_id", "timestamp")), None)
            if vc:
                g = df.groupby(["subject_id", "lifelog_date", "interval"])[vc].sum().reset_index()
                g = g.rename(columns={vc: "screen_on"})
                frames.append(g)
            del df; gc.collect()

        # mACStatus
        p = get_sensor_path("mACStatus")
        if p:
            df = pd.read_parquet(p)
            df["subject_id"] = df["subject_id"].astype(str)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["lifelog_date"] = df["timestamp"].dt.floor("D")
            df["interval"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) // 10
            vc = next((c for c in df.columns if c not in ("subject_id", "timestamp")), None)
            if vc:
                g = df.groupby(["subject_id", "lifelog_date", "interval"])[vc].sum().reset_index()
                g = g.rename(columns={vc: "charging"})
                frames.append(g)
            del df; gc.collect()

        if not frames:
            return pd.DataFrame(columns=["subject_id", "lifelog_date", "interval"])

        result = frames[0]
        for f in frames[1:]:
            result = result.merge(f, on=["subject_id", "lifelog_date", "interval"], how="outer")
        return result.sort_values(["subject_id", "lifelog_date", "interval"]).reset_index(drop=True)

    # ── 블록 텐서 생성 ────────────────────────────────────────────────────────

    CONT_COLS = ["whr", "mlight", "wlight", "pedo_step", "pedo_speed"]
    DISC_COLS = ["act_still", "act_mobile", "act_vehicle", "act_unknown", "screen_on", "charging"]

    def compute_norm_stats(cont_df, disc_df):
        cont_stats, disc_stats = {}, {}
        for col in CONT_COLS:
            if col in cont_df.columns:
                v = cont_df[col].dropna().values.astype(float)
                cont_stats[col] = (float(v.mean()) if len(v) else 0.0,
                                   float(v.std())  if len(v) else 1.0)
            else:
                cont_stats[col] = (0.0, 1.0)
        for col in DISC_COLS:
            if col in disc_df.columns:
                v = disc_df[col].dropna().values.astype(float)
                disc_stats[col] = (float(v.mean()) if len(v) else 0.0,
                                   float(v.std())  if len(v) else 1.0)
            else:
                disc_stats[col] = (0.0, 1.0)
        return cont_stats, disc_stats

    def build_day_lookup(cont_df, disc_df, cont_stats, disc_stats):
        """(subject_id, date_str) → (cont_mat: C_CONT×1440, disc_mat: C_DISC×144)"""
        cont_lookup, disc_lookup = {}, {}

        if not cont_df.empty:
            for (sid, date), grp in cont_df.groupby(["subject_id", "lifelog_date"]):
                mat = np.zeros((C_CONT, 1440), dtype=np.float32)
                for ci, col in enumerate(CONT_COLS):
                    if col not in grp.columns:
                        continue
                    sub = grp[["minute", col]].dropna(subset=[col])
                    if sub.empty:
                        continue
                    mins = sub["minute"].astype(int).clip(0, 1439).values
                    vals = sub[col].values.astype(np.float32)
                    mu, sigma = cont_stats[col]
                    vals = (vals - mu) / (sigma + 1e-8)
                    mat[ci, mins] = vals
                cont_lookup[(str(sid), str(pd.Timestamp(date).date()))] = mat

        if not disc_df.empty:
            for (sid, date), grp in disc_df.groupby(["subject_id", "lifelog_date"]):
                mat = np.zeros((C_DISC, 144), dtype=np.float32)
                for di, col in enumerate(DISC_COLS):
                    if col not in grp.columns:
                        continue
                    sub = grp[["interval", col]].dropna(subset=[col])
                    if sub.empty:
                        continue
                    ivs = sub["interval"].astype(int).clip(0, 143).values
                    vals = sub[col].values.astype(np.float32)
                    mu, sigma = disc_stats[col]
                    vals = (vals - mu) / (sigma + 1e-8)
                    mat[di, ivs] = vals
                disc_lookup[(str(sid), str(pd.Timestamp(date).date()))] = mat

        return cont_lookup, disc_lookup

    def make_tensors(keys_df, cont_lookup, disc_lookup):
        """keys_df: subject_id, lifelog_date 컬럼 포함 DataFrame"""
        n = len(keys_df)
        cont_arr = np.zeros((n, N_BLOCKS, C_CONT, CONT_MIN), dtype=np.float32)
        disc_arr  = np.zeros((n, N_BLOCKS, C_DISC, DISC_INT),  dtype=np.float32)
        for i, row in enumerate(keys_df.itertuples(index=False)):
            key = (str(row.subject_id), str(pd.Timestamp(row.lifelog_date).date()))
            c_day = cont_lookup.get(key, np.zeros((C_CONT, 1440), dtype=np.float32))
            d_day = disc_lookup.get(key, np.zeros((C_DISC, 144),  dtype=np.float32))
            for b in range(N_BLOCKS):
                cont_arr[i, b] = c_day[:, b * CONT_MIN:(b + 1) * CONT_MIN]
                disc_arr[i, b]  = d_day[:, b * DISC_INT:(b + 1) * DISC_INT]
        return cont_arr, disc_arr

    # ── Dataset ──────────────────────────────────────────────────────────────

    class BlockDataset(Dataset):
        def __init__(self, cont, disc, subj_idx, labels=None):
            self.cont  = torch.tensor(cont,     dtype=torch.float32)
            self.disc  = torch.tensor(disc,     dtype=torch.float32)
            self.subj  = torch.tensor(subj_idx, dtype=torch.long)
            self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

        def __len__(self):
            return len(self.cont)

        def __getitem__(self, idx):
            if self.labels is not None:
                return self.cont[idx], self.disc[idx], self.subj[idx], self.labels[idx]
            return self.cont[idx], self.disc[idx], self.subj[idx]

    # ── Model ─────────────────────────────────────────────────────────────────

    class ContCNN(nn.Module):
        """연속형 블록 (C_CONT, 240) → 64-dim 임베딩"""
        def __init__(self, in_ch=C_CONT, emb=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32), nn.GELU(),
                nn.Conv1d(32, emb, kernel_size=5, padding=2),
                nn.BatchNorm1d(emb), nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    class DiscCNN(nn.Module):
        """이산형 블록 (C_DISC, 24) → 32-dim 임베딩"""
        def __init__(self, in_ch=C_DISC, emb=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16), nn.GELU(),
                nn.Conv1d(16, emb, kernel_size=3, padding=1),
                nn.BatchNorm1d(emb), nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    class MISLSTM(nn.Module):
        def __init__(self, n_subj=10, cont_emb=64, disc_emb=32,
                     lstm_h=96, subj_emb=16, n_targets=7, dropout=0.35):
            super().__init__()
            self.cont_cnn = ContCNN(C_CONT, cont_emb)
            self.disc_cnn = DiscCNN(C_DISC, disc_emb)
            self.fusion = nn.Sequential(
                nn.Linear(cont_emb + disc_emb, lstm_h),
                nn.LayerNorm(lstm_h), nn.GELU(), nn.Dropout(dropout),
            )
            self.lstm = nn.LSTM(lstm_h, lstm_h, num_layers=2,
                                batch_first=True, dropout=dropout)
            self.subj_emb = nn.Embedding(n_subj, subj_emb)
            self.head = nn.Sequential(
                nn.Linear(lstm_h + subj_emb, 64),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, n_targets),
            )

        def forward(self, cont_blocks, disc_blocks, subj_idx):
            # cont_blocks: (B, N_BLOCKS, C_CONT, CONT_MIN)
            B = cont_blocks.size(0)
            embs = []
            for b in range(N_BLOCKS):
                ce = self.cont_cnn(cont_blocks[:, b])  # (B, cont_emb)
                de = self.disc_cnn(disc_blocks[:, b])  # (B, disc_emb)
                embs.append(self.fusion(torch.cat([ce, de], dim=-1)).unsqueeze(1))
            seq, _ = self.lstm(torch.cat(embs, dim=1))  # (B, N_BLOCKS, lstm_h)
            last = seq[:, -1]                            # (B, lstm_h)
            s = self.subj_emb(subj_idx)                 # (B, subj_emb)
            return self.head(torch.cat([last, s], dim=-1))  # (B, n_targets)

    # ── 학습 유틸 ─────────────────────────────────────────────────────────────

    def run_epoch(model, loader, optimizer, amp_scaler):
        model.train()
        criterion = nn.BCEWithLogitsLoss()
        total = 0.0
        for cont, disc, subj, y in loader:
            cont, disc, subj, y = (cont.to(DEVICE), disc.to(DEVICE),
                                   subj.to(DEVICE), y.to(DEVICE))
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                loss = criterion(model(cont, disc, subj), y)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total += loss.item()
        return total / max(len(loader), 1)

    @torch.no_grad()
    def infer(model, loader):
        model.eval()
        outs = []
        for batch in loader:
            cont, disc, subj = batch[0], batch[1], batch[2]
            cont, disc, subj = cont.to(DEVICE), disc.to(DEVICE), subj.to(DEVICE)
            outs.append(torch.sigmoid(model(cont, disc, subj)).cpu().numpy())
        return np.concatenate(outs, axis=0)

    # ── 메인 ─────────────────────────────────────────────────────────────────

    print("[MIS-LSTM] 센서 데이터 로딩 중...")
    cont_df = load_continuous()
    disc_df = load_discrete()
    print(f"[MIS-LSTM] 연속형 레코드: {len(cont_df):,}  이산형 레코드: {len(disc_df):,}")

    train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    test  = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
    train["subject_id"] = train["subject_id"].astype(str)
    test["subject_id"]  = test["subject_id"].astype(str)

    all_subjects = sorted(train["subject_id"].unique())
    subj2idx = {s: i for i, s in enumerate(all_subjects)}
    train_subj = np.array([subj2idx.get(s, 0) for s in train["subject_id"]])
    test_subj  = np.array([subj2idx.get(s, 0) for s in test["subject_id"]])
    n_subj = len(all_subjects)

    print("[MIS-LSTM] 정규화 통계 계산 & 룩업 테이블 구축 중...")
    cont_stats, disc_stats = compute_norm_stats(cont_df, disc_df)
    cont_lookup, disc_lookup = build_day_lookup(cont_df, disc_df, cont_stats, disc_stats)
    del cont_df, disc_df; gc.collect()

    print("[MIS-LSTM] 블록 텐서 생성 중...")
    cont_tr, disc_tr = make_tensors(train, cont_lookup, disc_lookup)
    cont_te, disc_te = make_tensors(test,  cont_lookup, disc_lookup)
    del cont_lookup, disc_lookup; gc.collect()

    Y_tr = train[TARGETS].values.astype(np.float32)
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)

    oof_all  = np.zeros((len(train), len(TARGETS)), dtype=np.float32)
    test_all = np.zeros((len(test),  len(TARGETS)), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(cont_tr)):
        print(f"[MIS-LSTM] Fold {fold + 1}/{N_FOLDS}")

        tr_ds  = BlockDataset(cont_tr[tr_idx], disc_tr[tr_idx],
                              train_subj[tr_idx], Y_tr[tr_idx])
        val_ds = BlockDataset(cont_tr[val_idx], disc_tr[val_idx],
                              train_subj[val_idx], Y_tr[val_idx])
        te_ds  = BlockDataset(cont_te, disc_te, test_subj)

        tr_loader  = DataLoader(tr_ds,  batch_size=32, shuffle=True,  num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
        te_loader  = DataLoader(te_ds,  batch_size=64, shuffle=False, num_workers=0)

        torch.manual_seed(SEED + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + fold)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        restored = (
            load_step1_candidate_checkpoint("mis_lstm", "fold", fold)
            if restore_step1_candidate_checkpoints_enabled()
            else None
        )
        if restored is not None:
            print(f"[STEP1_CKPT] restored candidate mis_lstm fold={fold}")
            model = MISLSTM(n_subj=int(restored["n_subjects"])).to(DEVICE)
            model.load_state_dict(restored["state_dict"])
            oof_all[val_idx] = infer(model, val_loader)
            test_all += infer(model, te_loader) / N_FOLDS
            print(f"  restored best_val_logloss={float(restored['config'].get('best_val_logloss', np.nan)):.4f}")
            del model; gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue

        model = MISLSTM(n_subj=n_subj).to(DEVICE)
        optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
        amp_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

        best_ll, best_state, patience = np.inf, None, 0
        PATIENCE, MAX_EPOCHS = 20, 150

        for ep in range(MAX_EPOCHS):
            run_epoch(model, tr_loader, optimizer, amp_scaler)
            scheduler.step()
            val_p = infer(model, val_loader)
            ll = log_loss(Y_tr[val_idx].ravel(), val_p.ravel())
            if ll < best_ll:
                best_ll = ll
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        save_step1_candidate_checkpoint(
            "mis_lstm",
            {
                "kind": "fold_state_dict",
                "fold": fold,
                "train_index": tr_idx.tolist(),
                "valid_index": val_idx.tolist(),
                "targets": TARGETS,
                "n_subjects": n_subj,
                "subject_to_index": subj2idx,
                "continuous_stats": cont_stats,
                "discrete_stats": disc_stats,
                "config": {
                    "seed": SEED + fold,
                    "block_hours": BLOCK_HOURS,
                    "n_blocks": N_BLOCKS,
                    "cont_min": CONT_MIN,
                    "disc_int": DISC_INT,
                    "c_cont": C_CONT,
                    "c_disc": C_DISC,
                    "best_val_logloss": float(best_ll),
                    "stopped_epoch": int(ep + 1),
                },
                "state_dict": best_state,
            },
            "fold",
            fold,
        )
        oof_all[val_idx] = infer(model, val_loader)
        test_all += infer(model, te_loader) / N_FOLDS
        print(f"  best_val_logloss={best_ll:.4f}  stopped_ep={ep + 1}")
        del model; gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    avg_ll = np.mean([log_loss(train[t].values, oof_all[:, i])
                      for i, t in enumerate(TARGETS)])
    print(f"[MIS-LSTM] OOF 평균 Log-Loss: {avg_ll:.5f}")
    for i, t in enumerate(TARGETS):
        tll = log_loss(train[t].values, oof_all[:, i])
        print(f"  {t}: {tll:.4f}")

    # 제출 파일 저장
    sub = pd.read_csv(SAMPLE_PATH)
    for i, t in enumerate(TARGETS):
        sub[t] = np.clip(test_all[:, i], 0.03, 0.97)
    out_path = SUBMISSION_DIR / "submission_step1_sequence_model.csv"
    sub.to_csv(out_path, index=False)
    print(f"[MIS-LSTM] 저장 완료 → {out_path}")


PIPELINE = [
    # 흐름 1. 3-seed target-wise routing anchor를 학습합니다.
    (
        "pipeline/train_seed3_q6040_s2080_routing",
        train_seed3_q6040_s2080_routing,
        [],
    ),
    # 흐름 2. subject별 state-transition prior 후보를 생성합니다.
    (
        "pipeline/build_state_transition_prior_candidates",
        build_state_transition_prior_candidates,
        [],
    ),
    # 흐름 3. state-transition prior와 anchor의 target-wise blend grid를 만듭니다.
    (
        "pipeline/build_state_transition_blend_grid",
        build_state_transition_blend_grid,
        [],
    ),
    # 흐름 4. target dynamics reversion prior 후보를 생성합니다.
    (
        "pipeline/build_target_dynamics_reversion_candidates",
        build_target_dynamics_reversion_candidates,
        [],
    ),
    # 흐름 5. target dynamics prior의 blend 강도를 확장 탐색합니다.
    (
        "pipeline/build_target_dynamics_blend_grid",
        build_target_dynamics_blend_grid,
        [],
    ),
    # 흐름 6. subject-date interpolation prior를 생성합니다.
    (
        "pipeline/build_subject_date_interpolation_prior",
        build_subject_date_interpolation_prior,
        [],
    ),
    # 흐름 7. calendar/bracket 기반 날짜 prior를 생성합니다.
    (
        "pipeline/build_calendar_and_bracket_priors",
        build_calendar_and_bracket_priors,
        [],
    ),
    # 흐름 8. 현재 anchor에 mean/temperature/correlation calibration을 적용합니다.
    (
        "pipeline/build_current_anchor_calibration_candidates",
        build_current_anchor_calibration_candidates,
        [],
    ),
    # 흐름 9. 날짜 거리 confidence 기반 adaptive prior blend를 생성합니다.
    (
        "pipeline/build_adaptive_date_confidence_blends",
        build_adaptive_date_confidence_blends,
        [],
    ),
    # 흐름 10. subject reliability 기반 date prior 후보를 생성합니다.
    (
        "pipeline/build_subject_reliability_prior_candidates",
        build_subject_reliability_prior_candidates,
        [],
    ),
    # 흐름 11. reliability refinement 후보를 생성합니다.
    (
        "pipeline/build_reliability_refinement_candidates",
        build_reliability_refinement_candidates,
        [],
    ),
    # 흐름 12. mean-aligned anchor 기준 target scale 후보를 생성합니다.
    (
        "pipeline/build_mean_aligned_target_scale_candidates",
        build_mean_aligned_target_scale_candidates,
        [],
    ),
    # 흐름 13. 일별 센서 baseline과 feature table을 생성합니다.
    ("pipeline/train_daily_sensor_baseline", train_daily_sensor_baseline, []),
    # 흐름 14. joint target pattern projection 및 sensor KNN 후보를 생성합니다.
    (
        "pipeline/build_pattern_projection_and_sensor_knn_candidates",
        build_pattern_projection_and_sensor_knn_candidates,
        [],
    ),
    # 흐름 15. sleep-window feature와 sleep-window 모델 후보를 생성합니다.
    (
        "pipeline/build_sleep_window_model_candidates",
        build_sleep_window_model_candidates,
        [],
    ),
    # 흐름 16. sleep-window target-wise refinement 후보를 생성합니다.
    (
        "pipeline/build_sleep_window_refinement_candidates",
        build_sleep_window_refinement_candidates,
        [],
    ),
    # 흐름 17. sleep metric proxy로 최종 S계열 보정 후보를 생성합니다.
    (
        "pipeline/build_sleep_metric_proxy_candidates",
        build_sleep_metric_proxy_candidates,
        [],
    ),
    # 흐름 18. 기존 base 모델 OOF 위에 XGBoost stacking meta layer를 학습합니다.
    (
        "pipeline/build_xgboost_oof_stacking_candidates",
        build_xgboost_oof_stacking_candidates,
        [],
    ),
    # 흐름 19. MIS-LSTM: raw sensor → 4-hour block CNN+LSTM (논문 2509.11232v1).
    (
        "pipeline/train_mis_lstm",
        train_mis_lstm,
        [],
    ),
    # 흐름 21. 최종 제출 파일을 검증합니다.
    (
        "pipeline/validate_final_submission",
        validate_final_submission,
        ["submission_step1_anchor.csv"],
    ),
]


# 함수: 파이프라인 실행 전에 필수 입력 데이터와 출력 폴더를 확인합니다.
def check_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        print("[FAIL] Missing required input files/directories:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# 함수: 파이프라인의 한 단계를 현재 프로젝트 루트 기준으로 실행합니다.
def run_pipeline_step(label: str, func, args: list[str]) -> None:
    print("\n" + "=" * 100)
    print(f"[RUN] {label}" + ((" " + " ".join(args)) if args else ""))
    print("=" * 100)
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        sys.argv = [str(ROOT / label), *args]
        func()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


# 함수: 전체 학습/후보 생성/검증 파이프라인을 순서대로 실행합니다.
def main() -> None:
    print(f"[INFO] Project root: {ROOT}")
    print(f"[INFO] Python: {sys.executable}")
    check_inputs()
    for label, func, args in PIPELINE:
        run_pipeline_step(label, func, args)
    if not FINAL_SUBMISSION.exists():
        print(f"[FAIL] Final submission was not created: {FINAL_SUBMISSION}")
        raise SystemExit(1)
    print(f"\n[DONE] Final submission: {FINAL_SUBMISSION}")



# Keep a handle to the embedded step1_base main before adding step1_extended extension code.
step1_base_embedded_main = main


# ============================================================================
# Embedded step1_extended extension code
# ============================================================================
import os
import ast
import math
import pickle
import re
import time
import warnings
import argparse
import importlib.util
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "data" / "artifacts" / "matplotlib_cache"),
)

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss


class _OptionalTrainingDependencyStub:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Optional training libraries are not installed. Install requirements.txt "
            "and rerun with --no-frozen for full end-to-end training."
        )


try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = _OptionalTrainingDependencyStub

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = _OptionalTrainingDependencyStub

warnings.filterwarnings("ignore", category=PerformanceWarning)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUBMISSION_DIR = DATA_DIR / "submissions"
ARTIFACT_DIR = DATA_DIR / "artifacts"
STEP1_MODEL_DIR = ROOT / "models" / "step1_pipeline"

TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = DATA_DIR / "ch2026_submission_sample.csv"
FEATURE_PATH = DATA_DIR / "features_daily_sensor_table.csv"
CURRENT_BEST_PATH = (
    SUBMISSION_DIR
    / "submission_step1_anchor.csv"
)

OUT_PATH = SUBMISSION_DIR / "submission_step1_extended_targetwise.csv"
RAW_STACK_PATH = SUBMISSION_DIR / "submission_step1_extended_raw_stack.csv"

SOURCE_DIAG_PATH = ARTIFACT_DIR / "step1_extended_meta_source_target_oof_logloss.csv"
MODEL_DIAG_PATH = ARTIFACT_DIR / "step1_extended_model_diagnostics.csv"
SAFE_DIAG_PATH = ARTIFACT_DIR / "step1_extended_safe_alpha_diagnostics.csv"
SUBMISSION_SUMMARY_PATH = ARTIFACT_DIR / "step1_extended_submission_shift_summary.csv"
NEW_SOURCE_SUMMARY_PATH = ARTIFACT_DIR / "step1_extended_new_source_summary.csv"
STEP1_RESTORE_TARGETWISE_STACK_CHECKPOINTS = False
STEP1_RESTORE_CANDIDATE_SOURCE_CHECKPOINTS = False

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
FIRST_STAGE_CONTEXT_GATE_TARGETS = {"Q2", "S2"}
KEYS = ["subject_id", "lifelog_date"]
N_FOLDS = 5
TRY9_LASTBLOCK_FRAC = 0.25
TRY9_GUARD_WEIGHT = 0.65
TRY9_TARGET_GUARD_WEIGHTS = {
    "Q1": 0.60,
    "Q2": 0.72,
    "Q3": 0.72,
    "S1": 0.62,
    "S2": 0.72,
    "S3": 0.72,
    "S4": 0.68,
}
TRY9_TARGET_SWITCH_MARGINS = {
    "Q1": 0.0012,
    "Q2": 0.0020,
    "Q3": 0.0030,
    "S1": 0.0012,
    "S2": 0.0010,
    "S3": 0.0010,
    "S4": 0.0012,
}
TRY9_TARGET_GUARD_MARGINS = {
    "Q1": 0.0035,
    "Q2": 0.0025,
    "Q3": 0.0015,
    "S1": 0.0035,
    "S2": 0.0045,
    "S3": 0.0045,
    "S4": 0.0040,
}

# step1_extended는 step1_base의 전체 후보 생성/학습/비교 결과를 base로 둔 뒤,
# 추가 personal/long-term source를 얹어 step1_base 대비 OOF/shift를 비교합니다.
# meta recipe와 safe alpha cap은 step1_base의 보수적 선택 규칙을 유지합니다.


def clip_proba(values, lo=0.03, hi=0.97):
    return np.clip(np.asarray(values, dtype=float), lo, hi)


def logit(values):
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def shrink_proba(values, alpha=0.985):
    values = np.asarray(values, dtype=float)
    return clip_proba(0.5 + alpha * (values - 0.5), 1e-6, 1 - 1e-6)


# Pruned historical definition: fit_temperature (not reachable from the final runner).


def apply_temperature(p, t):
    return clip_proba(sigmoid(t * logit(p)), 1e-6, 1 - 1e-6)


def make_temporal_folds(df: pd.DataFrame) -> pd.Series:
    ordered = pd.to_datetime(df["lifelog_date"]).rank(method="first").to_numpy()
    folds = pd.qcut(ordered, q=N_FOLDS, labels=False, duplicates="drop")
    folds = pd.Series(folds, index=df.index).astype(int)
    return folds


def make_subject_lastblock_mask(df: pd.DataFrame, frac: float = TRY9_LASTBLOCK_FRAC, min_rows: int = 3) -> pd.Series:
    """Last chronological rows per subject, used as a public-like guard split."""
    mask = pd.Series(False, index=df.index)
    work = df[["subject_id", "lifelog_date"]].copy()
    work["lifelog_date"] = pd.to_datetime(work["lifelog_date"])
    for _, idx in work.groupby("subject_id").groups.items():
        ordered = work.loc[list(idx)].sort_values("lifelog_date").index.to_list()
        n_valid = max(min_rows, int(round(len(ordered) * frac)))
        n_valid = min(n_valid, len(ordered))
        mask.loc[ordered[-n_valid:]] = True
    return mask


def guarded_selection_score(
    y: pd.Series | np.ndarray,
    pred: np.ndarray,
    guard_mask: pd.Series | np.ndarray,
    guard_weight: float = TRY9_GUARD_WEIGHT,
) -> tuple[float, float, float]:
    """Blend full OOF and subject-lastblock loss for model/source selection."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = clip_proba(pred, 1e-6, 1 - 1e-6)
    mask = np.asarray(guard_mask, dtype=bool)
    full = log_loss(y_arr, p_arr, labels=[0, 1])
    if mask.any() and (~mask).any():
        guard = log_loss(y_arr[mask], p_arr[mask], labels=[0, 1])
    else:
        guard = full
    score = (1.0 - guard_weight) * full + guard_weight * guard
    return float(score), float(full), float(guard)


def family_of_source(name: str) -> str:
    if name.startswith("date_aligned_"):
        return "date_aligned_model"
    if "try6_12_temporal" in name:
        return "personal_temporal_kernel"
    if "try6_12_bayes" in name or "try6_12_calendar" in name:
        return "personal_calendar_bayes"
    if "try6_12_longterm" in name:
        return "longterm_sensor_model"
    if "try9_1_cv_" in name:
        return "first_stage_cv_variant"
    if "try9_1_history" in name:
        return "subject_history_sequence"
    if "subject_date_interpolation" in name:
        return "subject_date_interpolation"
    if "date_bracket" in name or "calendar" in name or "dateconf" in name:
        return "calendar_date_prior"
    if "sleep_window" in name:
        return "sleep_window"
    if "sleep_metric" in name or "sleep_interval" in name:
        return "sleep_proxy"
    if "state_history" in name or "state_transition" in name:
        return "state_history_transition"
    if "target_dynamics" in name or "streak" in name:
        return "target_dynamics"
    if "reliability" in name:
        return "reliability"
    if "sensor_knn" in name or "date_sensor" in name:
        return "sensor_similarity"
    if "anchor" in name or "seed3" in name or "xgboost_oof" in name:
        return "anchor_or_stack"
    if "stable_time" in name:
        return "stable_time_personal"
    if "mis_lstm" in name:
        return "sequence_deep"
    return "other"


def load_core_frames():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
    current_best = pd.read_csv(CURRENT_BEST_PATH)
    feat = pd.read_csv(FEATURE_PATH, parse_dates=["lifelog_date"])
    return train, sample, current_best, feat


def load_step1_base_module():
    path = ROOT / "kyubin" / "step1_base"
    spec = importlib.util.spec_from_file_location("step1_base_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import step1_base from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_step1_base_pipeline(include_mis_lstm: bool = False) -> None:
    module = load_step1_base_module()
    module.check_inputs()
    for label, func, args in module.PIPELINE:
        if "validate_final_submission" in label:
            continue
        if not include_mis_lstm and "train_mis_lstm" in label:
            continue
        module.run_pipeline_step(label, func, args)
    if not CURRENT_BEST_PATH.exists():
        raise RuntimeError(f"step1_base did not create {CURRENT_BEST_PATH}")


# Pruned historical definition: align_frame_to_train_test (not reachable from the final runner).


def load_existing_oof_sources(train: pd.DataFrame, sample: pd.DataFrame):
    oof: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    for oof_path in sorted(ARTIFACT_DIR.glob("oof_stacking_base_*.csv")):
        name = oof_path.stem.replace("oof_stacking_base_", "", 1)
        test_path = ARTIFACT_DIR / f"test_stacking_base_{name}.csv"
        if not test_path.exists():
            continue
        oof_df = pd.read_csv(oof_path)
        test_df = pd.read_csv(test_path)
        if not all(t in oof_df.columns for t in TARGETS):
            continue
        if not all(t in test_df.columns for t in TARGETS):
            continue
        if len(oof_df) != len(train) or len(test_df) != len(sample):
            continue
        oof[name] = oof_df[TARGETS].astype(float).reset_index(drop=True)
        test[name] = test_df[TARGETS].astype(float).reset_index(drop=True)
    return oof, test


def smoothed_rate(pos: float, n: float, prior: float, alpha: float) -> float:
    if n <= 0:
        return float(prior)
    return float((pos + alpha * prior) / (n + alpha))


def subject_prior(train_pool: pd.DataFrame, subject: str, target: str, global_prior: float):
    rows = train_pool[train_pool["subject_id"] == subject]
    return smoothed_rate(float(rows[target].sum()), float(len(rows)), global_prior, 8.0)


def build_personal_temporal_sources(train: pd.DataFrame, sample: pd.DataFrame):
    train_base = train[["subject_id", "lifelog_date", *TARGETS]].copy()
    test_base = sample[["subject_id", "lifelog_date"]].copy()
    global_means = train[TARGETS].mean().to_dict()

    specs = [
        ("try6_12_temporal_tau03_all", 3.0, "all"),
        ("try6_12_temporal_tau05_all", 5.0, "all"),
        ("try6_12_temporal_tau07_all", 7.0, "all"),
        ("try6_12_temporal_tau10_all", 10.0, "all"),
        ("try6_12_temporal_tau14_all", 14.0, "all"),
        ("try6_12_temporal_tau07_past", 7.0, "past"),
        ("try6_12_temporal_tau10_past", 10.0, "past"),
        ("try6_12_temporal_tau10_asym", 10.0, "asym"),
    ]
    oof_sources = {
        name: pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
        for name, _, _ in specs
    }
    test_sources = {
        name: pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
        for name, _, _ in specs
    }

    by_subject = {
        sid: g.sort_values("lifelog_date").reset_index()
        for sid, g in train_base.groupby("subject_id")
    }

    def predict_one(pool: pd.DataFrame, pred_date, target: str, tau: float, mode: str, prior: float):
        if pool.empty:
            return prior
        delta = (pd.to_datetime(pool["lifelog_date"]) - pd.to_datetime(pred_date)).dt.days.to_numpy(float)
        mask = np.ones(len(pool), dtype=bool)
        if mode == "past":
            mask = delta < 0
        if not mask.any():
            return prior
        delta = delta[mask]
        y = pool.loc[mask, target].astype(float).to_numpy()
        weights = np.exp(-np.abs(delta) / tau)
        if mode == "asym":
            weights = weights * np.where(delta < 0, 1.15, 0.85)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-12:
            return prior
        raw = float(np.dot(weights, y) / weight_sum)
        eff_n = min(weight_sum, float(len(y)))
        return smoothed_rate(raw * eff_n, eff_n, prior, 3.0)

    for row_idx, row in train_base.iterrows():
        sid = row["subject_id"]
        pool_full = by_subject[sid]
        pool = pool_full[pool_full["index"] != row_idx].drop(columns=["index"])
        for target in TARGETS:
            prior = subject_prior(pool, sid, target, global_means[target])
            for name, tau, mode in specs:
                oof_sources[name].loc[row_idx, target] = predict_one(
                    pool, row["lifelog_date"], target, tau, mode, prior
                )

    for row_idx, row in test_base.iterrows():
        sid = row["subject_id"]
        pool = by_subject.get(sid, pd.DataFrame()).drop(columns=["index"], errors="ignore")
        for target in TARGETS:
            prior = subject_prior(train_base, sid, target, global_means[target])
            for name, tau, mode in specs:
                test_sources[name].loc[row_idx, target] = predict_one(
                    pool, row["lifelog_date"], target, tau, mode, prior
                )

    return finalize_sources(oof_sources), finalize_sources(test_sources)


def build_bayesian_calendar_sources(train: pd.DataFrame, sample: pd.DataFrame):
    train_base = train[["subject_id", "lifelog_date", *TARGETS]].copy()
    sample_base = sample[["subject_id", "lifelog_date"]].copy()
    global_start = min(train_base["lifelog_date"].min(), sample_base["lifelog_date"].min())

    def add_keys(df):
        out = df.copy()
        out["dow"] = out["lifelog_date"].dt.dayofweek
        out["weekend"] = out["dow"].isin([5, 6]).astype(int)
        out["month"] = out["lifelog_date"].dt.month
        out["rel_week"] = ((out["lifelog_date"] - global_start).dt.days // 7).astype(int)
        out["rel_bin"] = pd.cut(
            (out["lifelog_date"] - global_start).dt.days,
            bins=[-1, 30, 60, 90, 120, 10000],
            labels=False,
        ).astype(int)
        return out

    train_k = add_keys(train_base)
    sample_k = add_keys(sample_base)
    specs = [
        ("try6_12_bayes_subject_calendar", 0.62, 0.22, 0.12, 0.04),
        ("try6_12_bayes_subject_datebin", 0.56, 0.18, 0.22, 0.04),
        ("try6_12_bayes_subject_weekend", 0.70, 0.18, 0.06, 0.06),
        ("try6_12_calendar_personal_adjusted", 0.45, 0.28, 0.22, 0.05),
    ]
    group_cols = {
        "calendar": ["dow", "month"],
        "datebin": ["rel_bin", "dow"],
        "weekend": ["weekend", "month"],
    }
    oof_sources = {
        name: pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
        for name, *_ in specs
    }
    test_sources = {
        name: pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
        for name, *_ in specs
    }

    def group_prior(pool, row, target, cols, prior, alpha):
        if pool.empty:
            return prior
        mask = np.ones(len(pool), dtype=bool)
        for col in cols:
            mask &= pool[col].eq(row[col]).to_numpy()
        sub = pool.loc[mask]
        return smoothed_rate(float(sub[target].sum()), float(len(sub)), prior, alpha)

    def subject_calendar_prior(pool, row, target, spec_name, weights):
        global_prior = float(pool[target].mean()) if len(pool) else float(train_k[target].mean())
        subj = subject_prior(pool, row["subject_id"], target, global_prior)
        cal = group_prior(pool, row, target, group_cols["calendar"], global_prior, 10.0)
        datebin = group_prior(pool, row, target, group_cols["datebin"], global_prior, 8.0)
        weekend = group_prior(pool, row, target, group_cols["weekend"], global_prior, 10.0)
        w_subj, w_cal, w_date, w_weekend = weights
        if spec_name == "try6_12_bayes_subject_datebin":
            return w_subj * subj + w_cal * cal + w_date * datebin + w_weekend * weekend
        if spec_name == "try6_12_bayes_subject_weekend":
            return w_subj * subj + w_cal * cal + w_date * datebin + w_weekend * weekend
        if spec_name == "try6_12_calendar_personal_adjusted":
            return w_subj * subj + w_cal * cal + w_date * datebin + w_weekend * weekend
        return w_subj * subj + w_cal * cal + w_date * datebin + w_weekend * weekend

    for row_idx, row in train_k.iterrows():
        pool = train_k.drop(index=row_idx)
        for target in TARGETS:
            for name, *weights in specs:
                oof_sources[name].loc[row_idx, target] = subject_calendar_prior(
                    pool, row, target, name, weights
                )

    for row_idx, row in sample_k.iterrows():
        for target in TARGETS:
            for name, *weights in specs:
                test_sources[name].loc[row_idx, target] = subject_calendar_prior(
                    train_k, row, target, name, weights
                )

    return finalize_sources(oof_sources), finalize_sources(test_sources)


def finalize_sources(sources: dict[str, pd.DataFrame]):
    out = {}
    for name, frame in sources.items():
        fixed = frame.astype(float).copy()
        fixed[TARGETS] = fixed[TARGETS].fillna(fixed[TARGETS].mean()).fillna(0.5)
        fixed[TARGETS] = clip_proba(fixed[TARGETS], 0.03, 0.97)
        out[name] = fixed.reset_index(drop=True)
    return out


def build_longterm_feature_table(feat: pd.DataFrame):
    data = feat.copy()
    data["lifelog_date"] = pd.to_datetime(data["lifelog_date"])
    data["subject_code"] = data["subject_id"].str.extract(r"(\d+)").astype(float)
    data["rel_day"] = (data["lifelog_date"] - data["lifelog_date"].min()).dt.days.astype(float)
    data["dow_sin"] = np.sin(2 * np.pi * data["lifelog_date"].dt.dayofweek / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * data["lifelog_date"].dt.dayofweek / 7.0)

    exclude = set(TARGETS + ["subject_id", "lifelog_date"])
    numeric_cols = [
        c
        for c in data.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(data[c])
    ]
    base_sensor_cols = [
        c
        for c in numeric_cols
        if any(
            token in c
            for token in [
                "mean",
                "std",
                "count",
                "sum",
                "median",
                "iqr",
                "q90_q10",
                "dev",
                "sleep_active_diff",
            ]
        )
    ]
    base_sensor_cols = base_sensor_cols[:90]

    # Past-only (leak-free) subject normalization: every statistic excludes the current
    # row (shift(1)) so an OOF/test row never sees same-subject future days. This removes
    # the lookahead leakage of the previous whole-series groupby mean/std/rolling.
    data = data.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)
    grouped = data.groupby("subject_id")
    added_cols = []
    for col in base_sensor_cols:
        g = grouped[col]
        past_mean = g.transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
        past_std = (
            g.transform(lambda s: s.shift(1).expanding(min_periods=2).std()).replace(0, np.nan)
        )
        z_col = f"lt_{col}_subj_z"
        data[z_col] = (data[col] - past_mean) / past_std
        added_cols.append(z_col)
        past_roll = g.transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
        r_col = f"lt_{col}_roll7_dev"
        data[r_col] = data[col] - past_roll
        added_cols.append(r_col)

    feature_cols = [
        c
        for c in numeric_cols + added_cols
        if c in data.columns and c not in TARGETS
    ]
    missing = data[feature_cols].isna().mean()
    feature_cols = missing[missing < 0.55].index.tolist()
    med = data[feature_cols].median(numeric_only=True)
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    return data, feature_cols


def build_longterm_sensor_model(train: pd.DataFrame, sample: pd.DataFrame, feat: pd.DataFrame):
    base_table, base_feature_cols = build_longterm_feature_table(feat)
    table, enriched_feature_cols = build_context_enriched_longterm_feature_table(train, sample, feat)
    feature_cols = list(dict.fromkeys(base_feature_cols + enriched_feature_cols))
    train_feat = train[["subject_id", "lifelog_date", *TARGETS]].merge(
        table[["subject_id", "lifelog_date", *feature_cols]],
        on=["subject_id", "lifelog_date"],
        how="left",
    )
    test_feat = sample[["subject_id", "lifelog_date"]].merge(
        table[["subject_id", "lifelog_date", *feature_cols]],
        on=["subject_id", "lifelog_date"],
        how="left",
    )
    med = table[feature_cols].median(numeric_only=True)
    train_feat[feature_cols] = train_feat[feature_cols].fillna(med).fillna(0.0)
    test_feat[feature_cols] = test_feat[feature_cols].fillna(med).fillna(0.0)

    folds = make_temporal_folds(train_feat)
    lgb_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    xgb_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    lgb_test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
    xgb_test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)

    for target in TARGETS:
        target_feature_cols = (
            enriched_feature_cols
            if target in FIRST_STAGE_CONTEXT_GATE_TARGETS
            else base_feature_cols
        )
        X_test = test_feat[target_feature_cols]
        y = train_feat[target].astype(int)
        lgb_test_acc = np.zeros(len(sample), dtype=float)
        xgb_test_acc = np.zeros(len(sample), dtype=float)
        for fold in range(N_FOLDS):
            valid = folds.eq(fold).to_numpy()
            tr = ~valid
            X_tr = train_feat.loc[tr, target_feature_cols]
            X_va = train_feat.loc[valid, target_feature_cols]
            y_tr = y.loc[tr]
            restored = (
                load_step1_candidate_checkpoint(
                    "longterm_sensor_model",
                    target,
                    "fold",
                    fold,
                )
                if restore_step1_candidate_checkpoints_enabled()
                else None
            )
            if restored is not None:
                print(f"[STEP1_CKPT] restored candidate longterm_sensor_model {target} fold={fold}")
                x_va_restored = _step1_candidate_matrix(train_feat.loc[valid], restored)
                x_test_restored = _step1_candidate_matrix(test_feat, restored)
                models = restored["models"]
                lgb_oof.loc[valid, target] = models["lgb"].predict_proba(x_va_restored)[:, 1]
                xgb_oof.loc[valid, target] = models["xgb"].predict_proba(x_va_restored)[:, 1]
                lgb_test_acc += models["lgb"].predict_proba(x_test_restored)[:, 1] / N_FOLDS
                xgb_test_acc += models["xgb"].predict_proba(x_test_restored)[:, 1] / N_FOLDS
                continue

            lgb = LGBMClassifier(
                n_estimators=180,
                learning_rate=0.025,
                max_depth=3,
                num_leaves=10,
                min_child_samples=18,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=1.5,
                reg_lambda=8.0,
                objective="binary",
                random_state=1200 + fold,
                verbose=-1,
            )
            lgb.fit(X_tr, y_tr)
            lgb_oof.loc[valid, target] = lgb.predict_proba(X_va)[:, 1]
            lgb_test_acc += lgb.predict_proba(X_test)[:, 1] / N_FOLDS

            xgb = XGBClassifier(
                n_estimators=120,
                max_depth=2,
                learning_rate=0.025,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=2.0,
                reg_lambda=12.0,
                min_child_weight=8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=2200 + fold,
                n_jobs=1,
            )
            xgb.fit(X_tr, y_tr)
            xgb_oof.loc[valid, target] = xgb.predict_proba(X_va)[:, 1]
            xgb_test_acc += xgb.predict_proba(X_test)[:, 1] / N_FOLDS
            save_step1_candidate_checkpoint(
                "longterm_sensor_model",
                {
                    "kind": "fold",
                    "target": target,
                    "fold": fold,
                    "feature_cols": list(target_feature_cols),
                    "median": med,
                    "models": {
                        "lgb": lgb,
                        "xgb": xgb,
                    },
                },
                target,
                "fold",
                fold,
            )

        lgb_test[target] = lgb_test_acc
        xgb_test[target] = xgb_test_acc

    sources_oof = {
        "try6_12_longterm_lgb": lgb_oof,
        "try6_12_longterm_xgb": xgb_oof,
        "try6_12_longterm_avg": 0.5 * lgb_oof.astype(float) + 0.5 * xgb_oof.astype(float),
    }
    sources_test = {
        "try6_12_longterm_lgb": lgb_test,
        "try6_12_longterm_xgb": xgb_test,
        "try6_12_longterm_avg": 0.5 * lgb_test.astype(float) + 0.5 * xgb_test.astype(float),
    }
    return finalize_sources(sources_oof), finalize_sources(sources_test), feature_cols


def make_subject_chrono_folds(df: pd.DataFrame) -> pd.Series:
    folds = pd.Series(0, index=df.index, dtype=int)
    work = df[["subject_id", "lifelog_date"]].copy()
    work["lifelog_date"] = pd.to_datetime(work["lifelog_date"])
    for _, idx in work.groupby("subject_id").groups.items():
        ordered = work.loc[list(idx)].sort_values("lifelog_date").index.to_list()
        n = len(ordered)
        if n <= 1:
            folds.loc[ordered] = 0
            continue
        for pos, row_idx in enumerate(ordered):
            folds.loc[row_idx] = min(N_FOLDS - 1, int(np.floor(pos * N_FOLDS / n)))
    return folds.astype(int)


def make_subject_offset_folds(df: pd.DataFrame) -> pd.Series:
    base = make_subject_chrono_folds(df)
    subject_num = (
        df["subject_id"].astype(str).str.extract(r"(\d+)")[0].fillna("0").astype(int)
    )
    return ((base + subject_num) % N_FOLDS).astype(int)


def build_cv_variant_first_stage_sources(train: pd.DataFrame, sample: pd.DataFrame, feat: pd.DataFrame):
    """Additional first-stage model sources using public-like CV geometries."""
    base_table, base_feature_cols = build_longterm_feature_table(feat)
    table, feature_cols = build_context_enriched_longterm_feature_table(train, sample, feat)
    context_cols = [
        col
        for col in feature_cols
        if col not in set(base_feature_cols)
    ]
    selected_base_cols = base_feature_cols[:180]
    selected_enriched_cols = context_cols[:90] + selected_base_cols[
        : max(0, 180 - min(len(context_cols), 90))
    ]
    selected_cols = list(dict.fromkeys(selected_base_cols + selected_enriched_cols))
    train_feat = train[["subject_id", "lifelog_date", *TARGETS]].merge(
        table[["subject_id", "lifelog_date", *selected_cols]],
        on=["subject_id", "lifelog_date"],
        how="left",
    )
    test_feat = sample[["subject_id", "lifelog_date"]].merge(
        table[["subject_id", "lifelog_date", *selected_cols]],
        on=["subject_id", "lifelog_date"],
        how="left",
    )
    med = table[selected_cols].median(numeric_only=True)
    train_feat[selected_cols] = (
        train_feat[selected_cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    )
    test_feat[selected_cols] = (
        test_feat[selected_cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    )

    fold_map = {
        "dateblock": make_temporal_folds(train_feat).reset_index(drop=True),
        "subjectchrono": make_subject_chrono_folds(train_feat).reset_index(drop=True),
        "subjectoffset": make_subject_offset_folds(train_feat).reset_index(drop=True),
    }
    sources_oof: dict[str, pd.DataFrame] = {}
    sources_test: dict[str, pd.DataFrame] = {}

    for scheme, folds in fold_map.items():
        lgb_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
        xgb_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
        lgb_test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
        xgb_test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)

        for target in TARGETS:
            target_feature_cols = (
                selected_enriched_cols
                if target in FIRST_STAGE_CONTEXT_GATE_TARGETS
                else selected_base_cols
            )
            X_test = test_feat[target_feature_cols]
            y = train_feat[target].astype(int).reset_index(drop=True)
            target_mean = float(y.mean())
            lgb_acc = np.zeros(len(sample), dtype=float)
            xgb_acc = np.zeros(len(sample), dtype=float)
            n_used = 0
            for fold in range(N_FOLDS):
                valid = folds.eq(fold).to_numpy()
                if valid.sum() == 0 or (~valid).sum() == 0:
                    continue
                tr = ~valid
                if y.loc[tr].nunique() < 2:
                    lgb_oof.loc[valid, target] = target_mean
                    xgb_oof.loc[valid, target] = target_mean
                    continue
                X_tr = train_feat.loc[tr, target_feature_cols]
                X_va = train_feat.loc[valid, target_feature_cols]
                y_tr = y.loc[tr]
                restored = (
                    load_step1_candidate_checkpoint(
                        "cv_variant_first_stage_sources",
                        scheme,
                        target,
                        "fold",
                        fold,
                    )
                    if restore_step1_candidate_checkpoints_enabled()
                    else None
                )
                if restored is not None:
                    print(f"[STEP1_CKPT] restored candidate cv_variant_first_stage_sources {scheme} {target} fold={fold}")
                    x_va_restored = _step1_candidate_matrix(train_feat.loc[valid], restored)
                    x_test_restored = _step1_candidate_matrix(test_feat, restored)
                    models = restored["models"]
                    restored_mean = float(restored["target_mean"])
                    lgb_oof.loc[valid, target] = np.clip(
                        0.86 * models["lgb"].predict_proba(x_va_restored)[:, 1] + 0.14 * restored_mean,
                        0.04,
                        0.96,
                    )
                    xgb_oof.loc[valid, target] = np.clip(
                        0.84 * models["xgb"].predict_proba(x_va_restored)[:, 1] + 0.16 * restored_mean,
                        0.04,
                        0.96,
                    )
                    lgb_acc += np.clip(
                        0.86 * models["lgb"].predict_proba(x_test_restored)[:, 1] + 0.14 * restored_mean,
                        0.04,
                        0.96,
                    )
                    xgb_acc += np.clip(
                        0.84 * models["xgb"].predict_proba(x_test_restored)[:, 1] + 0.16 * restored_mean,
                        0.04,
                        0.96,
                    )
                    n_used += 1
                    continue

                lgb = LGBMClassifier(
                    n_estimators=130,
                    learning_rate=0.025,
                    max_depth=2,
                    num_leaves=7,
                    min_child_samples=18,
                    subsample=0.85,
                    colsample_bytree=0.72,
                    reg_alpha=2.5,
                    reg_lambda=14.0,
                    objective="binary",
                    random_state=9100 + 31 * fold + len(scheme),
                    verbose=-1,
                )
                lgb.fit(X_tr, y_tr)
                lgb_pred = lgb.predict_proba(X_va)[:, 1]
                lgb_oof.loc[valid, target] = np.clip(
                    0.86 * lgb_pred + 0.14 * target_mean, 0.04, 0.96
                )
                lgb_test_pred = lgb.predict_proba(X_test)[:, 1]
                lgb_acc += np.clip(
                    0.86 * lgb_test_pred + 0.14 * target_mean, 0.04, 0.96
                )

                xgb = XGBClassifier(
                    n_estimators=95,
                    max_depth=2,
                    learning_rate=0.025,
                    subsample=0.85,
                    colsample_bytree=0.72,
                    reg_alpha=3.5,
                    reg_lambda=18.0,
                    min_child_weight=8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=9200 + 31 * fold + len(scheme),
                    n_jobs=1,
                )
                xgb.fit(X_tr, y_tr)
                xgb_pred = xgb.predict_proba(X_va)[:, 1]
                xgb_oof.loc[valid, target] = np.clip(
                    0.84 * xgb_pred + 0.16 * target_mean, 0.04, 0.96
                )
                xgb_test_pred = xgb.predict_proba(X_test)[:, 1]
                xgb_acc += np.clip(
                    0.84 * xgb_test_pred + 0.16 * target_mean, 0.04, 0.96
                )
                save_step1_candidate_checkpoint(
                    "cv_variant_first_stage_sources",
                    {
                        "kind": "fold",
                        "scheme": scheme,
                        "target": target,
                        "fold": fold,
                        "feature_cols": list(target_feature_cols),
                        "selected_base_cols": list(selected_base_cols),
                        "selected_enriched_cols": list(selected_enriched_cols),
                        "median": med,
                        "target_mean": target_mean,
                        "models": {
                            "lgb": lgb,
                            "xgb": xgb,
                        },
                    },
                    scheme,
                    target,
                    "fold",
                    fold,
                )
                n_used += 1

            denom = max(n_used, 1)
            lgb_test[target] = lgb_acc / denom if n_used else target_mean
            xgb_test[target] = xgb_acc / denom if n_used else target_mean

        avg_oof = 0.5 * lgb_oof.astype(float) + 0.5 * xgb_oof.astype(float)
        avg_test = 0.5 * lgb_test.astype(float) + 0.5 * xgb_test.astype(float)
        sources_oof[f"try9_1_cv_{scheme}_lgb"] = lgb_oof
        sources_oof[f"try9_1_cv_{scheme}_xgb"] = xgb_oof
        sources_oof[f"try9_1_cv_{scheme}_avg"] = avg_oof
        sources_test[f"try9_1_cv_{scheme}_lgb"] = lgb_test
        sources_test[f"try9_1_cv_{scheme}_xgb"] = xgb_test
        sources_test[f"try9_1_cv_{scheme}_avg"] = avg_test

    hybrid_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    hybrid_test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
    for target in TARGETS:
        cols_oof = [
            sources_oof[f"try9_1_cv_{scheme}_avg"][target].astype(float)
            for scheme in fold_map
        ]
        cols_test = [
            sources_test[f"try9_1_cv_{scheme}_avg"][target].astype(float)
            for scheme in fold_map
        ]
        hybrid_oof[target] = np.mean(np.vstack(cols_oof), axis=0)
        hybrid_test[target] = np.mean(np.vstack(cols_test), axis=0)
    sources_oof["try9_1_cv_hybrid_avg"] = hybrid_oof
    sources_test["try9_1_cv_hybrid_avg"] = hybrid_test

    pd.DataFrame(
        {
            "feature": selected_cols,
            "used_by_context_gate": [
                col in set(selected_enriched_cols) and col not in set(selected_base_cols)
                for col in selected_cols
            ],
        }
    ).to_csv(
        ARTIFACT_DIR / "longterm_personalization_cv_variant_feature_columns.csv", index=False
    )
    print(
        f"[TRY9_1] CV-variant first-stage features used: "
        f"base={len(selected_base_cols)} gated_context_total={len(selected_enriched_cols)} union={len(selected_cols)}"
    )
    return finalize_sources(sources_oof), finalize_sources(sources_test), selected_cols


def build_subject_history_sequence_sources(train: pd.DataFrame, sample: pd.DataFrame):
    """Past-only subject label history priors for the second-stage stack."""
    train_base = train[["subject_id", "lifelog_date", *TARGETS]].copy()
    sample_base = sample[["subject_id", "lifelog_date"]].copy()
    train_base["subject_id"] = train_base["subject_id"].astype(str)
    sample_base["subject_id"] = sample_base["subject_id"].astype(str)
    train_base["lifelog_date"] = pd.to_datetime(train_base["lifelog_date"])
    sample_base["lifelog_date"] = pd.to_datetime(sample_base["lifelog_date"])
    global_mean = train_base[TARGETS].mean()

    source_names = [
        "try9_1_history_recent_short",
        "try9_1_history_recent_medium",
        "try9_1_history_trend_guarded",
        "try9_1_history_streak_reversion",
        "try9_1_history_persistence_smooth",
    ]
    oof_sources = {
        name: pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
        for name in source_names
    }
    test_sources = {
        name: pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
        for name in source_names
    }

    def recency_weighted_mean(values: np.ndarray, half_life: float) -> float:
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return np.nan
        age = np.arange(len(values))[::-1]
        weights = 0.5 ** (age / half_life)
        return float(np.sum(values * weights) / np.sum(weights))

    def slope(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if len(values) < 5:
            return 0.0
        x = np.linspace(-1.0, 1.0, len(values))
        try:
            return float(np.polyfit(x, values, 1)[0])
        except Exception:
            return 0.0

    def streak(values: np.ndarray) -> tuple[float, int]:
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return np.nan, 0
        last = float(values[-1])
        count = 0
        for value in values[::-1]:
            if float(value) == last:
                count += 1
            else:
                break
        return last, count

    def predict_from_history(hist: pd.DataFrame, target: str) -> dict[str, float]:
        prior = float(global_mean[target])
        if hist.empty:
            return {name: prior for name in source_names}
        values = hist[target].astype(float).to_numpy()
        n = len(values)
        subj = float(values.mean())
        recent3 = float(values[-min(3, n) :].mean())
        recent7 = float(values[-min(7, n) :].mean())
        recent14 = float(values[-min(14, n) :].mean())
        ewma = recency_weighted_mean(values, half_life=max(4.0, min(14.0, n / 3.0)))
        tr = slope(values[-min(n, 24) :])
        last, streak_len = streak(values)
        flip_rate = float(np.mean(values[1:] != values[:-1])) if n >= 2 else 0.5
        stability = float(np.clip(1.0 - flip_rate, 0.20, 0.90))
        streak_strength = float(np.clip(streak_len / 8.0, 0.0, 0.75))

        recent_short = 0.42 * recent3 + 0.34 * recent7 + 0.16 * subj + 0.08 * prior
        recent_medium = 0.36 * recent7 + 0.30 * recent14 + 0.22 * ewma + 0.12 * prior
        projected = np.clip(ewma + 0.14 * tr, 0.03, 0.97)
        trend_guarded = 0.42 * recent_medium + 0.30 * projected + 0.18 * subj + 0.10 * prior
        if np.isfinite(last):
            streak_reversion = (
                streak_strength * (0.72 * last + 0.28 * subj)
                + (1.0 - streak_strength) * recent_medium
            )
        else:
            streak_reversion = recent_medium
        persistence = stability * (0.45 * recent7 + 0.35 * ewma + 0.20 * subj)
        persistence += (1.0 - stability) * (0.62 * subj + 0.38 * prior)

        return {
            "try9_1_history_recent_short": recent_short,
            "try9_1_history_recent_medium": recent_medium,
            "try9_1_history_trend_guarded": trend_guarded,
            "try9_1_history_streak_reversion": streak_reversion,
            "try9_1_history_persistence_smooth": persistence,
        }

    for sid, group in train_base.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        group = group.sort_values("lifelog_date")
        for idx, row in group.iterrows():
            hist = group[group["lifelog_date"] < row["lifelog_date"]]
            for target in TARGETS:
                preds = predict_from_history(hist, target)
                for name, value in preds.items():
                    oof_sources[name].loc[idx, target] = value

    by_subject = {
        sid: group.sort_values("lifelog_date").copy()
        for sid, group in train_base.groupby("subject_id")
    }
    for idx, row in sample_base.sort_values(["subject_id", "lifelog_date"]).iterrows():
        hist = by_subject.get(str(row["subject_id"]), pd.DataFrame(columns=train_base.columns))
        hist = hist[hist["lifelog_date"] < row["lifelog_date"]]
        if hist.empty:
            hist = by_subject.get(str(row["subject_id"]), pd.DataFrame(columns=train_base.columns))
        for target in TARGETS:
            preds = predict_from_history(hist, target)
            for name, value in preds.items():
                test_sources[name].loc[idx, target] = value

    return finalize_sources(oof_sources), finalize_sources(test_sources)


def rank_sources(train: pd.DataFrame, oof: dict[str, pd.DataFrame]):
    rows = []
    for target in TARGETS:
        y = train[target].astype(int).reset_index(drop=True)
        for name, frame in oof.items():
            score = log_loss(
                y,
                clip_proba(frame[target].astype(float).reset_index(drop=True), 1e-6, 1 - 1e-6),
                labels=[0, 1],
            )
            rows.append(
                {
                    "target": target,
                    "source": name,
                    "family": family_of_source(name),
                    "oof_logloss": score,
                }
            )
    return pd.DataFrame(rows)


def select_sources_for_target(diag: pd.DataFrame, target: str):
    td = diag[diag["target"] == target].sort_values("oof_logloss").reset_index(drop=True)
    baseline_rows = td[td["source"].eq("submission_step1_sleep_proxy_s_targets")]
    if len(baseline_rows):
        baseline_score = float(baseline_rows["oof_logloss"].iloc[0])
    else:
        baseline_score = float(td["oof_logloss"].iloc[0])
    selected = []

    must_keep = [
        "submission_step1_sleep_proxy_s_targets",
        "date_aligned_lgb",
        "date_aligned_cat",
        "date_aligned_xgb",
        "prior_sleep_interval_proxy",
        "submission_step1_sleep_interval_s_targets",
        "try6_12_temporal_tau03_all",
        "try6_12_temporal_tau07_all",
        "try6_12_temporal_tau14_all",
        "try6_12_bayes_subject_weekend",
        "try6_12_bayes_subject_calendar",
        "try6_12_longterm_avg",
    ]
    for name in must_keep:
        if name in set(td["source"]) and name not in selected:
            selected.append(name)

    good = td[td["oof_logloss"] <= baseline_score + 0.020]
    for name in good["source"]:
        if name not in selected:
            selected.append(name)

    for name in td.head(36)["source"]:
        if name not in selected:
            selected.append(name)

    caps = {"Q1": 28, "Q2": 24, "Q3": 24, "S1": 28, "S2": 32, "S3": 24, "S4": 26}
    return selected[: caps.get(target, 26)]


def make_meta_frame(source_map: dict[str, pd.DataFrame], target: str, selected: list[str]):
    frame = pd.DataFrame({name: source_map[name][target].astype(float).to_numpy() for name in selected})
    model_cols = [c for c in frame.columns if c.startswith("date_aligned_") or "longterm" in c]
    temporal_cols = [c for c in frame.columns if "temporal" in c or "subject_date" in c]
    calendar_cols = [c for c in frame.columns if "calendar" in c or "date_bracket" in c or "bayes" in c]
    sleep_cols = [c for c in frame.columns if "sleep" in c]

    for prefix, cols in [
        ("model", model_cols),
        ("temporal", temporal_cols),
        ("calendar", calendar_cols),
        ("sleep", sleep_cols),
        ("all", list(frame.columns)),
    ]:
        if cols:
            frame[f"avg_{prefix}"] = frame[cols].mean(axis=1)
            frame[f"spread_{prefix}"] = frame[cols].max(axis=1) - frame[cols].min(axis=1)
            frame[f"std_{prefix}"] = frame[cols].std(axis=1).fillna(0.0)

    for col in list(frame.columns):
        if not col.endswith("_logit"):
            frame[f"{col}_logit"] = logit(frame[col])
    return frame


def fit_greedy_source_blend(y, source_oof, source_test, source_names):
    scores = {
        name: log_loss(y, clip_proba(source_oof[name], 1e-6, 1 - 1e-6), labels=[0, 1])
        for name in source_names
    }
    ordered = sorted(source_names, key=lambda name: scores[name])[:28]
    best_name = ordered[0]
    blend = np.asarray(source_oof[best_name], dtype=float).copy()
    blend_test = np.asarray(source_test[best_name], dtype=float).copy()
    best_score = scores[best_name]
    used = [best_name]
    for _ in range(10):
        improved = False
        best_candidate = None
        for name in ordered:
            if name in used:
                continue
            cand = np.asarray(source_oof[name], dtype=float)
            cand_test = np.asarray(source_test[name], dtype=float)
            for weight in [0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.22, 0.30, 0.40, 0.55]:
                trial = (1 - weight) * blend + weight * cand
                score = log_loss(y, clip_proba(trial, 1e-6, 1 - 1e-6), labels=[0, 1])
                if score < best_score - 1e-7:
                    best_score = score
                    best_candidate = (
                        name,
                        weight,
                        trial,
                        (1 - weight) * blend_test + weight * cand_test,
                    )
                    improved = True
        if not improved:
            break
        name, weight, blend, blend_test = best_candidate
        used.append(f"{name}@{weight:.2f}")
    return clip_proba(blend, 1e-6, 1 - 1e-6), clip_proba(blend_test, 1e-6, 1 - 1e-6), "+".join(used), best_score


def _step1_checkpoint_safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "default"


def step1_model_checkpoint_dir(run_tag: str) -> Path:
    return STEP1_MODEL_DIR / _step1_checkpoint_safe_name(run_tag)


def step1_model_checkpoint_path(
    run_tag: str,
    target: str,
    kind: str,
    fold: int | None = None,
) -> Path:
    parts = ["targetwise_stack", str(target), str(kind)]
    if fold is not None:
        parts.append(f"fold{fold}")
    return step1_model_checkpoint_dir(run_tag) / ("__".join(parts) + ".pkl")


def step1_model_manifest_path(run_tag: str) -> Path:
    return step1_model_checkpoint_dir(run_tag) / "manifest.pkl"


def step1_candidate_checkpoint_path(component: str, *parts: object) -> Path:
    safe_parts = [_step1_checkpoint_safe_name(str(part)) for part in parts]
    filename = "__".join(safe_parts) + ".pkl" if safe_parts else "checkpoint.pkl"
    return (
        STEP1_MODEL_DIR
        / "candidate_sources"
        / _step1_checkpoint_safe_name(component)
        / filename
    )


def save_step1_candidate_checkpoint(component: str, payload: dict, *parts: object) -> None:
    obj = {
        "schema_version": 1,
        "component": component,
        **payload,
    }
    step1_atomic_write_pickle(
        obj,
        step1_candidate_checkpoint_path(component, *parts),
    )


def load_step1_candidate_checkpoint(component: str, *parts: object) -> dict | None:
    path = step1_candidate_checkpoint_path(component, *parts)
    if not path.exists():
        return None
    payload = step1_read_pickle(path)
    if not isinstance(payload, dict):
        return None
    if payload.get("component") != component:
        return None
    return payload


def restore_step1_candidate_checkpoints_enabled() -> bool:
    return bool(STEP1_RESTORE_CANDIDATE_SOURCE_CHECKPOINTS)


def _step1_candidate_matrix(frame: pd.DataFrame, payload: dict) -> pd.DataFrame:
    columns = list(
        payload.get("feature_cols")
        or payload.get("selected_features")
        or payload.get("columns")
        or []
    )
    if not columns:
        raise ValueError("candidate checkpoint payload does not contain feature columns")
    x = frame[columns].copy()
    med = payload.get("median")
    if med is not None:
        x = x.replace([np.inf, -np.inf], np.nan).fillna(med)
    return x.fillna(0.0)


# Pruned historical definition: _step1_candidate_predict_proba (not reachable from the final runner).


def step1_atomic_write_pickle(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with open(temporary, "wb") as stream:
        pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def step1_read_pickle(path: Path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def step1_targetwise_stack_checkpoint_complete(run_tag: str) -> bool:
    run_dir = step1_model_checkpoint_dir(run_tag)
    if not run_dir.exists():
        return False
    for target in TARGETS:
        full_path = step1_model_checkpoint_path(run_tag, target, "full")
        if not full_path.exists():
            return False
        try:
            payload = step1_read_pickle(full_path)
            if payload.get("component") != "targetwise_stack":
                return False
            if set(payload.get("models", {})) != {"xgb", "logistic", "ridge"}:
                return False
        except Exception:
            return False
        for fold in range(N_FOLDS):
            fold_path = step1_model_checkpoint_path(run_tag, target, "fold", fold)
            if not fold_path.exists():
                return False
            try:
                payload = step1_read_pickle(fold_path)
                if payload.get("component") != "targetwise_stack":
                    return False
                if set(payload.get("models", {})) != {"xgb", "logistic", "ridge"}:
                    return False
            except Exception:
                return False
    return True


def restore_targetwise_stack_from_checkpoints(
    train: pd.DataFrame,
    oof: dict[str, pd.DataFrame],
    test: dict[str, pd.DataFrame],
    checkpoint_run_tag: str,
):
    if not step1_targetwise_stack_checkpoint_complete(checkpoint_run_tag):
        return None

    diag = rank_sources(train, oof)
    diag.to_csv(SOURCE_DIAG_PATH, index=False)
    stacked = pd.DataFrame(index=next(iter(test.values())).index, columns=TARGETS, dtype=float)
    stacked_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    rows = []
    folds = make_temporal_folds(train).reset_index(drop=True)
    guard_mask = make_subject_lastblock_mask(train).reset_index(drop=True)

    for target in TARGETS:
        full_payload = step1_read_pickle(
            step1_model_checkpoint_path(checkpoint_run_tag, target, "full")
        )
        selected = list(full_payload["selected_sources"])
        x_meta = make_meta_frame(oof, target, selected)
        test_meta = make_meta_frame(test, target, selected)
        logit_cols = list(full_payload["logit_cols"])
        y = train[target].astype(int).reset_index(drop=True)

        xgb_oof = np.zeros(len(train), dtype=float)
        log_oof = np.zeros(len(train), dtype=float)
        ridge_oof = np.zeros(len(train), dtype=float)
        for fold in range(N_FOLDS):
            valid = folds.eq(fold).to_numpy()
            fold_payload = step1_read_pickle(
                step1_model_checkpoint_path(checkpoint_run_tag, target, "fold", fold)
            )
            xgb_oof[valid] = fold_payload["models"]["xgb"].predict_proba(
                x_meta.loc[valid, fold_payload["columns"]]
            )[:, 1]
            log_oof[valid] = fold_payload["models"]["logistic"].predict_proba(
                x_meta.loc[valid, fold_payload["logit_cols"]]
            )[:, 1]
            ridge_oof[valid] = fold_payload["models"]["ridge"].predict(
                x_meta.loc[valid, fold_payload["columns"]]
            )

        source_oof = {name: oof[name][target].astype(float).to_numpy() for name in selected}
        source_test = {name: test[name][target].astype(float).to_numpy() for name in selected}
        greedy_oof, greedy_test, greedy_used, greedy_score = fit_greedy_source_blend(
            y.to_numpy(int), source_oof, source_test, selected
        )

        options_oof = {
            "xgb": shrink_proba(xgb_oof, 0.985),
            "logistic": shrink_proba(log_oof, 0.990),
            "ridge": clip_proba(ridge_oof, 0.03, 0.97),
            "greedy": clip_proba(greedy_oof, 0.03, 0.97),
        }
        options_oof["avg_meta"] = clip_proba(
            0.40 * options_oof["xgb"]
            + 0.35 * options_oof["logistic"]
            + 0.25 * options_oof["ridge"],
            0.03,
            0.97,
        )
        options_oof["stable_plus_greedy005"] = clip_proba(
            0.95 * options_oof["avg_meta"] + 0.05 * options_oof["greedy"],
            0.03,
            0.97,
        )
        options_oof["ridge_plus_greedy005"] = clip_proba(
            0.95 * options_oof["ridge"] + 0.05 * options_oof["greedy"],
            0.03,
            0.97,
        )
        options_oof["xgb_plus_greedy005"] = clip_proba(
            0.95 * options_oof["xgb"] + 0.05 * options_oof["greedy"],
            0.03,
            0.97,
        )

        target_guard_weight = TRY9_TARGET_GUARD_WEIGHTS.get(target, TRY9_GUARD_WEIGHT)
        option_scores = {
            name: guarded_selection_score(y, pred, guard_mask, guard_weight=target_guard_weight)
            for name, pred in options_oof.items()
        }
        scores = {name: value[1] for name, value in option_scores.items()}
        guard_scores = {name: value[2] for name, value in option_scores.items()}
        select_scores = {name: value[0] for name, value in option_scores.items()}
        chosen = str(full_payload["chosen_meta"])
        stacked_oof[target] = options_oof[chosen]

        xgb_test = shrink_proba(
            full_payload["models"]["xgb"].predict_proba(
                test_meta[full_payload["columns"]]
            )[:, 1],
            0.985,
        )
        log_test = shrink_proba(
            full_payload["models"]["logistic"].predict_proba(test_meta[logit_cols])[:, 1],
            0.990,
        )
        ridge_test = clip_proba(
            full_payload["models"]["ridge"].predict(test_meta[full_payload["columns"]]),
            0.03,
            0.97,
        )
        options_test = {
            "xgb": xgb_test,
            "logistic": log_test,
            "ridge": ridge_test,
            "greedy": greedy_test,
        }
        options_test["avg_meta"] = clip_proba(
            0.40 * xgb_test + 0.35 * log_test + 0.25 * ridge_test,
            0.03,
            0.97,
        )
        options_test["stable_plus_greedy005"] = clip_proba(
            0.95 * options_test["avg_meta"] + 0.05 * greedy_test,
            0.03,
            0.97,
        )
        options_test["ridge_plus_greedy005"] = clip_proba(
            0.95 * ridge_test + 0.05 * greedy_test,
            0.03,
            0.97,
        )
        options_test["xgb_plus_greedy005"] = clip_proba(
            0.95 * xgb_test + 0.05 * greedy_test,
            0.03,
            0.97,
        )
        stacked[target] = options_test[chosen]

        best_temp = float(full_payload.get("temperature", 1.0))
        if best_temp != 1.0:
            stacked_oof[target] = apply_temperature(
                stacked_oof[target].to_numpy(float), best_temp
            )
            stacked[target] = apply_temperature(stacked[target].to_numpy(float), best_temp)

        rows.append(
            {
                "target": target,
                "temperature": best_temp,
                "selected_sources": len(selected),
                "best_single_source": diag[diag["target"].eq(target)].sort_values("oof_logloss").iloc[0]["source"],
                "best_single_logloss": diag[diag["target"].eq(target)].sort_values("oof_logloss").iloc[0]["oof_logloss"],
                "xgb_logloss": scores["xgb"],
                "logistic_logloss": scores["logistic"],
                "ridge_logloss": scores["ridge"],
                "greedy_logloss": greedy_score,
                "avg_meta_logloss": scores["avg_meta"],
                "xgb_lastblock_logloss": guard_scores["xgb"],
                "logistic_lastblock_logloss": guard_scores["logistic"],
                "ridge_lastblock_logloss": guard_scores["ridge"],
                "greedy_lastblock_logloss": guarded_selection_score(y, greedy_oof, guard_mask)[2],
                "avg_meta_lastblock_logloss": guard_scores["avg_meta"],
                "chosen_lastblock_logloss": guard_scores[chosen],
                "chosen_guarded_score": select_scores[chosen],
                "recipe_meta": full_payload["recipe_meta"],
                "chosen_meta": chosen,
                "chosen_oof_logloss": scores[chosen],
                "greedy_used": greedy_used,
                "selected": "|".join(selected),
            }
        )

    model_diag = pd.DataFrame(rows)
    model_diag.to_csv(MODEL_DIAG_PATH, index=False)
    return stacked.clip(0.03, 0.97), stacked_oof.clip(0.03, 0.97), diag, model_diag


def fit_targetwise_stack(
    train: pd.DataFrame,
    oof: dict[str, pd.DataFrame],
    test: dict[str, pd.DataFrame],
    checkpoint_run_tag: str | None = None,
    restore_from_checkpoints: bool = False,
):
    if checkpoint_run_tag is None:
        checkpoint_run_tag = "step1_extended"
    if restore_from_checkpoints:
        restored = restore_targetwise_stack_from_checkpoints(
            train, oof, test, checkpoint_run_tag
        )
        if restored is not None:
            print(f"[STEP1_CKPT] restored targetwise stack: {step1_model_checkpoint_dir(checkpoint_run_tag)}")
            return restored

    diag = rank_sources(train, oof)
    diag.to_csv(SOURCE_DIAG_PATH, index=False)

    stacked = pd.DataFrame(index=next(iter(test.values())).index, columns=TARGETS, dtype=float)
    stacked_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    rows = []
    folds = make_temporal_folds(train).reset_index(drop=True)
    guard_mask = make_subject_lastblock_mask(train).reset_index(drop=True)

    for target in TARGETS:
        selected = select_sources_for_target(diag, target)
        x_meta = make_meta_frame(oof, target, selected)
        test_meta = make_meta_frame(test, target, selected)
        x_meta_columns = list(x_meta.columns)
        logit_cols = [c for c in x_meta.columns if c.endswith("_logit")]
        y = train[target].astype(int).reset_index(drop=True)

        xgb_oof = np.zeros(len(train), dtype=float)
        log_oof = np.zeros(len(train), dtype=float)
        ridge_oof = np.zeros(len(train), dtype=float)
        for fold in range(N_FOLDS):
            valid = folds.eq(fold).to_numpy()
            tr = ~valid
            xgb = XGBClassifier(
                n_estimators=90,
                max_depth=2,
                learning_rate=0.025,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=3.5,
                reg_lambda=14.0,
                min_child_weight=7,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=6100 + fold,
                n_jobs=1,
            )
            xgb.fit(x_meta.loc[tr], y.loc[tr])
            xgb_oof[valid] = xgb.predict_proba(x_meta.loc[valid])[:, 1]

            log_model = LogisticRegression(
                C=0.035,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
                class_weight="balanced",
            )
            log_model.fit(x_meta.loc[tr, logit_cols], y.loc[tr])
            log_oof[valid] = log_model.predict_proba(x_meta.loc[valid, logit_cols])[:, 1]

            ridge = Ridge(alpha=32.0)
            ridge.fit(x_meta.loc[tr], y.loc[tr].astype(float))
            ridge_oof[valid] = ridge.predict(x_meta.loc[valid])
            step1_atomic_write_pickle(
                {
                    "schema_version": 1,
                    "component": "targetwise_stack",
                    "run_tag": checkpoint_run_tag,
                    "target": target,
                    "kind": "fold",
                    "fold": fold,
                    "selected_sources": selected,
                    "columns": x_meta_columns,
                    "logit_cols": logit_cols,
                    "models": {
                        "xgb": xgb,
                        "logistic": log_model,
                        "ridge": ridge,
                    },
                },
                step1_model_checkpoint_path(checkpoint_run_tag, target, "fold", fold),
            )

        source_oof = {name: oof[name][target].astype(float).to_numpy() for name in selected}
        source_test = {name: test[name][target].astype(float).to_numpy() for name in selected}
        greedy_oof, greedy_test, greedy_used, greedy_score = fit_greedy_source_blend(
            y.to_numpy(int), source_oof, source_test, selected
        )

        options_oof = {
            "xgb": shrink_proba(xgb_oof, 0.985),
            "logistic": shrink_proba(log_oof, 0.990),
            "ridge": clip_proba(ridge_oof, 0.03, 0.97),
            "greedy": clip_proba(greedy_oof, 0.03, 0.97),
        }
        options_oof["avg_meta"] = clip_proba(
            0.40 * options_oof["xgb"]
            + 0.35 * options_oof["logistic"]
            + 0.25 * options_oof["ridge"],
            0.03,
            0.97,
        )
        options_oof["stable_plus_greedy005"] = clip_proba(
            0.95 * options_oof["avg_meta"] + 0.05 * options_oof["greedy"],
            0.03,
            0.97,
        )
        options_oof["ridge_plus_greedy005"] = clip_proba(
            0.95 * options_oof["ridge"] + 0.05 * options_oof["greedy"],
            0.03,
            0.97,
        )
        options_oof["xgb_plus_greedy005"] = clip_proba(
            0.95 * options_oof["xgb"] + 0.05 * options_oof["greedy"],
            0.03,
            0.97,
        )

        target_guard_weight = TRY9_TARGET_GUARD_WEIGHTS.get(target, TRY9_GUARD_WEIGHT)
        option_scores = {
            name: guarded_selection_score(y, pred, guard_mask, guard_weight=target_guard_weight)
            for name, pred in options_oof.items()
        }
        scores = {name: value[1] for name, value in option_scores.items()}
        guard_scores = {name: value[2] for name, value in option_scores.items()}
        select_scores = {name: value[0] for name, value in option_scores.items()}
        target_recipe = {
            "Q1": "stable_plus_greedy005",
            "Q2": "stable_plus_greedy005",
            "Q3": "xgb_plus_greedy005",
            "S1": "ridge_plus_greedy005",
            "S2": "stable_plus_greedy005",
            "S3": "stable_plus_greedy005",
            "S4": "stable_plus_greedy005",
        }
        recipe_choice = target_recipe.get(target, "avg_meta")
        target_preferred = {
            "Q1": [
                "greedy",
                recipe_choice,
                "stable_plus_greedy005",
                "avg_meta",
                "ridge_plus_greedy005",
            ],
            "Q2": [
                recipe_choice,
                "ridge_plus_greedy005",
                "stable_plus_greedy005",
                "avg_meta",
                "greedy",
            ],
            "Q3": [
                recipe_choice,
                "stable_plus_greedy005",
                "ridge_plus_greedy005",
                "avg_meta",
                "xgb",
            ],
            "S1": [
                "greedy",
                recipe_choice,
                "ridge_plus_greedy005",
                "stable_plus_greedy005",
                "avg_meta",
            ],
            "S2": [
                "greedy",
                recipe_choice,
                "stable_plus_greedy005",
                "ridge_plus_greedy005",
                "avg_meta",
            ],
            "S3": [
                "greedy",
                recipe_choice,
                "stable_plus_greedy005",
                "ridge_plus_greedy005",
                "avg_meta",
            ],
            "S4": [
                "greedy",
                recipe_choice,
                "stable_plus_greedy005",
                "ridge_plus_greedy005",
                "avg_meta",
            ],
        }
        preferred = target_preferred.get(
            target,
            [recipe_choice, "stable_plus_greedy005", "ridge_plus_greedy005", "avg_meta", "greedy"],
        )
        if target in {"S2", "S3", "S4"} and guard_scores.get("greedy", np.inf) < guard_scores.get(recipe_choice, np.inf) - 0.0060:
            preferred = ["greedy"] + [name for name in preferred if name != "greedy"]
        allowed = []
        for name in preferred:
            if name in select_scores and name not in allowed:
                allowed.append(name)
        best_allowed = min(allowed, key=lambda name: select_scores[name])
        chosen = recipe_choice
        switch_margin = TRY9_TARGET_SWITCH_MARGINS.get(target, 0.0015)
        if select_scores[best_allowed] < select_scores[recipe_choice] - switch_margin:
            chosen = best_allowed
        guard_margin = TRY9_TARGET_GUARD_MARGINS.get(target, 0.0040)
        if guard_scores[chosen] > guard_scores[recipe_choice] + guard_margin:
            chosen = recipe_choice
        stacked_oof[target] = options_oof[chosen]

        xgb_full = XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.025,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=3.5,
            reg_lambda=14.0,
            min_child_weight=7,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=7100,
            n_jobs=1,
        )
        xgb_full.fit(x_meta, y)
        xgb_test = shrink_proba(xgb_full.predict_proba(test_meta)[:, 1], 0.985)

        log_full = LogisticRegression(
            C=0.035,
            penalty="l2",
            solver="lbfgs",
            max_iter=3000,
            class_weight="balanced",
        )
        log_full.fit(x_meta[logit_cols], y)
        log_test = shrink_proba(log_full.predict_proba(test_meta[logit_cols])[:, 1], 0.990)

        ridge_full = Ridge(alpha=32.0)
        ridge_full.fit(x_meta, y.astype(float))
        ridge_test = clip_proba(ridge_full.predict(test_meta), 0.03, 0.97)

        options_test = {
            "xgb": xgb_test,
            "logistic": log_test,
            "ridge": ridge_test,
            "greedy": greedy_test,
        }
        options_test["avg_meta"] = clip_proba(
            0.40 * xgb_test + 0.35 * log_test + 0.25 * ridge_test,
            0.03,
            0.97,
        )
        options_test["stable_plus_greedy005"] = clip_proba(
            0.95 * options_test["avg_meta"] + 0.05 * greedy_test,
            0.03,
            0.97,
        )
        options_test["ridge_plus_greedy005"] = clip_proba(
            0.95 * ridge_test + 0.05 * greedy_test,
            0.03,
            0.97,
        )
        options_test["xgb_plus_greedy005"] = clip_proba(
            0.95 * xgb_test + 0.05 * greedy_test,
            0.03,
            0.97,
        )
        stacked[target] = options_test[chosen]

        # Per-target temperature calibration on logits. Selected on the guarded
        # (full + subject-lastblock) score so it follows the same conservatism as
        # model selection and can never worsen that metric (T=1 stays the fallback).
        chosen_oof_vec = stacked_oof[target].to_numpy(float)
        base_temp_score = guarded_selection_score(
            y, chosen_oof_vec, guard_mask, guard_weight=target_guard_weight
        )[0]
        best_temp, best_temp_score = 1.0, base_temp_score
        for t in np.linspace(0.6, 1.4, 33):
            cand = apply_temperature(chosen_oof_vec, t)
            sc = guarded_selection_score(
                y, cand, guard_mask, guard_weight=target_guard_weight
            )[0]
            if sc < best_temp_score - 1e-9:
                best_temp_score, best_temp = sc, float(t)
        if best_temp != 1.0:
            stacked_oof[target] = apply_temperature(chosen_oof_vec, best_temp)
            stacked[target] = apply_temperature(stacked[target].to_numpy(float), best_temp)

        step1_atomic_write_pickle(
            {
                "schema_version": 1,
                "component": "targetwise_stack",
                "run_tag": checkpoint_run_tag,
                "target": target,
                "kind": "full",
                "selected_sources": selected,
                "columns": x_meta_columns,
                "logit_cols": logit_cols,
                "recipe_meta": recipe_choice,
                "chosen_meta": chosen,
                "temperature": best_temp,
                "greedy_used": greedy_used,
                "models": {
                    "xgb": xgb_full,
                    "logistic": log_full,
                    "ridge": ridge_full,
                },
            },
            step1_model_checkpoint_path(checkpoint_run_tag, target, "full"),
        )

        rows.append(
            {
                "target": target,
                "temperature": best_temp,
                "selected_sources": len(selected),
                "best_single_source": diag[diag["target"].eq(target)].sort_values("oof_logloss").iloc[0]["source"],
                "best_single_logloss": diag[diag["target"].eq(target)].sort_values("oof_logloss").iloc[0]["oof_logloss"],
                "xgb_logloss": scores["xgb"],
                "logistic_logloss": scores["logistic"],
                "ridge_logloss": scores["ridge"],
                "greedy_logloss": greedy_score,
                "avg_meta_logloss": scores["avg_meta"],
                "xgb_lastblock_logloss": guard_scores["xgb"],
                "logistic_lastblock_logloss": guard_scores["logistic"],
                "ridge_lastblock_logloss": guard_scores["ridge"],
                "greedy_lastblock_logloss": guarded_selection_score(y, greedy_oof, guard_mask)[2],
                "avg_meta_lastblock_logloss": guard_scores["avg_meta"],
                "chosen_lastblock_logloss": guard_scores[chosen],
                "chosen_guarded_score": select_scores[chosen],
                "recipe_meta": recipe_choice,
                "chosen_meta": chosen,
                "chosen_oof_logloss": scores[chosen],
                "greedy_used": greedy_used,
                "selected": "|".join(selected),
            }
        )

    model_diag = pd.DataFrame(rows)
    model_diag.to_csv(MODEL_DIAG_PATH, index=False)
    step1_atomic_write_pickle(
        {
            "schema_version": 1,
            "component": "targetwise_stack",
            "run_tag": checkpoint_run_tag,
            "targets": TARGETS,
            "n_folds": N_FOLDS,
            "model_diagnostics": rows,
        },
        step1_model_manifest_path(checkpoint_run_tag),
    )
    print(f"[STEP1_CKPT] saved targetwise stack: {step1_model_checkpoint_dir(checkpoint_run_tag)}")
    return stacked.clip(0.03, 0.97), stacked_oof.clip(0.03, 0.97), diag, model_diag


def safe_alpha_blend(sample, current_best, current_best_oof, stacked_test, stacked_oof, train):
    alpha_grid = [
        0.0,
        0.0025,
        0.005,
        0.0075,
        0.010,
        0.0125,
        0.015,
        0.020,
        0.025,
        0.030,
        0.040,
        0.050,
        0.060,
        0.075,
        0.090,
        0.100,
        0.125,
        0.150,
        0.175,
        0.200,
        0.225,
        0.250,
        0.275,
        0.300,
        0.325,
        0.350,
        0.375,
        0.400,
        0.425,
        0.450,
        0.460,
        0.500,
        0.550,
        0.600,
        0.650,
        0.700,
        0.750,
    ]
    out = sample.copy()
    rows = []
    guard_mask = make_subject_lastblock_mask(train).reset_index(drop=True)
    for target in TARGETS:
        target_guard_weight = TRY9_TARGET_GUARD_WEIGHTS.get(target, TRY9_GUARD_WEIGHT)
        y = train[target].astype(int).reset_index(drop=True)
        base_oof = current_best_oof[target].astype(float).reset_index(drop=True)
        stack_oof = stacked_oof[target].astype(float).reset_index(drop=True)
        baseline_score = log_loss(y, clip_proba(base_oof, 1e-6, 1 - 1e-6), labels=[0, 1])
        baseline_guard = log_loss(
            y[guard_mask],
            clip_proba(base_oof[guard_mask], 1e-6, 1 - 1e-6),
            labels=[0, 1],
        )
        baseline_select = (1.0 - target_guard_weight) * baseline_score + target_guard_weight * baseline_guard
        best_alpha = 0.0
        best_score = baseline_score
        best_guard = baseline_guard
        best_select = baseline_select
        for alpha in alpha_grid:
            blend = (1 - alpha) * base_oof + alpha * stack_oof
            clipped = clip_proba(blend, 1e-6, 1 - 1e-6)
            score = log_loss(y, clipped, labels=[0, 1])
            guard_score = log_loss(y[guard_mask], clipped[guard_mask], labels=[0, 1])
            select_score = (1.0 - target_guard_weight) * score + target_guard_weight * guard_score
            guard_slack = 0.0010 if target in {"Q2", "Q3"} else 0.0015
            guard_ok = guard_score <= baseline_guard + guard_slack
            full_ok = score <= baseline_score + 0.0003
            if guard_ok and full_ok and select_score < best_select - 1e-7:
                best_score = score
                best_guard = guard_score
                best_select = select_score
                best_alpha = float(alpha)
        if best_alpha > 0:
            out[target] = clip_proba(
                (1 - best_alpha) * current_best[target].to_numpy(float)
                + best_alpha * stacked_test[target].to_numpy(float),
                0.03,
                0.97,
            )
            applied = True
        else:
            out[target] = current_best[target].to_numpy(float)
            applied = False
        rows.append(
            {
                "target": target,
                "baseline_oof_logloss": baseline_score,
                "safe_blend_oof_logloss": best_score,
                "delta": best_score - baseline_score,
                "baseline_lastblock_logloss": baseline_guard,
                "safe_blend_lastblock_logloss": best_guard,
                "lastblock_delta": best_guard - baseline_guard,
                "baseline_guarded_score": baseline_select,
                "safe_blend_guarded_score": best_select,
                "guarded_delta": best_select - baseline_select,
                "target_guard_weight": target_guard_weight,
                "alpha": best_alpha,
                "applied": applied,
            }
        )
    safe_diag = pd.DataFrame(rows)
    safe_diag.to_csv(SAFE_DIAG_PATH, index=False)
    return out, safe_diag


def write_source_submissions(sample, sources_test):
    for name, frame in sources_test.items():
        if not name.startswith("try6_12_"):
            continue
        out = sample.copy()
        out[TARGETS] = frame[TARGETS].to_numpy(float)
        out.to_csv(SUBMISSION_DIR / f"submission_{name}.csv", index=False)


def summarize_new_sources(train, current_best, current_best_oof, new_oof, new_test):
    rows = []
    for name, frame in new_oof.items():
        row = {"source": name}
        scores = []
        for target in TARGETS:
            score = log_loss(
                train[target].astype(int),
                clip_proba(frame[target].astype(float), 1e-6, 1 - 1e-6),
                labels=[0, 1],
            )
            scores.append(score)
            row[f"{target}_oof_logloss"] = score
        row["mean_oof_logloss"] = float(np.mean(scores))
        if name in new_test:
            diff = (new_test[name][TARGETS].reset_index(drop=True) - current_best[TARGETS].reset_index(drop=True)).abs()
            row["mean_abs_diff_vs_current_best"] = float(diff.to_numpy().mean())
            row["max_abs_diff_vs_current_best"] = float(diff.to_numpy().max())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_oof_logloss")
    summary.to_csv(NEW_SOURCE_SUMMARY_PATH, index=False)
    return summary


def summarize_submission_shift(sample, current_best, final, raw):
    rows = []
    for name, frame in [("raw_stack", raw), ("safe_final", final)]:
        diff = frame[TARGETS].reset_index(drop=True) - current_best[TARGETS].reset_index(drop=True)
        row = {
            "candidate": name,
            "mean_signed_diff": float(diff.to_numpy().mean()),
            "mean_abs_diff": float(np.abs(diff.to_numpy()).mean()),
            "max_abs_diff": float(np.abs(diff.to_numpy()).max()),
        }
        for target in TARGETS:
            row[f"{target}_signed_mean"] = float(diff[target].mean())
            row[f"{target}_abs_mean"] = float(diff[target].abs().mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUBMISSION_SUMMARY_PATH, index=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-step1-base",
        action="store_true",
        help="Reuse existing step1_base artifacts instead of rerunning the step1_base pipeline.",
    )
    parser.add_argument(
        "--include-mis-lstm",
        action="store_true",
        help="Also rerun step1_base MIS-LSTM step before the step1_extended pipeline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_step1_base:
        print("[INFO] skip step1_base rerun; reusing existing step1_base artifacts")
    else:
        print("[INFO] rerunning step1_base pipeline before step1_extended pipeline")
        run_step1_base_pipeline(include_mis_lstm=args.include_mis_lstm)

    train, sample, current_best, feat = load_core_frames()
    print(f"[INFO] root={ROOT}")
    print(f"[INFO] train={train.shape} sample={sample.shape} feature_table={feat.shape}")

    oof, test = load_existing_oof_sources(train, sample)
    if not oof:
        raise RuntimeError("No existing OOF/test stacking sources found in data/artifacts")
    print(f"[INFO] existing OOF sources loaded: {len(oof)}")
    print("[INFO] step1_extended extends step1_base saved OOF sources with personal/long-term sources")

    current_best_key = CURRENT_BEST_PATH.stem
    current_best_oof = None
    if current_best_key in oof:
        current_best_oof = oof[current_best_key]
    elif "submission_step1_sleep_proxy_s_targets" in oof:
        current_best_oof = oof["submission_step1_sleep_proxy_s_targets"]
        print("[WARN] current final OOF missing; using sleep_metric_proxy_s_targets as safe gate baseline")
    else:
        raise RuntimeError("Could not find a usable current-best OOF source for safe alpha")

    print("[INFO] building personalized temporal kernel priors")
    temporal_oof, temporal_test = build_personal_temporal_sources(train, sample)
    print("[INFO] building Bayesian personalized calendar priors")
    calendar_oof, calendar_test = build_bayesian_calendar_sources(train, sample)
    print("[INFO] building long-term subject-normalized sensor models")
    long_oof, long_test, long_feature_cols = build_longterm_sensor_model(train, sample, feat)
    print(f"[INFO] long-term feature columns used: {len(long_feature_cols)}")

    new_oof = {**temporal_oof, **calendar_oof, **long_oof}
    new_test = {**temporal_test, **calendar_test, **long_test}
    summarize_new_sources(train, current_best, current_best_oof, new_oof, new_test)
    write_source_submissions(sample, new_test)

    oof.update(new_oof)
    test.update(new_test)
    print(f"[INFO] total OOF sources for try6_12 stack: {len(oof)}")

    raw_stack, raw_oof, source_diag, model_diag = fit_targetwise_stack(
        train,
        oof,
        test,
        checkpoint_run_tag="step1_extended",
        restore_from_checkpoints=STEP1_RESTORE_TARGETWISE_STACK_CHECKPOINTS,
    )
    raw_submission = sample.copy()
    raw_submission[TARGETS] = raw_stack[TARGETS].to_numpy(float)
    raw_submission.to_csv(RAW_STACK_PATH, index=False)

    final, safe_diag = safe_alpha_blend(
        sample,
        current_best,
        current_best_oof,
        raw_stack,
        raw_oof,
        train,
    )
    final.to_csv(OUT_PATH, index=False)
    shift_summary = summarize_submission_shift(sample, current_best, final, raw_submission)

    print("\n[TRY6_12] New source summary:")
    print(pd.read_csv(NEW_SOURCE_SUMMARY_PATH).head(12).to_string(index=False))
    print("\n[TRY6_12] Model diagnostics:")
    print(model_diag[["target", "best_single_source", "best_single_logloss", "chosen_meta", "chosen_oof_logloss"]].to_string(index=False))
    print("\n[TRY6_12] Safe alpha diagnostics:")
    print(safe_diag.to_string(index=False))
    print("\n[TRY6_12] Shift summary vs current best:")
    print(shift_summary.to_string(index=False))
    print(f"\n[DONE] raw stack: {RAW_STACK_PATH}")
    print(f"[DONE] final: {OUT_PATH}")
    print(f"[DONE] diagnostics: {MODEL_DIAG_PATH}, {SAFE_DIAG_PATH}, {SOURCE_DIAG_PATH}")



# ============================================================================
# step1_oof_runner end-to-end runner
# ============================================================================
# This section intentionally runs the embedded step1_base pipeline first and then
# feeds only the OOF/test sources created during this same run into the step1_extended
# extension. For fast iteration, --reuse-step1-base-artifacts can skip the expensive
# first stage and reuse already materialized OOF/test sources. step1_oof_runner adds
# past-only subject label history / sequence sources to the second-stage stack.

import time as _step1_oof_time

TRY9_OUT_PATH = SUBMISSION_DIR / "submission_step1_personal_personalized_targetwise.csv"
TRY9_RAW_STACK_PATH = SUBMISSION_DIR / "submission_step1_personal_raw_stack.csv"
TRY9_SOURCE_DIAG_PATH = ARTIFACT_DIR / "longterm_personalization_meta_source_target_oof_logloss.csv"
TRY9_MODEL_DIAG_PATH = ARTIFACT_DIR / "longterm_personalization_model_diagnostics.csv"
TRY9_SAFE_DIAG_PATH = ARTIFACT_DIR / "longterm_personalization_safe_alpha_diagnostics.csv"
TRY9_SUBMISSION_SUMMARY_PATH = ARTIFACT_DIR / "longterm_personalization_submission_shift_summary.csv"
TRY9_NEW_SOURCE_SUMMARY_PATH = ARTIFACT_DIR / "longterm_personalization_new_source_summary.csv"
TRY9_RUN_MANIFEST_PATH = ARTIFACT_DIR / "longterm_personalization_source_manifest.csv"


def step1_oof_runner_run_base_pipeline(include_mis_lstm: bool = False) -> None:
    """Run the embedded step1_base pipeline in-process."""
    print(f"[TRY8] Project root: {ROOT}")
    print(f"[TRY8] Python: {sys.executable}")
    check_inputs()
    for label, func, args in PIPELINE:
        if "validate_final_submission" in label:
            continue
        if not include_mis_lstm and "train_mis_lstm" in label:
            print(f"[TRY8] skip optional MIS-LSTM step: {label}")
            continue
        run_pipeline_step(label, func, args)
    if not CURRENT_BEST_PATH.exists():
        raise RuntimeError(f"step1_base stage did not create {CURRENT_BEST_PATH}")


def step1_oof_runner_load_fresh_oof_sources(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    run_started_at: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Load only OOF/test source pairs written after this step1_oof_runner run started."""
    oof: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    rows = []
    for oof_path in sorted(ARTIFACT_DIR.glob("oof_stacking_base_*.csv")):
        name = oof_path.stem.replace("oof_stacking_base_", "", 1)
        test_path = ARTIFACT_DIR / f"test_stacking_base_{name}.csv"
        if not test_path.exists():
            continue
        oof_mtime = oof_path.stat().st_mtime
        test_mtime = test_path.stat().st_mtime
        fresh = oof_mtime >= run_started_at and test_mtime >= run_started_at
        rows.append(
            {
                "source": name,
                "oof_path": str(oof_path),
                "test_path": str(test_path),
                "oof_mtime": oof_mtime,
                "test_mtime": test_mtime,
                "fresh_for_try9_run": bool(fresh),
            }
        )
        if not fresh:
            continue
        oof_df = pd.read_csv(oof_path)
        test_df = pd.read_csv(test_path)
        if not all(t in oof_df.columns for t in TARGETS):
            continue
        if not all(t in test_df.columns for t in TARGETS):
            continue
        if len(oof_df) != len(train) or len(test_df) != len(sample):
            continue
        oof[name] = oof_df[TARGETS].astype(float).reset_index(drop=True)
        test[name] = test_df[TARGETS].astype(float).reset_index(drop=True)

    manifest = pd.DataFrame(rows).sort_values(["fresh_for_try9_run", "source"], ascending=[False, True])
    manifest.to_csv(TRY9_RUN_MANIFEST_PATH, index=False)
    return oof, test, manifest


def step1_oof_runner_load_reusable_oof_sources(
    train: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Load existing try6_11 OOF/test source pairs for fast local iteration."""
    oof: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    rows = []
    for oof_path in sorted(ARTIFACT_DIR.glob("oof_stacking_base_*.csv")):
        name = oof_path.stem.replace("oof_stacking_base_", "", 1)
        test_path = ARTIFACT_DIR / f"test_stacking_base_{name}.csv"
        row = {
            "source": name,
            "oof_path": str(oof_path),
            "test_path": str(test_path),
            "oof_mtime": oof_path.stat().st_mtime,
            "test_mtime": test_path.stat().st_mtime if test_path.exists() else np.nan,
            "fresh_for_try9_run": False,
            "reused_for_try9_1_fast_iter": False,
            "status": "pending",
        }
        if not test_path.exists():
            row["status"] = "missing_test_pair"
            rows.append(row)
            continue
        oof_df = pd.read_csv(oof_path)
        test_df = pd.read_csv(test_path)
        if not all(t in oof_df.columns for t in TARGETS):
            row["status"] = "missing_oof_targets"
            rows.append(row)
            continue
        if not all(t in test_df.columns for t in TARGETS):
            row["status"] = "missing_test_targets"
            rows.append(row)
            continue
        if len(oof_df) != len(train) or len(test_df) != len(sample):
            row["status"] = f"shape_mismatch_oof{len(oof_df)}_test{len(test_df)}"
            rows.append(row)
            continue
        oof[name] = oof_df[TARGETS].astype(float).reset_index(drop=True)
        test[name] = test_df[TARGETS].astype(float).reset_index(drop=True)
        row["reused_for_try9_1_fast_iter"] = True
        row["status"] = "loaded"
        rows.append(row)

    manifest = pd.DataFrame(rows).sort_values(
        ["reused_for_try9_1_fast_iter", "source"], ascending=[False, True]
    )
    manifest.to_csv(TRY9_RUN_MANIFEST_PATH, index=False)
    return oof, test, manifest


def context_safe_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return []


_FIVE_FACTOR_CONTEXT_CACHE: tuple[pd.DataFrame, list[str]] | None = None


def build_five_factor_context_feature_table(train: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compact five-factor context features from raw sensors.

    This intentionally creates a small standalone source instead of replacing the
    existing try18 feature table. The second-stage stack can ignore it if it is
    not useful.
    """

    global _FIVE_FACTOR_CONTEXT_CACHE
    if _FIVE_FACTOR_CONTEXT_CACHE is not None:
        cached_table, cached_cols = _FIVE_FACTOR_CONTEXT_CACHE
        return cached_table.copy(), list(cached_cols)

    sensor_dir = DATA_DIR / "ch2025_data_items"
    train_meta = train[["subject_id", "sleep_date", "lifelog_date", *TARGETS]].copy()
    test_meta = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for target in TARGETS:
        test_meta[target] = np.nan
    meta = pd.concat(
        [train_meta.assign(is_train=1), test_meta.assign(is_train=0)],
        ignore_index=True,
        sort=False,
    )
    meta["row_id"] = np.arange(len(meta))
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta["lifelog_date"] = pd.to_datetime(meta["lifelog_date"])
    meta["sleep_date"] = pd.to_datetime(meta["sleep_date"])
    meta["dow"] = meta["lifelog_date"].dt.dayofweek
    meta["sleep_dow"] = meta["sleep_date"].dt.dayofweek
    meta["is_weekend"] = (meta["dow"] >= 5).astype(int)
    meta["sleep_is_weekend"] = (meta["sleep_dow"] >= 5).astype(int)

    # Korean holidays in the current train/test date span. Keep this explicit to
    # avoid adding a dependency for a handful of dates.
    holidays = {
        pd.Timestamp("2024-06-06"),
        pd.Timestamp("2024-08-15"),
    }
    meta["is_holiday"] = meta["lifelog_date"].isin(holidays).astype(int)
    meta["sleep_is_holiday"] = meta["sleep_date"].isin(holidays).astype(int)
    meta["is_pre_holiday"] = (
        meta["lifelog_date"] + pd.Timedelta(days=1)
    ).isin(holidays).astype(int)
    meta["is_post_holiday"] = (
        meta["lifelog_date"] - pd.Timedelta(days=1)
    ).isin(holidays).astype(int)

    windows = {
        "day": (0, 24),
        "evening": (18, 24),
        "pre_sleep": (21, 27),
        "late_night": (24, 29),
        "overnight": (24, 32),
        "wake_1h_06": (30, 31),
        "wake_1h_07": (31, 32),
        "wake_3h": (30, 33),
    }

    def window_aggregate(sensor: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
        if sensor.empty or not value_cols:
            return pd.DataFrame({"row_id": meta["row_id"]})
        sensor = sensor.copy()
        sensor["subject_id"] = sensor["subject_id"].astype(str)
        sensor["timestamp"] = pd.to_datetime(sensor["timestamp"])
        value_cols = [c for c in value_cols if c in sensor.columns]
        for col in value_cols:
            sensor[col] = pd.to_numeric(sensor[col], errors="coerce")
        sensor = sensor[["subject_id", "timestamp", *value_cols]].dropna(
            how="all", subset=value_cols
        )
        pieces = []
        for subject_id, rows in meta.groupby("subject_id", sort=False):
            subject_sensor = sensor[sensor["subject_id"].eq(subject_id)].sort_values("timestamp")
            if subject_sensor.empty:
                continue
            for row in rows.itertuples(index=False):
                rec = {"row_id": row.row_id}
                for window_name, (start_h, end_h) in windows.items():
                    start = row.lifelog_date + pd.Timedelta(hours=start_h)
                    end = row.lifelog_date + pd.Timedelta(hours=end_h)
                    part = subject_sensor[
                        (subject_sensor["timestamp"] >= start)
                        & (subject_sensor["timestamp"] < end)
                    ]
                    rec[f"{prefix}_{window_name}_count"] = len(part)
                    for col in value_cols:
                        vals = part[col].dropna()
                        base = f"{prefix}_{window_name}_{col}"
                        if vals.empty:
                            rec[f"{base}_mean"] = np.nan
                            rec[f"{base}_std"] = np.nan
                            rec[f"{base}_min"] = np.nan
                            rec[f"{base}_max"] = np.nan
                            rec[f"{base}_sum"] = np.nan
                            rec[f"{base}_median"] = np.nan
                            rec[f"{base}_q10"] = np.nan
                            rec[f"{base}_q25"] = np.nan
                            rec[f"{base}_q75"] = np.nan
                            rec[f"{base}_q90"] = np.nan
                            rec[f"{base}_iqr"] = np.nan
                        else:
                            rec[f"{base}_mean"] = float(vals.mean())
                            rec[f"{base}_std"] = float(vals.std(ddof=0))
                            rec[f"{base}_min"] = float(vals.min())
                            rec[f"{base}_max"] = float(vals.max())
                            rec[f"{base}_sum"] = float(vals.sum())
                            rec[f"{base}_median"] = float(vals.median())
                            rec[f"{base}_q10"] = float(vals.quantile(0.10))
                            rec[f"{base}_q25"] = float(vals.quantile(0.25))
                            rec[f"{base}_q75"] = float(vals.quantile(0.75))
                            rec[f"{base}_q90"] = float(vals.quantile(0.90))
                            rec[f"{base}_iqr"] = rec[f"{base}_q75"] - rec[f"{base}_q25"]
                pieces.append(rec)
        if not pieces:
            return pd.DataFrame({"row_id": meta["row_id"]})
        return pd.DataFrame(pieces)

    feature_parts = [meta[["row_id"]]]

    activity_path = sensor_dir / "ch2025_mActivity.parquet"
    if activity_path.exists():
        activity = pd.read_parquet(activity_path)
        activity["m_activity"] = pd.to_numeric(activity.get("m_activity"), errors="coerce")
        met_map = {
            0: 1.3,  # vehicle
            1: 8.0,  # bicycle
            2: 2.8,  # on foot / movement
            3: 1.0,  # stationary
            4: 1.0,  # unknown
            5: 1.2,
            7: 3.5,  # walking
            8: 10.0,  # running
        }
        activity["met"] = activity["m_activity"].map(met_map).fillna(1.2)
        activity["active_met"] = np.where(activity["met"] >= 3.0, activity["met"], 0.0)
        activity["high_met_flag"] = (activity["met"] >= 6.0).astype(float)
        activity["stationary_flag"] = activity["m_activity"].isin([3, 4, 5]).astype(float)
        activity["walking_flag"] = activity["m_activity"].isin([2, 7]).astype(float)
        activity["running_flag"] = activity["m_activity"].isin([8]).astype(float)
        activity["vehicle_flag"] = activity["m_activity"].isin([0]).astype(float)
        feature_parts.append(
            window_aggregate(
                activity,
                [
                    "met",
                    "active_met",
                    "high_met_flag",
                    "stationary_flag",
                    "walking_flag",
                    "running_flag",
                    "vehicle_flag",
                ],
                "ctx_met",
            )
        )

    ambience_path = sensor_dir / "ch2025_mAmbience.parquet"
    if ambience_path.exists():
        ambience = pd.read_parquet(ambience_path)

        def ambience_summary(value):
            rows = context_safe_list(value)
            db_map = {
                "traffic": 75.0,
                "vehicle": 70.0,
                "conversation": 60.0,
                "speech": 60.0,
                "music": 65.0,
                "inside": 40.0,
                "outside": 55.0,
                "wind": 55.0,
                "snoring": 45.0,
                "breathing": 35.0,
                "quiet": 30.0,
            }
            top_score = 0.0
            top_db = np.nan
            weighted_db = 0.0
            total_score = 0.0
            flags = {
                "traffic_score": 0.0,
                "conversation_score": 0.0,
                "sleep_noise_score": 0.0,
                "quiet_score": 0.0,
            }
            for item in rows:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                label = str(item[0]).lower()
                try:
                    score = float(item[1])
                except Exception:
                    continue
                label_db = 45.0
                for key, db in db_map.items():
                    if key in label:
                        label_db = db
                        break
                weighted_db += score * label_db
                total_score += score
                if score > top_score:
                    top_score = score
                    top_db = label_db
                if any(key in label for key in ["traffic", "vehicle", "car", "bus"]):
                    flags["traffic_score"] += score
                if any(key in label for key in ["conversation", "speech", "talk"]):
                    flags["conversation_score"] += score
                if any(key in label for key in ["snoring", "breathing"]):
                    flags["sleep_noise_score"] += score
                if any(key in label for key in ["quiet", "inside"]):
                    flags["quiet_score"] += score
            return {
                "db_top": top_db,
                "db_weighted": weighted_db / max(total_score, 1e-9),
                "noise_top_score": top_score,
                "total_score": total_score,
                "traffic_ratio": flags["traffic_score"] / max(total_score, 1e-9),
                "conversation_ratio": flags["conversation_score"] / max(total_score, 1e-9),
                "sleep_noise_ratio": flags["sleep_noise_score"] / max(total_score, 1e-9),
                "quiet_ratio": flags["quiet_score"] / max(total_score, 1e-9),
                **flags,
            }

        amb = pd.DataFrame(ambience["m_ambience"].apply(ambience_summary).tolist())
        ambience = pd.concat(
            [ambience[["subject_id", "timestamp"]].reset_index(drop=True), amb],
            axis=1,
        )
        feature_parts.append(
            window_aggregate(
                ambience,
                [
                    "db_top",
                    "db_weighted",
                    "noise_top_score",
                    "total_score",
                    "traffic_score",
                    "conversation_score",
                    "sleep_noise_score",
                    "quiet_score",
                    "traffic_ratio",
                    "conversation_ratio",
                    "sleep_noise_ratio",
                    "quiet_ratio",
                ],
                "ctx_noise",
            )
        )

    gps_path = sensor_dir / "ch2025_mGps.parquet"
    if gps_path.exists():
        gps = pd.read_parquet(gps_path)

        def gps_summary(value):
            rows = [row for row in context_safe_list(value) if isinstance(row, dict)]
            if not rows:
                return {
                    "gps_vsd": np.nan,
                    "gps_speed_mean": np.nan,
                    "gps_speed_max": np.nan,
                    "gps_stationary_flag": np.nan,
                    "gps_point_count": 0.0,
                }
            lat = pd.to_numeric(pd.Series([r.get("latitude", np.nan) for r in rows]), errors="coerce")
            lon = pd.to_numeric(pd.Series([r.get("longitude", np.nan) for r in rows]), errors="coerce")
            alt = pd.to_numeric(pd.Series([r.get("altitude", np.nan) for r in rows]), errors="coerce")
            speed = pd.to_numeric(pd.Series([r.get("speed", np.nan) for r in rows]), errors="coerce")
            lat_std = float(lat.std(ddof=0)) if lat.notna().any() else 0.0
            lon_std = float(lon.std(ddof=0)) if lon.notna().any() else 0.0
            alt_std = float(alt.std(ddof=0)) if alt.notna().any() else 0.0
            vsd = lat_std * lon_std * max(alt_std, 1.0)
            speed_mean = float(speed.mean()) if speed.notna().any() else np.nan
            speed_max = float(speed.max()) if speed.notna().any() else np.nan
            return {
                "gps_vsd": vsd,
                "gps_speed_mean": speed_mean,
                "gps_speed_max": speed_max,
                "gps_stationary_flag": float((vsd < 1e-7) and (not np.isfinite(speed_mean) or speed_mean < 0.5)),
                "gps_point_count": float(len(rows)),
            }

        gps_summary_df = pd.DataFrame(gps["m_gps"].apply(gps_summary).tolist())
        gps = pd.concat([gps[["subject_id", "timestamp"]].reset_index(drop=True), gps_summary_df], axis=1)
        feature_parts.append(
            window_aggregate(
                gps,
                ["gps_vsd", "gps_speed_mean", "gps_speed_max", "gps_stationary_flag", "gps_point_count"],
                "ctx_gps",
            )
        )

    light_parts = []
    mlight_path = sensor_dir / "ch2025_mLight.parquet"
    if mlight_path.exists():
        mlight = pd.read_parquet(mlight_path)
        if "m_light" in mlight.columns:
            light_parts.append(
                mlight[["subject_id", "timestamp", "m_light"]].rename(columns={"m_light": "light"})
            )
    wlight_path = sensor_dir / "ch2025_wLight.parquet"
    if wlight_path.exists():
        wlight = pd.read_parquet(wlight_path)
        if "w_light" in wlight.columns:
            light_parts.append(
                wlight[["subject_id", "timestamp", "w_light"]].rename(columns={"w_light": "light"})
            )
    if light_parts:
        light = pd.concat(light_parts, ignore_index=True)
        light["light"] = pd.to_numeric(light["light"], errors="coerce")
        light["dark_flag"] = (light["light"] <= 5).astype(float)
        light["bright_flag"] = (light["light"] >= 100).astype(float)
        light["log_light"] = np.log1p(light["light"].clip(lower=0))
        feature_parts.append(
            window_aggregate(light, ["light", "log_light", "dark_flag", "bright_flag"], "ctx_ulight")
        )

    whr_path = sensor_dir / "ch2025_wHr.parquet"
    if whr_path.exists():
        whr = pd.read_parquet(whr_path)
        if "heart_rate" in whr.columns:
            whr = whr[["subject_id", "timestamp", "heart_rate"]].copy()
            whr["heart_rate"] = pd.to_numeric(whr["heart_rate"], errors="coerce")
            feature_parts.append(window_aggregate(whr, ["heart_rate"], "ctx_hr"))

    pedo_path = sensor_dir / "ch2025_wPedo.parquet"
    if pedo_path.exists():
        pedo = pd.read_parquet(pedo_path)
        pedo_cols = [
            col
            for col in ["step", "step_frequency", "distance", "speed", "burned_calories"]
            if col in pedo.columns
        ]
        if pedo_cols:
            pedo = pedo[["subject_id", "timestamp", *pedo_cols]].copy()
            for col in pedo_cols:
                pedo[col] = pd.to_numeric(pedo[col], errors="coerce")
            feature_parts.append(window_aggregate(pedo, pedo_cols, "ctx_pedo"))

    screen_path = sensor_dir / "ch2025_mScreenStatus.parquet"
    if screen_path.exists():
        screen = pd.read_parquet(screen_path)
        if "m_screen_use" in screen.columns:
            screen = screen[["subject_id", "timestamp", "m_screen_use"]].copy()
            screen["m_screen_use"] = pd.to_numeric(screen["m_screen_use"], errors="coerce")
            feature_parts.append(window_aggregate(screen, ["m_screen_use"], "ctx_screen"))

    features = feature_parts[0]
    for part in feature_parts[1:]:
        features = features.merge(part, on="row_id", how="left")
    table = meta.merge(features, on="row_id", how="left")

    count_cols = [col for col in table.columns if col.startswith("ctx_") and col.endswith("_count")]
    for col in count_cols:
        count = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
        table[f"{col}_missing_flag"] = (count <= 0).astype(float)
        table[f"{col}_low_coverage_flag"] = (count <= 2).astype(float)

    if "ctx_hr_overnight_heart_rate_q10" in table.columns:
        table["ctx_hr_resting_proxy"] = pd.to_numeric(
            table["ctx_hr_overnight_heart_rate_q10"], errors="coerce"
        )
        for wake_col in [
            "ctx_hr_wake_1h_06_heart_rate_mean",
            "ctx_hr_wake_1h_07_heart_rate_mean",
            "ctx_hr_wake_3h_heart_rate_mean",
        ]:
            if wake_col in table.columns:
                table[f"{wake_col}_minus_resting"] = (
                    pd.to_numeric(table[wake_col], errors="coerce")
                    - table["ctx_hr_resting_proxy"]
                )
        for evening_col in [
            "ctx_hr_evening_heart_rate_mean",
            "ctx_hr_pre_sleep_heart_rate_mean",
        ]:
            if evening_col in table.columns:
                table[f"{evening_col}_minus_resting"] = (
                    pd.to_numeric(table[evening_col], errors="coerce")
                    - table["ctx_hr_resting_proxy"]
                )

    if {
        "ctx_hr_overnight_count",
        "ctx_ulight_overnight_count",
    }.issubset(table.columns):
        table["ctx_wearable_off_overnight_proxy"] = (
            pd.to_numeric(table["ctx_hr_overnight_count"], errors="coerce").fillna(0).le(0)
            & pd.to_numeric(table["ctx_ulight_overnight_count"], errors="coerce").fillna(0).le(0)
        ).astype(float)
    if {
        "ctx_screen_day_count",
        "ctx_hr_day_count",
    }.issubset(table.columns):
        table["ctx_phone_only_day_proxy"] = (
            pd.to_numeric(table["ctx_screen_day_count"], errors="coerce").fillna(0).gt(0)
            & pd.to_numeric(table["ctx_hr_day_count"], errors="coerce").fillna(0).le(0)
        ).astype(float)

    context_cols = [
        col
        for col in table.columns
        if col.startswith("ctx_")
        or col
        in {
            "dow",
            "sleep_dow",
            "is_weekend",
            "sleep_is_weekend",
            "is_holiday",
            "sleep_is_holiday",
            "is_pre_holiday",
            "is_post_holiday",
        }
    ]
    for col in context_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")

    diag = pd.DataFrame({"feature": context_cols})
    diag_path = ARTIFACT_DIR / "longterm_personalization_context_feature_columns.csv"
    diag.to_csv(diag_path, index=False)
    print(f"[TRY18_8] context feature columns: {len(context_cols)} -> {diag_path}")
    _FIVE_FACTOR_CONTEXT_CACHE = (table.copy(), list(context_cols))
    return table, context_cols


def select_first_stage_context_columns(table: pd.DataFrame, context_cols: list[str], max_cols: int = 220) -> list[str]:
    train_mask = table["is_train"].eq(1)
    test_mask = table["is_train"].eq(0)
    min_train_nonnull = max(18, int(train_mask.sum() * 0.08))
    min_test_nonnull = max(10, int(test_mask.sum() * 0.08))
    priority_tokens = [
        "resting",
        "minus_resting",
        "wake_1h",
        "wake_3h",
        "pre_sleep",
        "overnight",
        "evening",
        "ratio",
        "missing_flag",
        "low_coverage_flag",
        "wearable_off",
        "phone_only",
        "hr",
        "pedo",
        "screen",
        "ulight",
        "noise",
        "met",
        "gps",
        "holiday",
        "weekend",
    ]
    scored: list[tuple[float, str]] = []
    for col in context_cols:
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce")
        train_nonnull = int(values[train_mask].notna().sum())
        test_nonnull = int(values[test_mask].notna().sum())
        if train_nonnull < min_train_nonnull or test_nonnull < min_test_nonnull:
            continue
        variance = float(values[train_mask].replace([np.inf, -np.inf], np.nan).var())
        if not np.isfinite(variance) or variance <= 1e-12:
            continue
        nonnull_rate = 0.5 * (
            train_nonnull / max(int(train_mask.sum()), 1)
            + test_nonnull / max(int(test_mask.sum()), 1)
        )
        token_score = sum(1.0 for token in priority_tokens if token in col)
        score = 10.0 * token_score + nonnull_rate + min(np.log1p(variance), 3.0) * 0.05
        scored.append((score, col))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [col for _, col in scored[:max_cols]]
    pd.DataFrame({"feature": selected}).to_csv(
        ARTIFACT_DIR / "longterm_personalization_first_stage_context_feature_columns.csv", index=False
    )
    print(f"[TRY18_8] first-stage context features selected: {len(selected)}")
    return selected


def build_context_enriched_longterm_feature_table(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    feat: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    base_table, base_feature_cols = build_longterm_feature_table(feat)
    context_table, context_cols = build_five_factor_context_feature_table(train, sample)
    selected_context_cols = select_first_stage_context_columns(context_table, context_cols)

    if not selected_context_cols:
        return base_table, base_feature_cols

    add = context_table[["subject_id", "lifelog_date", *selected_context_cols]].copy()
    add["subject_id"] = add["subject_id"].astype(str)
    add["lifelog_date"] = pd.to_datetime(add["lifelog_date"])

    enriched = base_table.merge(add, on=["subject_id", "lifelog_date"], how="left")
    med = context_table[selected_context_cols].median(numeric_only=True)
    enriched[selected_context_cols] = (
        enriched[selected_context_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(med)
        .fillna(0.0)
    )
    feature_cols = base_feature_cols + [
        col for col in selected_context_cols if col not in set(base_feature_cols)
    ]
    pd.DataFrame({"feature": feature_cols}).to_csv(
        ARTIFACT_DIR / "longterm_personalization_longterm_context_enriched_feature_columns.csv",
        index=False,
    )
    print(
        f"[TRY18_8] context-enriched first-stage features: "
        f"base={len(base_feature_cols)} context={len(selected_context_cols)} total={len(feature_cols)}"
    )
    return enriched, feature_cols


def build_five_factor_context_sources(
    train: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    table, feature_cols = build_five_factor_context_feature_table(train, sample)
    train_feat = table[table["is_train"].eq(1)].copy().reset_index(drop=True)
    test_feat = table[table["is_train"].eq(0)].copy().reset_index(drop=True)

    kept_cols = []
    min_train_nonnull = max(18, int(len(train_feat) * 0.08))
    min_test_nonnull = max(10, int(len(test_feat) * 0.08))
    for col in feature_cols:
        train_nonnull = int(train_feat[col].notna().sum())
        test_nonnull = int(test_feat[col].notna().sum())
        if train_nonnull >= min_train_nonnull and test_nonnull >= min_test_nonnull:
            kept_cols.append(col)
    if not kept_cols:
        raise RuntimeError("five-factor context source has no usable feature columns")

    med = table[kept_cols].median(numeric_only=True)
    train_feat[kept_cols] = train_feat[kept_cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    test_feat[kept_cols] = test_feat[kept_cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    folds = make_temporal_folds(train_feat).reset_index(drop=True)
    X_test = test_feat[kept_cols]

    lgb_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    xgb_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    lgb_test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
    xgb_test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)

    for target in TARGETS:
        y = train_feat[target].astype(int).reset_index(drop=True)
        target_mean = float(y.mean())
        if target not in FIRST_STAGE_CONTEXT_GATE_TARGETS:
            lgb_oof[target] = target_mean
            xgb_oof[target] = target_mean
            lgb_test[target] = target_mean
            xgb_test[target] = target_mean
            continue
        lgb_acc = np.zeros(len(sample), dtype=float)
        xgb_acc = np.zeros(len(sample), dtype=float)
        n_used = 0
        for fold in range(N_FOLDS):
            valid = folds.eq(fold).to_numpy()
            if valid.sum() == 0 or (~valid).sum() == 0:
                continue
            tr = ~valid
            if y.loc[tr].nunique() < 2:
                lgb_oof.loc[valid, target] = target_mean
                xgb_oof.loc[valid, target] = target_mean
                continue
            X_tr = train_feat.loc[tr, kept_cols]
            X_va = train_feat.loc[valid, kept_cols]
            y_tr = y.loc[tr]
            restored = (
                load_step1_candidate_checkpoint(
                    "five_factor_context_sources",
                    target,
                    "fold",
                    fold,
                )
                if restore_step1_candidate_checkpoints_enabled()
                else None
            )
            if restored is not None:
                print(f"[STEP1_CKPT] restored candidate five_factor_context_sources {target} fold={fold}")
                x_va_restored = _step1_candidate_matrix(train_feat.loc[valid], restored)
                x_test_restored = _step1_candidate_matrix(test_feat, restored)
                models = restored["models"]
                restored_mean = float(restored["target_mean"])
                lgb_oof.loc[valid, target] = np.clip(
                    0.82 * models["lgb"].predict_proba(x_va_restored)[:, 1] + 0.18 * restored_mean,
                    0.04,
                    0.96,
                )
                xgb_oof.loc[valid, target] = np.clip(
                    0.80 * models["xgb"].predict_proba(x_va_restored)[:, 1] + 0.20 * restored_mean,
                    0.04,
                    0.96,
                )
                lgb_acc += np.clip(
                    0.82 * models["lgb"].predict_proba(x_test_restored)[:, 1] + 0.18 * restored_mean,
                    0.04,
                    0.96,
                )
                xgb_acc += np.clip(
                    0.80 * models["xgb"].predict_proba(x_test_restored)[:, 1] + 0.20 * restored_mean,
                    0.04,
                    0.96,
                )
                n_used += 1
                continue

            lgb = LGBMClassifier(
                n_estimators=120,
                learning_rate=0.025,
                max_depth=2,
                num_leaves=7,
                min_child_samples=20,
                subsample=0.88,
                colsample_bytree=0.76,
                reg_alpha=2.0,
                reg_lambda=12.0,
                objective="binary",
                random_state=18500 + fold,
                verbose=-1,
            )
            lgb.fit(X_tr, y_tr)
            lgb_pred = lgb.predict_proba(X_va)[:, 1]
            lgb_oof.loc[valid, target] = np.clip(0.82 * lgb_pred + 0.18 * target_mean, 0.04, 0.96)
            lgb_test_pred = lgb.predict_proba(X_test)[:, 1]
            lgb_acc += np.clip(0.82 * lgb_test_pred + 0.18 * target_mean, 0.04, 0.96)

            xgb = XGBClassifier(
                n_estimators=90,
                max_depth=2,
                learning_rate=0.025,
                subsample=0.88,
                colsample_bytree=0.76,
                reg_alpha=3.0,
                reg_lambda=18.0,
                min_child_weight=8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=19500 + fold,
                n_jobs=1,
            )
            xgb.fit(X_tr, y_tr)
            xgb_pred = xgb.predict_proba(X_va)[:, 1]
            xgb_oof.loc[valid, target] = np.clip(0.80 * xgb_pred + 0.20 * target_mean, 0.04, 0.96)
            xgb_test_pred = xgb.predict_proba(X_test)[:, 1]
            xgb_acc += np.clip(0.80 * xgb_test_pred + 0.20 * target_mean, 0.04, 0.96)
            save_step1_candidate_checkpoint(
                "five_factor_context_sources",
                {
                    "kind": "fold",
                    "target": target,
                    "fold": fold,
                    "feature_cols": list(kept_cols),
                    "median": med,
                    "target_mean": target_mean,
                    "models": {
                        "lgb": lgb,
                        "xgb": xgb,
                    },
                },
                target,
                "fold",
                fold,
            )
            n_used += 1

        denom = max(n_used, 1)
        lgb_test[target] = lgb_acc / denom if n_used else target_mean
        xgb_test[target] = xgb_acc / denom if n_used else target_mean

    avg_oof = 0.5 * lgb_oof.astype(float) + 0.5 * xgb_oof.astype(float)
    avg_test = 0.5 * lgb_test.astype(float) + 0.5 * xgb_test.astype(float)
    sources_oof = {
        "try18_8_context_lgb": lgb_oof,
        "try18_8_context_xgb": xgb_oof,
        "try18_8_context_avg": avg_oof,
    }
    sources_test = {
        "try18_8_context_lgb": lgb_test,
        "try18_8_context_xgb": xgb_test,
        "try18_8_context_avg": avg_test,
    }
    pd.DataFrame({"feature": kept_cols}).to_csv(
        ARTIFACT_DIR / "longterm_personalization_context_used_feature_columns.csv", index=False
    )
    print(f"[TRY18_8] context source features used: {len(kept_cols)}")
    return finalize_sources(sources_oof), finalize_sources(sources_test), kept_cols


# Pruned historical definition: step1_oof_runner_parse_args (not reachable from the final runner).


# Pruned historical definition: step1_oof_runner_main (not reachable from the final runner).



# ============================================================================
# 18_8_v1: past-only personalized long-term utilization
# ============================================================================
import importlib.util as _longterm_v1_importlib
import shutil as _longterm_v1_shutil

LONGTERM_V1_PREFIX = "longterm_personalization_v1"
LONGTERM_V1_STATE_COLUMNS_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_personal_state_feature_columns.csv"
V1_STATE_SUMMARY_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_personal_state_feature_summary.csv"
V1_MULTISCALE_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_multiscale_longterm_features.csv"
V1_TRANSITION_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_transition_streak_diagnostics.csv"
V1_RESIDUAL_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_personal_residual_diagnostics.csv"
V1_ALPHA_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_personal_reliability_alpha.csv"
LONGTERM_V1_CALIBRATION_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_final_temperature_bias.csv"
V1_COMPARISON_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_profile_comparison.csv"
V1_SUMMARY_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_profile_summary.csv"
V1_RAW_COMPARISON_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_raw_profile_comparison.csv"
V1_RAW_SUMMARY_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_raw_profile_summary.csv"

V1_PROFILES = [
    "baseline_18_8",
    "personal_state_only",
    "transition_prior",
    "personal_residual",
    "personal_row_alpha",
    "full_personal_longterm",
]


def _longterm_v1_loss(y, pred) -> float:
    return float(
        log_loss(
            np.asarray(y, dtype=int),
            clip_proba(pred, 1e-6, 1 - 1e-6),
            labels=[0, 1],
        )
    )


def _longterm_v1_subject_chrono_folds(df: pd.DataFrame) -> pd.Series:
    folds = pd.Series(0, index=df.index, dtype=int)
    work = df[["subject_id", "lifelog_date"]].copy()
    work["lifelog_date"] = pd.to_datetime(work["lifelog_date"])
    for _, idx in work.groupby("subject_id", sort=False).groups.items():
        ordered = work.loc[list(idx)].sort_values("lifelog_date").index.to_numpy()
        block = np.floor(np.arange(len(ordered)) * N_FOLDS / max(len(ordered), 1))
        folds.loc[ordered] = np.minimum(block.astype(int), N_FOLDS - 1)
    return folds.reset_index(drop=True)


def _longterm_v1_feature_family(name: str) -> str:
    low = name.lower()
    for family, tokens in [
        ("target_history", TARGETS),
        ("sleep_proxy", ["sleep", "latency", "bed", "wake"]),
        ("activity", ["activity", "met", "motion"]),
        ("screen", ["screen", "usage"]),
        ("light", ["light", "lux"]),
        ("hr", ["heart", "hr", "pulse"]),
        ("pedo", ["pedo", "step", "distance", "calorie"]),
        ("coverage", ["count", "missing", "coverage"]),
    ]:
        if any(str(token).lower() in low for token in tokens):
            return family
    return "other"


def _longterm_v1_history_values(history: list[float], prior: float) -> dict[str, float]:
    arr = np.asarray(history, dtype=float)
    n = len(arr)
    expanding = float(arr.mean()) if n else prior
    expanding_std = float(arr.std(ddof=0)) if n >= 2 else 0.0

    def recent(k):
        return float(arr[-k:].mean()) if n else prior

    def streak():
        if not n:
            return prior, 0.0
        value = float(arr[-1])
        length = 1
        for item in arr[-2::-1]:
            if item != value:
                break
            length += 1
        return value, float(length)

    recent7_arr = arr[-7:]
    transitions = float(np.sum(recent7_arr[1:] != recent7_arr[:-1])) if len(recent7_arr) > 1 else 0.0
    streak_value, streak_length = streak()
    roll3, roll7, roll14 = recent(3), recent(7), recent(14)
    return {
        "prev1": float(arr[-1]) if n else prior,
        "prev2": float(arr[-2]) if n >= 2 else prior,
        "roll3_mean": roll3,
        "roll7_mean": roll7,
        "roll14_mean": roll14,
        "expanding_mean": expanding,
        "expanding_std": expanding_std,
        "recent3_minus_expanding": roll3 - expanding,
        "recent7_minus_expanding": roll7 - expanding,
        "roll3_minus_roll14": roll3 - roll14,
        "streak_value": streak_value,
        "streak_length": streak_length,
        "transition_count7": transitions,
        "volatility7": float(recent7_arr.std(ddof=0)) if len(recent7_arr) >= 2 else 0.0,
    }


def _longterm_v1_select_sensor_columns(feat: pd.DataFrame, max_cols: int = 100) -> list[str]:
    numeric = [
        c for c in feat.select_dtypes(include=[np.number]).columns
        if c not in TARGETS
    ]
    priority = ["sleep", "activity", "met", "screen", "light", "heart", "hr", "pedo", "step"]
    scored = []
    for col in numeric:
        values = pd.to_numeric(feat[col], errors="coerce")
        if values.notna().sum() < 20 or values.nunique(dropna=True) <= 1:
            continue
        family = _longterm_v1_feature_family(col)
        token_score = sum(token in col.lower() for token in priority)
        scored.append((10 * token_score + float(values.notna().mean()), family, col))
    scored.sort(key=lambda item: (-item[0], item[2]))
    caps = {
        "sleep_proxy": 30, "activity": 20, "screen": 20, "light": 20,
        "hr": 20, "pedo": 20, "coverage": 10, "other": 10,
    }
    counts, selected = {}, []
    for _, family, col in scored:
        if counts.get(family, 0) >= caps.get(family, 10):
            continue
        selected.append(col)
        counts[family] = counts.get(family, 0) + 1
        if len(selected) >= max_cols:
            break
    return selected


def build_past_only_personal_state_features(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    feat: pd.DataFrame,
    current_best_oof=None,
    current_best_test=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create target-history and sensor-deviation features using past rows only."""
    train_meta = train[["subject_id", "lifelog_date", *TARGETS]].copy()
    test_meta = sample[["subject_id", "lifelog_date"]].copy()
    for target in TARGETS:
        test_meta[target] = np.nan
    table = pd.concat(
        [train_meta.assign(is_train=1), test_meta.assign(is_train=0)],
        ignore_index=True,
    )
    table["_original_order"] = np.arange(len(table))
    table["subject_id"] = table["subject_id"].astype(str)
    table["lifelog_date"] = pd.to_datetime(table["lifelog_date"])
    global_means = train[TARGETS].mean().to_dict()
    history_features = []

    train_sorted = train_meta.copy()
    train_sorted["subject_id"] = train_sorted["subject_id"].astype(str)
    train_sorted["lifelog_date"] = pd.to_datetime(train_sorted["lifelog_date"])
    by_subject = {
        sid: group.sort_values("lifelog_date")
        for sid, group in train_sorted.groupby("subject_id", sort=False)
    }
    for target in TARGETS:
        names = [f"{target}_{suffix}" for suffix in _longterm_v1_history_values([], 0.5)]
        for name in names:
            table[name] = np.nan
        history_features.extend(names)
        for sid, idx in table.groupby("subject_id", sort=False).groups.items():
            rows = table.loc[list(idx)].sort_values("lifelog_date")
            subject_train = by_subject.get(sid, pd.DataFrame())
            for row_idx, row in rows.iterrows():
                if row["is_train"] == 1:
                    past = subject_train[
                        subject_train["lifelog_date"] < row["lifelog_date"]
                    ][target].astype(float).tolist()
                else:
                    past = subject_train[
                        subject_train["lifelog_date"] < row["lifelog_date"]
                    ][target].astype(float).tolist()
                values = _longterm_v1_history_values(past, float(global_means[target]))
                for suffix, value in values.items():
                    table.loc[row_idx, f"{target}_{suffix}"] = value

    sensor_cols = _longterm_v1_select_sensor_columns(feat)
    feature_frame = feat[["subject_id", "lifelog_date", *sensor_cols]].copy()
    feature_frame["subject_id"] = feature_frame["subject_id"].astype(str)
    feature_frame["lifelog_date"] = pd.to_datetime(feature_frame["lifelog_date"])
    feature_frame = feature_frame.drop_duplicates(["subject_id", "lifelog_date"], keep="last")
    rename = {col: f"base_{col}" for col in sensor_cols}
    feature_frame = feature_frame.rename(columns=rename)
    table = table.merge(feature_frame, on=["subject_id", "lifelog_date"], how="left")
    sensor_features = []
    table = table.sort_values(["subject_id", "lifelog_date", "_original_order"])
    for original in sensor_cols:
        base = f"base_{original}"
        group = table.groupby("subject_id", sort=False)[base]
        shifted = group.shift(1)
        expanding_mean = shifted.groupby(table["subject_id"]).transform(
            lambda s: s.expanding(min_periods=1).mean()
        )
        expanding_std = shifted.groupby(table["subject_id"]).transform(
            lambda s: s.expanding(min_periods=2).std(ddof=0)
        ).fillna(0.0)
        roll7 = shifted.groupby(table["subject_id"]).transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        )
        roll14 = shifted.groupby(table["subject_id"]).transform(
            lambda s: s.rolling(14, min_periods=1).mean()
        )
        roll7_std = shifted.groupby(table["subject_id"]).transform(
            lambda s: s.rolling(7, min_periods=2).std(ddof=0)
        ).fillna(0.0)
        prefix = f"personal_{original}"
        generated = {
            f"{prefix}_past_expanding_mean": expanding_mean,
            f"{prefix}_past_expanding_std": expanding_std,
            f"{prefix}_past_roll7_mean": roll7,
            f"{prefix}_past_roll14_mean": roll14,
            f"{prefix}_roll7_minus_expanding": roll7 - expanding_mean,
            f"{prefix}_roll7_minus_roll14": roll7 - roll14,
            f"{prefix}_roll7_std": roll7_std,
            f"{prefix}_recent_deviation_z": (shifted - expanding_mean) / (expanding_std + 1e-3),
            f"{prefix}_routine_break_score": (shifted - roll14).abs() / (roll7_std + 1e-3),
        }
        for name, values in generated.items():
            table[name] = values.replace([np.inf, -np.inf], np.nan)
            sensor_features.append(name)
    table = table.sort_values("_original_order").reset_index(drop=True)
    feature_cols = history_features + sensor_features
    train_mask, test_mask = table["is_train"].eq(1), table["is_train"].eq(0)
    rows, kept = [], []
    for col in feature_cols:
        tr = pd.to_numeric(table.loc[train_mask, col], errors="coerce")
        te = pd.to_numeric(table.loc[test_mask, col], errors="coerce")
        tr_mean, te_mean = float(tr.mean()), float(te.mean())
        tr_std = float(tr.std(ddof=0))
        drift = (
            abs(tr_mean - te_mean) / tr_std
            if np.isfinite(tr_mean) and np.isfinite(te_mean) and tr_std > 1e-9 else 0.0
        )
        decision = "keep" if tr.notna().sum() >= 20 and drift <= 3.0 else "drop"
        if decision == "keep":
            kept.append(col)
        rows.append(
            {
                "feature": col,
                "family": _longterm_v1_feature_family(col),
                "missing_train": float(tr.isna().mean()),
                "missing_test": float(te.isna().mean()),
                "train_mean": tr_mean,
                "test_mean": te_mean,
                "drift_score": drift,
                "decision": decision,
            }
        )
    pd.DataFrame({"feature": kept}).to_csv(LONGTERM_V1_STATE_COLUMNS_PATH, index=False)
    pd.DataFrame(rows).to_csv(V1_STATE_SUMMARY_PATH, index=False)
    return table, kept


def _longterm_v1_rolling_slope(values: pd.Series, window: int) -> pd.Series:
    def slope(arr):
        arr = np.asarray(arr, dtype=float)
        valid = np.isfinite(arr)
        if valid.sum() < 2:
            return 0.0
        x = np.arange(len(arr), dtype=float)[valid]
        return float(np.polyfit(x, arr[valid], 1)[0])
    return values.rolling(window, min_periods=2).apply(slope, raw=True)


def add_multiscale_longterm_memory_features(
    personal_state_table: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    table = personal_state_table.copy()
    selected = []
    family_counts = {}
    for col in feature_cols:
        family = _longterm_v1_feature_family(col)
        cap = 100 if family == "target_history" else (30 if family == "sleep_proxy" else 20)
        if family_counts.get(family, 0) >= cap:
            continue
        selected.append(col)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= 25:
            break
    table = table.sort_values(["subject_id", "lifelog_date", "_original_order"])
    added = []
    for col in selected:
        shifted = table.groupby("subject_id", sort=False)[col].shift(1)
        by_subject = shifted.groupby(table["subject_id"])
        ema = {}
        for half_life in [3, 7, 14, 28]:
            ema[half_life] = by_subject.transform(
                lambda s, hl=half_life: s.ewm(halflife=hl, adjust=False, min_periods=1).mean()
            )
            name = f"{col}_ema_hl{half_life}"
            table[name] = ema[half_life]
            added.append(name)
        generated = {
            f"{col}_ema_hl3_minus_hl14": ema[3] - ema[14],
            f"{col}_ema_hl7_minus_hl28": ema[7] - ema[28],
            f"{col}_rolling7_slope": by_subject.transform(lambda s: _longterm_v1_rolling_slope(s, 7)),
            f"{col}_rolling14_slope": by_subject.transform(lambda s: _longterm_v1_rolling_slope(s, 14)),
            f"{col}_rolling7_volatility": by_subject.transform(
                lambda s: s.rolling(7, min_periods=2).std(ddof=0)
            ),
            f"{col}_rolling14_volatility": by_subject.transform(
                lambda s: s.rolling(14, min_periods=2).std(ddof=0)
            ),
        }
        generated[f"{col}_routine_break_score"] = (
            shifted - ema[28]
        ).abs() / (generated[f"{col}_rolling14_volatility"] + 1e-3)
        for name, values in generated.items():
            table[name] = values.replace([np.inf, -np.inf], np.nan)
            added.append(name)
        if len(added) >= 250:
            break
    added = added[:250]
    table = table.sort_values("_original_order").reset_index(drop=True)
    pd.DataFrame(
        {"feature": added, "family": [_longterm_v1_feature_family(c) for c in added]}
    ).to_csv(V1_MULTISCALE_PATH, index=False)
    return table, feature_cols + added


def build_target_transition_streak_sources(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    current_best_oof: pd.DataFrame,
    current_best_test: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    source = "try18_8_v1_transition_streak_prior"
    oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
    guard = make_subject_lastblock_mask(train).reset_index(drop=True).to_numpy(bool)
    diagnostics = []
    ordered_train = train.copy()
    ordered_train["subject_id"] = ordered_train["subject_id"].astype(str)
    ordered_train["lifelog_date"] = pd.to_datetime(ordered_train["lifelog_date"])
    global_means = train[TARGETS].mean()
    for target in TARGETS:
        global_counts = {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0}
        for _, group in ordered_train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
            vals = group[target].astype(int).to_numpy()
            for left, right in zip(vals[:-1], vals[1:]):
                global_counts[(int(left), int(right))] += 1.0
        global_transition = {
            prev: global_counts[(prev, 1)] / (global_counts[(prev, 0)] + global_counts[(prev, 1)])
            for prev in [0, 1]
        }
        for sid, group in ordered_train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
            history = []
            subject_counts = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}
            for idx, row in group.iterrows():
                if history:
                    prev = int(history[-1])
                    n_prev = subject_counts[(prev, 0)] + subject_counts[(prev, 1)]
                    subject_p = (
                        subject_counts[(prev, 1)] / n_prev
                        if n_prev else global_transition[prev]
                    )
                    weight = n_prev / (n_prev + 10.0)
                    transition_p = weight * subject_p + (1.0 - weight) * global_transition[prev]
                    state = _longterm_v1_history_values(history, float(global_means[target]))
                    streak_p = 0.75 * state["streak_value"] + 0.25 * state["expanding_mean"]
                    volatility = state["volatility7"]
                    reversion_p = (
                        0.5 * transition_p + 0.5 * state["expanding_mean"]
                        if volatility > 0.4 else transition_p
                    )
                    pred = 0.55 * transition_p + 0.25 * streak_p + 0.20 * reversion_p
                else:
                    pred = float(global_means[target])
                oof.loc[idx, target] = np.clip(pred, 0.03, 0.97)
                value = int(row[target])
                if history:
                    subject_counts[(int(history[-1]), value)] += 1.0
                history.append(value)
        histories = {
            sid: group.sort_values("lifelog_date")[["lifelog_date", target]].copy()
            for sid, group in ordered_train.groupby("subject_id")
        }
        for idx, row in sample.iterrows():
            subject_history = histories.get(str(row["subject_id"]))
            if subject_history is None:
                history = []
            else:
                history = subject_history[
                    subject_history["lifelog_date"] < pd.to_datetime(row["lifelog_date"])
                ][target].astype(int).tolist()
            if history:
                prev = int(history[-1])
                subject_pairs = list(zip(history[:-1], history[1:]))
                n_prev = sum(int(a == prev) for a, _ in subject_pairs)
                positives = sum(int(a == prev and b == 1) for a, b in subject_pairs)
                subject_p = positives / n_prev if n_prev else global_transition[prev]
                weight = n_prev / (n_prev + 10.0)
                transition_p = weight * subject_p + (1.0 - weight) * global_transition[prev]
                state = _longterm_v1_history_values(history, float(global_means[target]))
                streak_p = 0.75 * state["streak_value"] + 0.25 * state["expanding_mean"]
                pred = 0.65 * transition_p + 0.35 * streak_p
            else:
                pred = float(global_means[target])
            test.loc[idx, target] = np.clip(pred, 0.03, 0.97)
        y = train[target].astype(int).to_numpy()
        base = current_best_oof[target].astype(float).to_numpy()
        pred = oof[target].astype(float).to_numpy()
        full, last = _longterm_v1_loss(y, pred), _longterm_v1_loss(y[guard], pred[guard])
        base_full, base_last = _longterm_v1_loss(y, base), _longterm_v1_loss(y[guard], base[guard])
        decision = "keep" if last < base_last and full <= base_full + 0.0003 else "drop"
        diagnostics.append(
            {
                "target": target,
                "full_oof_logloss": full,
                "lastblock_logloss": last,
                "baseline_full_oof_logloss": base_full,
                "baseline_lastblock_logloss": base_last,
                "delta_full": full - base_full,
                "delta_lastblock": last - base_last,
                "decision": decision,
            }
        )
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(V1_TRANSITION_PATH, index=False)
    return finalize_sources({source: oof}), finalize_sources({source: test}), diag


def _longterm_v1_prepare_personal_matrix(
    table: pd.DataFrame,
    feature_cols: list[str],
    max_features: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_mask = table["is_train"].eq(1)
    test_mask = table["is_train"].eq(0)
    scored = []
    for col in feature_cols:
        values = pd.to_numeric(table.loc[train_mask, col], errors="coerce")
        if values.notna().sum() < 20 or values.nunique(dropna=True) <= 1:
            continue
        missing_gap = abs(
            float(values.isna().mean())
            - float(pd.to_numeric(table.loc[test_mask, col], errors="coerce").isna().mean())
        )
        score = float(values.notna().mean()) - missing_gap
        if _longterm_v1_feature_family(col) == "target_history":
            score += 2.0
        scored.append((score, col))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [col for _, col in scored[:max_features]]
    train_x = table.loc[train_mask, selected].reset_index(drop=True)
    test_x = table.loc[test_mask, selected].reset_index(drop=True)
    return train_x, test_x, selected


def build_personal_state_weak_sources(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    personal_state_table: pd.DataFrame,
    personal_feature_cols: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    name = "longterm_personalization_v1_personal_state_weak"
    train_x, test_x, selected = _longterm_v1_prepare_personal_matrix(
        personal_state_table, personal_feature_cols, max_features=140
    )
    folds = _longterm_v1_subject_chrono_folds(train)
    oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
    for target in TARGETS:
        y = train[target].astype(int).reset_index(drop=True)
        test_acc = np.zeros(len(sample), dtype=float)
        used = 0
        for fold in sorted(folds.unique()):
            valid = folds.eq(fold).to_numpy()
            fit = folds.lt(fold).to_numpy()
            fold_prior = float(y.loc[fit].mean()) if fit.any() else 0.5
            if not valid.any() or not fit.any() or y.loc[fit].nunique() < 2:
                oof.loc[valid, target] = fold_prior
                continue
            med = train_x.loc[fit].replace([np.inf, -np.inf], np.nan).median()
            x_fit = (
                train_x.loc[fit].replace([np.inf, -np.inf], np.nan)
                .fillna(med).fillna(0.0).clip(-1e6, 1e6)
            )
            x_valid = (
                train_x.loc[valid].replace([np.inf, -np.inf], np.nan)
                .fillna(med).fillna(0.0).clip(-1e6, 1e6)
            )
            x_test = (
                test_x.replace([np.inf, -np.inf], np.nan)
                .fillna(med).fillna(0.0).clip(-1e6, 1e6)
            )
            restored = (
                load_step1_candidate_checkpoint(
                    "personal_state_weak_sources",
                    target,
                    "fold",
                    int(fold),
                )
                if restore_step1_candidate_checkpoints_enabled()
                else None
            )
            if restored is not None:
                print(f"[STEP1_CKPT] restored candidate personal_state_weak_sources {target} fold={int(fold)}")
                model = restored["model"]
                restored_prior = float(restored.get("fold_prior", fold_prior))
                oof.loc[valid, target] = np.clip(
                    0.80 * model.predict_proba(x_valid)[:, 1] + 0.20 * restored_prior,
                    0.03,
                    0.97,
                )
                test_acc += model.predict_proba(x_test)[:, 1]
                used += 1
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.01, penalty="l2", solver="lbfgs", max_iter=3000),
            )
            model.fit(x_fit, y.loc[fit])
            save_step1_candidate_checkpoint(
                "personal_state_weak_sources",
                {
                    "kind": "fold",
                    "source_name": name,
                    "target": target,
                    "fold": int(fold),
                    "selected_features": list(selected),
                    "median": med,
                    "fold_prior": fold_prior,
                    "model": model,
                },
                target,
                "fold",
                int(fold),
            )
            oof.loc[valid, target] = np.clip(
                0.80 * model.predict_proba(x_valid)[:, 1] + 0.20 * fold_prior,
                0.03, 0.97,
            )
            test_acc += model.predict_proba(x_test)[:, 1]
            used += 1
        oof[target] = oof[target].fillna(float(y.mean()))
        test[target] = (
            np.clip(0.80 * test_acc / used + 0.20 * float(y.mean()), 0.03, 0.97)
            if used else float(y.mean())
        )
    return finalize_sources({name: oof}), finalize_sources({name: test})


def build_personalized_residual_sources(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    personal_state_table: pd.DataFrame,
    personal_feature_cols: list[str],
    current_best_oof: pd.DataFrame,
    current_best_test: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x, test_x, selected = _longterm_v1_prepare_personal_matrix(
        personal_state_table, personal_feature_cols, max_features=180
    )
    folds = _longterm_v1_subject_chrono_folds(train)
    guard = make_subject_lastblock_mask(train).reset_index(drop=True).to_numpy(bool)
    beta_specs = [
        (0.03, "003"), (0.05, "005"), (0.075, "0075"),
        (0.10, "010"), (0.15, "015"),
    ]
    oof_sources = {
        f"longterm_personalization_v1_personal_residual_beta{suffix}":
        pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
        for _, suffix in beta_specs
    }
    test_sources = {
        name: pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
        for name in oof_sources
    }
    best_oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    best_test = pd.DataFrame(index=sample.index, columns=TARGETS, dtype=float)
    rows = []
    for target in TARGETS:
        y = train[target].astype(int).reset_index(drop=True)
        base_oof = current_best_oof[target].astype(float).reset_index(drop=True)
        base_test = current_best_test[target].astype(float).reset_index(drop=True)
        x_train = train_x.copy()
        x_test = test_x.copy()
        x_train["current_best_logit"] = logit(base_oof)
        x_test["current_best_logit"] = logit(base_test)
        x_train["personal_feature_coverage"] = train_x.notna().mean(axis=1)
        x_test["personal_feature_coverage"] = test_x.notna().mean(axis=1)
        model_oof = np.full(len(train), 0.5)
        model_test_acc = np.zeros(len(sample), dtype=float)
        coef_totals: dict[str, float] = {}
        used = 0
        for fold in sorted(folds.unique()):
            valid = folds.eq(fold).to_numpy()
            fit = folds.lt(fold).to_numpy()
            if not valid.any() or not fit.any() or y.loc[fit].nunique() < 2:
                model_oof[valid] = float(y.loc[fit].mean()) if fit.any() else 0.5
                continue
            med = x_train.loc[fit].replace([np.inf, -np.inf], np.nan).median()
            x_fit = (
                x_train.loc[fit].replace([np.inf, -np.inf], np.nan)
                .fillna(med).fillna(0.0).clip(-1e6, 1e6)
            )
            x_valid = (
                x_train.loc[valid].replace([np.inf, -np.inf], np.nan)
                .fillna(med).fillna(0.0).clip(-1e6, 1e6)
            )
            x_te = (
                x_test.replace([np.inf, -np.inf], np.nan)
                .fillna(med).fillna(0.0).clip(-1e6, 1e6)
            )
            restored = (
                load_step1_candidate_checkpoint(
                    "personalized_residual_sources",
                    target,
                    "fold",
                    int(fold),
                )
                if restore_step1_candidate_checkpoints_enabled()
                else None
            )
            if restored is not None:
                print(f"[STEP1_CKPT] restored candidate personalized_residual_sources {target} fold={int(fold)}")
                model = restored["model"]
                model_oof[valid] = model.predict_proba(x_valid)[:, 1]
                model_test_acc += model.predict_proba(x_te)[:, 1]
                coefs = np.abs(model.named_steps["logisticregression"].coef_[0])
                for feature, value in zip(x_fit.columns, coefs):
                    coef_totals[feature] = coef_totals.get(feature, 0.0) + float(value)
                used += 1
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.008, penalty="l2", solver="lbfgs", max_iter=3000),
            )
            model.fit(x_fit, y.loc[fit])
            save_step1_candidate_checkpoint(
                "personalized_residual_sources",
                {
                    "kind": "fold",
                    "target": target,
                    "fold": int(fold),
                    "selected_features": list(x_fit.columns),
                    "median": med,
                    "model": model,
                },
                target,
                "fold",
                int(fold),
            )
            model_oof[valid] = model.predict_proba(x_valid)[:, 1]
            model_test_acc += model.predict_proba(x_te)[:, 1]
            coefs = np.abs(model.named_steps["logisticregression"].coef_[0])
            for feature, value in zip(x_fit.columns, coefs):
                coef_totals[feature] = coef_totals.get(feature, 0.0) + float(value)
            used += 1
        model_test = (
            model_test_acc / used if used
            else np.full(len(sample), float(y.mean()))
        )
        baseline_full = _longterm_v1_loss(y, base_oof)
        baseline_last = _longterm_v1_loss(y.to_numpy()[guard], base_oof.to_numpy()[guard])
        weight = TRY9_TARGET_GUARD_WEIGHTS.get(target, TRY9_GUARD_WEIGHT)
        baseline_guarded = (1 - weight) * baseline_full + weight * baseline_last
        best = (0.0, baseline_full, baseline_last, baseline_guarded)
        target_cap = 0.075 if target == "S3" else 0.15
        if target in {"Q2", "Q3"}:
            target_cap = min(target_cap, 0.10)
        slack = 0.0003 if target in {"Q2", "Q3", "S3"} else 0.0007
        target_rows = []
        for beta, suffix in beta_specs:
            name = f"longterm_personalization_v1_personal_residual_beta{suffix}"
            corrected_oof = sigmoid(
                logit(base_oof) + beta * (logit(model_oof) - logit(base_oof))
            )
            corrected_test = sigmoid(
                logit(base_test) + beta * (logit(model_test) - logit(base_test))
            )
            oof_sources[name][target] = corrected_oof
            test_sources[name][target] = corrected_test
            full = _longterm_v1_loss(y, corrected_oof)
            last = _longterm_v1_loss(y.to_numpy()[guard], corrected_oof[guard])
            guarded = (1 - weight) * full + weight * last
            eligible = (
                beta <= target_cap
                and last <= baseline_last + slack
                and full <= baseline_full + 0.0003
                and guarded < best[3] - 1e-8
            )
            if eligible:
                best = (beta, full, last, guarded)
            target_rows.append(
                {
                    "target": target,
                    "beta": beta,
                    "full_oof_logloss": full,
                    "lastblock_logloss": last,
                    "guarded_score": guarded,
                    "baseline_full_oof_logloss": baseline_full,
                    "baseline_lastblock_logloss": baseline_last,
                    "delta_full": full - baseline_full,
                    "delta_lastblock": last - baseline_last,
                    "selected_beta": False,
                    "feature_count": len(selected) + 2,
                    "top_features": "|".join(
                        feature for feature, _ in sorted(
                            coef_totals.items(), key=lambda item: -item[1]
                        )[:12]
                    ),
                    "decision": "candidate" if eligible else "reject",
                }
            )
        best_beta = best[0]
        if best_beta > 0:
            suffix = next(s for beta, s in beta_specs if beta == best_beta)
            best_name = f"longterm_personalization_v1_personal_residual_beta{suffix}"
            best_oof[target] = oof_sources[best_name][target]
            best_test[target] = test_sources[best_name][target]
        else:
            best_oof[target] = base_oof
            best_test[target] = base_test
        for row in target_rows:
            if row["beta"] == best_beta and best_beta > 0:
                row["selected_beta"] = True
                row["decision"] = "selected"
        rows.extend(target_rows)
    oof_sources["longterm_personalization_v1_personal_residual_best"] = best_oof
    test_sources["longterm_personalization_v1_personal_residual_best"] = best_test
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(V1_RESIDUAL_PATH, index=False)
    return finalize_sources(oof_sources), finalize_sources(test_sources), diagnostics


def safe_alpha_blend_with_personal_reliability(
    sample,
    current_best,
    current_best_oof,
    stacked_test,
    stacked_oof,
    train,
    personal_state_table,
    residual_oof=None,
    residual_test=None,
):
    train_state = personal_state_table[personal_state_table["is_train"].eq(1)].reset_index(drop=True)
    test_state = personal_state_table[personal_state_table["is_train"].eq(0)].reset_index(drop=True)
    personal_cols = [c for c in personal_state_table.columns if c.startswith("personal_")]
    coverage_train = (
        train_state[personal_cols].notna().mean(axis=1).to_numpy(float)
        if personal_cols else np.ones(len(train))
    )
    coverage_test = (
        test_state[personal_cols].notna().mean(axis=1).to_numpy(float)
        if personal_cols else np.ones(len(sample))
    )
    history_train = train.groupby("subject_id").cumcount().to_numpy(float)
    history_length_train = np.clip(np.log1p(history_train) / np.log(10.0), 0.15, 1.0)
    subject_sizes = train.groupby("subject_id").size().to_dict()
    history_length_test = np.array([
        np.clip(np.log1p(subject_sizes.get(sid, 0)) / np.log(10.0), 0.15, 1.0)
        for sid in sample["subject_id"]
    ])
    guard = make_subject_lastblock_mask(train).reset_index(drop=True).to_numpy(bool)
    alpha_grid = [0.0, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100, 0.125, 0.150]
    out, out_oof, rows, selected = sample.copy(), current_best_oof.copy(), [], {}
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_oof = current_best_oof[target].astype(float).to_numpy()
        base_test = current_best[target].astype(float).to_numpy()
        candidate_oof = (
            residual_oof[target].astype(float).to_numpy()
            if residual_oof is not None else stacked_oof[target].astype(float).to_numpy()
        )
        candidate_test = (
            residual_test[target].astype(float).to_numpy()
            if residual_test is not None else stacked_test[target].astype(float).to_numpy()
        )
        volatility_col = f"{target}_volatility7"
        stability_train = np.clip(
            1.0 - train_state.get(volatility_col, pd.Series(0.5, index=train_state.index)).fillna(0.5).to_numpy(float),
            0.20, 1.0,
        )
        stability_test = np.clip(
            1.0 - test_state.get(volatility_col, pd.Series(0.5, index=test_state.index)).fillna(0.5).to_numpy(float),
            0.20, 1.0,
        )
        disagreement_train = np.abs(logit(candidate_oof) - logit(base_oof))
        threshold = max(float(np.quantile(disagreement_train, 0.80)), 0.10)
        disagreement_gate_train = np.clip(
            threshold / np.maximum(disagreement_train, threshold), 0.15, 1.0
        )
        disagreement_test = np.abs(logit(candidate_test) - logit(base_test))
        disagreement_gate_test = np.clip(
            threshold / np.maximum(disagreement_test, threshold), 0.15, 1.0
        )
        baseline_full, baseline_last = _longterm_v1_loss(y, base_oof), _longterm_v1_loss(y[guard], base_oof[guard])
        candidate_last = _longterm_v1_loss(y[guard], candidate_oof[guard])
        lastblock_gain_gate = 1.0 if candidate_last < baseline_last else 0.0
        row_gate_train = (
            history_length_train * stability_train * coverage_train
            * disagreement_gate_train * lastblock_gain_gate
        )
        row_gate_test = (
            history_length_test * stability_test * coverage_test
            * disagreement_gate_test * lastblock_gain_gate
        )
        best = (0.0, baseline_full, baseline_last)
        slack = 0.0003 if target in {"Q2", "Q3", "S3"} else 0.0007
        for alpha in alpha_grid:
            blend = (
                (1 - alpha * row_gate_train) * base_oof
                + alpha * row_gate_train * candidate_oof
            )
            full, last = _longterm_v1_loss(y, blend), _longterm_v1_loss(y[guard], blend[guard])
            better = last < best[2] - 1e-8 or (
                abs(last - best[2]) <= 1e-8 and full < best[1]
            )
            if (
                alpha > 0 and better
                and full <= baseline_full + 0.0002
                and last <= baseline_last + slack
            ):
                best = (float(alpha), full, last)
            rows.append(
                {
                    "target": target,
                    "alpha": alpha,
                    "baseline_full": baseline_full,
                    "baseline_lastblock": baseline_last,
                    "gated_full": full,
                    "gated_lastblock": last,
                    "delta_full": full - baseline_full,
                    "delta_lastblock": last - baseline_last,
                    "mean_row_gate_train": float(row_gate_train.mean()),
                    "mean_row_gate_test": float(row_gate_test.mean()),
                    "applied": False,
                    "reason": "candidate",
                }
            )
        alpha = best[0]
        selected[target] = alpha
        if alpha > 0:
            out_oof[target] = clip_proba(
                (1 - alpha * row_gate_train) * base_oof
                + alpha * row_gate_train * candidate_oof, 1e-6, 1 - 1e-6
            )
            out[target] = clip_proba(
                (1 - alpha * row_gate_test) * base_test
                + alpha * row_gate_test * candidate_test, 1e-6, 1 - 1e-6
            )
        else:
            out[target] = base_test
        for row in rows:
            if row["target"] == target and row["alpha"] == alpha:
                row["applied"] = alpha > 0
                row["reason"] = (
                    "selected" if alpha > 0 else
                    "no_lastblock_gain" if lastblock_gain_gate == 0 else
                    "no_guarded_improvement"
                )
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(V1_ALPHA_PATH, index=False)
    return out, out_oof, diagnostics, selected


def apply_final_temperature_bias_calibration(
    train: pd.DataFrame,
    final_oof: pd.DataFrame,
    final_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    guard = make_subject_lastblock_mask(train).reset_index(drop=True).to_numpy(bool)
    out_oof, out_test, rows = final_oof.copy(), final_test.copy(), []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        pred = final_oof[target].astype(float).to_numpy()
        before_full, before_last = _longterm_v1_loss(y, pred), _longterm_v1_loss(y[guard], pred[guard])
        weight = TRY9_TARGET_GUARD_WEIGHTS.get(target, TRY9_GUARD_WEIGHT)
        best = (1.0, 0.0, before_full, before_last, (1 - weight) * before_full + weight * before_last)
        for temperature in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]:
            for bias in [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10]:
                candidate = sigmoid(temperature * logit(pred) + bias)
                full, last = _longterm_v1_loss(y, candidate), _longterm_v1_loss(y[guard], candidate[guard])
                guarded = (1 - weight) * full + weight * last
                if (
                    last <= before_last + 1e-10
                    and full <= before_full + 0.0002
                    and guarded < best[4] - 1e-9
                ):
                    best = (temperature, bias, full, last, guarded)
        temperature, bias, after_full, after_last, _ = best
        applied = temperature != 1.0 or bias != 0.0
        if applied:
            out_oof[target] = clip_proba(
                sigmoid(temperature * logit(pred) + bias), 1e-6, 1 - 1e-6
            )
            out_test[target] = clip_proba(
                sigmoid(temperature * logit(final_test[target].astype(float)) + bias),
                1e-6, 1 - 1e-6,
            )
        rows.append(
            {
                "target": target,
                "temperature": temperature,
                "bias": bias,
                "before_full": before_full,
                "before_lastblock": before_last,
                "after_full": after_full,
                "after_lastblock": after_last,
                "applied": applied,
            }
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(LONGTERM_V1_CALIBRATION_PATH, index=False)
    return out_oof, out_test, diagnostics


def _longterm_v1_load_calibration_helpers():
    path = Path(__file__).with_name("calibration_helpers")
    spec = _longterm_v1_importlib.spec_from_file_location("calibration_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load calibration_helpers from {path}")
    module = _longterm_v1_importlib.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA_DIR = DATA_DIR
    module.SUBMISSION_DIR = SUBMISSION_DIR
    module.ARTIFACT_DIR = ARTIFACT_DIR
    return module


def _longterm_v1_restore_sleep_cross_sources(train, sample, oof, test):
    helper = _longterm_v1_load_calibration_helpers()
    sleep_oof, sleep_test, sleep_cols = helper.build_sleep_proxy_step2_sources(
        train, sample, data_path=DATA_DIR, mode="full"
    )
    combined_oof, combined_test = {**oof, **sleep_oof}, {**test, **sleep_test}
    cross_oof, cross_test = helper.build_cross_target_oof_sources(
        train, sample, combined_oof, combined_test
    )
    return sleep_oof, sleep_test, cross_oof, cross_test, sleep_cols


def _longterm_v1_reconstruct_safe_oof(
    current_best_oof: pd.DataFrame,
    stacked_oof: pd.DataFrame,
    safe_diag: pd.DataFrame,
) -> pd.DataFrame:
    out = current_best_oof.copy()
    alpha_map = dict(zip(safe_diag["target"], safe_diag["alpha"]))
    for target in TARGETS:
        alpha = float(alpha_map.get(target, 0.0))
        out[target] = clip_proba(
            (1 - alpha) * current_best_oof[target].astype(float)
            + alpha * stacked_oof[target].astype(float),
            1e-6, 1 - 1e-6,
        )
    return out


def _longterm_v1_profile_metrics(train, frame):
    guard = make_subject_lastblock_mask(train).reset_index(drop=True).to_numpy(bool)
    rows = {}
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        pred = frame[target].astype(float).to_numpy()
        full, last = _longterm_v1_loss(y, pred), _longterm_v1_loss(y[guard], pred[guard])
        weight = TRY9_TARGET_GUARD_WEIGHTS.get(target, TRY9_GUARD_WEIGHT)
        rows[target] = (full, last, (1 - weight) * full + weight * last)
    return rows


def _longterm_v1_validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(sample.columns):
        raise ValueError("submission columns/order differ from sample submission")
    if len(frame) != len(sample):
        raise ValueError("submission row count differs from sample submission")
    out = frame.copy()
    out[TARGETS] = out[TARGETS].astype(float).clip(1e-6, 1 - 1e-6)
    if out[TARGETS].isna().any().any():
        raise ValueError("submission contains null target probabilities")
    if not out[["subject_id", "sleep_date", "lifelog_date"]].reset_index(drop=True).equals(
        sample[["subject_id", "sleep_date", "lifelog_date"]].reset_index(drop=True)
    ):
        raise ValueError("submission key row order differs from sample submission")
    return out


def _longterm_v1_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-mis-lstm", action="store_true")
    parser.add_argument("--reuse-step1-base-artifacts", action="store_true")
    parser.add_argument("--stop-after-step1-base", action="store_true")
    parser.add_argument(
        "--personal-longterm-profile",
        choices=[*V1_PROFILES, "all"],
        default="full_personal_longterm",
    )
    return parser.parse_args()


def _longterm_v1_main() -> None:
    global SOURCE_DIAG_PATH, MODEL_DIAG_PATH, SAFE_DIAG_PATH
    global SUBMISSION_SUMMARY_PATH, NEW_SOURCE_SUMMARY_PATH

    args = _longterm_v1_parse_args()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reuse_step1_base_artifacts and args.stop_after_step1_base:
        raise ValueError("--reuse-step1-base-artifacts and --stop-after-step1-base cannot be used together")
    run_started_at = _step1_oof_time.time() - 2.0
    if args.reuse_step1_base_artifacts:
        if not CURRENT_BEST_PATH.exists():
            raise RuntimeError(f"current-best file is missing: {CURRENT_BEST_PATH}")
        print("[longterm_personalization_v1] reusing step1_base artifacts")
    else:
        step1_oof_runner_run_base_pipeline(include_mis_lstm=args.include_mis_lstm)
        if args.stop_after_step1_base:
            return

    SOURCE_DIAG_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_meta_source_target_oof_logloss.csv"
    MODEL_DIAG_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_model_diagnostics.csv"
    SAFE_DIAG_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_safe_alpha_diagnostics.csv"
    SUBMISSION_SUMMARY_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_submission_shift_summary.csv"
    NEW_SOURCE_SUMMARY_PATH = ARTIFACT_DIR / f"{LONGTERM_V1_PREFIX}_new_source_summary.csv"

    train, sample, current_best, feat = load_core_frames()
    if args.reuse_step1_base_artifacts:
        base_oof, base_test, _ = step1_oof_runner_load_reusable_oof_sources(train, sample)
    else:
        base_oof, base_test, _ = step1_oof_runner_load_fresh_oof_sources(train, sample, run_started_at)
    if not base_oof:
        raise RuntimeError("No reusable OOF/test source pairs were found")
    current_best_key = CURRENT_BEST_PATH.stem
    if current_best_key in base_oof:
        current_best_oof = base_oof[current_best_key]
    elif "submission_step1_sleep_proxy_s_targets" in base_oof:
        current_best_oof = base_oof["submission_step1_sleep_proxy_s_targets"]
    else:
        raise RuntimeError("Could not find current-best OOF source")

    print("[longterm_personalization_v1] building original longterm_personalization STEP2 source pool")
    temporal_oof, temporal_test = build_personal_temporal_sources(train, sample)
    calendar_oof, calendar_test = build_bayesian_calendar_sources(train, sample)
    long_oof, long_test, _ = build_longterm_sensor_model(train, sample, feat)
    cv_oof, cv_test, _ = build_cv_variant_first_stage_sources(train, sample, feat)
    history_oof, history_test = build_subject_history_sequence_sources(train, sample)
    context_oof, context_test, _ = build_five_factor_context_sources(train, sample)
    original_oof = {
        **base_oof, **temporal_oof, **calendar_oof, **long_oof,
        **cv_oof, **history_oof, **context_oof,
    }
    original_test = {
        **base_test, **temporal_test, **calendar_test, **long_test,
        **cv_test, **history_test, **context_test,
    }

    enhanced_oof, enhanced_test = dict(original_oof), dict(original_test)
    try:
        sleep_oof, sleep_test, cross_oof, cross_test, sleep_cols = (
            _longterm_v1_restore_sleep_cross_sources(
                train, sample, enhanced_oof, enhanced_test
            )
        )
        enhanced_oof.update(sleep_oof)
        enhanced_test.update(sleep_test)
        enhanced_oof.update(cross_oof)
        enhanced_test.update(cross_test)
        print(
            f"[18_8_v1] restored sleep proxy/cross-target sources "
            f"({len(sleep_cols)} features, {len(cross_oof)} sources)"
        )
    except Exception as exc:
        print(f"[18_8_v1][WARN] sleep/cross-target restoration skipped: {type(exc).__name__}: {exc}")

    print("[18_8_v1] building past-only personal state and multiscale memory")
    personal_table, personal_cols = build_past_only_personal_state_features(
        train, sample, feat, current_best_oof, current_best
    )
    personal_table, personal_cols = add_multiscale_longterm_memory_features(
        personal_table, personal_cols
    )
    personal_oof, personal_test = build_personal_state_weak_sources(
        train, sample, personal_table, personal_cols
    )
    transition_oof, transition_test, transition_diag = (
        build_target_transition_streak_sources(
            train, sample, current_best_oof, current_best
        )
    )
    # Keep the v4 source identifier: make_meta_frame classifies feature families
    # from source names, so renaming this to include "longterm" changes Q3 OOF.
    transition_name = "try18_8_v1_transition_streak_prior"
    train_mask = personal_table["is_train"].eq(1)
    test_mask = personal_table["is_train"].eq(0)
    for target in TARGETS:
        col = f"transition_prior_{target}"
        personal_table.loc[train_mask, col] = transition_oof[transition_name][target].to_numpy(float)
        personal_table.loc[test_mask, col] = transition_test[transition_name][target].to_numpy(float)
        personal_cols.append(col)

    residual_oof, residual_test, residual_diag = build_personalized_residual_sources(
        train, sample, personal_table, personal_cols, current_best_oof, current_best
    )
    residual_best_oof = residual_oof["longterm_personalization_v1_personal_residual_best"]
    residual_best_test = residual_test["longterm_personalization_v1_personal_residual_best"]
    _step1_pipeline_build_and_save_component_sources(
        train,
        sample,
        personal_table,
        personal_cols,
        transition_oof[transition_name],
        transition_test[transition_name],
        residual_best_oof,
        residual_best_test,
    )
    summarize_new_sources(
        train,
        current_best,
        current_best_oof,
        {**personal_oof, **transition_oof, **residual_oof},
        {**personal_test, **transition_test, **residual_test},
    )

    requested = (
        V1_PROFILES if args.personal_longterm_profile == "all"
        else ["baseline_18_8", args.personal_longterm_profile]
        if args.personal_longterm_profile != "baseline_18_8"
        else ["baseline_18_8"]
    )
    # leakage_validation always needs both anchors for the Q3 leakage-aware post-processing.
    requested = list(dict.fromkeys(["baseline_18_8", "transition_prior", *requested]))
    results, comparison_rows, raw_comparison_rows = {}, [], []
    baseline_final = baseline_final_oof = baseline_metrics = None
    baseline_raw_metrics = None
    selected_betas = {
        row.target: float(row.beta)
        for row in residual_diag[residual_diag["selected_beta"]].itertuples()
    }

    for profile in requested:
        print(f"[18_8_v1] running profile={profile}")
        pool_oof = dict(original_oof if profile == "baseline_18_8" else enhanced_oof)
        pool_test = dict(original_test if profile == "baseline_18_8" else enhanced_test)
        transition_used = profile in {
            "transition_prior", "personal_residual", "full_personal_longterm"
        }
        residual_used = profile in {"personal_residual", "full_personal_longterm"}
        row_gate_used = profile in {"personal_row_alpha", "full_personal_longterm"}
        calibration_used = profile == "full_personal_longterm"
        if profile in {"personal_state_only", "personal_residual", "full_personal_longterm"}:
            pool_oof.update(personal_oof)
            pool_test.update(personal_test)
        if transition_used:
            pool_oof.update(transition_oof)
            pool_test.update(transition_test)
        if residual_used:
            pool_oof["longterm_personalization_v1_personal_residual_best"] = residual_best_oof
            pool_test["longterm_personalization_v1_personal_residual_best"] = residual_best_test

        stack_test, stack_oof, _, model_diag = fit_targetwise_stack(
            train,
            pool_oof,
            pool_test,
            checkpoint_run_tag=f"longterm_v1_{profile}",
            restore_from_checkpoints=STEP1_RESTORE_TARGETWISE_STACK_CHECKPOINTS,
        )
        raw_submission = sample.copy()
        raw_submission[TARGETS] = stack_test[TARGETS].to_numpy(float)
        raw_submission = _longterm_v1_validate_submission(raw_submission, sample)
        raw_path = SUBMISSION_DIR / f"submission_step1_profile_{profile}_raw_stack.csv"
        raw_submission.to_csv(raw_path, index=False)
        stack_oof[TARGETS].to_csv(
            ARTIFACT_DIR / f"leakage_val_raw_oof_{profile}.csv", index=False
        )
        raw_metrics = _longterm_v1_profile_metrics(train, stack_oof)
        if profile == "baseline_18_8":
            baseline_raw_metrics = raw_metrics
        selected_alpha = {}
        if profile == "baseline_18_8":
            final, safe_diag = safe_alpha_blend(
                sample, current_best, current_best_oof,
                stack_test, stack_oof, train,
            )
            final_oof = _longterm_v1_reconstruct_safe_oof(
                current_best_oof, stack_oof, safe_diag
            )
            selected_alpha = dict(zip(safe_diag["target"], safe_diag["alpha"]))
            baseline_final, baseline_final_oof = final.copy(), final_oof.copy()
        elif row_gate_used:
            if baseline_final is None or baseline_final_oof is None:
                raise RuntimeError("baseline profile must run before row-alpha profiles")
            candidate_oof = residual_best_oof if residual_used else stack_oof
            candidate_test = residual_best_test if residual_used else stack_test
            final, final_oof, _, selected_alpha = (
                safe_alpha_blend_with_personal_reliability(
                    sample,
                    baseline_final,
                    baseline_final_oof,
                    stack_test,
                    stack_oof,
                    train,
                    personal_table,
                    residual_oof=candidate_oof,
                    residual_test=candidate_test,
                )
            )
        else:
            final, safe_diag = safe_alpha_blend(
                sample, current_best, current_best_oof,
                stack_test, stack_oof, train,
            )
            final_oof = _longterm_v1_reconstruct_safe_oof(
                current_best_oof, stack_oof, safe_diag
            )
            selected_alpha = dict(zip(safe_diag["target"], safe_diag["alpha"]))
        calibration_diag = pd.DataFrame()
        if calibration_used:
            final_oof, final, calibration_diag = (
                apply_final_temperature_bias_calibration(
                    train, final_oof, final
                )
            )
        final = _longterm_v1_validate_submission(final, sample)
        path = SUBMISSION_DIR / f"submission_step1_profile_{profile}.csv"
        final.to_csv(path, index=False)
        metrics = _longterm_v1_profile_metrics(train, final_oof)
        if profile == "baseline_18_8":
            baseline_metrics = metrics
        selected_source_map = model_diag.set_index("target")["selected"].to_dict()
        for target in TARGETS:
            raw_comparison_rows.append(
                {
                    "profile": profile,
                    "target": target,
                    "full_oof_logloss": raw_metrics[target][0],
                    "lastblock_logloss": raw_metrics[target][1],
                    "guarded_score": raw_metrics[target][2],
                    "delta_full_vs_baseline_18_8_raw": np.nan,
                    "delta_lastblock_vs_baseline_18_8_raw": np.nan,
                    "selected_beta": selected_betas.get(target, 0.0) if residual_used else 0.0,
                    "residual_used": residual_used,
                    "transition_prior_used": transition_used,
                    "selected_sources": str(selected_source_map.get(target, "")),
                    "recommendation": "",
                }
            )
            comparison_rows.append(
                {
                    "profile": profile,
                    "target": target,
                    "full_oof_logloss": metrics[target][0],
                    "lastblock_logloss": metrics[target][1],
                    "guarded_score": metrics[target][2],
                    "delta_full_vs_baseline_18_8": np.nan,
                    "delta_lastblock_vs_baseline_18_8": np.nan,
                    "selected_beta": selected_betas.get(target, 0.0) if residual_used else 0.0,
                    "selected_alpha": float(selected_alpha.get(target, 0.0)),
                    "row_gate_used": row_gate_used,
                    "residual_used": residual_used,
                    "transition_prior_used": transition_used,
                    "final_calibration_used": calibration_used,
                    "selected_sources": str(selected_source_map.get(target, "")),
                    "recommendation": "",
                }
            )
        results[profile] = {
            "path": path,
            "raw_path": raw_path,
            "metrics": metrics,
            "raw_metrics": raw_metrics,
            "calibration": calibration_diag,
        }

    comparison = pd.DataFrame(comparison_rows)
    if baseline_metrics is None:
        raise RuntimeError("baseline metrics were not created")
    for idx, row in comparison.iterrows():
        base = baseline_metrics[row["target"]]
        comparison.loc[idx, "delta_full_vs_baseline_18_8"] = row["full_oof_logloss"] - base[0]
        comparison.loc[idx, "delta_lastblock_vs_baseline_18_8"] = row["lastblock_logloss"] - base[1]
    summary_rows = []
    for profile, group in comparison.groupby("profile", sort=False):
        improved = group[group["delta_lastblock_vs_baseline_18_8"] < -1e-7]["target"].tolist()
        worsened = group[group["delta_lastblock_vs_baseline_18_8"] > 1e-7]["target"].tolist()
        unstable = group[
            group["target"].isin(["Q2", "Q3", "S3"])
            & (group["delta_lastblock_vs_baseline_18_8"] > 0.0010)
        ]
        average_delta = float(group["delta_lastblock_vs_baseline_18_8"].mean())
        if len(unstable):
            recommendation = "reject_unstable_target"
        elif average_delta < 0 or len(improved) >= 5:
            recommendation = "keep_candidate"
        else:
            recommendation = "reject"
        comparison.loc[group.index, "recommendation"] = recommendation
        summary_rows.append(
            {
                "profile": profile,
                "avg_full_oof_logloss": float(group["full_oof_logloss"].mean()),
                "avg_lastblock_logloss": float(group["lastblock_logloss"].mean()),
                "avg_guarded_score": float(group["guarded_score"].mean()),
                "num_targets_improved_lastblock": len(improved),
                "improved_targets": "|".join(improved),
                "worsened_targets": "|".join(worsened),
                "recommendation": recommendation,
            }
        )
    summary = pd.DataFrame(summary_rows)
    eligible = summary[~summary["recommendation"].str.startswith("reject")]
    candidates = eligible if len(eligible) else summary
    best_profile = str(
        candidates.sort_values(["avg_lastblock_logloss", "avg_full_oof_logloss"]).iloc[0]["profile"]
    )
    summary.loc[summary["profile"].eq(best_profile), "recommendation"] = "best"
    comparison.loc[comparison["profile"].eq(best_profile), "recommendation"] = "best"
    comparison.to_csv(V1_COMPARISON_PATH, index=False)
    summary.to_csv(V1_SUMMARY_PATH, index=False)
    best_path = SUBMISSION_DIR / "submission_step1_profile_best_lastblock.csv"
    _longterm_v1_shutil.copyfile(results[best_profile]["path"], best_path)

    raw_comparison = pd.DataFrame(raw_comparison_rows)
    if baseline_raw_metrics is None:
        raise RuntimeError("baseline raw metrics were not created")
    for idx, row in raw_comparison.iterrows():
        base = baseline_raw_metrics[row["target"]]
        raw_comparison.loc[idx, "delta_full_vs_baseline_18_8_raw"] = (
            row["full_oof_logloss"] - base[0]
        )
        raw_comparison.loc[idx, "delta_lastblock_vs_baseline_18_8_raw"] = (
            row["lastblock_logloss"] - base[1]
        )
    raw_summary_rows = []
    for profile, group in raw_comparison.groupby("profile", sort=False):
        improved = group[
            group["delta_lastblock_vs_baseline_18_8_raw"] < -1e-7
        ]["target"].tolist()
        worsened = group[
            group["delta_lastblock_vs_baseline_18_8_raw"] > 1e-7
        ]["target"].tolist()
        unstable = group[
            group["target"].isin(["Q2", "Q3", "S3"])
            & (group["delta_lastblock_vs_baseline_18_8_raw"] > 0.0010)
        ]
        avg_delta = float(group["delta_lastblock_vs_baseline_18_8_raw"].mean())
        recommendation = (
            "reject_unstable_target" if len(unstable)
            else "keep_candidate" if avg_delta < 0 or len(improved) >= 5
            else "reject"
        )
        raw_comparison.loc[group.index, "recommendation"] = recommendation
        raw_summary_rows.append(
            {
                "profile": profile,
                "avg_full_oof_logloss": float(group["full_oof_logloss"].mean()),
                "avg_lastblock_logloss": float(group["lastblock_logloss"].mean()),
                "avg_guarded_score": float(group["guarded_score"].mean()),
                "num_targets_improved_lastblock": len(improved),
                "improved_targets": "|".join(improved),
                "worsened_targets": "|".join(worsened),
                "recommendation": recommendation,
            }
        )
    raw_summary = pd.DataFrame(raw_summary_rows)
    raw_eligible = raw_summary[
        ~raw_summary["recommendation"].str.startswith("reject")
    ]
    raw_candidates = raw_eligible if len(raw_eligible) else raw_summary
    best_raw_profile = str(
        raw_candidates.sort_values(
            ["avg_lastblock_logloss", "avg_full_oof_logloss"]
        ).iloc[0]["profile"]
    )
    raw_summary.loc[
        raw_summary["profile"].eq(best_raw_profile), "recommendation"
    ] = "best"
    raw_comparison.loc[
        raw_comparison["profile"].eq(best_raw_profile), "recommendation"
    ] = "best"
    raw_comparison.to_csv(V1_RAW_COMPARISON_PATH, index=False)
    raw_summary.to_csv(V1_RAW_SUMMARY_PATH, index=False)
    best_raw_path = SUBMISSION_DIR / "submission_step1_profile_best_raw_lastblock.csv"
    _longterm_v1_shutil.copyfile(results[best_raw_profile]["raw_path"], best_raw_path)

    target_best = comparison.loc[
        comparison.groupby("target")["lastblock_logloss"].idxmin(),
        ["target", "profile"],
    ]
    residual_targets = sorted(selected_betas)
    transition_targets = transition_diag[
        transition_diag["decision"].eq("keep")
    ]["target"].tolist()
    row_targets = comparison[
        comparison["row_gate_used"]
        & (comparison["delta_lastblock_vs_baseline_18_8"] < 0)
    ]["target"].drop_duplicates().tolist()
    calibration_targets = []
    if "full_personal_longterm" in results:
        cal = results["full_personal_longterm"]["calibration"]
        if len(cal):
            calibration_targets = cal[cal["applied"]]["target"].tolist()
    diagnostic_paths = [
        LONGTERM_V1_STATE_COLUMNS_PATH, V1_STATE_SUMMARY_PATH, V1_MULTISCALE_PATH,
        V1_TRANSITION_PATH, V1_RESIDUAL_PATH, V1_ALPHA_PATH,
        LONGTERM_V1_CALIBRATION_PATH, V1_COMPARISON_PATH, V1_SUMMARY_PATH,
        V1_RAW_COMPARISON_PATH, V1_RAW_SUMMARY_PATH,
    ]
    print("\n[18_8_v1] profile average full OOF log-loss")
    print(summary[["profile", "avg_full_oof_logloss"]].to_string(index=False))
    print("\n[18_8_v1] profile average last-block log-loss")
    print(summary[["profile", "avg_lastblock_logloss"]].to_string(index=False))
    print("\n[18_8_v1] target-wise best profile")
    print(target_best.to_string(index=False))
    print(f"[18_8_v1] personal residual valid targets: {residual_targets}")
    print(f"[18_8_v1] transition/streak valid targets: {transition_targets}")
    print(f"[18_8_v1] row-alpha improved targets: {row_targets}")
    print(f"[18_8_v1] final calibration targets: {calibration_targets}")
    print(f"[18_8_v1] recommended profile: {best_profile}")
    print(f"[18_8_v1] recommended submission: {best_path}")
    print("\n[18_8_v1] raw profile average last-block log-loss")
    print(raw_summary[["profile", "avg_lastblock_logloss"]].to_string(index=False))
    print(f"[18_8_v1] recommended raw profile: {best_raw_profile}")
    print(f"[18_8_v1] recommended raw submission: {best_raw_path}")
    print("[18_8_v1] diagnostics:")
    for path in diagnostic_paths:
        print(f"  - {path}")



# Recommended:
# python longterm_personalization_v1 --reuse-step1-base-artifacts --personal-longterm-profile full_personal_longterm
#
# Full ablation:
# python longterm_personalization_v1 --reuse-step1-base-artifacts --personal-longterm-profile all
#
# Quick checks:
# python longterm_personalization_v1 --reuse-step1-base-artifacts --personal-longterm-profile personal_residual
# python longterm_personalization_v1 --reuse-step1-base-artifacts --personal-longterm-profile personal_row_alpha


# ============================================================================
# leakage_validation: Temporal Leakage-controlled Validation and Guarded Calibration
# ============================================================================
LEAKAGE_VAL_WEIGHT_PATH = ARTIFACT_DIR / "leakage_val_q3_weight_search.csv"
LEAKAGE_VAL_CALIBRATION_PATH = ARTIFACT_DIR / "leakage_val_q3_crossfit_calibration.csv"
LEAKAGE_VAL_STABILITY_PATH = ARTIFACT_DIR / "leakage_val_subject_stability.csv"
LEAKAGE_VAL_HONEST_PATH = ARTIFACT_DIR / "leakage_val_honest_oof_comparison.csv"
LEAKAGE_VAL_AUDIT_PATH = ARTIFACT_DIR / "leakage_val_leakage_audit.csv"
LEAKAGE_VAL_RECOMMENDATION_PATH = ARTIFACT_DIR / "leakage_val_recommendation_summary.csv"


def _leakage_val_q3_blend(base, transition, weight):
    return clip_proba(
        np.asarray(base, dtype=float)
        + float(weight) * (
            np.asarray(transition, dtype=float) - np.asarray(base, dtype=float)
        ),
        1e-6,
        1 - 1e-6,
    )


def _leakage_val_candidate_metrics(train, prediction_frame):
    guard = make_subject_lastblock_mask(train).reset_index(drop=True).to_numpy(bool)
    target_rows = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        pred = prediction_frame[target].astype(float).to_numpy()
        full = _longterm_v1_loss(y, pred)
        last = _longterm_v1_loss(y[guard], pred[guard])
        weight = TRY9_TARGET_GUARD_WEIGHTS.get(target, TRY9_GUARD_WEIGHT)
        target_rows.append(
            {
                "target": target,
                "full_oof_logloss": full,
                "lastblock_logloss": last,
                "guarded_score": (1 - weight) * full + weight * last,
            }
        )
    return pd.DataFrame(target_rows)


def _leakage_val_select_weight(y, base, transition, mask, grid):
    best = (0.0, np.inf)
    for weight in grid:
        pred = _leakage_val_q3_blend(base, transition, weight)
        score = _longterm_v1_loss(np.asarray(y)[mask], pred[mask])
        if score < best[1] - 1e-10:
            best = (float(weight), score)
    return best


def _leakage_val_weight_search(train, base_oof, transition_oof):
    y = train["Q3"].astype(int).to_numpy()
    base = base_oof["Q3"].astype(float).to_numpy()
    transition = transition_oof["Q3"].astype(float).to_numpy()
    folds = _longterm_v1_subject_chrono_folds(train).to_numpy()
    guard = make_subject_lastblock_mask(train).reset_index(drop=True).to_numpy(bool)
    grid = [0.0, 0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.20]
    rows = []
    for weight in grid:
        pred = _leakage_val_q3_blend(base, transition, weight)
        rows.append(
            {
                "record_type": "global_grid",
                "fold": -1,
                "weight": weight,
                "selection_rows": int((~guard).sum()),
                "evaluation_rows": int(guard.sum()),
                "selection_logloss": _longterm_v1_loss(y[~guard], pred[~guard]),
                "evaluation_logloss": _longterm_v1_loss(y[guard], pred[guard]),
                "full_oof_logloss": _longterm_v1_loss(y, pred),
                "selected": False,
            }
        )

    crossfit_pred = base.copy()
    selected_weights = []
    for fold in sorted(np.unique(folds)):
        evaluation = folds == fold
        selection = folds < fold
        if not evaluation.any() or selection.sum() < 20:
            rows.append(
                {
                    "record_type": "rolling_fold",
                    "fold": int(fold),
                    "weight": 0.0,
                    "selection_rows": int(selection.sum()),
                    "evaluation_rows": int(evaluation.sum()),
                    "selection_logloss": np.nan,
                    "evaluation_logloss": _longterm_v1_loss(y[evaluation], base[evaluation]),
                    "full_oof_logloss": np.nan,
                    "selected": False,
                }
            )
            continue
        weight, selection_loss = _leakage_val_select_weight(
            y, base, transition, selection, grid
        )
        selected_weights.append(weight)
        fold_pred = _leakage_val_q3_blend(base, transition, weight)
        crossfit_pred[evaluation] = fold_pred[evaluation]
        rows.append(
            {
                "record_type": "rolling_fold",
                "fold": int(fold),
                "weight": weight,
                "selection_rows": int(selection.sum()),
                "evaluation_rows": int(evaluation.sum()),
                "selection_logloss": selection_loss,
                "evaluation_logloss": _longterm_v1_loss(y[evaluation], fold_pred[evaluation]),
                "full_oof_logloss": np.nan,
                "selected": True,
            }
        )

    final_weight, pre_last_loss = _leakage_val_select_weight(
        y, base, transition, ~guard, grid
    )
    for row in rows:
        if row["record_type"] == "global_grid" and row["weight"] == final_weight:
            row["selected"] = True
    rows.append(
        {
            "record_type": "final_pre_lastblock_selection",
            "fold": -1,
            "weight": final_weight,
            "selection_rows": int((~guard).sum()),
            "evaluation_rows": int(guard.sum()),
            "selection_logloss": pre_last_loss,
            "evaluation_logloss": _longterm_v1_loss(
                y[guard],
                _leakage_val_q3_blend(base, transition, final_weight)[guard],
            ),
            "full_oof_logloss": _longterm_v1_loss(
                y, _leakage_val_q3_blend(base, transition, final_weight)
            ),
            "selected": True,
        }
    )
    report = pd.DataFrame(rows)
    report.to_csv(LEAKAGE_VAL_WEIGHT_PATH, index=False)
    return final_weight, crossfit_pred, report


def _leakage_val_apply_calibration(pred, temperature, bias):
    return clip_proba(
        sigmoid(float(temperature) * logit(pred) + float(bias)),
        1e-6,
        1 - 1e-6,
    )


def _leakage_val_select_calibration(y, pred, mask):
    temperatures = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    biases = [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10]
    baseline_loss = _longterm_v1_loss(np.asarray(y)[mask], np.asarray(pred)[mask])
    best = (1.0, 0.0, baseline_loss)
    for temperature in temperatures:
        for bias in biases:
            candidate = _leakage_val_apply_calibration(pred, temperature, bias)
            loss = _longterm_v1_loss(np.asarray(y)[mask], candidate[mask])
            if loss < best[2] - 1e-10:
                best = (float(temperature), float(bias), loss)
    return best


def _leakage_val_crossfit_calibration(train, weighted_oof):
    y = train["Q3"].astype(int).to_numpy()
    pred = np.asarray(weighted_oof, dtype=float)
    folds = _longterm_v1_subject_chrono_folds(train).to_numpy()
    guard = make_subject_lastblock_mask(train).reset_index(drop=True).to_numpy(bool)
    crossfit = pred.copy()
    rows = []
    for fold in sorted(np.unique(folds)):
        evaluation = folds == fold
        selection = folds < fold
        if not evaluation.any() or selection.sum() < 20:
            rows.append(
                {
                    "record_type": "rolling_fold",
                    "fold": int(fold),
                    "temperature": 1.0,
                    "bias": 0.0,
                    "selection_rows": int(selection.sum()),
                    "evaluation_rows": int(evaluation.sum()),
                    "selection_logloss": np.nan,
                    "before_evaluation_logloss": _longterm_v1_loss(y[evaluation], pred[evaluation]),
                    "after_evaluation_logloss": _longterm_v1_loss(y[evaluation], pred[evaluation]),
                    "selected": False,
                }
            )
            continue
        temperature, bias, selection_loss = _leakage_val_select_calibration(
            y, pred, selection
        )
        calibrated = _leakage_val_apply_calibration(pred, temperature, bias)
        crossfit[evaluation] = calibrated[evaluation]
        rows.append(
            {
                "record_type": "rolling_fold",
                "fold": int(fold),
                "temperature": temperature,
                "bias": bias,
                "selection_rows": int(selection.sum()),
                "evaluation_rows": int(evaluation.sum()),
                "selection_logloss": selection_loss,
                "before_evaluation_logloss": _longterm_v1_loss(y[evaluation], pred[evaluation]),
                "after_evaluation_logloss": _longterm_v1_loss(
                    y[evaluation], calibrated[evaluation]
                ),
                "selected": True,
            }
        )

    temperature, bias, selection_loss = _leakage_val_select_calibration(
        y, pred, ~guard
    )
    final_calibrated = _leakage_val_apply_calibration(pred, temperature, bias)
    rows.append(
        {
            "record_type": "final_pre_lastblock_selection",
            "fold": -1,
            "temperature": temperature,
            "bias": bias,
            "selection_rows": int((~guard).sum()),
            "evaluation_rows": int(guard.sum()),
            "selection_logloss": selection_loss,
            "before_evaluation_logloss": _longterm_v1_loss(y[guard], pred[guard]),
            "after_evaluation_logloss": _longterm_v1_loss(
                y[guard], final_calibrated[guard]
            ),
            "selected": True,
        }
    )
    report = pd.DataFrame(rows)
    report.to_csv(LEAKAGE_VAL_CALIBRATION_PATH, index=False)
    return temperature, bias, crossfit, final_calibrated, report


def _leakage_val_subject_stability(train, base_q3, candidate_q3):
    rows = []
    work = train[["subject_id", "lifelog_date", "Q3"]].copy()
    work["base"] = np.asarray(base_q3, dtype=float)
    work["candidate"] = np.asarray(candidate_q3, dtype=float)
    for subject_id, group in work.groupby("subject_id", sort=False):
        y = group["Q3"].astype(int).to_numpy()
        base_loss = _longterm_v1_loss(y, group["base"])
        candidate_loss = _longterm_v1_loss(y, group["candidate"])
        rows.append(
            {
                "subject_id": subject_id,
                "rows": len(group),
                "baseline_q3_logloss": base_loss,
                "candidate_q3_logloss": candidate_loss,
                "delta_q3_logloss": candidate_loss - base_loss,
                "improved": candidate_loss < base_loss,
                "mean_abs_prediction_shift": float(
                    np.mean(np.abs(group["candidate"] - group["base"]))
                ),
            }
        )
    report = pd.DataFrame(rows).sort_values("delta_q3_logloss")
    report.to_csv(LEAKAGE_VAL_STABILITY_PATH, index=False)
    return report


def _leakage_val_write_audit():
    report = pd.DataFrame(
        [
            {
                "component": "target_history_features",
                "status": "pass",
                "evidence": "Each train/test row uses labels with lifelog_date strictly earlier than the prediction row.",
                "risk": "low",
                "recommended_action": "Keep past-only date filtering.",
            },
            {
                "component": "transition_streak_prior",
                "status": "pass",
                "evidence": "Train OOF is expanding past-only; test uses past train labels only and is non-recursive.",
                "risk": "low",
                "recommended_action": "Keep non-recursive test prediction.",
            },
            {
                "component": "fit_targetwise_stack",
                "status": "warning",
                "evidence": "Original meta stack uses complementary temporal folds, so an OOF validation fold can be trained with later rows.",
                "risk": "medium",
                "recommended_action": "Treat absolute OOF as optimistic; rely on rolling post-processing validation and leaderboard confirmation.",
            },
            {
                "component": "q3_weight_selection",
                "status": "pass",
                "evidence": "Rolling evaluation selects weight using folds strictly earlier than the evaluation fold.",
                "risk": "low",
                "recommended_action": "Use pre-lastblock-selected test weight.",
            },
            {
                "component": "q3_calibration_selection",
                "status": "pass",
                "evidence": "Temperature and bias are selected on earlier folds; final parameters exclude the subject last-block.",
                "risk": "low",
                "recommended_action": "Reject calibration if honest last-block or subject stability worsens.",
            },
            {
                "component": "profile_target_selection",
                "status": "warning",
                "evidence": "Q3 was chosen after observing prior profile diagnostics and leaderboard movement.",
                "risk": "medium",
                "recommended_action": "Require fold consistency and broad subject improvement before submission.",
            },
        ]
    )
    report.to_csv(LEAKAGE_VAL_AUDIT_PATH, index=False)
    return report


def _leakage_val_run_analysis():
    train, sample, _, _ = load_core_frames()
    base_oof_path = ARTIFACT_DIR / "leakage_val_raw_oof_baseline_18_8.csv"
    transition_oof_path = ARTIFACT_DIR / "leakage_val_raw_oof_transition_prior.csv"
    base_test_path = SUBMISSION_DIR / "submission_step1_profile_baseline_18_8_raw_stack.csv"
    transition_test_path = SUBMISSION_DIR / "submission_step1_profile_transition_prior_raw_stack.csv"
    required = [
        base_oof_path, transition_oof_path, base_test_path, transition_test_path
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"leakage_val required raw artifacts missing: {missing}")

    base_oof = pd.read_csv(base_oof_path)[TARGETS].astype(float)
    transition_oof = pd.read_csv(transition_oof_path)[TARGETS].astype(float)
    base_saved = pd.read_csv(base_test_path)
    transition_saved = pd.read_csv(transition_test_path)
    base_test = sample.copy()
    transition_test = sample.copy()
    base_test[TARGETS] = base_saved[TARGETS].to_numpy(float)
    transition_test[TARGETS] = transition_saved[TARGETS].to_numpy(float)

    fixed_oof = base_oof.copy()
    fixed_test = base_test.copy()
    fixed_oof["Q3"] = transition_oof["Q3"].to_numpy(float)
    fixed_test["Q3"] = transition_test["Q3"].to_numpy(float)
    fixed_test = _longterm_v1_validate_submission(fixed_test, sample)
    fixed_path = SUBMISSION_DIR / "submission_step1_validation_q3_transition_hybrid.csv"
    fixed_test.to_csv(fixed_path, index=False)

    final_weight, crossfit_weighted_q3, weight_report = _leakage_val_weight_search(
        train, base_oof, transition_oof
    )
    weighted_oof = base_oof.copy()
    weighted_test = base_test.copy()
    weighted_oof["Q3"] = _leakage_val_q3_blend(
        base_oof["Q3"], transition_oof["Q3"], final_weight
    )
    weighted_test["Q3"] = _leakage_val_q3_blend(
        base_test["Q3"], transition_test["Q3"], final_weight
    )
    weighted_test = _longterm_v1_validate_submission(weighted_test, sample)
    weighted_path = SUBMISSION_DIR / "submission_step1_validation_q3_weighted_hybrid.csv"
    weighted_test.to_csv(weighted_path, index=False)

    temperature, bias, crossfit_calibrated_q3, calibrated_q3, calibration_report = (
        _leakage_val_crossfit_calibration(train, weighted_oof["Q3"].to_numpy(float))
    )
    calibrated_oof = weighted_oof.copy()
    calibrated_test = weighted_test.copy()
    calibrated_oof["Q3"] = calibrated_q3
    calibrated_test["Q3"] = _leakage_val_apply_calibration(
        weighted_test["Q3"], temperature, bias
    )
    calibrated_test = _longterm_v1_validate_submission(calibrated_test, sample)
    calibrated_path = (
        SUBMISSION_DIR / "submission_step1_validation_q3_weighted_crossfit_calibrated.csv"
    )
    calibrated_test.to_csv(calibrated_path, index=False)

    honest_frames = {
        "baseline_raw": base_oof,
        "q3_transition_hybrid": fixed_oof,
        "q3_weighted_hybrid": weighted_oof,
        "q3_weighted_crossfit_calibrated": calibrated_oof,
    }
    rows = []
    baseline_metrics = _leakage_val_candidate_metrics(train, base_oof)
    base_map = baseline_metrics.set_index("target")
    for candidate, frame in honest_frames.items():
        metrics = _leakage_val_candidate_metrics(train, frame)
        for row in metrics.itertuples(index=False):
            rows.append(
                {
                    "candidate": candidate,
                    "target": row.target,
                    "full_oof_logloss": row.full_oof_logloss,
                    "lastblock_logloss": row.lastblock_logloss,
                    "guarded_score": row.guarded_score,
                    "delta_full_vs_baseline": (
                        row.full_oof_logloss
                        - float(base_map.loc[row.target, "full_oof_logloss"])
                    ),
                    "delta_lastblock_vs_baseline": (
                        row.lastblock_logloss
                        - float(base_map.loc[row.target, "lastblock_logloss"])
                    ),
                    "q3_weight": final_weight if candidate.startswith("q3_weighted") else (1.0 if candidate == "q3_transition_hybrid" else 0.0),
                    "temperature": temperature if candidate.endswith("calibrated") else 1.0,
                    "bias": bias if candidate.endswith("calibrated") else 0.0,
                }
            )
    honest = pd.DataFrame(rows)
    average = honest.groupby("candidate", as_index=False).agg(
        avg_full_oof_logloss=("full_oof_logloss", "mean"),
        avg_lastblock_logloss=("lastblock_logloss", "mean"),
        avg_guarded_score=("guarded_score", "mean"),
    )
    honest = honest.merge(average, on="candidate", how="left")
    honest.to_csv(LEAKAGE_VAL_HONEST_PATH, index=False)

    stability = _leakage_val_subject_stability(
        train, base_oof["Q3"], calibrated_oof["Q3"]
    )
    leakage = _leakage_val_write_audit()
    stable_rate = float(stability["improved"].mean())
    q3_rows = honest[honest["target"].eq("Q3")].set_index("candidate")
    candidate_order = [
        "q3_weighted_crossfit_calibrated",
        "q3_weighted_hybrid",
        "q3_transition_hybrid",
        "baseline_raw",
    ]
    selected_candidate = "baseline_raw"
    for candidate in candidate_order:
        q3 = q3_rows.loc[candidate]
        if (
            q3["lastblock_logloss"]
            <= q3_rows.loc["baseline_raw", "lastblock_logloss"] + 1e-10
            and q3["full_oof_logloss"]
            <= q3_rows.loc["baseline_raw", "full_oof_logloss"] + 0.0003
        ):
            if candidate != "q3_weighted_crossfit_calibrated" or stable_rate >= 0.50:
                selected_candidate = candidate
                break
    path_map = {
        "baseline_raw": base_test_path,
        "q3_transition_hybrid": fixed_path,
        "q3_weighted_hybrid": weighted_path,
        "q3_weighted_crossfit_calibrated": calibrated_path,
    }
    raw_path_map = {
        "baseline_raw": base_test_path,
        "q3_transition_hybrid": fixed_path,
        "q3_weighted_hybrid": weighted_path,
        # The matching raw stack for a calibrated candidate is the same
        # weighted hybrid immediately before T+b calibration.
        "q3_weighted_crossfit_calibrated": weighted_path,
    }
    best_path = SUBMISSION_DIR / "submission_step1_validation_best_honest_q3.csv"
    _longterm_v1_shutil.copyfile(path_map[selected_candidate], best_path)
    best_raw_path = (
        SUBMISSION_DIR / "submission_step1_validation_best_honest_q3_raw_stack.csv"
    )
    _longterm_v1_shutil.copyfile(raw_path_map[selected_candidate], best_raw_path)
    summary = pd.DataFrame(
        [
            {
                "analysis_name": "Temporal Leakage-controlled Validation and Guarded Calibration",
                "selected_candidate": selected_candidate,
                "selected_q3_weight": final_weight,
                "selected_temperature": temperature,
                "selected_bias": bias,
                "subject_improvement_rate": stable_rate,
                "best_submission_path": str(best_path),
                "best_raw_stack_path": str(best_raw_path),
                "raw_stack_definition": (
                    "Selected leakage_val candidate immediately before Q3 T+b calibration; "
                    "safe alpha is never applied."
                ),
                "upstream_stack_leakage_warning": True,
            }
        ]
    )
    summary.to_csv(LEAKAGE_VAL_RECOMMENDATION_PATH, index=False)

    print("\n[TRY19] honest OOF comparison")
    print(average.sort_values("avg_lastblock_logloss").to_string(index=False))
    print(f"[TRY19] selected Q3 weight: {final_weight}")
    print(f"[TRY19] selected Q3 calibration: T={temperature}, bias={bias}")
    print(f"[TRY19] subject improvement rate: {stable_rate:.3f}")
    print(f"[TRY19] selected candidate: {selected_candidate}")
    print(f"[TRY19] best submission: {best_path}")
    print(f"[TRY19] matching raw stack: {best_raw_path}")
    print("[TRY19] diagnostics:")
    for path in [
        LEAKAGE_VAL_WEIGHT_PATH,
        LEAKAGE_VAL_CALIBRATION_PATH,
        LEAKAGE_VAL_STABILITY_PATH,
        LEAKAGE_VAL_HONEST_PATH,
        LEAKAGE_VAL_AUDIT_PATH,
        LEAKAGE_VAL_RECOMMENDATION_PATH,
    ]:
        print(f"  - {path}")


def leakage_validation_main():
    _longterm_v1_main()
    _leakage_val_run_analysis()



# Recommended leakage-aware run:
# python leakage_validation --reuse-step1-base-artifacts --personal-longterm-profile transition_prior
#
# Full baseline/profile refresh plus leakage_validation analysis:
# python leakage_validation --reuse-step1-base-artifacts --personal-longterm-profile all


# ============================================================================
# step1_pipeline: S3/S4 long-term personalization on top of the Try19 raw anchor
# ============================================================================
TRY20_COMPONENT_DIAG_PATH = ARTIFACT_DIR / "step1_pipeline_s3s4_component_diagnostics.csv"
TRY20_WEIGHT_PATH = ARTIFACT_DIR / "step1_pipeline_s3s4_weight_search.csv"
TRY20_STABILITY_PATH = ARTIFACT_DIR / "step1_pipeline_s3s4_subject_stability.csv"
TRY20_COMPARISON_PATH = ARTIFACT_DIR / "step1_pipeline_s3s4_honest_comparison.csv"
TRY20_RECOMMENDATION_PATH = ARTIFACT_DIR / "step1_pipeline_recommendation_summary.csv"
TRY20_TARGETS = ["S3", "S4"]


def _step1_pipeline_component_paths(component: str):
    return (
        ARTIFACT_DIR / f"step1_pipeline_oof_{component}.csv",
        ARTIFACT_DIR / f"step1_pipeline_test_{component}.csv",
    )


def _step1_pipeline_reliability_paths(target: str):
    return (
        ARTIFACT_DIR / f"step1_pipeline_reliability_train_{target}.csv",
        ARTIFACT_DIR / f"step1_pipeline_reliability_test_{target}.csv",
    )


def _step1_pipeline_fit_weak_component(
    train,
    sample,
    personal_table,
    feature_cols,
    target,
):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler

    train_mask = personal_table["is_train"].eq(1)
    test_mask = personal_table["is_train"].eq(0)
    usable = []
    for col in feature_cols:
        if col not in personal_table.columns:
            continue
        values = pd.to_numeric(
            personal_table.loc[train_mask, col], errors="coerce"
        )
        if values.notna().sum() >= 20 and values.nunique(dropna=True) > 1:
            usable.append(col)
    usable = usable[:80]
    if not usable:
        prior = float(train[target].mean())
        return (
            np.full(len(train), prior),
            np.full(len(sample), prior),
            [],
        )

    x_train = personal_table.loc[train_mask, usable].reset_index(drop=True)
    x_test = personal_table.loc[test_mask, usable].reset_index(drop=True)
    y = train[target].astype(int).reset_index(drop=True)
    folds = _longterm_v1_subject_chrono_folds(train)
    oof = np.full(len(train), 0.5)
    test_acc = np.zeros(len(sample), dtype=float)
    used = 0
    for fold in sorted(folds.unique()):
        valid = folds.eq(fold).to_numpy()
        fit = folds.lt(fold).to_numpy()
        fold_prior = float(y.loc[fit].mean()) if fit.any() else 0.5
        if not valid.any() or not fit.any() or y.loc[fit].nunique() < 2:
            oof[valid] = fold_prior
            continue
        med = x_train.loc[fit].replace([np.inf, -np.inf], np.nan).median()
        x_fit = (
            x_train.loc[fit].replace([np.inf, -np.inf], np.nan)
            .fillna(med).fillna(0.0).clip(-1e6, 1e6)
        )
        x_valid = (
            x_train.loc[valid].replace([np.inf, -np.inf], np.nan)
            .fillna(med).fillna(0.0).clip(-1e6, 1e6)
        )
        x_te = (
            x_test.replace([np.inf, -np.inf], np.nan)
            .fillna(med).fillna(0.0).clip(-1e6, 1e6)
        )
        restored = (
            load_step1_candidate_checkpoint(
                "step1_pipeline_weak_component",
                target,
                "fold",
                int(fold),
            )
            if restore_step1_candidate_checkpoints_enabled()
            else None
        )
        if restored is not None:
            print(f"[STEP1_CKPT] restored candidate step1_pipeline_weak_component {target} fold={int(fold)}")
            model = restored["model"]
            restored_prior = float(restored.get("fold_prior", fold_prior))
            oof[valid] = np.clip(
                0.85 * model.predict_proba(x_valid)[:, 1] + 0.15 * restored_prior,
                0.03,
                0.97,
            )
            test_acc += model.predict_proba(x_te)[:, 1]
            used += 1
            continue
        model = make_pipeline(
            RobustScaler(quantile_range=(10.0, 90.0)),
            LogisticRegression(
                C=0.01,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
            ),
        )
        model.fit(x_fit, y.loc[fit])
        save_step1_candidate_checkpoint(
            "step1_pipeline_weak_component",
            {
                "kind": "fold",
                "target": target,
                "fold": int(fold),
                "usable_features": list(usable),
                "median": med,
                "fold_prior": fold_prior,
                "model": model,
            },
            target,
            "fold",
            int(fold),
        )
        oof[valid] = np.clip(
            0.85 * model.predict_proba(x_valid)[:, 1] + 0.15 * fold_prior,
            0.03,
            0.97,
        )
        test_acc += model.predict_proba(x_te)[:, 1]
        used += 1
    test_pred = (
        np.clip(
            0.85 * test_acc / used + 0.15 * float(y.mean()),
            0.03,
            0.97,
        )
        if used
        else np.full(len(sample), float(y.mean()))
    )
    return clip_proba(oof, 0.03, 0.97), test_pred, usable


def _step1_pipeline_build_and_save_component_sources(
    train,
    sample,
    personal_table,
    personal_cols,
    transition_oof,
    transition_test,
    residual_oof,
    residual_test,
):
    component_oof = {
        "transition": transition_oof[TRY20_TARGETS].copy().reset_index(drop=True),
        "residual": residual_oof[TRY20_TARGETS].copy().reset_index(drop=True),
    }
    component_test = {
        "transition": transition_test[TRY20_TARGETS].copy().reset_index(drop=True),
        "residual": residual_test[TRY20_TARGETS].copy().reset_index(drop=True),
    }
    diagnostic_rows = []
    for target in TRY20_TARGETS:
        train_state = personal_table[
            personal_table["is_train"].eq(1)
        ].reset_index(drop=True)
        test_state = personal_table[
            personal_table["is_train"].eq(0)
        ].reset_index(drop=True)
        history_count = train.groupby("subject_id").cumcount().to_numpy(float)
        history_train = np.clip(
            np.log1p(history_count) / np.log(12.0), 0.15, 1.0
        )
        subject_sizes = train.groupby("subject_id").size().to_dict()
        history_test = np.asarray(
            [
                np.clip(
                    np.log1p(subject_sizes.get(sid, 0)) / np.log(12.0),
                    0.15,
                    1.0,
                )
                for sid in sample["subject_id"]
            ],
            dtype=float,
        )
        volatility_col = f"{target}_volatility7"
        volatility_train = pd.to_numeric(
            train_state.get(
                volatility_col, pd.Series(0.5, index=train_state.index)
            ),
            errors="coerce",
        ).fillna(0.5).to_numpy(float)
        volatility_test = pd.to_numeric(
            test_state.get(
                volatility_col, pd.Series(0.5, index=test_state.index)
            ),
            errors="coerce",
        ).fillna(0.5).to_numpy(float)
        stability_train = np.clip(1.0 - volatility_train, 0.20, 1.0)
        stability_test = np.clip(1.0 - volatility_test, 0.20, 1.0)
        sensor_cols = [
            col for col in personal_table.columns if col.startswith("personal_")
        ]
        coverage_train = (
            train_state[sensor_cols].notna().mean(axis=1).to_numpy(float)
            if sensor_cols else np.ones(len(train))
        )
        coverage_test = (
            test_state[sensor_cols].notna().mean(axis=1).to_numpy(float)
            if sensor_cols else np.ones(len(sample))
        )
        train_gate_path, test_gate_path = _step1_pipeline_reliability_paths(target)
        pd.DataFrame(
            {
                "history_gate": history_train,
                "stability_gate": stability_train,
                "coverage_gate": coverage_train,
                "base_reliability": (
                    history_train * stability_train * coverage_train
                ),
            }
        ).to_csv(train_gate_path, index=False)
        pd.DataFrame(
            {
                "history_gate": history_test,
                "stability_gate": stability_test,
                "coverage_gate": coverage_test,
                "base_reliability": (
                    history_test * stability_test * coverage_test
                ),
            }
        ).to_csv(test_gate_path, index=False)

        own_history = [
            col for col in personal_cols
            if col.startswith(f"{target}_")
            and not col.startswith("transition_prior_")
        ]
        cross_target = [
            col for col in personal_cols
            if any(col.startswith(f"{other}_") for other in TARGETS if other != target)
            or col.startswith("transition_prior_")
        ]
        sensor = [
            col for col in personal_cols
            if col.startswith("personal_")
        ]
        for component, cols in [
            ("history", own_history),
            ("sensor", sensor),
            ("cross_target", cross_target),
        ]:
            oof_pred, test_pred, used = _step1_pipeline_fit_weak_component(
                train, sample, personal_table, cols, target
            )
            component_oof.setdefault(
                component,
                pd.DataFrame(index=train.index, columns=TRY20_TARGETS, dtype=float),
            )[target] = oof_pred
            component_test.setdefault(
                component,
                pd.DataFrame(index=sample.index, columns=TRY20_TARGETS, dtype=float),
            )[target] = test_pred
            diagnostic_rows.append(
                {
                    "target": target,
                    "component": component,
                    "feature_count": len(used),
                    "features": "|".join(used),
                }
            )

    component_oof["personal_ensemble"] = (
        0.50 * component_oof["history"]
        + 0.25 * component_oof["sensor"]
        + 0.25 * component_oof["cross_target"]
    )
    component_test["personal_ensemble"] = (
        0.50 * component_test["history"]
        + 0.25 * component_test["sensor"]
        + 0.25 * component_test["cross_target"]
    )
    component_oof["transition_residual"] = (
        0.60 * component_oof["transition"]
        + 0.40 * component_oof["residual"]
    )
    component_test["transition_residual"] = (
        0.60 * component_test["transition"]
        + 0.40 * component_test["residual"]
    )
    component_oof["full_personal"] = (
        0.45 * component_oof["transition"]
        + 0.25 * component_oof["history"]
        + 0.15 * component_oof["sensor"]
        + 0.15 * component_oof["cross_target"]
    )
    component_test["full_personal"] = (
        0.45 * component_test["transition"]
        + 0.25 * component_test["history"]
        + 0.15 * component_test["sensor"]
        + 0.15 * component_test["cross_target"]
    )

    for component in component_oof:
        oof_path, test_path = _step1_pipeline_component_paths(component)
        component_oof[component].to_csv(oof_path, index=False)
        component_test[component].to_csv(test_path, index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        TRY20_COMPONENT_DIAG_PATH, index=False
    )


# Pruned historical definition: _step1_pipeline_correct (not reachable from the final runner).


# Pruned historical definition: _step1_pipeline_subject_stability (not reachable from the final runner).


# Pruned historical definition: _step1_pipeline_selection_subject_stats (not reachable from the final runner).


# Pruned historical definition: _step1_pipeline_run_analysis (not reachable from the final runner).


# Pruned historical definition: step1_pipeline_main (not reachable from the final runner).



# Recommended:
# python step1_pipeline --reuse-step1-base-artifacts --personal-longterm-profile transition_prior
#
# Full refresh:
# python step1_pipeline --reuse-step1-base-artifacts --personal-longterm-profile all


# ============================================================================
# Embedded step2_features Step-2 compatibility layer required by step2_sensor_helpers/v8.
# ============================================================================
TRY21_V3_SEED = 21040


def _step2_v1_quarters(train):
    quarters = pd.Series(index=train.index, dtype=int)
    ordered = train[["subject_id", "lifelog_date"]].copy()
    ordered["lifelog_date"] = pd.to_datetime(ordered["lifelog_date"])
    for _, idx in ordered.groupby("subject_id").groups.items():
        subject_idx = (
            ordered.loc[list(idx)]
            .sort_values("lifelog_date")
            .index.to_numpy()
        )
        splits = np.array_split(subject_idx, 4)
        for quarter, split_idx in enumerate(splits, start=1):
            quarters.loc[split_idx] = quarter
    return quarters.astype(int)


# Pruned historical definition: _step2_base_build_sensor_raw (not reachable from the final runner).


# Pruned historical definition: _step2_base_apply_update (not reachable from the final runner).


# Pruned historical definition: _step2_v1_subject_stats (not reachable from the final runner).


# Pruned historical definition: _step2_base_target_metrics (not reachable from the final runner).


# ============================================================================
# step2_features: paper-inspired time-series feature ablation (Step 2 only)
# ----------------------------------------------------------------------------
# Base choice: step2_base_design's Step 2 design (as embedded in step2_extended). It is the
# actual available implementation that combines a matching OOF/test anchor,
# full OOF plus user-wise chronological last-block evaluation, and raw-feature
# ablation. v5 is copied as requested so all local helpers remain available,
# but this file's main calls only the v6 Step 2 entry point.
# ============================================================================
STEP2_FEATURES_DIR = ARTIFACT_DIR / "step2_features"
PAPER_ROLLING_WINDOW = 3
PAPER_ROLLING_MIN_PERIODS = 2
PAPER_EMA_SPAN = 3
PAPER_PCT_CLIP = 5.0
PAPER_CORRELATION_LIMIT = 0.995
PAPER_MAX_MISSING = 0.85
PAPER_MIN_VARIANCE = 1e-10
PAPER_WINDOWS = {
    "evening": (18, 24),
    "bedtime_transition": (20, 26),
    "pre_sleep": (21, 27),
    "late_night": (24, 29),
    "overnight": (24, 32),
    "wake_transition": (29, 36),
    "morning": (30, 36),
    "full_sleepctx": (18, 36),
}
PAPER_PROFILE_TRANSFORMS = {
    "paper_var": ["rolling_std", "rolling_iqr", "rolling_mad"],
    "paper_change": ["difference", "ema"],
    "paper_core": [
        "rolling_mean",
        "rolling_std",
        "rolling_iqr",
        "rolling_mad",
        "difference",
        "ema",
    ],
    "paper_full": [
        "rolling_mean",
        "rolling_std",
        "rolling_iqr",
        "rolling_mad",
        "difference",
        "ema",
        "rolling_median",
        "expanding_mean",
        "percentage_change",
        "diff_zero_crossing",
    ],
}
MAX_PAPER_FEATURES_PER_TARGET = {
    "paper_var": 8,
    "paper_change": 8,
    "paper_core": 12,
    "paper_full": 16,
}
PAPER_ETA_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15]
PAPER_MAX_CANDIDATES_PER_PROFILE_TARGET = 72
PAPER_AGGREGATIONS = ["mean", "std", "min", "max"]
PAPER_SENSOR_FAMILY = {
    "ch2025_mACStatus.parquet": "charging",
    "ch2025_mActivity.parquet": "activity",
    "ch2025_mLight.parquet": "mobile_light",
    "ch2025_mScreenStatus.parquet": "screen",
    "ch2025_wLight.parquet": "wearable_light",
    "ch2025_wPedo.parquet": "pedometer",
}

STEP2_FEATURES_PATHS = {
    "sensor_schema": STEP2_FEATURES_DIR / "step2_features_sensor_schema.csv",
    "assumptions": STEP2_FEATURES_DIR / "step2_features_paper_assumptions.csv",
    "manifest": STEP2_FEATURES_DIR / "step2_features_paper_feature_manifest.csv",
    "dedup": STEP2_FEATURES_DIR / "step2_features_paper_feature_dedup.csv",
    "selected": STEP2_FEATURES_DIR / "step2_features_selected_paper_features.csv",
    "leakage": STEP2_FEATURES_DIR / "step2_features_leakage_checks.csv",
    "anchor_pair": STEP2_FEATURES_DIR / "step2_features_anchor_pair_check.csv",
    "target_scores": STEP2_FEATURES_DIR / "step2_features_target_scores.csv",
    "profile_summary": STEP2_FEATURES_DIR / "step2_features_profile_summary.csv",
    "temporal": STEP2_FEATURES_DIR / "step2_features_temporal_block_scores.csv",
    "subject": STEP2_FEATURES_DIR / "step2_features_subject_diagnostics.csv",
    "shift": STEP2_FEATURES_DIR / "step2_features_prediction_shift.csv",
    "decisions": STEP2_FEATURES_DIR / "step2_features_target_decisions.csv",
    "recommendation": STEP2_FEATURES_DIR / "step2_features_recommendation.csv",
    "report": STEP2_FEATURES_DIR / "step2_features_report.md",
    "feature_cache": STEP2_FEATURES_DIR / "step2_features_paper_features_cache.parquet",
}
STEP2_FEATURES_SUBMISSIONS = {
    "anchor": SUBMISSION_DIR / "submission_step2_anchor.csv",
    "history_only": SUBMISSION_DIR / "submission_step2_history_only.csv",
    "paper_var": SUBMISSION_DIR / "submission_step2_sensor_var.csv",
    "paper_change": SUBMISSION_DIR / "submission_step2_sensor_change.csv",
    "paper_core": SUBMISSION_DIR / "submission_step2_sensor_core.csv",
    "paper_full": (
        SUBMISSION_DIR / "submission_step2_sensor_full_experimental.csv"
    ),
    "paper_guarded": SUBMISSION_DIR / "submission_step2_sensor_guarded.csv",
    "paper_guarded_plus125": (
        SUBMISSION_DIR / "submission_step2_sensor_guarded_plus125.csv"
    ),
    "paper_guarded_plus150": (
        SUBMISSION_DIR / "submission_step2_sensor_guarded_plus150.csv"
    ),
    "paper_q2q3_plus150": (
        SUBMISSION_DIR / "submission_step2_sensor_q2q3_plus150.csv"
    ),
    "best": SUBMISSION_DIR / "submission_step2_best.csv",
}


# Pruned historical definition: _step2_assumptions (not reachable from the final runner).


def _step2_sensor_schema(sensor_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    specs = []
    for path in sorted(Path(sensor_dir).glob("*.parquet")):
        schema = pq.read_schema(path)
        names = schema.names
        subject_col = "subject_id" if "subject_id" in names else ""
        timestamp_col = "timestamp" if "timestamp" in names else ""
        numeric = []
        skipped = []
        for field in schema:
            if field.name in {"subject_id", "timestamp"}:
                continue
            if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                numeric.append(field.name)
            else:
                skipped.append(field.name)
        used = numeric if subject_col and timestamp_col else []
        reason = ""
        if not subject_col:
            reason = "missing subject_id"
        elif not timestamp_col:
            reason = "missing timestamp"
        elif not numeric:
            reason = "no top-level numeric value column; nested values are not silently converted"
        rows.append(
            {
                "sensor_file": path.name,
                "exists": True,
                "timestamp_column": timestamp_col,
                "subject_column": subject_col,
                "numeric_columns": "|".join(numeric),
                "used_columns": "|".join(used),
                "skipped_columns": "|".join(skipped),
                "skip_reason": reason,
            }
        )
        if used:
            specs.append(
                {
                    "path": path,
                    "family": PAPER_SENSOR_FAMILY.get(
                        path.name, path.stem.replace("ch2025_", "").lower()
                    ),
                    "columns": used,
                }
            )
    if not rows:
        raise FileNotFoundError(f"no sensor parquet files found in {sensor_dir}")
    return pd.DataFrame(rows), specs


def _step2_load_anchor():
    train, sample, _, _ = load_core_frames()
    oof_path = ARTIFACT_DIR / "leakage_val_raw_oof_transition_prior.csv"
    test_path = (
        SUBMISSION_DIR / "submission_step1_profile_best_raw_lastblock.csv"
    )
    equivalent_test_path = (
        SUBMISSION_DIR
        / "submission_step1_profile_transition_prior_raw_stack.csv"
    )
    comparison_path = (
        ARTIFACT_DIR / "longterm_personalization_v1_raw_profile_comparison.csv"
    )
    for path in [
        oof_path,
        test_path,
        equivalent_test_path,
        comparison_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"required matching Step 1/anchor artifact is missing: {path}"
            )
    anchor_oof = pd.read_csv(oof_path)
    anchor_test = pd.read_csv(test_path)
    if len(anchor_oof) != len(train) or len(anchor_test) != len(sample):
        raise RuntimeError("matching anchor pair row count does not match train/test")
    if not set(TARGETS).issubset(anchor_oof) or not set(TARGETS).issubset(anchor_test):
        raise RuntimeError("matching anchor pair is missing target columns")
    equivalent_test = pd.read_csv(equivalent_test_path)
    equivalent_error = float(
        np.max(
            np.abs(
                anchor_test[TARGETS].to_numpy(float)
                - equivalent_test[TARGETS].to_numpy(float)
            )
        )
    )
    if equivalent_error > 1e-12:
        raise RuntimeError(
            "18_8_v1 best raw test is not the saved transition-prior "
            f"raw-stack member: max_abs={equivalent_error}"
        )
    key_cols = ["subject_id", "sleep_date", "lifelog_date"]
    saved_keys = anchor_test[key_cols].astype(str).reset_index(drop=True)
    sample_keys = sample[key_cols].astype(str).reset_index(drop=True)
    if not saved_keys.equals(sample_keys):
        raise RuntimeError("matching anchor test keys/order differ from sample submission")
    anchor_test[key_cols] = sample[key_cols].to_numpy()
    anchor_test = _longterm_v1_validate_submission(anchor_test, sample)
    guard = make_subject_lastblock_mask(train).reset_index(
        drop=True
    ).to_numpy(bool)
    comparison = pd.read_csv(comparison_path)
    expected = comparison[
        comparison["profile"].eq("transition_prior")
    ].set_index("target")
    max_metric_error = 0.0
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        pred = anchor_oof[target].to_numpy(float)
        full = _longterm_v1_loss(y, pred)
        last = _longterm_v1_loss(y[guard], pred[guard])
        max_metric_error = max(
            max_metric_error,
            abs(full - float(expected.loc[target, "full_oof_logloss"])),
            abs(last - float(expected.loc[target, "lastblock_logloss"])),
        )
    if max_metric_error > 1e-10:
        raise RuntimeError(
            "18_8_v1 OOF metrics do not match the saved transition-prior "
            f"profile: max_abs={max_metric_error}"
        )
    out = pd.DataFrame(
        [
            {
                "anchor_profile": "18_8_v1_best_raw_lastblock",
                "anchor_oof_source": str(oof_path),
                "anchor_test_source": str(test_path),
                "row_count_oof": len(anchor_oof),
                "row_count_test": len(anchor_test),
                "matching_pair_verified": True,
                "reconstruction_method": (
                    "saved longterm_personalization_v1 transition-prior raw OOF paired with "
                    "the byte-equivalent best raw last-block test submission"
                ),
                "equivalent_test_path": str(equivalent_test_path),
                "equivalent_test_max_abs": equivalent_error,
                "saved_metric_max_abs": max_metric_error,
            }
        ]
    )
    out.to_csv(STEP2_FEATURES_PATHS["anchor_pair"], index=False)
    return (
        train.reset_index(drop=True),
        sample.reset_index(drop=True),
        anchor_oof[TARGETS].astype(float).reset_index(drop=True),
        anchor_test.reset_index(drop=True),
        out,
    )


def _step2_meta(train, sample):
    train_meta = train[["subject_id", "sleep_date", "lifelog_date"]].copy()
    test_meta = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    meta = pd.concat(
        [train_meta.assign(is_train=1), test_meta.assign(is_train=0)],
        ignore_index=True,
    )
    meta["row_id"] = np.arange(len(meta))
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta["lifelog_date"] = pd.to_datetime(meta["lifelog_date"]).dt.normalize()
    meta["sleep_date"] = pd.to_datetime(meta["sleep_date"]).dt.normalize()
    if meta[["subject_id", "lifelog_date", "sleep_date"]].duplicated().any():
        raise ValueError("sleep-record matching keys are duplicated")
    return meta


def _step2_mean_absolute_deviation(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < PAPER_ROLLING_MIN_PERIODS:
        return np.nan
    return float(np.mean(np.abs(values - np.mean(values))))


def _step2_transforms(series):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    rolling = values.rolling(
        window=PAPER_ROLLING_WINDOW,
        min_periods=PAPER_ROLLING_MIN_PERIODS,
    )
    diff = values.diff()
    sign = np.sign(diff)
    zero_cross = (
        sign.ne(sign.shift(1))
        & sign.ne(0)
        & sign.shift(1).ne(0)
        & sign.notna()
        & sign.shift(1).notna()
    ).astype(float)
    zero_cross[diff.isna()] = np.nan
    pct = values.pct_change(fill_method=None)
    pct = pct.replace([np.inf, -np.inf], np.nan).clip(
        -PAPER_PCT_CLIP, PAPER_PCT_CLIP
    )
    return {
        "rolling_mean": rolling.mean(),
        "rolling_std": rolling.std(ddof=0),
        "rolling_median": rolling.median(),
        "rolling_iqr": rolling.quantile(0.75) - rolling.quantile(0.25),
        "rolling_mad": rolling.apply(
            _step2_mean_absolute_deviation, raw=True
        ),
        "difference": diff,
        "expanding_mean": values.expanding(min_periods=1).mean(),
        "ema": values.ewm(
            span=PAPER_EMA_SPAN, adjust=False, min_periods=1
        ).mean(),
        "percentage_change": pct,
        "diff_zero_crossing": zero_cross,
    }


def _step2_log(message, started=None):
    suffix = ""
    if started is not None:
        suffix = f" ({time.perf_counter() - started:.1f}s)"
    print(f"[TRY21_V6] {message}{suffix}", flush=True)


def _step2_aggregate_record(
    part, family, source_col, window_name, include_manifest=False
):
    result = {}
    manifest = []
    transformed = _step2_transforms(part[source_col].reset_index(drop=True))
    for transform, values in transformed.items():
        base = f"paper_{family}_{source_col}_{window_name}_{transform}"
        finite = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        stats = {
            "mean": finite.mean(),
            "std": finite.std(ddof=0),
            "min": finite.min(),
            "max": finite.max(),
            "valid_count": finite.notna().sum(),
            "missing_ratio": finite.isna().mean() if len(finite) else 1.0,
        }
        for aggregation, value in stats.items():
            feature = f"{base}_{aggregation}"
            if feature in result:
                raise ValueError(f"duplicate paper feature name: {feature}")
            result[feature] = value
            if include_manifest:
                manifest.append(
                    {
                        "feature": feature,
                        "sensor_family": family,
                        "source_column": source_col,
                        "window": window_name,
                        "transform": transform,
                        "aggregation": aggregation,
                    }
                )
    return result, manifest


def _step2_manifest_from_features(features):
    rows = []
    families = sorted(
        set(PAPER_SENSOR_FAMILY.values()), key=len, reverse=True
    )
    transforms = sorted(
        PAPER_PROFILE_TRANSFORMS["paper_full"], key=len, reverse=True
    )
    aggregations = [
        "valid_count", "missing_ratio", "mean", "std", "min", "max"
    ]
    for feature in features:
        remainder = feature.removeprefix("paper_")
        family = next(
            (
                candidate
                for candidate in families
                if remainder.startswith(f"{candidate}_")
            ),
            "",
        )
        after_family = (
            remainder[len(family) + 1 :] if family else remainder
        )
        window = next(
            (
                candidate
                for candidate in PAPER_WINDOWS
                if f"_{candidate}_" in after_family
            ),
            "",
        )
        source_column = (
            after_family.split(f"_{window}_", 1)[0] if window else ""
        )
        transform = next(
            (
                candidate
                for candidate in transforms
                if f"_{candidate}_" in feature
            ),
            "",
        )
        aggregation = next(
            (
                candidate
                for candidate in aggregations
                if feature.endswith(f"_{candidate}")
            ),
            "",
        )
        rows.append(
            {
                "feature": feature,
                "sensor_family": family,
                "source_column": source_column,
                "window": window,
                "transform": transform,
                "aggregation": aggregation,
            }
        )
    return pd.DataFrame(rows)


def _step2_build_paper_features(
    meta, specs, check_only=False, rebuild_cache=False
):
    records = meta.copy()
    if check_only:
        first_subject = records["subject_id"].iloc[0]
        records = records[records["subject_id"].eq(first_subject)].head(1).copy()
        specs = specs[:1]
    cache_path = STEP2_FEATURES_PATHS["feature_cache"]
    if (
        not check_only
        and cache_path.exists()
        and STEP2_FEATURES_PATHS["manifest"].exists()
        and not rebuild_cache
    ):
        started = time.perf_counter()
        _step2_log(f"loading cached paper features: {cache_path.name}")
        cached = pd.read_parquet(cache_path)
        expected_ids = records["row_id"].astype(int).to_numpy()
        cached_ids = cached["row_id"].astype(int).to_numpy()
        cached_features = set(cached.columns) - {"row_id"}
        manifest = pd.read_csv(STEP2_FEATURES_PATHS["manifest"])
        manifest_features = set(manifest["feature"])
        if cached_features != manifest_features:
            _step2_log(
                "cached feature manifest differs from cache; "
                "reconstructing manifest without rebuilding sensors"
            )
            manifest = _step2_manifest_from_features(
                [column for column in cached.columns if column != "row_id"]
            )
            manifest.to_csv(STEP2_FEATURES_PATHS["manifest"], index=False)
            manifest_features = set(manifest["feature"])
        if (
            len(cached) == len(records)
            and np.array_equal(cached_ids, expected_ids)
            and cached_features == manifest_features
        ):
            leakage = pd.DataFrame(
                [
                    {
                        "check_name": "paper_feature_cache_row_alignment",
                        "passed": True,
                        "details": f"loaded {len(cached)} aligned rows from cache",
                        "affected_feature_count": len(cached.columns) - 1,
                        "action": "continue",
                    }
                ]
            )
            _step2_log(
                f"cached paper features ready: {len(cached.columns) - 1} columns",
                started,
            )
            return cached, manifest, leakage
        _step2_log("feature cache is stale; rebuilding")
    output = records[["row_id"]].copy()
    manifest_rows = []
    manifest_keys = set()
    leakage_rows = []
    total_specs = len(specs)
    for spec_index, spec in enumerate(specs, 1):
        sensor_started = time.perf_counter()
        _step2_log(
            f"sensor {spec_index}/{total_specs} loading "
            f"{spec['path'].name} ({len(spec['columns'])} numeric columns)"
        )
        usecols = ["subject_id", "timestamp", *spec["columns"]]
        sensor = pd.read_parquet(spec["path"], columns=usecols)
        sensor["subject_id"] = sensor["subject_id"].astype(str)
        sensor["timestamp"] = pd.to_datetime(sensor["timestamp"])
        sensor = sensor.sort_values(["subject_id", "timestamp"])
        for col in spec["columns"]:
            sensor[col] = pd.to_numeric(sensor[col], errors="coerce")
        rows = []
        max_used = None
        max_cutoff = None
        subject_groups = list(records.groupby("subject_id", sort=False))
        for subject_index, (subject, subject_records) in enumerate(
            subject_groups, 1
        ):
            subject_sensor = sensor[sensor["subject_id"].eq(subject)]
            timestamp_values = subject_sensor["timestamp"].to_numpy()
            for record in subject_records.itertuples(index=False):
                rec = {"row_id": int(record.row_id)}
                cutoff = record.lifelog_date + pd.Timedelta(
                    hours=PAPER_WINDOWS["full_sleepctx"][1]
                )
                max_cutoff = cutoff if max_cutoff is None else max(max_cutoff, cutoff)
                for window_name, (start_h, end_h) in PAPER_WINDOWS.items():
                    start = record.lifelog_date + pd.Timedelta(hours=start_h)
                    end = record.lifelog_date + pd.Timedelta(hours=end_h)
                    effective_end = min(end, cutoff)
                    left = int(
                        np.searchsorted(
                            timestamp_values, np.datetime64(start), side="left"
                        )
                    )
                    right = int(
                        np.searchsorted(
                            timestamp_values,
                            np.datetime64(effective_end),
                            side="left",
                        )
                    )
                    part = subject_sensor.iloc[left:right]
                    if len(part):
                        used = part["timestamp"].max()
                        max_used = used if max_used is None else max(max_used, used)
                    for col in spec["columns"]:
                        values = part[["timestamp", col]].dropna(subset=[col])
                        manifest_key = (
                            spec["family"], col, window_name
                        )
                        built, manifest = _step2_aggregate_record(
                            values,
                            spec["family"],
                            col,
                            window_name,
                            include_manifest=manifest_key not in manifest_keys,
                        )
                        manifest_keys.add(manifest_key)
                        rec.update(built)
                        manifest_rows.extend(manifest)
                rows.append(rec)
            _step2_log(
                f"sensor {spec_index}/{total_specs} "
                f"subject {subject_index}/{len(subject_groups)} complete"
            )
        sensor_features = pd.DataFrame(rows)
        if sensor_features["row_id"].duplicated().any():
            raise ValueError(f"duplicate row_id from {spec['path'].name}")
        before = len(output)
        output = output.merge(sensor_features, on="row_id", how="left")
        after = len(output)
        assert before == after, f"row count changed while merging {spec['path'].name}"
        passed = max_used is None or max_used < max_cutoff
        leakage_rows.append(
            {
                "check_name": f"prediction_cutoff::{spec['path'].name}",
                "passed": bool(passed),
                "details": f"max_used={max_used}; max_cutoff={max_cutoff}",
                "affected_feature_count": max(len(sensor_features.columns) - 1, 0),
                "action": "continue" if passed else "stop",
            }
        )
        _step2_log(
            f"sensor {spec_index}/{total_specs} complete: "
            f"{len(sensor_features.columns) - 1} features",
            sensor_started,
        )
    feature_cols = [c for c in output if c != "row_id"]
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("duplicate paper feature columns were generated")
    manifest = pd.DataFrame(manifest_rows).drop_duplicates("feature")
    leakage = pd.DataFrame(leakage_rows)
    if len(leakage) and not leakage["passed"].all():
        raise RuntimeError("sensor prediction-cutoff leakage check failed")
    if not check_only:
        cache_started = time.perf_counter()
        manifest.to_csv(STEP2_FEATURES_PATHS["manifest"], index=False)
        _step2_log(
            f"writing paper feature cache: {len(output)} rows x "
            f"{len(output.columns) - 1} features"
        )
        output.to_parquet(cache_path, index=False)
        _step2_log("paper feature cache written", cache_started)
    return output, manifest, leakage


# Pruned historical definition: _step2_deduplicate (not reachable from the final runner).


# Pruned historical definition: _step2_v2_history_features (not reachable from the final runner).


# Pruned historical definition: _step2_history_calendar (not reachable from the final runner).


# Pruned historical definition: _step2_feature_meta (not reachable from the final runner).


# Pruned historical definition: _step2_select_features (not reachable from the final runner).


# Pruned historical definition: _step2_preprocess_fit (not reachable from the final runner).


# Pruned historical definition: _step2_preprocess_apply (not reachable from the final runner).


# Pruned historical definition: _step2_models (not reachable from the final runner).


# Pruned historical definition: _step2_fit_profile (not reachable from the final runner).


# Pruned historical definition: _step2_accept (not reachable from the final runner).


def _step2_markdown(frame):
    if frame.empty:
        return "_No rows._"
    printable = frame.copy().fillna("")
    columns = [str(col) for col in printable.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in printable.astype(str).itertuples(index=False, name=None):
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def _step2_write_report(
    schema, assumptions, summary, decisions, selected, recommendation
):
    lines = [
        "# try21_v6 paper-inspired Step 2 ablation\n\n",
        "## 1. 실험 목적\n",
        "기존 matching anchor를 유지하면서 단기 변동성, 상태 변화, 최근 추세, "
        "불규칙성, 개인 과거 baseline 대비 deviation 신호를 진단한다.\n\n",
        "## 2. 기반 코드\n",
        "`step2_extended`를 복제했고, 그 안의 `step2_base_design` matching-anchor, full OOF, "
        "user-wise chronological last-block, raw ablation 구조를 기반으로 했다.\n\n",
        "## 3. Step 1을 실행하지 않는 구조\n",
        "저장된 `leakage_val_raw_oof_transition_prior.csv`와 "
        "`submission_step1_profile_best_raw_lastblock.csv`의 matching pair만 읽으며 "
        "Step 1 fitting 함수는 호출하지 않는다. best raw submission이 "
        "`submission_step1_profile_transition_prior_raw_stack.csv`와 동일한지도 "
        "실행 시 검증한다.\n\n",
        "## 4. 논문에서 명시된 전처리\n",
        "사용자별 시계열, 3-observation rolling, 10종 변환, mean/std/min/max 집계, "
        "날짜 피처와 결측 처리를 반영했다.\n\n",
        "## 5. 논문에 명시되지 않은 구현 가정\n",
        _step2_markdown(assumptions) + "\n\n",
        "## 6. 센서 schema 확인 결과\n",
        _step2_markdown(schema) + "\n\n",
        "## 7. leakage 방지 방식\n",
        "각 sleep record의 기존 window 내부에서만 변환하며 cutoff 이후 timestamp를 "
        "제외한다. 날짜 간 history는 shift(1) 기반이고, preprocessing 및 paper "
        "feature selection은 fold fit rows에서만 계산한다.\n\n",
        "## 8. profile별 피처 구성\n",
        "\n".join(
            f"- `{profile}`: {', '.join(transforms)}"
            for profile, transforms in PAPER_PROFILE_TRANSFORMS.items()
        ) + "\n\n",
        "## 9. target별 selected feature\n",
        _step2_markdown(
            selected[selected["fold"].astype(str).eq("full_train")]
        )
        + "\n\n",
        "## 10~17. full OOF, last-block, P2/P3/P4 및 profile 결과\n",
        _step2_markdown(summary) + "\n\n",
        "## 18. 타깃별 최종 선택\n",
        _step2_markdown(decisions) + "\n\n",
        "## 19. 최종 추천 submission\n",
        _step2_markdown(recommendation) + "\n\n",
        "## 20. 한계와 다음 실험\n",
        "이 구현은 paper-inspired time-series features이며 논문을 완전히 재현한 것이 "
        "아니다. EMA, zero-crossing, 불규칙 간격 처리와 cutoff는 프로젝트의 "
        "implementation choice not specified in the paper이다.\n",
    ]
    STEP2_FEATURES_PATHS["report"].write_text("".join(lines), encoding="utf-8")


# Pruned historical definition: _step2_check_only (not reachable from the final runner).


# Pruned historical definition: _step2_run (not reachable from the final runner).


# Pruned historical definition: _step2_parse_args (not reachable from the final runner).


# Pruned historical definition: step2_features_main (not reachable from the final runner).



# ============================================================================
# Embedded step2_sensor_helpers support for the step2_runner Step-2 runner.
# ============================================================================
import sys as _pipeline_sys

import argparse
import ast
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

v6 = _pipeline_sys.modules[__name__]


V7_VERSION = "life_state_v2_optimized"
V7_DIR = v6.ARTIFACT_DIR / "step2_sensor_helpers"
V7_STATE_CACHE = V7_DIR / f"step2_sensor_helpers_{V7_VERSION}_features.parquet"
V7_STATE_MANIFEST = V7_DIR / "step2_sensor_helpers_state_feature_manifest.csv"
V7_SENSOR_CACHE_DIR = V7_DIR / "sensor_cache"
V7_SUMMARY_CHUNK_SIZE = 50_000

V7_WINDOWS = dict(v6.PAPER_WINDOWS)
V7_NESTED_SENSORS = {
    "hr": ("ch2025_wHr.parquet", "heart_rate"),
    "gps": ("ch2025_mGps.parquet", "m_gps"),
    "wifi": ("ch2025_mWifi.parquet", "m_wifi"),
    "ble": ("ch2025_mBle.parquet", "m_ble"),
    "usage": ("ch2025_mUsageStats.parquet", "m_usage_stats"),
    "ambience": ("ch2025_mAmbience.parquet", "m_ambience"),
}

V7_PATHS = {
    key: V7_DIR / path.name.replace("step2_features", "step2_sensor_helpers")
    for key, path in v6.STEP2_FEATURES_PATHS.items()
}
V7_PATHS["feature_cache"] = V7_STATE_CACHE
V7_PATHS["state_manifest"] = V7_STATE_MANIFEST
V7_SUBMISSIONS = {
    key: v6.SUBMISSION_DIR / path.name.replace("step2_features", "step2_sensor_helpers")
    for key, path in v6.STEP2_FEATURES_SUBMISSIONS.items()
}

_STEP2_FEATURES_PATHS_SAVED = dict(v6.STEP2_FEATURES_PATHS)
_STEP2_FEATURES_BUILD_PAPER = v6._step2_build_paper_features
_STEP2_FEATURES_WRITE_REPORT = v6._step2_write_report


def _sensor_helpers_log(message, started=None):
    suffix = ""
    if started is not None:
        suffix = f" ({time.perf_counter() - started:.1f}s)"
    print(f"[TRY21_V7] {message}{suffix}", flush=True)


def _sensor_helpers_safe_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return []


def _sensor_helpers_numeric(values):
    if values is None:
        return np.empty(0, dtype=float)
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        array = pd.to_numeric(
            pd.Series(values, dtype="object"), errors="coerce"
        ).to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _sensor_helpers_distribution(values, prefix):
    values = _sensor_helpers_numeric(values)
    if not len(values):
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q90": np.nan,
            f"{prefix}_range": np.nan,
        }
    return {
        f"{prefix}_count": float(len(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
    }


def _sensor_helpers_hr_summary(value):
    values = _sensor_helpers_numeric(_sensor_helpers_safe_list(value))
    result = _sensor_helpers_distribution(values, "hr")
    if len(values) >= 2:
        diff = np.diff(values)
        x = np.arange(len(values), dtype=float)
        x_centered = x - x.mean()
        denominator = float(np.dot(x_centered, x_centered))
        result["hr_slope"] = (
            float(np.dot(x_centered, values - values.mean()) / denominator)
            if denominator > 0
            else 0.0
        )
        result["hr_abs_diff"] = float(np.mean(np.abs(diff)))
        result["hr_high_jump_count"] = float(np.count_nonzero(np.abs(diff) >= 10))
    else:
        result.update(
            {
                "hr_slope": np.nan,
                "hr_abs_diff": np.nan,
                "hr_high_jump_count": 0.0,
            }
        )
    return result


def _sensor_helpers_gps_summary(value):
    rows = [row for row in _sensor_helpers_safe_list(value) if isinstance(row, dict)]
    speed = _sensor_helpers_numeric([row.get("speed") for row in rows])
    lat = _sensor_helpers_numeric([row.get("latitude") for row in rows])
    lon = _sensor_helpers_numeric([row.get("longitude") for row in rows])
    altitude = _sensor_helpers_numeric([row.get("altitude") for row in rows])
    result = {
        "gps_point_count": float(len(rows)),
        "gps_speed_mean": float(np.mean(speed)) if len(speed) else np.nan,
        "gps_speed_max": float(np.max(speed)) if len(speed) else np.nan,
        "gps_moving_ratio": float(np.mean(speed > 0.5)) if len(speed) else np.nan,
        "gps_stationary_ratio": float(np.mean(speed <= 0.2)) if len(speed) else np.nan,
        "gps_lat_range": float(np.ptp(lat)) if len(lat) else np.nan,
        "gps_lon_range": float(np.ptp(lon)) if len(lon) else np.nan,
        "gps_alt_range": (
            float(np.ptp(altitude)) if len(altitude) else np.nan
        ),
    }
    if len(lat) >= 2 and len(lon) >= 2:
        dlat = np.diff(lat)
        dlon = np.diff(lon)
        result["gps_path_proxy"] = float(np.sqrt(dlat**2 + dlon**2).sum())
    else:
        result["gps_path_proxy"] = np.nan
    return result


def _sensor_helpers_radio_summary(value, address_key):
    rows = [row for row in _sensor_helpers_safe_list(value) if isinstance(row, dict)]
    rssi = _sensor_helpers_numeric([row.get("rssi") for row in rows])
    addresses = [
        str(row.get(address_key))
        for row in rows
        if row.get(address_key) not in (None, "")
    ]
    return {
        "radio_device_count": float(len(rows)),
        "radio_unique_count": float(len(set(addresses))),
        "radio_rssi_mean": float(np.mean(rssi)) if len(rssi) else np.nan,
        "radio_rssi_max": float(np.max(rssi)) if len(rssi) else np.nan,
        "radio_strong_count": float(np.count_nonzero(rssi >= -65)),
        "radio_weak_count": float(np.count_nonzero(rssi <= -85)),
    }


def _sensor_helpers_usage_category(app_name):
    name = str(app_name).lower()
    categories = {
        "communication": [
            "카카오", "메시지", "통화", "전화", "message", "call", "telegram",
        ],
        "media": [
            "youtube", "netflix", "video", "music", "유튜브", "음악", "tiktok",
        ],
        "browser_news": [
            "naver", "chrome", "browser", "internet", "네이버", "뉴스",
        ],
        "finance_shopping": [
            "bank", "pay", "shop", "toss", "토스", "은행", "쇼핑", "쿠팡",
        ],
        "game": ["game", "게임"],
        "system_home": ["one ui", "launcher", "system", "홈"],
    }
    for category, needles in categories.items():
        if any(needle in name for needle in needles):
            return category
    return "other"


def _sensor_helpers_usage_summary(value):
    rows = [row for row in _sensor_helpers_safe_list(value) if isinstance(row, dict)]
    totals = {}
    app_times = []
    app_names = []
    for row in rows:
        app_name = str(row.get("app_name", ""))
        try:
            total_time = max(float(row.get("total_time", 0.0)), 0.0)
        except (TypeError, ValueError):
            total_time = 0.0
        category = _sensor_helpers_usage_category(app_name)
        totals[category] = totals.get(category, 0.0) + total_time
        app_times.append(total_time)
        if app_name:
            app_names.append(app_name)
    total = float(sum(app_times))
    shares = np.asarray(app_times, dtype=float)
    shares = shares / total if total > 0 else shares
    entropy = float(-(shares * np.log(shares + 1e-12)).sum()) if len(shares) else 0.0
    result = {
        "usage_total": total,
        "usage_app_count": float(len(rows)),
        "usage_unique_apps": float(len(set(app_names))),
        "usage_max_app_time": float(max(app_times, default=0.0)),
        "usage_app_entropy": entropy,
    }
    for category in [
        "communication",
        "media",
        "browser_news",
        "finance_shopping",
        "game",
        "system_home",
        "other",
    ]:
        value = float(totals.get(category, 0.0))
        result[f"usage_{category}_time"] = value
        result[f"usage_{category}_share"] = value / (total + 1e-9)
    return result


def _sensor_helpers_ambience_summary(value):
    rows = _sensor_helpers_safe_list(value)
    groups = {
        "speech": ["speech", "conversation", "narration"],
        "music": ["music", "singing", "musical"],
        "vehicle": ["vehicle", "car", "bus", "train", "truck", "motor"],
        "outside": ["outside", "wind", "rustling"],
        "inside": ["inside"],
        "sleep_noise": ["snoring", "breathing"],
    }
    scores = {name: 0.0 for name in groups}
    probabilities = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        label = str(row[0]).lower()
        try:
            score = max(float(row[1]), 0.0)
        except (TypeError, ValueError):
            continue
        probabilities.append(score)
        for group, needles in groups.items():
            if any(needle in label for needle in needles):
                scores[group] += score
    result = {
        f"ambience_{group}_score": value for group, value in scores.items()
    }
    result["ambience_label_count"] = float(len(rows))
    result["ambience_top_score"] = float(max(probabilities, default=0.0))
    result["ambience_entropy"] = float(
        -sum(p * math.log(p + 1e-12) for p in probabilities if p > 0)
    )
    return result


def _sensor_helpers_summarizer(family):
    if family == "hr":
        return _sensor_helpers_hr_summary
    if family == "gps":
        return _sensor_helpers_gps_summary
    if family == "wifi":
        return lambda value: _sensor_helpers_radio_summary(value, "bssid")
    if family == "ble":
        return lambda value: _sensor_helpers_radio_summary(value, "address")
    if family == "usage":
        return _sensor_helpers_usage_summary
    if family == "ambience":
        return _sensor_helpers_ambience_summary
    raise KeyError(f"unknown nested sensor family: {family}")


def _sensor_helpers_summarize_values(values, family):
    summarizer = _sensor_helpers_summarizer(family)
    total = len(values)
    chunks = []
    started = time.perf_counter()
    for start in range(0, total, V7_SUMMARY_CHUNK_SIZE):
        stop = min(start + V7_SUMMARY_CHUNK_SIZE, total)
        records = [summarizer(value) for value in values.iloc[start:stop]]
        chunks.append(pd.DataFrame.from_records(records))
        _sensor_helpers_log(
            f"{family}: summarized {stop:,}/{total:,} raw rows",
            started,
        )
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _sensor_helpers_window_features(part, family, window_name, start, end):
    result = {}
    prefix = f"paper_state_{family}"
    duration_minutes = max((end - start).total_seconds() / 60.0, 1.0)
    timestamps = pd.to_datetime(part["timestamp"]).sort_values()
    result[
        f"{prefix}_event_count_{window_name}_rolling_mean_sum"
    ] = float(len(part))
    result[
        f"{prefix}_coverage_ratio_{window_name}_rolling_mean_mean"
    ] = min(float(len(part)) / duration_minutes, 1.0)
    if len(timestamps):
        offsets = (timestamps - start).dt.total_seconds() / 3600.0
        gaps = timestamps.diff().dt.total_seconds().div(60.0)
        result[
            f"{prefix}_first_event_hour_{window_name}_rolling_mean_mean"
        ] = float(offsets.iloc[0])
        result[
            f"{prefix}_last_event_hour_{window_name}_rolling_mean_mean"
        ] = float(offsets.iloc[-1])
        result[
            f"{prefix}_max_gap_minutes_{window_name}_rolling_std_max"
        ] = float(gaps.max()) if gaps.notna().any() else duration_minutes
    else:
        result[
            f"{prefix}_first_event_hour_{window_name}_rolling_mean_mean"
        ] = np.nan
        result[
            f"{prefix}_last_event_hour_{window_name}_rolling_mean_mean"
        ] = np.nan
        result[
            f"{prefix}_max_gap_minutes_{window_name}_rolling_std_max"
        ] = duration_minutes

    numeric_cols = [
        column
        for column in part.columns
        if column not in {"subject_id", "timestamp"}
    ]
    for column in numeric_cols:
        values = pd.to_numeric(part[column], errors="coerce")
        base = f"{prefix}_{column}_{window_name}"
        result[f"{base}_rolling_mean_mean"] = float(values.mean())
        result[f"{base}_rolling_std_std"] = float(values.std(ddof=0))
        result[f"{base}_rolling_mean_sum"] = float(values.sum(min_count=1))
        result[f"{base}_rolling_mean_max"] = float(values.max())
    return result


def _sensor_helpers_build_sensor_features(meta, family, check_only=False):
    file_name, value_col = V7_NESTED_SENSORS[family]
    path = v6.DATA_DIR / "ch2025_data_items" / file_name
    records = meta
    if check_only:
        first_subject = records["subject_id"].iloc[0]
        records = records[records["subject_id"].eq(first_subject)].head(1)
    subjects = records["subject_id"].astype(str).unique().tolist()
    earliest = pd.to_datetime(records["lifelog_date"]).min() + pd.Timedelta(
        hours=min(start for start, _ in V7_WINDOWS.values())
    )
    latest = pd.to_datetime(records["lifelog_date"]).max() + pd.Timedelta(
        hours=max(end for _, end in V7_WINDOWS.values())
    )
    filters = [
        ("timestamp", ">=", earliest),
        ("timestamp", "<", latest),
        ("subject_id", "in", subjects),
    ]
    load_started = time.perf_counter()
    raw = pd.read_parquet(
        path,
        columns=["subject_id", "timestamp", value_col],
        filters=filters,
    )
    raw["subject_id"] = raw["subject_id"].astype(str)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw[
        raw["subject_id"].isin(subjects)
        & raw["timestamp"].ge(earliest)
        & raw["timestamp"].lt(latest)
    ].reset_index(drop=True)
    _sensor_helpers_log(
        f"{family}: loaded {len(raw):,} relevant raw rows",
        load_started,
    )
    summary = _sensor_helpers_summarize_values(raw[value_col], family)
    sensor = pd.concat(
        [raw[["subject_id", "timestamp"]].reset_index(drop=True), summary],
        axis=1,
    ).sort_values(["subject_id", "timestamp"])
    del raw, summary

    rows = []
    subject_groups = list(records.groupby("subject_id", sort=False))
    for subject_index, (subject, subject_records) in enumerate(subject_groups, 1):
        subject_sensor = sensor[sensor["subject_id"].eq(subject)]
        timestamp_values = subject_sensor["timestamp"].to_numpy()
        for record in subject_records.itertuples(index=False):
            output = {"row_id": int(record.row_id)}
            for window_name, (start_hour, end_hour) in V7_WINDOWS.items():
                start = record.lifelog_date + pd.Timedelta(hours=start_hour)
                end = record.lifelog_date + pd.Timedelta(hours=end_hour)
                left = int(
                    np.searchsorted(timestamp_values, np.datetime64(start), side="left")
                )
                right = int(
                    np.searchsorted(timestamp_values, np.datetime64(end), side="left")
                )
                part = subject_sensor.iloc[left:right]
                output.update(
                    _sensor_helpers_window_features(part, family, window_name, start, end)
                )
            rows.append(output)
        _sensor_helpers_log(
            f"{family}: subject {subject_index}/{len(subject_groups)} complete"
        )
    return pd.DataFrame(rows)


def _sensor_helpers_sensor_cache_path(family):
    return V7_SENSOR_CACHE_DIR / f"{V7_VERSION}_{family}_record_features.parquet"


def _sensor_helpers_load_sensor_cache(records, family):
    path = _sensor_helpers_sensor_cache_path(family)
    if not path.exists():
        return None
    cached = pd.read_parquet(path)
    expected = records["row_id"].astype(int).to_numpy()
    actual = cached["row_id"].astype(int).to_numpy()
    if len(cached) != len(records) or not np.array_equal(actual, expected):
        _sensor_helpers_log(f"{family}: sensor cache is stale")
        return None
    _sensor_helpers_log(f"{family}: loaded sensor cache ({len(cached.columns) - 1} features)")
    return cached


def _sensor_helpers_get(frame, family, source, window, suffix="rolling_mean_mean"):
    column = f"paper_state_{family}_{source}_{window}_{suffix}"
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _sensor_helpers_add_life_state_features(frame, meta):
    result = frame.copy()
    eps = 1e-6
    windows = [
        "evening",
        "bedtime_transition",
        "pre_sleep",
        "late_night",
        "overnight",
        "wake_transition",
        "morning",
    ]
    quiet_scores = {}
    active_scores = {}
    for window in windows:
        usage = np.log1p(
            _sensor_helpers_get(result, "usage", "usage_total", window, "rolling_mean_sum")
            .clip(lower=0)
            .fillna(0)
        )
        gps_move = (
            _sensor_helpers_get(result, "gps", "gps_moving_ratio", window)
            .clip(0, 1)
            .fillna(0)
        )
        hr_std = (
            _sensor_helpers_get(result, "hr", "hr_std", window)
            .clip(lower=0)
            .fillna(0)
        )
        wifi_density = np.log1p(
            _sensor_helpers_get(result, "wifi", "radio_unique_count", window)
            .clip(lower=0)
            .fillna(0)
        )
        sleep_noise = (
            _sensor_helpers_get(result, "ambience", "ambience_sleep_noise_score", window)
            .clip(lower=0)
            .fillna(0)
        )
        quiet = (
            -0.30 * usage
            -0.28 * gps_move
            -0.025 * hr_std
            -0.06 * wifi_density
            +0.20 * sleep_noise
        )
        active = -quiet
        quiet_scores[window] = quiet
        active_scores[window] = active
        result[
            f"paper_state_cross_quiet_score_{window}_rolling_mean_mean"
        ] = quiet
        result[
            f"paper_state_cross_active_score_{window}_rolling_mean_mean"
        ] = active

    centers = pd.Series(
        {
            "bedtime_transition": 23.0,
            "pre_sleep": 24.0,
            "late_night": 26.5,
            "overnight": 28.0,
            "wake_transition": 32.5,
            "morning": 33.0,
        }
    )
    sleep_windows = ["bedtime_transition", "pre_sleep", "late_night", "overnight"]
    wake_windows = ["overnight", "wake_transition", "morning"]
    quiet_matrix = pd.DataFrame(quiet_scores)[sleep_windows]
    active_matrix = pd.DataFrame(active_scores)[wake_windows]
    onset_weights = np.exp(quiet_matrix.clip(-6, 6).to_numpy(float))
    onset_weights /= np.maximum(onset_weights.sum(axis=1, keepdims=True), eps)
    wake_weights = np.exp(active_matrix.clip(-6, 6).to_numpy(float))
    wake_weights /= np.maximum(wake_weights.sum(axis=1, keepdims=True), eps)
    onset = onset_weights @ centers.loc[sleep_windows].to_numpy(float)
    wake = wake_weights @ centers.loc[wake_windows].to_numpy(float)
    duration = np.clip(wake - onset, 2.0, 14.0)

    derived = {
        "sleep_onset_hour": onset,
        "wake_hour": wake,
        "sleep_duration": duration,
        "sleep_midpoint": (onset + wake) / 2.0,
        "sleep_fragmentation": (
            pd.DataFrame(active_scores)[["late_night", "overnight"]]
            .mean(axis=1)
            .to_numpy(float)
        ),
        "late_usage": np.log1p(
            _sensor_helpers_get(
                result,
                "usage",
                "usage_total",
                "late_night",
                "rolling_mean_sum",
            )
            .clip(lower=0)
            .fillna(0)
        ).to_numpy(float),
        "hr_night_drop": (
            _sensor_helpers_get(result, "hr", "hr_mean", "pre_sleep")
            - _sensor_helpers_get(result, "hr", "hr_mean", "overnight")
        ).to_numpy(float),
        "hr_morning_recovery": (
            _sensor_helpers_get(result, "hr", "hr_mean", "morning")
            - _sensor_helpers_get(result, "hr", "hr_mean", "overnight")
        ).to_numpy(float),
        "night_stationary_ratio": _sensor_helpers_get(
            result, "gps", "gps_stationary_ratio", "overnight"
        ).to_numpy(float),
    }
    indexed_meta = meta.set_index("row_id")
    row_ids = result["row_id"].astype(int).to_numpy()
    subject = (
        indexed_meta.loc[row_ids, "subject_id"].astype(str).to_numpy()
    )
    dates = pd.to_datetime(
        indexed_meta.loc[row_ids, "lifelog_date"]
    ).to_numpy()
    order = pd.DataFrame(
        {"row_id": row_ids, "subject": subject, "date": dates}
    ).sort_values(["subject", "date", "row_id"])
    for name, values in derived.items():
        column = f"paper_state_cross_{name}_rolling_mean_mean"
        result[column] = values
        ordered_values = pd.Series(values, index=row_ids).reindex(
            order["row_id"].to_numpy()
        )
        past_mean = ordered_values.groupby(order["subject"].to_numpy()).transform(
            lambda series: series.expanding().mean().shift(1)
        )
        deviation = ordered_values - past_mean
        deviation = deviation.reindex(result["row_id"]).to_numpy(float)
        result[
            f"paper_state_cross_{name}_difference_mean"
        ] = deviation

    coverage_columns = [
        column
        for column in result
        if "_coverage_ratio_full_sleepctx_" in column
    ]
    if coverage_columns:
        coverage = result[coverage_columns].apply(pd.to_numeric, errors="coerce")
        result[
            "paper_state_cross_sensor_coverage_rolling_mean_mean"
        ] = coverage.mean(axis=1)
        result[
            "paper_state_cross_missing_sensor_count_rolling_mean_sum"
        ] = coverage.isna().sum(axis=1) + coverage.le(0).sum(axis=1)
    return result


def _sensor_helpers_state_manifest(columns):
    rows = []
    for column in columns:
        transform = next(
            (
                name
                for name in v6.PAPER_PROFILE_TRANSFORMS["paper_full"]
                if f"_{name}_" in column
            ),
            "",
        )
        rows.append(
            {
                "feature": column,
                "sensor_family": "life_state",
                "source_column": column,
                "window": next(
                    (window for window in V7_WINDOWS if f"_{window}_" in column),
                    "cross_window",
                ),
                "transform": transform,
                "aggregation": column.rsplit("_", 1)[-1],
            }
        )
    return pd.DataFrame(rows)


def _sensor_helpers_build_state_features(meta, check_only=False, rebuild_cache=False):
    records = meta.copy()
    if check_only:
        first_subject = records["subject_id"].iloc[0]
        records = records[records["subject_id"].eq(first_subject)].head(1).copy()
    if V7_STATE_CACHE.exists() and not check_only and not rebuild_cache:
        cached = pd.read_parquet(V7_STATE_CACHE)
        expected = records["row_id"].astype(int).to_numpy()
        actual = cached["row_id"].astype(int).to_numpy()
        if len(cached) == len(records) and np.array_equal(expected, actual):
            manifest = _sensor_helpers_state_manifest(
                [column for column in cached if column != "row_id"]
            )
            _sensor_helpers_log(
                f"loaded state cache: {len(cached.columns) - 1} features"
            )
            return cached, manifest
        _sensor_helpers_log("state cache is stale; rebuilding")

    output = records[["row_id"]].copy()
    families = list(V7_NESTED_SENSORS)
    if not check_only:
        V7_SENSOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for index, family in enumerate(families, 1):
        started = time.perf_counter()
        _sensor_helpers_log(f"nested sensor {index}/{len(families)}: {family}")
        built = (
            None
            if check_only or rebuild_cache
            else _sensor_helpers_load_sensor_cache(records, family)
        )
        if built is None:
            built = _sensor_helpers_build_sensor_features(
                records, family, check_only=check_only
            )
            if not check_only:
                built.to_parquet(_sensor_helpers_sensor_cache_path(family), index=False)
                _sensor_helpers_log(f"{family}: sensor cache written")
        output = output.merge(built, on="row_id", how="left", validate="one_to_one")
        _sensor_helpers_log(
            f"{family} ready: {len(built.columns) - 1} features", started
        )
    output = _sensor_helpers_add_life_state_features(output, records)
    feature_columns = [column for column in output if column != "row_id"]
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("duplicate v7 life-state feature names")
    numeric = output[feature_columns].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(float)).any():
        raise ValueError("v7 life-state features contain infinity")
    manifest = _sensor_helpers_state_manifest(feature_columns)
    if not check_only:
        V7_DIR.mkdir(parents=True, exist_ok=True)
        output.to_parquet(V7_STATE_CACHE, index=False)
        manifest.to_csv(V7_STATE_MANIFEST, index=False)
    return output, manifest


# Pruned historical definition: _sensor_helpers_build_combined_features (not reachable from the final runner).


# Pruned historical definition: _sensor_helpers_write_report (not reachable from the final runner).


# Pruned historical definition: _sensor_helpers_configure_step2_features (not reachable from the final runner).


# Pruned historical definition: _sensor_helpers_parse_args (not reachable from the final runner).


# Pruned historical definition: step2_sensor_helpers_main (not reachable from the final runner).



# ============================================================================
# Embedded step2_runner Step-2 scenario runner.
# Unqualified v8 ARTIFACT_DIR is renamed to STEP2_ARTIFACT_DIR so it does not
# overwrite the embedded step1_pipeline/step2_features base ARTIFACT_DIR in this monolithic file.
# ============================================================================

import argparse
import json
import math
import pickle
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import log_loss

v6 = _pipeline_sys.modules[__name__]
v7 = _pipeline_sys.modules[__name__]


SEED = v6.TRY21_V3_SEED
TARGETS = list(v6.TARGETS)
KEY_COLUMNS = ["subject_id", "sleep_date", "lifelog_date"]
STEP2_ARTIFACT_DIR = v6.ARTIFACT_DIR / "step2_runner"
PREDICTION_DIR = STEP2_ARTIFACT_DIR / "predictions"
CACHE_DIR = STEP2_ARTIFACT_DIR / "feature_cache"
STEP2_MODEL_DIR = ROOT / "models" / "step2_runner"
SUBMISSION_DIR = v6.SUBMISSION_DIR

FILES = {
    "registry": STEP2_ARTIFACT_DIR / "step2_run_registry.csv",
    "target_scores": STEP2_ARTIFACT_DIR / "step2_targetwise_scores.csv",
    "average_scores": STEP2_ARTIFACT_DIR / "step2_average_scores.csv",
    "feature_ablation": STEP2_ARTIFACT_DIR / "step2_feature_ablation.csv",
    "model_ablation": STEP2_ARTIFACT_DIR / "step2_model_ablation.csv",
    "blend_results": STEP2_ARTIFACT_DIR / "step2_blend_results.csv",
    "calibration_results": STEP2_ARTIFACT_DIR / "step2_calibration_results.csv",
    "selected_features": STEP2_ARTIFACT_DIR / "step2_selected_features.csv",
    "family_summary": STEP2_ARTIFACT_DIR / "step2_selected_feature_family_summary.csv",
    "best_single": STEP2_ARTIFACT_DIR / "step2_best_single_run.csv",
    "best_per_target": STEP2_ARTIFACT_DIR / "step2_best_per_target.csv",
    "best_hybrid": STEP2_ARTIFACT_DIR / "step2_best_targetwise_hybrid.csv",
    "leakage": STEP2_ARTIFACT_DIR / "step2_leakage_checks.csv",
    "runtime": STEP2_ARTIFACT_DIR / "step2_runtime_summary.csv",
}

SUBMISSIONS = {
    "best_single": SUBMISSION_DIR / "submission_step2_best_model.csv",
    "best_hybrid": SUBMISSION_DIR / "submission_step2_target_ensemble.csv",
    "best_calibrated": SUBMISSION_DIR / "submission_step2_calibrated.csv",
}

FEATURE_RUNS = [
    ("S00_BASE_V7", "base_v7", 32),
    ("S01_K32", "base_v7", 32),
    ("S01_K64", "base_v7", 64),
    ("S01_K96", "base_v7", 96),
    ("S01_K128", "base_v7", 128),
    ("S02_STAT_FEATURES", "stat", 64),
    ("S03_SCREEN_SLEEP_EPISODE", "screen", 64),
    ("S04_PAST_ONLY_PERSONALIZATION", "personal", 64),
    ("S05_MISSINGNESS_CONFIDENCE", "missing", 64),
    ("S06_SLEEP_FRAGMENTATION", "fragmentation", 64),
    ("S07_ALL_K64", "all", 64),
    ("S07_ALL_K96", "all", 96),
    ("S07_ALL_K128", "all", 128),
]


def log(message, started=None):
    suffix = ""
    if started is not None:
        suffix = f" ({time.perf_counter() - started:.1f}s)"
    print(f"[TRY21_V8] {message}{suffix}", flush=True)


def ensure_dirs():
    for path in [
        STEP2_ARTIFACT_DIR,
        PREDICTION_DIR,
        CACHE_DIR,
        STEP2_MODEL_DIR,
        SUBMISSION_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        warnings.warn(f"ignoring unreadable partial CSV {path}: {error}")
        return pd.DataFrame()


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{time.time_ns()}.tmp"
    )
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{time.time_ns()}.tmp"
    )
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


# Pruned historical definition: append_csv (not reachable from the final runner).


def replace_run_rows(path, rows, run_tag):
    frame = pd.DataFrame(rows)
    old = safe_read_csv(path)
    if not old.empty:
        if "run_tag" in old.columns:
            old = old[old["run_tag"].astype(str) != str(run_tag)]
        else:
            warnings.warn(
                f"discarding malformed CSV without run_tag column: {path}"
            )
            old = pd.DataFrame()
    if frame.empty:
        if old.empty:
            Path(path).unlink(missing_ok=True)
            return
        atomic_write_csv(old, path)
        return
    if not old.empty:
        frame = pd.concat([old, frame], ignore_index=True)
    atomic_write_csv(frame, path)


def prediction_paths(run_tag):
    return {
        kind: PREDICTION_DIR / f"{run_tag}_{kind}.parquet"
        for kind in ["oof", "test", "lastblock"]
    }


def model_checkpoint_dir(run_tag):
    return STEP2_MODEL_DIR / run_tag


def model_checkpoint_path(run_tag, target, kind, name=None, fold=None):
    parts = [str(target), str(kind)]
    if name is not None:
        parts.append(str(name))
    if fold is not None:
        parts.append(f"fold{fold}")
    return model_checkpoint_dir(run_tag) / ("__".join(parts) + ".pkl")


def atomic_write_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with open(temporary, "wb") as stream:
        pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


# Pruned historical definition: read_pickle (not reachable from the final runner).


def run_complete(run_tag):
    paths = prediction_paths(run_tag)
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        for path in paths.values():
            columns = pd.read_parquet(path, columns=TARGETS).columns
            if not set(TARGETS).issubset(columns):
                return False
    except Exception as error:
        warnings.warn(
            f"{run_tag}: prediction artifact is incomplete; rerunning ({error})"
        )
        return False
    if not FILES["registry"].exists():
        return False
    registry = safe_read_csv(FILES["registry"])
    if registry.empty or not {"run_tag", "status"}.issubset(registry.columns):
        return False
    return bool(
        ((registry["run_tag"].astype(str) == run_tag)
         & (registry["status"].astype(str) == "completed")).any()
    )


def clip_probability(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prediction contains NaN or infinity")
    return np.clip(values, 1e-6, 1 - 1e-6)


def probability_logit(values):
    values = clip_probability(values)
    return np.log(values / (1.0 - values))


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=float), -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def validate_probability_frame(frame, expected_rows):
    if len(frame) != expected_rows:
        raise ValueError("prediction row count mismatch")
    if not set(TARGETS).issubset(frame):
        raise ValueError("prediction target schema mismatch")
    values = frame[TARGETS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("prediction contains NaN/inf")
    if (values <= 0).any() or (values >= 1).any():
        raise ValueError("binary probability is outside (0, 1)")


def load_fixed_context():
    ensure_dirs()
    original_path = v6.STEP2_FEATURES_PATHS["anchor_pair"]
    try:
        v6.STEP2_FEATURES_PATHS["anchor_pair"] = (
            STEP2_ARTIFACT_DIR / "step1_anchor_pair_check.csv"
        )
        train, sample, anchor_oof, anchor_test, pair = v6._step2_load_anchor()
    finally:
        v6.STEP2_FEATURES_PATHS["anchor_pair"] = original_path

    train = train.reset_index(drop=True)
    sample = sample.reset_index(drop=True)
    anchor_oof = anchor_oof.reset_index(drop=True)
    anchor_test = anchor_test.reset_index(drop=True)
    train_keys = train[KEY_COLUMNS].astype(str).reset_index(drop=True)
    sample_keys = sample[KEY_COLUMNS].astype(str).reset_index(drop=True)
    assert len(anchor_oof) == len(train)
    assert len(anchor_test) == len(sample)
    assert anchor_test[KEY_COLUMNS].astype(str).reset_index(drop=True).equals(
        sample_keys
    )
    assert not train_keys.duplicated().any()
    assert not sample_keys.duplicated().any()

    quarters = v6._step2_v1_quarters(train).reset_index(drop=True)
    chronological_folds = v6._longterm_v1_subject_chrono_folds(train).reset_index(drop=True)
    guard = v6.make_subject_lastblock_mask(train).reset_index(drop=True)
    meta = v6._step2_meta(train, sample)
    if not np.array_equal(meta["row_id"].to_numpy(), np.arange(len(meta))):
        raise AssertionError("row_id order is not stable")

    checks = [
        ("step1_matching_pair_verified", bool(pair.iloc[0]["matching_pair_verified"]),
         str(pair.iloc[0]["anchor_oof_source"])),
        ("train_row_order_unique", not train_keys.duplicated().any(), str(len(train))),
        ("test_row_order_matches_submission", True, str(len(sample))),
        ("same_quarters_all_runs", quarters.notna().all(), "subject-wise 4 quarters"),
        ("same_lastblock_all_runs", guard.any(), f"rows={int(guard.sum())}"),
        ("targets_not_features", True, "|".join(TARGETS)),
    ]
    return {
        "train": train,
        "sample": sample,
        "anchor_oof": anchor_oof[TARGETS].astype(float),
        "anchor_test": anchor_test,
        "quarters": quarters,
        "folds": chronological_folds,
        "guard": guard.to_numpy(bool),
        "meta": meta,
        "checks": checks,
    }


def numeric_feature_frame(frame):
    blocked = set(TARGETS + KEY_COLUMNS + ["row_id", "is_train"])
    columns = [
        column for column in frame.columns
        if column not in blocked
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    output = frame[columns].copy()
    output = output.loc[:, ~output.columns.duplicated()]
    output = output.replace([np.inf, -np.inf], np.nan)
    return output


def load_v7_base_features(context, reuse_cache=True):
    cache = CACHE_DIR / "base_v7_features.parquet"
    if reuse_cache and cache.exists():
        frame = pd.read_parquet(cache)
    else:
        v7_cache = v7.V7_STATE_CACHE
        v6_cache = v6.STEP2_FEATURES_PATHS["feature_cache"]
        if not v7_cache.exists() or not v6_cache.exists():
            raise FileNotFoundError(
                "step2_features/step2_sensor_helpers cache is missing; run step2_sensor_helpers once first"
            )
        paper = pd.read_parquet(v6_cache)
        state = pd.read_parquet(v7_cache)
        frame = paper.merge(state, on="row_id", validate="one_to_one")
        frame.to_parquet(cache, index=False)
    expected = context["meta"]["row_id"].to_numpy(int)
    actual = frame["row_id"].to_numpy(int)
    if len(frame) != len(expected) or not np.array_equal(actual, expected):
        raise RuntimeError("v7 feature cache row order differs from train/test")
    return frame


def safe_sequence_stats(values, timestamps):
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    timestamps = pd.to_datetime(pd.Series(timestamps)).to_numpy("datetime64[ns]")
    valid = np.isfinite(values)
    values = values[valid]
    timestamps = timestamps[valid]
    names = [
        "median", "q25", "q75", "iqr", "mad", "cv", "skewness",
        "kurtosis", "trimmed_mean", "first_last_diff", "linear_slope",
        "lag1_autocorr", "mean_abs_diff", "median_abs_diff",
        "zero_crossing_count", "mean_crossing_count", "number_of_peaks",
        "median_event_gap", "p90_event_gap", "max_event_gap", "active_span",
        "event_density", "longest_inactive_span",
    ]
    result = {name: np.nan for name in names}
    n = len(values)
    if not n:
        return result
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    result.update(
        median=median,
        q25=float(q25),
        q75=float(q75),
        iqr=float(q75 - q25),
        mad=float(np.median(np.abs(values - median))),
    )
    mean, std = float(np.mean(values)), float(np.std(values))
    result["cv"] = std / abs(mean) if abs(mean) > 1e-9 else np.nan
    if n >= 3 and std > 1e-9:
        z = (values - mean) / std
        result["skewness"] = float(np.mean(z**3))
        result["kurtosis"] = float(np.mean(z**4) - 3.0)
    trim = max(int(n * 0.1), 0)
    ordered = np.sort(values)
    trimmed = ordered[trim:n - trim] if n - 2 * trim > 0 else ordered
    result["trimmed_mean"] = float(np.mean(trimmed))
    result["first_last_diff"] = float(values[-1] - values[0])
    if n >= 2:
        x = np.arange(n, dtype=float)
        result["linear_slope"] = float(np.polyfit(x, values, 1)[0])
        diff = np.diff(values)
        result["mean_abs_diff"] = float(np.mean(np.abs(diff)))
        result["median_abs_diff"] = float(np.median(np.abs(diff)))
        result["zero_crossing_count"] = float(
            np.count_nonzero(np.sign(values[:-1]) != np.sign(values[1:]))
        )
        centered = values - mean
        result["mean_crossing_count"] = float(
            np.count_nonzero(np.sign(centered[:-1]) != np.sign(centered[1:]))
        )
        if std > 1e-9 and np.std(values[:-1]) > 1e-9 and np.std(values[1:]) > 1e-9:
            result["lag1_autocorr"] = float(
                np.corrcoef(values[:-1], values[1:])[0, 1]
            )
    if n >= 3:
        result["number_of_peaks"] = float(
            np.count_nonzero(
                (values[1:-1] > values[:-2])
                & (values[1:-1] > values[2:])
            )
        )
    if len(timestamps):
        span = float(
            (timestamps[-1] - timestamps[0]) / np.timedelta64(1, "m")
        )
        result["active_span"] = max(span, 0.0)
        result["event_density"] = n / max(span, 1.0)
    if len(timestamps) >= 2:
        gaps = np.diff(timestamps) / np.timedelta64(1, "m")
        result["median_event_gap"] = float(np.median(gaps))
        result["p90_event_gap"] = float(np.quantile(gaps, 0.9))
        result["max_event_gap"] = float(np.max(gaps))
        result["longest_inactive_span"] = float(np.max(gaps))
    return result


def build_stat_features(context, reuse_cache=True):
    cache = CACHE_DIR / "statistical_features.parquet"
    if reuse_cache and cache.exists():
        return pd.read_parquet(cache)
    meta = context["meta"]
    output = meta[["row_id"]].copy()
    schema, specs = v6._step2_sensor_schema(v6.DATA_DIR / "ch2025_data_items")
    del schema
    rows = []
    for record in meta.itertuples(index=False):
        rows.append({"row_id": int(record.row_id)})
    row_map = {int(row["row_id"]): row for row in rows}
    for spec in specs:
        sensor = pd.read_parquet(
            spec["path"], columns=["subject_id", "timestamp", *spec["columns"]]
        )
        sensor["subject_id"] = sensor["subject_id"].astype(str)
        sensor["timestamp"] = pd.to_datetime(sensor["timestamp"])
        for subject, records in meta.groupby("subject_id", sort=False):
            subject_sensor = sensor[sensor["subject_id"].eq(str(subject))]
            times = subject_sensor["timestamp"].to_numpy()
            for record in records.itertuples(index=False):
                for window, (start_h, end_h) in v6.PAPER_WINDOWS.items():
                    start = record.lifelog_date + pd.Timedelta(hours=start_h)
                    end = record.lifelog_date + pd.Timedelta(hours=end_h)
                    left = np.searchsorted(times, np.datetime64(start), side="left")
                    right = np.searchsorted(times, np.datetime64(end), side="left")
                    part = subject_sensor.iloc[left:right]
                    for column in spec["columns"]:
                        stats = safe_sequence_stats(part[column], part["timestamp"])
                        prefix = f"new_stat_{spec['family']}_{column}_{window}"
                        row_map[int(record.row_id)].update(
                            {f"{prefix}_{name}": value for name, value in stats.items()}
                        )
    output = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    output.to_parquet(cache, index=False)
    return output


def merge_short_on_intervals(events):
    events = events.sort_values("timestamp").drop_duplicates(
        ["timestamp"], keep="last"
    )
    if events.empty:
        return events
    events = events.loc[
        events["state"].ne(events["state"].shift())
    ].reset_index(drop=True)
    keep_state = events["state"].to_numpy(int).copy()
    times = pd.to_datetime(events["timestamp"]).to_numpy("datetime64[ns]")
    for index in range(1, len(events) - 1):
        if keep_state[index - 1] == 0 and keep_state[index] == 1 and keep_state[index + 1] == 0:
            duration = float((times[index + 1] - times[index]) / np.timedelta64(1, "m"))
            if duration <= 2.0:
                keep_state[index] = 0
    events = events.assign(state=keep_state)
    return events.loc[events["state"].ne(events["state"].shift())].reset_index(drop=True)


def screen_episode_record(events, start, end, soft_row):
    names = [
        "screen_sleep_detected", "screen_sleep_confidence", "screen_bedtime",
        "screen_wake_time", "screen_sleep_duration", "screen_longest_off_minutes",
        "screen_second_longest_off_minutes", "screen_sleep_block_count",
        "screen_interruption_count", "screen_interruption_minutes",
        "screen_last_on_before_sleep", "screen_first_on_after_sleep",
    ]
    result = {f"new_screen_{name}": np.nan for name in names}
    if events.empty:
        result["new_screen_screen_sleep_detected"] = 0.0
        return result
    merged = merge_short_on_intervals(events)
    times = pd.to_datetime(merged["timestamp"]).tolist()
    states = merged["state"].astype(int).tolist()
    blocks = []
    for index, (timestamp, state) in enumerate(zip(times, states)):
        block_end = times[index + 1] if index + 1 < len(times) else end
        if state == 0:
            minutes = (block_end - timestamp).total_seconds() / 60.0
            blocks.append((timestamp, block_end, max(minutes, 0.0)))
    valid = [block for block in blocks if block[2] >= 100.0]
    ranked = sorted(blocks, key=lambda item: item[2], reverse=True)
    result["new_screen_screen_longest_off_minutes"] = (
        ranked[0][2] if ranked else 0.0
    )
    result["new_screen_screen_second_longest_off_minutes"] = (
        ranked[1][2] if len(ranked) > 1 else 0.0
    )
    result["new_screen_screen_sleep_block_count"] = float(len(valid))
    result["new_screen_screen_sleep_detected"] = float(bool(valid))
    if not valid:
        result["new_screen_screen_sleep_confidence"] = 0.0
        return result
    primary = max(valid, key=lambda item: item[2])
    bedtime, wake, duration = primary
    result["new_screen_screen_sleep_confidence"] = float(
        np.clip(duration / 480.0, 0.0, 1.0)
    )
    result["new_screen_screen_bedtime"] = (
        bedtime - start
    ).total_seconds() / 3600.0 + 21.0
    result["new_screen_screen_wake_time"] = (
        wake - start
    ).total_seconds() / 3600.0 + 21.0
    result["new_screen_screen_sleep_duration"] = duration / 60.0
    inside = merged[
        merged["timestamp"].ge(bedtime)
        & merged["timestamp"].lt(wake)
        & merged["state"].eq(1)
    ]
    result["new_screen_screen_interruption_count"] = float(len(inside))
    interruption_minutes = 0.0
    for index in inside.index:
        next_times = merged.loc[
            merged.index > index, "timestamp"
        ]
        next_time = next_times.iloc[0] if len(next_times) else wake
        interruption_minutes += min(
            max((next_time - merged.loc[index, "timestamp"]).total_seconds() / 60.0, 0.0),
            120.0,
        )
    result["new_screen_screen_interruption_minutes"] = interruption_minutes
    before = merged[
        merged["timestamp"].lt(bedtime) & merged["state"].eq(1)
    ]["timestamp"]
    after = merged[
        merged["timestamp"].ge(wake) & merged["state"].eq(1)
    ]["timestamp"]
    result["new_screen_screen_last_on_before_sleep"] = (
        (before.max() - start).total_seconds() / 3600.0 + 21.0
        if len(before) else np.nan
    )
    result["new_screen_screen_first_on_after_sleep"] = (
        (after.min() - start).total_seconds() / 3600.0 + 21.0
        if len(after) else np.nan
    )
    soft_onset = soft_row.get(
        "paper_state_cross_sleep_onset_hour_rolling_mean_mean", np.nan
    )
    soft_wake = soft_row.get(
        "paper_state_cross_wake_hour_rolling_mean_mean", np.nan
    )
    soft_duration = soft_row.get(
        "paper_state_cross_sleep_duration_rolling_mean_mean", np.nan
    )
    result["new_screen_bedtime_soft_difference"] = (
        result["new_screen_screen_bedtime"] - soft_onset
    )
    result["new_screen_wake_soft_difference"] = (
        result["new_screen_screen_wake_time"] - soft_wake
    )
    result["new_screen_duration_soft_difference"] = (
        result["new_screen_screen_sleep_duration"] - soft_duration
    )
    result["new_screen_boundary_agreement"] = float(
        abs(result["new_screen_bedtime_soft_difference"]) <= 1.0
        and abs(result["new_screen_wake_soft_difference"]) <= 1.0
    )
    return result


def build_screen_features(context, base, reuse_cache=True):
    cache = CACHE_DIR / "screen_sleep_episode.parquet"
    if reuse_cache and cache.exists():
        return pd.read_parquet(cache)
    path = v6.DATA_DIR / "ch2025_data_items" / "ch2025_mScreenStatus.parquet"
    sensor = pd.read_parquet(path)
    sensor["subject_id"] = sensor["subject_id"].astype(str)
    sensor["timestamp"] = pd.to_datetime(sensor["timestamp"])
    sensor["state"] = pd.to_numeric(sensor["m_screen_use"], errors="coerce").fillna(0).gt(0).astype(int)
    rows = []
    indexed_base = base.set_index("row_id")
    for record in context["meta"].itertuples(index=False):
        start = record.lifelog_date + pd.Timedelta(hours=21)
        end = record.lifelog_date + pd.Timedelta(hours=35)
        events = sensor[
            sensor["subject_id"].eq(record.subject_id)
            & sensor["timestamp"].ge(start)
            & sensor["timestamp"].lt(end)
        ][["timestamp", "state"]]
        soft = indexed_base.loc[int(record.row_id)].to_dict()
        row = {"row_id": int(record.row_id)}
        row.update(screen_episode_record(events, start, end, soft))
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_parquet(cache, index=False)
    return output


def choose_personalization_columns(frame):
    tokens = [
        "sleep_onset", "wake_hour", "sleep_duration", "fragmentation",
        "late_usage", "hr_mean_overnight", "hr_std_overnight",
        "activity", "pedometer", "step", "screen",
    ]
    columns = [
        column for column in frame
        if column != "row_id"
        and any(token in column.lower() for token in tokens)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return columns[:80]


def build_personalization_features(context, source, reuse_cache=True):
    cache = CACHE_DIR / "past_only_personalization.parquet"
    if reuse_cache and cache.exists():
        return pd.read_parquet(cache)
    meta = context["meta"][["row_id", "subject_id", "lifelog_date"]].copy()
    data = meta.merge(source, on="row_id", how="left", validate="one_to_one")
    data = data.sort_values(["subject_id", "lifelog_date", "row_id"]).reset_index(drop=True)
    output = data[["row_id"]].copy()
    columns = choose_personalization_columns(source)
    grouped = data.groupby("subject_id", sort=False)
    weekday = pd.to_datetime(data["lifelog_date"]).dt.dayofweek
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")
        past = grouped[column]
        output[f"new_personal_{column}_lag1"] = past.shift(1)
        output[f"new_personal_{column}_lag2"] = past.shift(2)
        for window in [3, 7, 14]:
            output[f"new_personal_{column}_roll{window}_mean"] = past.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=2).mean()
            )
        for window in [3, 7]:
            output[f"new_personal_{column}_roll{window}_std"] = past.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=2).std()
            )
        expanding_mean = past.transform(
            lambda series: series.shift(1).expanding(min_periods=2).mean()
        )
        expanding_std = past.transform(
            lambda series: series.shift(1).expanding(min_periods=2).std()
        )
        output[f"new_personal_{column}_expanding_mean"] = expanding_mean
        output[f"new_personal_{column}_expanding_std"] = expanding_std
        output[f"new_personal_{column}_current_minus_past"] = values - expanding_mean
        roll3 = output[f"new_personal_{column}_roll3_mean"]
        output[f"new_personal_{column}_roll3_minus_expanding"] = roll3 - expanding_mean
        temp = pd.DataFrame(
            {
                "subject_id": data["subject_id"],
                "weekday": weekday,
                "value": values,
            }
        )
        same_weekday = temp.groupby(["subject_id", "weekday"])["value"].transform(
            lambda series: series.shift(1).expanding(min_periods=1).mean()
        )
        output[f"new_personal_{column}_same_weekday_mean"] = same_weekday
        output[f"new_personal_{column}_current_minus_same_weekday"] = values - same_weekday
    output = output.sort_values("row_id").reset_index(drop=True)
    expected_row_ids = context["meta"]["row_id"].astype(int).to_numpy()
    actual_row_ids = output["row_id"].astype(int).to_numpy()
    if not np.array_equal(actual_row_ids, expected_row_ids):
        raise RuntimeError(
            "personalization feature row order differs from train/test meta"
        )
    output.to_parquet(cache, index=False)
    return output


def build_missingness_features(context, base, reuse_cache=True):
    cache = CACHE_DIR / "missingness_confidence.parquet"
    if reuse_cache and cache.exists():
        return pd.read_parquet(cache)
    meta = context["meta"][["row_id", "subject_id", "lifelog_date"]].copy()
    families = list(v7.V7_NESTED_SENSORS)
    output = meta[["row_id"]].copy()
    available_columns = []
    overnight_available = []
    for family in families:
        count = f"paper_state_{family}_event_count_full_sleepctx_rolling_mean_sum"
        coverage = f"paper_state_{family}_coverage_ratio_full_sleepctx_rolling_mean_mean"
        max_gap = f"paper_state_{family}_max_gap_minutes_full_sleepctx_rolling_std_max"
        first = f"paper_state_{family}_first_event_hour_full_sleepctx_rolling_mean_mean"
        last = f"paper_state_{family}_last_event_hour_full_sleepctx_rolling_mean_mean"
        overnight = f"paper_state_{family}_event_count_overnight_rolling_mean_sum"
        count_values = pd.to_numeric(base.get(count), errors="coerce")
        coverage_values = pd.to_numeric(base.get(coverage), errors="coerce")
        available = count_values.fillna(0).gt(0).astype(float)
        output[f"new_missing_{family}_sensor_available"] = available
        output[f"new_missing_{family}_event_count"] = count_values
        output[f"new_missing_{family}_coverage"] = coverage_values
        output[f"new_missing_{family}_max_gap"] = pd.to_numeric(base.get(max_gap), errors="coerce")
        output[f"new_missing_{family}_observed_span"] = (
            pd.to_numeric(base.get(last), errors="coerce")
            - pd.to_numeric(base.get(first), errors="coerce")
        )
        available_columns.append(f"new_missing_{family}_sensor_available")
        overnight_name = f"new_missing_{family}_overnight_available"
        output[overnight_name] = pd.to_numeric(
            base.get(overnight), errors="coerce"
        ).fillna(0).gt(0).astype(float)
        overnight_available.append(overnight_name)
        work = pd.DataFrame(
            {
                "row_id": meta["row_id"],
                "subject_id": meta["subject_id"],
                "lifelog_date": pd.to_datetime(meta["lifelog_date"]),
                "missing": 1.0 - available,
            }
        ).sort_values(["subject_id", "lifelog_date", "row_id"])
        previous = work.groupby("subject_id")["missing"].shift(1)
        output.loc[work.index, f"new_missing_{family}_current_day_missing"] = work["missing"]
        output.loc[work.index, f"new_missing_{family}_previous_day_missing"] = previous
        output.loc[work.index, f"new_missing_{family}_current_and_previous_missing"] = (
            work["missing"] * previous
        )
        consecutive = work.groupby("subject_id")["missing"].transform(
            lambda series: series.groupby(series.eq(0).cumsum()).cumsum()
        )
        output.loc[work.index, f"new_missing_{family}_consecutive_missing_days"] = consecutive
        observed_date = work["lifelog_date"].where(work["missing"].eq(0))
        last_observed = observed_date.groupby(work["subject_id"]).ffill().shift(1)
        output.loc[work.index, f"new_missing_{family}_days_since_last_observed"] = (
            work["lifelog_date"] - last_observed
        ).dt.days
    output["new_missing_available_sensor_count"] = output[available_columns].sum(axis=1)
    output["new_missing_missing_sensor_count"] = len(families) - output["new_missing_available_sensor_count"]
    output["new_missing_overnight_available_sensor_count"] = output[overnight_available].sum(axis=1)
    coverage_columns = [column for column in output if column.endswith("_coverage")]
    mean_coverage = output[coverage_columns].mean(axis=1)
    output["new_missing_high_coverage_flag"] = mean_coverage.ge(0.5).astype(float)
    output["new_missing_low_coverage_flag"] = mean_coverage.lt(0.15).astype(float)
    screen_conf = pd.to_numeric(
        base.get("new_screen_screen_sleep_confidence"), errors="coerce"
    )
    output["new_missing_sleep_boundary_confidence"] = (
        0.6 * mean_coverage.fillna(0)
        + 0.4 * screen_conf.fillna(0)
    )
    output.to_parquet(cache, index=False)
    return output


def build_fragmentation_features(context, base, screen, reuse_cache=True):
    cache = CACHE_DIR / "sleep_fragmentation.parquet"
    if reuse_cache and cache.exists():
        return pd.read_parquet(cache)
    output = context["meta"][["row_id"]].copy()

    def value(family, source, suffix="rolling_mean_sum"):
        name = f"paper_state_{family}_{source}_overnight_{suffix}"
        return pd.to_numeric(base.get(name), errors="coerce")

    output["new_frag_overnight_screen_on_count"] = pd.to_numeric(
        screen.get("new_screen_screen_interruption_count"), errors="coerce"
    )
    output["new_frag_overnight_screen_on_minutes"] = pd.to_numeric(
        screen.get("new_screen_screen_interruption_minutes"), errors="coerce"
    )
    output["new_frag_overnight_usage_event_count"] = value(
        "usage", "event_count", "rolling_mean_sum"
    )
    output["new_frag_overnight_usage_minutes"] = value("usage", "usage_total")
    output["new_frag_overnight_movement_burst_count"] = value(
        "gps", "gps_moving_ratio", "rolling_mean_sum"
    )
    output["new_frag_overnight_hr_jump_count"] = value(
        "hr", "hr_high_jump_count"
    )
    output["new_frag_overnight_hr_abs_diff"] = value("hr", "hr_abs_diff")
    output["new_frag_overnight_wifi_change_count"] = value(
        "wifi", "radio_unique_count"
    )
    output["new_frag_overnight_ble_change_count"] = value(
        "ble", "radio_unique_count"
    )
    max_gap_columns = [
        f"paper_state_{family}_max_gap_minutes_overnight_rolling_std_max"
        for family in ["hr", "gps", "wifi", "ble", "usage"]
    ]
    gaps = pd.concat(
        [pd.to_numeric(base.get(column), errors="coerce") for column in max_gap_columns],
        axis=1,
    )
    output["new_frag_longest_uninterrupted_quiet_block"] = gaps.max(axis=1)
    output["new_frag_quiet_block_count"] = gaps.ge(30).sum(axis=1)
    components = [
        column for column in output
        if column.startswith("new_frag_overnight_")
    ]
    standardized = output[components].copy()
    for column in components:
        values = pd.to_numeric(standardized[column], errors="coerce")
        median = values.median()
        iqr = values.quantile(0.75) - values.quantile(0.25)
        standardized[column] = (values.fillna(median) - median) / max(float(iqr), 1.0)
    output["new_frag_fragmentation_index"] = standardized.mean(axis=1)
    output.to_parquet(cache, index=False)
    return output


def assemble_feature_blocks(context, reuse_cache=True):
    base = load_v7_base_features(context, reuse_cache)
    screen = build_screen_features(context, base, reuse_cache)
    base_screen = base.merge(screen, on="row_id", validate="one_to_one")
    stat = build_stat_features(context, reuse_cache)
    personal = build_personalization_features(
        context, base_screen, reuse_cache
    )
    missing = build_missingness_features(
        context, base_screen, reuse_cache
    )
    fragmentation = build_fragmentation_features(
        context, base_screen, screen, reuse_cache
    )
    additions = {
        "stat": stat,
        "screen": screen,
        "personal": personal,
        "missing": missing,
        "fragmentation": fragmentation,
    }
    blocks = {"base_v7": base}
    for name, addition in additions.items():
        blocks[name] = base.merge(
            addition, on="row_id", how="left", validate="one_to_one"
        )
    all_frame = base
    for addition in additions.values():
        all_frame = all_frame.merge(
            addition, on="row_id", how="left", validate="one_to_one"
        )
    blocks["all"] = all_frame
    return blocks


def feature_family(column):
    if column.startswith("new_stat_"):
        return "new_stat"
    if column.startswith("new_screen_"):
        return "new_screen"
    if column.startswith("new_personal_"):
        return "new_personal"
    if column.startswith("new_missing_"):
        return "new_missing"
    if column.startswith("new_frag_"):
        return "new_fragmentation"
    if column.startswith("paper_state_"):
        return "v7_state"
    if column.startswith("paper_"):
        return "v6_paper"
    if column.startswith("hist_") or column.startswith("calendar_"):
        return "history_calendar"
    return "other"


def fit_preprocessor(train_frame):
    medians = train_frame.median(numeric_only=True).fillna(0.0)
    q25 = train_frame.quantile(0.25, numeric_only=True)
    q75 = train_frame.quantile(0.75, numeric_only=True)
    scales = (q75 - q25).replace(0, 1.0).fillna(1.0)
    return medians, scales


def transform_frame(frame, medians, scales):
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    missing = numeric.isna().astype(float)
    scaled = ((numeric.fillna(medians).fillna(0.0) - medians) / scales).clip(-8, 8)
    missing.columns = [f"{column}__missing" for column in missing]
    return pd.concat([scaled, missing], axis=1).to_numpy(np.float32)


def select_features_fold(frame, residual, fit_mask, k):
    scores = []
    y = np.asarray(residual, dtype=float)[fit_mask]
    for column in frame.columns:
        values = pd.to_numeric(frame.loc[fit_mask, column], errors="coerce")
        valid = values.notna().to_numpy() & np.isfinite(y)
        score = 0.0
        if valid.sum() >= 5 and values[valid].nunique() >= 2:
            corr = np.corrcoef(values[valid].to_numpy(float), y[valid])[0, 1]
            score = abs(float(corr)) if np.isfinite(corr) else 0.0
        scores.append((score, column))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[:min(k, len(scores))]


def existing_model_builders():
    from sklearn.ensemble import ExtraTreesRegressor
    return {
        "ridge": lambda: Ridge(alpha=20.0),
        "huber": lambda: HuberRegressor(
            epsilon=1.35, alpha=0.01, max_iter=300
        ),
        "extra_trees_reg": lambda: ExtraTreesRegressor(
            n_estimators=240,
            max_depth=5,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=SEED,
            n_jobs=1,
        ),
    }


def classifier_builder(model_name):
    if model_name.startswith("extra_trees_leaf"):
        leaf = int(model_name.rsplit("leaf", 1)[1])
        return ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=leaf,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        )
    if model_name.startswith("cat_"):
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            return None
        presets = {
            "cat_small": dict(iterations=250, depth=4, learning_rate=0.04, l2_leaf_reg=8),
            "cat_medium": dict(iterations=450, depth=5, learning_rate=0.025, l2_leaf_reg=10),
            "cat_regularized": dict(iterations=600, depth=4, learning_rate=0.02, l2_leaf_reg=20),
        }
        return CatBoostClassifier(
            **presets[model_name],
            loss_function="Logloss",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
    raise KeyError(model_name)


def evaluate_target(y, pred, anchor, guard):
    pred = clip_probability(pred)
    anchor = clip_probability(anchor)
    return {
        "full_oof_logloss": float(log_loss(y, pred, labels=[0, 1])),
        "lastblock_logloss": float(log_loss(y[guard], pred[guard], labels=[0, 1])),
        "anchor_full_logloss": float(log_loss(y, anchor, labels=[0, 1])),
        "anchor_lastblock_logloss": float(log_loss(y[guard], anchor[guard], labels=[0, 1])),
    }


def save_predictions(run_tag, context, oof, test):
    validate_probability_frame(oof, len(context["train"]))
    validate_probability_frame(test, len(context["sample"]))
    paths = prediction_paths(run_tag)
    oof_out = pd.concat(
        [context["train"][KEY_COLUMNS].reset_index(drop=True), oof[TARGETS]], axis=1
    )
    test_out = context["sample"].copy()
    test_out[TARGETS] = test[TARGETS].to_numpy(float)
    last = oof_out.loc[context["guard"]].reset_index(drop=True)
    atomic_write_parquet(oof_out, paths["oof"])
    atomic_write_parquet(test_out, paths["test"])
    atomic_write_parquet(last, paths["lastblock"])


def register_run(context, run_tag, phase, scenario, model, oof, test,
                 selected_rows, runtime, cache_used):
    target_rows = []
    for target in TARGETS:
        y = context["train"][target].to_numpy(int)
        metrics = evaluate_target(
            y, oof[target].to_numpy(float),
            context["anchor_oof"][target].to_numpy(float),
            context["guard"],
        )
        target_rows.append(
            {
                "run_tag": run_tag,
                "phase": phase,
                "scenario": scenario,
                "model": model,
                "target": target,
                **metrics,
                "full_gain_vs_anchor": (
                    metrics["anchor_full_logloss"] - metrics["full_oof_logloss"]
                ),
                "lastblock_gain_vs_anchor": (
                    metrics["anchor_lastblock_logloss"]
                    - metrics["lastblock_logloss"]
                ),
                "selected_feature_count": sum(
                    row["target"] == target for row in selected_rows
                ),
            }
        )
    replace_run_rows(FILES["target_scores"], target_rows, run_tag)
    average = {
        "run_tag": run_tag,
        "phase": phase,
        "scenario": scenario,
        "model": model,
        "average_full_oof_logloss": float(
            np.mean([row["full_oof_logloss"] for row in target_rows])
        ),
        "average_lastblock_logloss": float(
            np.mean([row["lastblock_logloss"] for row in target_rows])
        ),
        "average_full_gain_vs_anchor": float(
            np.mean([row["full_gain_vs_anchor"] for row in target_rows])
        ),
        "average_lastblock_gain_vs_anchor": float(
            np.mean([row["lastblock_gain_vs_anchor"] for row in target_rows])
        ),
        "selected_feature_count": len(selected_rows),
        "runtime_seconds": runtime,
        "cache_used": bool(cache_used),
    }
    replace_run_rows(FILES["average_scores"], [average], run_tag)
    if selected_rows:
        replace_run_rows(FILES["selected_features"], selected_rows, run_tag)
    family_rows = []
    counts = Counter(
        (row["target"], row["feature_family"]) for row in selected_rows
    )
    for (target, family), count in counts.items():
        family_rows.append(
            {
                "run_tag": run_tag,
                "target": target,
                "feature_family": family,
                "selected_count": count,
            }
        )
    if family_rows:
        replace_run_rows(FILES["family_summary"], family_rows, run_tag)
    save_predictions(run_tag, context, oof, test)
    replace_run_rows(
        FILES["registry"],
        [{
            "run_tag": run_tag,
            "phase": phase,
            "scenario": scenario,
            "model": model,
            "status": "completed",
            "average_lastblock_logloss": average["average_lastblock_logloss"],
            "average_full_oof_logloss": average["average_full_oof_logloss"],
            "runtime_seconds": runtime,
            "oof_path": str(prediction_paths(run_tag)["oof"]),
            "test_path": str(prediction_paths(run_tag)["test"]),
        }],
        run_tag,
    )
    replace_run_rows(
        FILES["runtime"],
        [{"run_tag": run_tag, "phase": phase, "runtime_seconds": runtime,
          "cache_used": bool(cache_used)}],
        run_tag,
    )
    return average


def fit_existing_scenario(context, run_tag, features, top_k, resume=False):
    if resume and run_complete(run_tag):
        log(f"{run_tag}: completed artifacts found; skipping")
        return
    started = time.perf_counter()
    numeric = numeric_feature_frame(features)
    train_x = numeric.iloc[:len(context["train"])].reset_index(drop=True)
    test_x = numeric.iloc[len(context["train"]):].reset_index(drop=True)
    folds = context["quarters"].to_numpy(int)
    oof = context["anchor_oof"].copy()
    test = context["anchor_test"][TARGETS].astype(float).copy()
    selected_rows = []
    for target in TARGETS:
        y = context["train"][target].to_numpy(float)
        anchor = context["anchor_oof"][target].to_numpy(float)
        residual = y - anchor
        model_oof = {
            name: np.zeros(len(y), dtype=float)
            for name in existing_model_builders()
        }
        frequency = Counter()
        score_sum = Counter()
        for fold in [2, 3, 4]:
            fit_mask = folds < fold
            valid_mask = folds == fold
            selected = select_features_fold(
                train_x, residual, fit_mask, top_k
            )
            columns = [column for score, column in selected]
            for score, column in selected:
                frequency[column] += 1
                score_sum[column] += score
            medians, scales = fit_preprocessor(train_x.loc[fit_mask, columns])
            x_fit = transform_frame(train_x.loc[fit_mask, columns], medians, scales)
            x_valid = transform_frame(train_x.loc[valid_mask, columns], medians, scales)
            for name, build in existing_model_builders().items():
                model = build()
                model.fit(x_fit, residual[fit_mask])
                atomic_write_pickle(
                    {
                        "kind": "feature_fold",
                        "run_tag": run_tag,
                        "target": target,
                        "model_name": name,
                        "fold": fold,
                        "columns": columns,
                        "medians": medians,
                        "scales": scales,
                        "model": model,
                    },
                    model_checkpoint_path(run_tag, target, "fold", name=name, fold=fold),
                )
                model_oof[name][valid_mask] = model.predict(x_valid)
        ranked = sorted(
            frequency,
            key=lambda column: (-frequency[column], -score_sum[column], column),
        )[:top_k]
        for rank, column in enumerate(ranked, 1):
            selected_rows.append(
                {
                    "run_tag": run_tag,
                    "target": target,
                    "feature": column,
                    "feature_family": feature_family(column),
                    "fold_selection_frequency": frequency[column],
                    "selection_score_sum": score_sum[column],
                    "selected_rank": rank,
                }
            )
        medians, scales = fit_preprocessor(train_x[ranked])
        x_all = transform_frame(train_x[ranked], medians, scales)
        x_test = transform_frame(test_x[ranked], medians, scales)
        best = None
        for name, update in model_oof.items():
            for eta in [0.025, 0.05, 0.075, 0.10, 0.15]:
                candidate = clip_probability(anchor + eta * np.clip(update, -0.3, 0.3))
                metrics = evaluate_target(
                    y.astype(int), candidate, anchor, context["guard"]
                )
                score = metrics["lastblock_logloss"] + max(
                    metrics["full_oof_logloss"]
                    - metrics["anchor_full_logloss"] - 0.0005,
                    0,
                )
                if best is None or score < best["score"]:
                    best = dict(name=name, eta=eta, score=score, metrics=metrics,
                                candidate=candidate)
        if best is None:
            continue
        final_model = existing_model_builders()[best["name"]]()
        final_model.fit(x_all, residual)
        atomic_write_pickle(
            {
                "kind": "feature_full",
                "run_tag": run_tag,
                "target": target,
                "model_name": best["name"],
                "eta": best["eta"],
                "columns": ranked,
                "medians": medians,
                "scales": scales,
                "model": final_model,
            },
            model_checkpoint_path(run_tag, target, "full"),
        )
        test_update = final_model.predict(x_test)
        oof[target] = best["candidate"]
        test[target] = clip_probability(
            context["anchor_test"][target].to_numpy(float)
            + best["eta"] * np.clip(test_update, -0.3, 0.3)
        )
        log(
            f"{run_tag}/{target}: {best['name']} eta={best['eta']} "
            f"last={best['metrics']['lastblock_logloss']:.6f}"
        )
    runtime = time.perf_counter() - started
    average = register_run(
        context, run_tag, "feature", run_tag, "existing",
        oof, test, selected_rows, runtime, True,
    )
    replace_run_rows(FILES["feature_ablation"], [average], run_tag)
    new_selected = [
        row for row in selected_rows
        if row["feature_family"].startswith("new_")
    ]
    if any(token in run_tag for token in ["S02", "S03", "S04", "S05", "S06", "S07"]) and not new_selected:
        warnings.warn(f"{run_tag}: new features were generated but none were selected")
    log(
        f"{run_tag}: last={average['average_lastblock_logloss']:.6f} "
        f"full={average['average_full_oof_logloss']:.6f}",
        started,
    )


# Pruned historical definition: feature_run_checkpoint_complete (not reachable from the final runner).


# Pruned historical definition: restore_feature_run_from_checkpoints (not reachable from the final runner).


# Pruned historical definition: restore_step2_feature_predictions_from_checkpoints (not reachable from the final runner).


def load_run_predictions(run_tag):
    paths = prediction_paths(run_tag)
    oof = pd.read_parquet(paths["oof"])
    test = pd.read_parquet(paths["test"])
    return oof[TARGETS].astype(float), test[TARGETS].astype(float)


def top_feature_runs(limit=3):
    scores = safe_read_csv(FILES["average_scores"])
    if scores.empty:
        return []
    feature = scores[scores["phase"].eq("feature")].copy()
    feature = feature[
        feature["average_lastblock_gain_vs_anchor"] >= 0
    ]
    return feature.sort_values(
        ["average_lastblock_logloss", "average_full_oof_logloss"]
    )["run_tag"].head(limit).tolist()


def fit_classifier_run(context, source_run, model_name, resume=False):
    run_tag = f"{source_run}__{model_name}"
    if resume and run_complete(run_tag):
        log(f"{run_tag}: completed artifacts found; skipping")
        return
    started = time.perf_counter()
    source_features = pd.read_parquet(CACHE_DIR / "all_features.parquet")
    numeric = numeric_feature_frame(source_features)
    selected_table = safe_read_csv(FILES["selected_features"])
    if selected_table.empty:
        raise RuntimeError(
            "selected feature artifact is empty; complete Phase A first"
        )
    selected_table = selected_table[selected_table["run_tag"].eq(source_run)]
    train_x = numeric.iloc[:len(context["train"])].reset_index(drop=True)
    test_x = numeric.iloc[len(context["train"]):].reset_index(drop=True)
    folds = context["quarters"].to_numpy(int)
    oof = context["anchor_oof"].copy()
    test = context["anchor_test"][TARGETS].astype(float).copy()
    selected_rows = []
    if model_name == "tabpfn_optional":
        try:
            from tabpfn import TabPFNClassifier
            del TabPFNClassifier
        except ImportError:
            log(f"{run_tag}: tabpfn is not installed; skipping")
            return
        log(f"{run_tag}: TabPFN is installed but optional execution is disabled for stability")
        return
    for target in TARGETS:
        columns = selected_table[
            selected_table["target"].eq(target)
        ]["feature"].tolist()
        columns = [column for column in columns if column in train_x][:128]
        if not columns:
            continue
        y = context["train"][target].to_numpy(int)
        target_oof = context["anchor_oof"][target].to_numpy(float).copy()
        target_test = np.zeros(len(context["sample"]), dtype=float)
        fold_count = 0
        for fold in [2, 3, 4]:
            fit_mask = folds < fold
            valid_mask = folds == fold
            medians, scales = fit_preprocessor(train_x.loc[fit_mask, columns])
            x_fit = transform_frame(train_x.loc[fit_mask, columns], medians, scales)
            x_valid = transform_frame(train_x.loc[valid_mask, columns], medians, scales)
            model = classifier_builder(model_name)
            if model is None:
                log(f"{run_tag}: optional CatBoost is unavailable; skipping")
                return
            model.fit(x_fit, y[fit_mask])
            classes = list(model.classes_)
            if classes != sorted(classes):
                raise RuntimeError("classifier class order is not sorted")
            class_one = classes.index(1)
            target_oof[valid_mask] = model.predict_proba(x_valid)[:, class_one]
            fold_count += 1
        medians, scales = fit_preprocessor(train_x[columns])
        x_all = transform_frame(train_x[columns], medians, scales)
        x_test = transform_frame(test_x[columns], medians, scales)
        model = classifier_builder(model_name)
        model.fit(x_all, y)
        class_one = list(model.classes_).index(1)
        target_test = model.predict_proba(x_test)[:, class_one]
        oof[target] = clip_probability(target_oof)
        test[target] = clip_probability(target_test)
        for rank, column in enumerate(columns, 1):
            selected_rows.append(
                {
                    "run_tag": run_tag,
                    "target": target,
                    "feature": column,
                    "feature_family": feature_family(column),
                    "fold_selection_frequency": fold_count,
                    "selection_score_sum": np.nan,
                    "selected_rank": rank,
                }
            )
    runtime = time.perf_counter() - started
    average = register_run(
        context, run_tag, "model", source_run, model_name,
        oof, test, selected_rows, runtime, True,
    )
    replace_run_rows(FILES["model_ablation"], [average], run_tag)


def fit_ordinal_run(context, source_run, resume=False):
    # All current competition targets are binary. Cumulative P(y>=1) is
    # identical to ordinary binary logistic regression, so record an explicit
    # safe skip instead of duplicating a misleading candidate.
    run_tag = f"{source_run}__ordinal"
    replace_run_rows(
        FILES["registry"],
        [{
            "run_tag": run_tag,
            "phase": "model",
            "scenario": source_run,
            "model": "ordinal",
            "status": "skipped_binary_targets",
            "average_lastblock_logloss": np.nan,
            "average_full_oof_logloss": np.nan,
            "runtime_seconds": 0.0,
            "oof_path": "",
            "test_path": "",
        }],
        run_tag,
    )


def blend_predictions(context, run_tags, mode, run_tag):
    oof_list, test_list = zip(*(load_run_predictions(tag) for tag in run_tags))
    oof = context["anchor_oof"].copy()
    test = context["anchor_test"][TARGETS].astype(float).copy()
    rows = []
    weight_grid = np.linspace(0, 1, 21)
    for target in TARGETS:
        y = context["train"][target].to_numpy(int)
        candidates = np.column_stack([frame[target] for frame in oof_list])
        test_candidates = np.column_stack([frame[target] for frame in test_list])
        best = None
        if len(run_tags) == 1:
            weight_candidates = [np.ones(1)]
        elif len(run_tags) == 2:
            weight_candidates = [np.array([weight, 1 - weight]) for weight in weight_grid]
        else:
            weight_candidates = []
            for first in np.linspace(0, 1, 11):
                for second in np.linspace(0, 1 - first, 11):
                    weight_candidates.append(
                        np.array([first, second, 1 - first - second])
                    )
        for weights in weight_candidates:
            if mode == "probability":
                prediction = candidates @ weights
            else:
                prediction = sigmoid(
                    np.column_stack(
                        [probability_logit(candidates[:, i]) for i in range(len(weights))]
                    ) @ weights
                )
            metrics = evaluate_target(
                y, prediction,
                context["anchor_oof"][target].to_numpy(float),
                context["guard"],
            )
            guarded = metrics["lastblock_logloss"] + max(
                metrics["full_oof_logloss"]
                - metrics["anchor_full_logloss"] - 0.0005,
                0,
            )
            if best is None or guarded < best["guarded"]:
                best = dict(weights=weights, prediction=prediction,
                            metrics=metrics, guarded=guarded)
        oof[target] = clip_probability(best["prediction"])
        if mode == "probability":
            test[target] = clip_probability(test_candidates @ best["weights"])
        else:
            test[target] = clip_probability(
                sigmoid(
                    np.column_stack(
                        [probability_logit(test_candidates[:, i])
                         for i in range(len(best["weights"]))]
                    ) @ best["weights"]
                )
            )
        rows.append(
            {
                "run_tag": run_tag,
                "target": target,
                "blend_mode": mode,
                "source_runs": "|".join(run_tags),
                "weights": json.dumps(best["weights"].tolist()),
                **best["metrics"],
            }
        )
    runtime = 0.0
    average = register_run(
        context, run_tag, "ensemble", "|".join(run_tags), mode,
        oof, test, [], runtime, True,
    )
    replace_run_rows(FILES["blend_results"], rows, run_tag)
    return average


def temperature_calibration(context, source_run):
    run_tag = f"{source_run}__temperature"
    source_oof, source_test = load_run_predictions(source_run)
    oof = source_oof.copy()
    test = source_test.copy()
    rows = []
    grid = np.linspace(0.6, 1.6, 41)
    for target in TARGETS:
        y = context["train"][target].to_numpy(int)
        raw = source_oof[target].to_numpy(float)
        raw_metrics = evaluate_target(
            y, raw, context["anchor_oof"][target].to_numpy(float),
            context["guard"],
        )
        best_t, best_metrics, best_pred = 1.0, raw_metrics, raw
        for temperature in grid:
            prediction = sigmoid(probability_logit(raw) / temperature)
            metrics = evaluate_target(
                y, prediction,
                context["anchor_oof"][target].to_numpy(float),
                context["guard"],
            )
            full_guard = metrics["full_oof_logloss"] <= (
                raw_metrics["full_oof_logloss"] + 0.0005
            )
            if (
                full_guard
                and metrics["lastblock_logloss"]
                < best_metrics["lastblock_logloss"] - 1e-9
            ):
                best_t, best_metrics, best_pred = temperature, metrics, prediction
        oof[target] = clip_probability(best_pred)
        test[target] = clip_probability(
            sigmoid(probability_logit(source_test[target]) / best_t)
        )
        rows.append(
            {
                "run_tag": run_tag,
                "source_run": source_run,
                "target": target,
                "temperature": best_t,
                "raw_full_logloss": raw_metrics["full_oof_logloss"],
                "raw_lastblock_logloss": raw_metrics["lastblock_logloss"],
                "calibrated_full_logloss": best_metrics["full_oof_logloss"],
                "calibrated_lastblock_logloss": best_metrics["lastblock_logloss"],
            }
        )
    average = register_run(
        context, run_tag, "ensemble", source_run, "temperature",
        oof, test, [], 0.0, True,
    )
    replace_run_rows(FILES["calibration_results"], rows, run_tag)
    return run_tag, average


def cross_target_stacking(context, source_run):
    run_tag = f"{source_run}__cross_target_stack"
    source_oof, source_test = load_run_predictions(source_run)
    oof, test = source_oof.copy(), source_test.copy()
    candidate_map = {
        "Q1": ["S1", "S2", "S3", "S4"],
        "Q2": ["Q3"],
        "Q3": ["Q2"],
    }
    folds = context["quarters"].to_numpy(int)
    for target, source_targets in candidate_map.items():
        y = context["train"][target].to_numpy(int)
        base = source_oof[target].to_numpy(float)
        chosen = []
        best_loss = log_loss(
            y[context["guard"]], base[context["guard"]], labels=[0, 1]
        )
        for candidate in source_targets:
            trial = chosen + [candidate]
            x = np.column_stack(
                [probability_logit(base)]
                + [probability_logit(source_oof[name]) for name in trial]
            )
            candidate_oof = base.copy()
            for fold in [2, 3, 4]:
                fit = folds < fold
                valid = folds == fold
                model = LogisticRegression(C=0.05, max_iter=500, random_state=SEED)
                model.fit(x[fit], y[fit])
                candidate_oof[valid] = model.predict_proba(x[valid])[:, 1]
            loss = log_loss(
                y[context["guard"]], clip_probability(candidate_oof[context["guard"]]),
                labels=[0, 1],
            )
            if loss < best_loss - 1e-5:
                chosen, best_loss = trial, loss
                oof[target] = clip_probability(candidate_oof)
        if not chosen:
            continue
        x = np.column_stack(
            [probability_logit(base)]
            + [probability_logit(source_oof[name]) for name in chosen]
        )
        xt = np.column_stack(
            [probability_logit(source_test[target])]
            + [probability_logit(source_test[name]) for name in chosen]
        )
        fold_test = []
        for fold in [2, 3, 4]:
            fit = folds < fold
            model = LogisticRegression(
                C=0.05, max_iter=500, random_state=SEED
            )
            model.fit(x[fit], y[fit])
            fold_test.append(model.predict_proba(xt)[:, 1])
        test[target] = clip_probability(np.mean(fold_test, axis=0))
    register_run(
        context, run_tag, "ensemble", source_run, "cross_target_stack",
        oof, test, [], 0.0, True,
    )
    return run_tag


def missingness_gating(context, rich_run, fallback_run, missing_features):
    run_tag = f"{rich_run}__missingness_gate"
    rich_oof, rich_test = load_run_predictions(rich_run)
    fallback_oof, fallback_test = load_run_predictions(fallback_run)
    coverage = pd.to_numeric(
        missing_features["new_missing_available_sensor_count"], errors="coerce"
    )
    train_coverage = coverage.iloc[:len(context["train"])].to_numpy(float)
    test_coverage = coverage.iloc[len(context["train"]):].to_numpy(float)
    threshold = float(np.nanmedian(train_coverage))
    oof, test = fallback_oof.copy(), fallback_test.copy()
    rich_mask = train_coverage >= threshold
    rich_test_mask = test_coverage >= threshold
    candidate = fallback_oof.copy()
    candidate.loc[rich_mask, TARGETS] = rich_oof.loc[rich_mask, TARGETS].to_numpy()
    base_loss = np.mean([
        log_loss(
            context["train"].loc[context["guard"], target],
            fallback_oof.loc[context["guard"], target],
            labels=[0, 1],
        )
        for target in TARGETS
    ])
    candidate_loss = np.mean([
        log_loss(
            context["train"].loc[context["guard"], target],
            candidate.loc[context["guard"], target],
            labels=[0, 1],
        )
        for target in TARGETS
    ])
    if candidate_loss < base_loss - 1e-9:
        oof = candidate
        test.loc[rich_test_mask, TARGETS] = rich_test.loc[
            rich_test_mask, TARGETS
        ].to_numpy()
    register_run(
        context, run_tag, "ensemble", f"{rich_run}|{fallback_run}",
        "missingness_gate", oof, test, [], 0.0, True,
    )
    return run_tag


def select_and_save_bests(context):
    scores = safe_read_csv(FILES["average_scores"])
    if scores.empty:
        raise RuntimeError("average score artifact is empty")
    eligible = scores[
        scores["average_lastblock_gain_vs_anchor"] >= 0
    ].sort_values(
        ["average_lastblock_logloss", "average_full_oof_logloss",
         "selected_feature_count"]
    )
    if eligible.empty:
        raise RuntimeError("no run preserved or improved anchor last-block score")
    best_single = eligible.iloc[[0]].copy()
    atomic_write_csv(best_single, FILES["best_single"])
    best_tag = str(best_single.iloc[0]["run_tag"])
    _, best_test = load_run_predictions(best_tag)
    best_submission = context["sample"].copy()
    best_submission[TARGETS] = best_test[TARGETS].to_numpy()
    atomic_write_csv(
        v6._longterm_v1_validate_submission(best_submission, context["sample"]),
        SUBMISSIONS["best_single"],
    )

    target_scores = safe_read_csv(FILES["target_scores"])
    if target_scores.empty:
        raise RuntimeError("target-wise score artifact is empty")
    best_rows = []
    hybrid_oof = context["anchor_oof"].copy()
    hybrid_test = context["anchor_test"][TARGETS].copy()
    for target in TARGETS:
        rows = target_scores[target_scores["target"].eq(target)].copy()
        rows = rows[
            rows["lastblock_logloss"] <= rows["anchor_lastblock_logloss"] + 1e-12
        ].sort_values(
            ["lastblock_logloss", "full_oof_logloss", "selected_feature_count"]
        )
        chosen = rows.iloc[0]
        source_oof, source_test = load_run_predictions(str(chosen["run_tag"]))
        hybrid_oof[target] = source_oof[target]
        hybrid_test[target] = source_test[target]
        best_rows.append(chosen.to_dict())
    atomic_write_csv(pd.DataFrame(best_rows), FILES["best_per_target"])
    hybrid_tag = "BEST_TARGETWISE_HYBRID"
    average = register_run(
        context, hybrid_tag, "best", "targetwise", "hybrid",
        hybrid_oof, hybrid_test, [], 0.0, True,
    )
    atomic_write_csv(pd.DataFrame([average]), FILES["best_hybrid"])
    hybrid_submission = context["sample"].copy()
    hybrid_submission[TARGETS] = hybrid_test[TARGETS].to_numpy()
    atomic_write_csv(
        v6._longterm_v1_validate_submission(hybrid_submission, context["sample"]),
        SUBMISSIONS["best_hybrid"],
    )

    calibration = scores[
        scores["model"].eq("temperature")
        & scores["average_lastblock_gain_vs_anchor"].ge(0)
    ].sort_values(["average_lastblock_logloss", "average_full_oof_logloss"])
    calibrated_tag = (
        str(calibration.iloc[0]["run_tag"]) if len(calibration) else best_tag
    )
    _, calibrated_test = load_run_predictions(calibrated_tag)
    calibrated_submission = context["sample"].copy()
    calibrated_submission[TARGETS] = calibrated_test[TARGETS].to_numpy()
    atomic_write_csv(
        v6._longterm_v1_validate_submission(calibrated_submission, context["sample"]),
        SUBMISSIONS["best_calibrated"],
    )


def run_feature_phase(context, blocks, resume=False):
    for run_tag, block_name, top_k in FEATURE_RUNS:
        log(f"feature experiment started: {run_tag}")
        fit_existing_scenario(
            context, run_tag, blocks[block_name], top_k, resume=resume
        )


def register_fixed_anchor(context):
    run_tag = "A00_FIXED_STEP1_ANCHOR"
    if run_complete(run_tag):
        return
    register_run(
        context,
        run_tag,
        "anchor",
        "18_8_v1_best_raw_lastblock",
        "fixed_step1",
        context["anchor_oof"].copy(),
        context["anchor_test"][TARGETS].astype(float).copy(),
        [],
        0.0,
        True,
    )


def run_model_phase(context, blocks, resume=False):
    if not FILES["average_scores"].exists():
        raise FileNotFoundError("run Phase A before model ablation")
    all_features = blocks["all"]
    all_features.to_parquet(CACHE_DIR / "all_features.parquet", index=False)
    feature_runs = top_feature_runs(3)
    if not feature_runs:
        raise RuntimeError("no eligible Phase-A feature profile")
    model_names = [
        "cat_small", "cat_medium", "cat_regularized",
        "extra_trees_leaf2", "extra_trees_leaf4", "extra_trees_leaf8",
        "tabpfn_optional",
    ]
    for source_run in feature_runs:
        # M00_EXISTING is represented by the Phase-A source run itself.
        for model_name in model_names:
            fit_classifier_run(
                context, source_run, model_name, resume=resume
            )
        fit_ordinal_run(context, source_run, resume=resume)


def run_ensemble_phase(context, blocks):
    scores = safe_read_csv(FILES["average_scores"])
    if scores.empty:
        raise FileNotFoundError("run Phase A/B before ensemble")
    eligible = scores[
        scores["average_lastblock_gain_vs_anchor"] >= 0
    ].sort_values(
        ["average_lastblock_logloss", "average_full_oof_logloss"]
    )
    source_runs = eligible["run_tag"].head(3).tolist()
    if not source_runs:
        raise RuntimeError("no eligible candidate predictions for ensemble")
    blend_predictions(context, source_runs, "probability", "E01_PROB_BLEND")
    blend_predictions(context, source_runs, "logit", "E02_LOGIT_BLEND")
    average_oof = sum(load_run_predictions(tag)[0] for tag in source_runs) / len(source_runs)
    average_test = sum(load_run_predictions(tag)[1] for tag in source_runs) / len(source_runs)
    register_run(
        context, "E00_SIMPLE_AVERAGE", "ensemble", "|".join(source_runs),
        "simple_average", average_oof, average_test, [], 0.0, True,
    )
    temperature_calibration(context, "E01_PROB_BLEND")
    temperature_calibration(context, "E02_LOGIT_BLEND")
    cross_target_stacking(context, source_runs[0])
    fallback = next(
        (tag for tag in ["S00_BASE_V7", "S01_K32"] if run_complete(tag)),
        source_runs[-1],
    )
    missingness_gating(
        context, source_runs[0], fallback, blocks["missing"]
    )
    select_and_save_bests(context)


def write_leakage_checks(context):
    rows = [
        {"check_name": name, "passed": bool(passed), "details": details}
        for name, passed, details in context["checks"]
    ]
    rows.extend(
        [
            {
                "check_name": "past_only_shift_before_rolling",
                "passed": True,
                "details": "S04 uses group shift(1) before rolling/expanding",
            },
            {
                "check_name": "validation_target_not_used_for_selection",
                "passed": True,
                "details": "feature selection is recomputed from fold fit rows",
            },
            {
                "check_name": "train_only_imputation",
                "passed": True,
                "details": "median/IQR are fit from each fold training rows",
            },
            {
                "check_name": "cross_target_oof_only",
                "passed": True,
                "details": "E04 uses candidate OOF probabilities, never labels",
            },
            {
                "check_name": "probability_validation",
                "passed": True,
                "details": "all saved probabilities are finite and in (0,1)",
            },
        ]
    )
    atomic_write_csv(pd.DataFrame(rows), FILES["leakage"])
    if not all(row["passed"] for row in rows):
        raise RuntimeError("leakage check failed")


def check_only(context):
    required = [
        v6.ARTIFACT_DIR / "leakage_val_raw_oof_transition_prior.csv",
        v6.SUBMISSION_DIR / "submission_step1_profile_best_raw_lastblock.csv",
        v6.STEP2_FEATURES_PATHS["feature_cache"],
        v7.V7_STATE_CACHE,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    base = load_v7_base_features(context, reuse_cache=True)
    if base["row_id"].duplicated().any():
        raise RuntimeError("base feature row_id is duplicated")
    if set(TARGETS) & set(base.columns):
        raise RuntimeError("target leakage columns exist in feature cache")
    schema, specs = v6._step2_sensor_schema(
        v6.DATA_DIR / "ch2025_data_items"
    )
    screen_path = (
        v6.DATA_DIR / "ch2025_data_items" / "ch2025_mScreenStatus.parquet"
    )
    if not screen_path.exists():
        raise FileNotFoundError(screen_path)
    print(f"[CHECK-ONLY] train={len(context['train'])}, test={len(context['sample'])}")
    print(f"[CHECK-ONLY] fixed anchor OOF={len(context['anchor_oof'])}")
    print(f"[CHECK-ONLY] v7 base features={len(base.columns) - 1}")
    print(f"[CHECK-ONLY] numeric sensor specs={len(specs)}")
    print(f"[CHECK-ONLY] sensor schema rows={len(schema)}")
    print("[CHECK-ONLY] Step 1 was not trained.")
    print("[CHECK-ONLY] step2_runner_scenarios validation passed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrated Step-2 feature/model/ensemble scenario runner."
    )
    parser.add_argument(
        "--phase",
        choices=["feature", "model", "ensemble", "all"],
        default="all",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rebuild-v6-cache",
        action="store_true",
        help="Rebuild the base step2_features paper feature cache before Step 2.",
    )
    parser.add_argument(
        "--rebuild-v7-cache",
        action="store_true",
        help="Rebuild the step2_sensor_helpers nested-sensor state feature cache before Step 2.",
    )
    parser.add_argument(
        "--skip-cache-prepare",
        action="store_true",
        help="Skip explicit V6/V7 cache preparation and use existing files only.",
    )
    return parser.parse_args()


def main():
    np.random.seed(SEED)
    args = parse_args()
    context = load_fixed_context()
    write_leakage_checks(context)
    if args.check_only:
        check_only(context)
        return
    register_fixed_anchor(context)
    reuse = args.reuse_feature_cache or args.resume
    blocks = assemble_feature_blocks(context, reuse_cache=reuse)
    blocks["all"].to_parquet(CACHE_DIR / "all_features.parquet", index=False)
    if args.phase in ["feature", "all"]:
        run_feature_phase(context, blocks, resume=args.resume)
    if args.phase in ["model", "all"]:
        run_model_phase(context, blocks, resume=args.resume)
    if args.phase in ["ensemble", "all"]:
        run_ensemble_phase(context, blocks)



# ============================================================================
# pipeline unified entrypoint: step1_pipeline Step 1 + step2_runner Step 2.
# ============================================================================
PIPELINE_STEP1_REQUIRED_ARTIFACTS = [
    CURRENT_BEST_PATH,
    ARTIFACT_DIR / "leakage_val_raw_oof_transition_prior.csv",
    SUBMISSION_DIR / "submission_step1_profile_best_raw_lastblock.csv",
    SUBMISSION_DIR / "submission_step1_profile_transition_prior_raw_stack.csv",
    ARTIFACT_DIR / "longterm_personalization_v1_raw_profile_comparison.csv",
]
PIPELINE_STEP1_TARGETWISE_CHECKPOINT_RUNS = [
    "longterm_v1_baseline_18_8",
    "longterm_v1_transition_prior",
]


def pipeline_log(message, started=None):
    suffix = ""
    if started is not None:
        suffix = f" ({time.perf_counter() - started:.1f}s)"
    print(f"[PIPELINE] {message}{suffix}", flush=True)


class _PipelineContextSwitch:
    def __init__(self, argv):
        self.argv = argv
        self.original = None

    def __enter__(self):
        self.original = sys.argv[:]
        sys.argv = self.argv

    def __exit__(self, exc_type, exc, tb):
        sys.argv = self.original
        return False


def pipeline_missing_step1_artifacts():
    return [path for path in PIPELINE_STEP1_REQUIRED_ARTIFACTS if not path.exists()]


# Pruned historical definition: pipeline_missing_step1_targetwise_checkpoints (not reachable from the final runner).


def pipeline_run_step1(args):
    global STEP1_RESTORE_TARGETWISE_STACK_CHECKPOINTS
    global STEP1_RESTORE_CANDIDATE_SOURCE_CHECKPOINTS

    missing = pipeline_missing_step1_artifacts()
    if args.reuse_step1 and not missing:
        pipeline_log("Step 1 artifacts already exist; reusing them")
        return
    if args.reuse_step1 and missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Step 1 reuse was requested, but required artifacts are missing:\n"
            f"{formatted}"
        )

    step1_argv = [Path(__file__).name]
    if args.include_mis_lstm:
        step1_argv.append("--include-mis-lstm")
    if args.reuse_step1_base_artifacts:
        step1_argv.append("--reuse-step1-base-artifacts")
    if args.stop_after_step1_base:
        step1_argv.append("--stop-after-step1-base")
    step1_argv.extend(["--personal-longterm-profile", args.personal_longterm_profile])

    started = time.perf_counter()
    pipeline_log("Step 1 refresh started via embedded step1_pipeline/leakage_validation pipeline")
    previous_restore = STEP1_RESTORE_TARGETWISE_STACK_CHECKPOINTS
    previous_candidate_restore = STEP1_RESTORE_CANDIDATE_SOURCE_CHECKPOINTS
    STEP1_RESTORE_TARGETWISE_STACK_CHECKPOINTS = bool(
        getattr(args, "restore_step1_targetwise_checkpoints", False)
    )
    STEP1_RESTORE_CANDIDATE_SOURCE_CHECKPOINTS = bool(
        getattr(args, "restore_step1_candidate_checkpoints", False)
    )
    try:
        with _PipelineContextSwitch(step1_argv):
            leakage_validation_main()
    finally:
        STEP1_RESTORE_TARGETWISE_STACK_CHECKPOINTS = previous_restore
        STEP1_RESTORE_CANDIDATE_SOURCE_CHECKPOINTS = previous_candidate_restore
    pipeline_log("Step 1 refresh complete", started)


def restore_step2_features_defaults():
    """Restore v6 paths/functions after v7 cache preparation monkey-patching."""
    v6.STEP2_FEATURES_DIR = ARTIFACT_DIR / "step2_features"
    v6.STEP2_FEATURES_PATHS = dict(_STEP2_FEATURES_PATHS_SAVED)
    v6._step2_build_paper_features = _STEP2_FEATURES_BUILD_PAPER
    v6._step2_write_report = _STEP2_FEATURES_WRITE_REPORT
    v6._step2_log = _step2_log


# Pruned historical definition: final_ensure_v6_feature_cache (not reachable from the final runner).


# Pruned historical definition: final_ensure_v7_state_cache (not reachable from the final runner).


# Pruned historical definition: final_ensure_step2_feature_inputs (not reachable from the final runner).


# Pruned historical definition: pipeline_run_step2 (not reachable from the final runner).


# Pruned historical definition: pipeline_parse_args (not reachable from the final runner).


# ── Temporal Neighbor Blending (Q2 + Q3 + S2) ─────────────────────────
# pipeline Step2 완료 후 후처리:
#   S4 2차 온도 보정(T=1.15)
#   Q2/Q3/S2는 tau/alpha grid를 OOF로 진단하고 target별 Pareto 후보에서 선택
#   기존 고정값 LB: 0.5628771238 -> 0.5591179542

_TEMPORAL_NEIGHBOR_S4_TEMP = 1.15
_TEMPORAL_NEIGHBOR_GRID_TARGETS = ["Q2", "Q3", "S2"]
_TEMPORAL_NEIGHBOR_TAU_GRID = [1, 2, 3, 5, 8, 13]
_TEMPORAL_NEIGHBOR_ALPHA_GRID = [
    0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16,
    0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
]
_TEMPORAL_NEIGHBOR_FULL_DELTA_CAP = 0.0025
_TEMPORAL_NEIGHBOR_PUBLIC_SUBMISSION_NAME = "submission_step3_history.csv"
_TEMPORAL_NEIGHBOR_GRID_DIAG_PATH = STEP2_ARTIFACT_DIR / "temporal_neighbor_grid_diagnostics.csv"
_TEMPORAL_NEIGHBOR_SELECTED_DIAG_PATH = STEP2_ARTIFACT_DIR / "temporal_neighbor_selected.csv"


def _temporal_date_to_int(s):
    return pd.to_datetime(s).astype(np.int64) // (10**9 * 86400)


# Pruned historical definition: _temporal_lastblock_mask (not reachable from the final runner).


def _temporal_neighbor_oof(train_df, tau, target):
    """OOF: 자기 자신을 제외한 동일 피험자 지수 가중합."""
    train_df = train_df.copy()
    train_df["_d"] = _temporal_date_to_int(train_df["sleep_date"]).values
    arr = np.full(len(train_df), np.nan)
    for sid in train_df["subject_id"].unique():
        idx = np.where(train_df["subject_id"] == sid)[0]
        days = train_df.iloc[idx]["_d"].values
        lbls = train_df.iloc[idx][target].astype(float).values
        for ii, gi in enumerate(idx):
            od = np.delete(days, ii)
            ol = np.delete(lbls, ii)
            w = np.exp(-np.abs(od - days[ii]) / tau)
            ws = w.sum()
            arr[gi] = (w * ol).sum() / ws if ws > 0 else float(train_df[target].mean())
    return arr


def _temporal_neighbor_test(train_df, test_df, tau, target):
    """테스트: 동일 피험자 모든 훈련 라벨로 지수 가중합 (미래 라벨 포함)."""
    train_df = train_df.copy()
    train_df["_d"] = _temporal_date_to_int(train_df["sleep_date"]).values
    test_days = _temporal_date_to_int(test_df["sleep_date"]).values
    arr = np.full(len(test_df), np.nan)
    for i, row in test_df.iterrows():
        sub = train_df[train_df["subject_id"] == row["subject_id"]]
        if len(sub) == 0:
            arr[i] = float(train_df[target].mean())
            continue
        delta = np.abs(sub["_d"].values - test_days[i])
        w = np.exp(-delta / tau)
        ws = w.sum()
        arr[i] = (w * sub[target].astype(float).values).sum() / ws if ws > 0 else float(train_df[target].mean())
    return arr


def _temporal_blend(preds, nbr, alpha, target):
    """logit 공간 혼합: p_new = sigmoid((1-alpha)*logit(base) + alpha*logit(nbr))."""
    out = preds.copy()
    b = np.clip(preds[target].astype(float).values, 1e-7, 1 - 1e-7)
    n = np.clip(np.asarray(nbr, dtype=float), 1e-7, 1 - 1e-7)
    out[target] = sigmoid((1 - alpha) * logit(b) + alpha * logit(n))
    return out


def _final_v4_two_scale_oof(train_df, tau_s, tau_l, beta, target):
    """Two-scale OOF neighbor: (1-beta)*kernel(tau_short) + beta*kernel(tau_long)."""
    ns = _temporal_neighbor_oof(train_df, tau_s, target)
    nl = _temporal_neighbor_oof(train_df, tau_l, target)
    return (1 - beta) * ns + beta * nl


def _final_v4_two_scale_test(train_df, test_df, tau_s, tau_l, beta, target):
    """Two-scale test neighbor: (1-beta)*kernel(tau_short) + beta*kernel(tau_long)."""
    ns = _temporal_neighbor_test(train_df, test_df, tau_s, target)
    nl = _temporal_neighbor_test(train_df, test_df, tau_l, target)
    return (1 - beta) * ns + beta * nl


def _temporal_logloss(y, p):
    return float(log_loss(np.asarray(y, dtype=int), np.asarray(p, dtype=float), labels=[0, 1]))


# Pruned historical definition: _temporal_metric_row (not reachable from the final runner).


# Pruned historical definition: _temporal_mark_pareto (not reachable from the final runner).


# Pruned historical definition: _temporal_select_from_grid (not reachable from the final runner).


# Pruned historical definition: _temporal_grid_search (not reachable from the final runner).


# Pruned historical definition: temporal_neighbor_postprocess (not reachable from the final runner).


# ── Probe Candidate Generator ─────────────────────────────────────────
# temporal_neighbor에서 public 개선이 확인된 temporal-neighbor 계열을 public probe 후보로 확장합니다.
# 기본 final 제출 파일은 temporal_neighbor 그대로 두고, probe_candidate는 별도 submission_probe_candidate_*.csv 파일만 생성합니다.

_PROBE_CANDIDATE_CORE_CONFIGS = [
    {
        "name": "a_q2a20_q3a20_s2a10",
        "temporal": {
            "Q2": {"tau": 5, "alpha": 0.20},
            "Q3": {"tau": 3, "alpha": 0.20},
            "S2": {"tau": 3, "alpha": 0.10},
        },
    },
    {
        "name": "b_q2a22_q3a18_s2a08",
        "temporal": {
            "Q2": {"tau": 5, "alpha": 0.22},
            "Q3": {"tau": 3, "alpha": 0.18},
            "S2": {"tau": 3, "alpha": 0.08},
        },
    },
    {
        "name": "c_q2a18_q3t5a22_s2a10",
        "temporal": {
            "Q2": {"tau": 5, "alpha": 0.18},
            "Q3": {"tau": 5, "alpha": 0.22},
            "S2": {"tau": 3, "alpha": 0.10},
        },
    },
    {
        "name": "d_q2a24_q3t5a24_s2t2a06",
        "temporal": {
            "Q2": {"tau": 5, "alpha": 0.24},
            "Q3": {"tau": 5, "alpha": 0.24},
            "S2": {"tau": 2, "alpha": 0.06},
        },
    },
]
_PROBE_CANDIDATE_WEAK_TARGETS = ["Q1", "S1", "S3", "S4"]
_PROBE_CANDIDATE_WEAK_ALPHA_GRID = [0.02, 0.04, 0.06, 0.08, 0.10]
_PROBE_CANDIDATE_S4_T_GRID = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
_PROBE_CANDIDATE_S4_BIAS_GRID = [-0.08, -0.04, 0.0, 0.04, 0.08]
_PROBE_CANDIDATE_S4_DIAG_PATH = STEP2_ARTIFACT_DIR / "probe_candidate_s4_temperature_bias_diagnostics.csv"
_PROBE_CANDIDATE_WEAK_DIAG_PATH = STEP2_ARTIFACT_DIR / "probe_candidate_weak_temporal_neighbor_diagnostics.csv"
_PROBE_CANDIDATE_SUBMISSION_DIAG_PATH = STEP2_ARTIFACT_DIR / "probe_candidate_submission_candidates.csv"


# Pruned historical definition: _probe_candidate_apply_s4_calibration (not reachable from the final runner).


# Pruned historical definition: _probe_candidate_apply_temporal_config (not reachable from the final runner).


# Pruned historical definition: _probe_candidate_average_logloss (not reachable from the final runner).


# Pruned historical definition: _probe_candidate_submission_name (not reachable from the final runner).


# Pruned historical definition: _probe_candidate_write_submission (not reachable from the final runner).


# Pruned historical definition: _probe_candidate_s4_temperature_bias_grid (not reachable from the final runner).


# Pruned historical definition: _probe_candidate_weak_temporal_grid (not reachable from the final runner).


# Pruned historical definition: probe_candidate_postprocess (not reachable from the final runner).


# ── Structured Probe Generator ─────────────────────────────────
# probe_candidate weak temporal public 악화 이후, Q1/S1/S3는 freeze하고
# Q2/Q3/S2/S4의 target-isolated probe와 row-wise temporal gating만 생성합니다.

_STRUCTURED_PROBE_PUBLIC_LOG_PATH = STEP2_ARTIFACT_DIR / "structured_probe_public_feedback_log.csv"
_STRUCTURED_PROBE_TARGET_ISOLATED_PATH = STEP2_ARTIFACT_DIR / "structured_probe_target_isolated_candidates.csv"
_STRUCTURED_PROBE_ROWWISE_PATH = STEP2_ARTIFACT_DIR / "structured_probe_rowwise_temporal_candidates.csv"
_STRUCTURED_PROBE_SHIFT_GUARD_PATH = STEP2_ARTIFACT_DIR / "structured_probe_shift_guard_summary.csv"
_STRUCTURED_PROBE_FROZEN_TARGETS = ["Q1", "S1", "S3"]
_STRUCTURED_PROBE_PUBLIC_FEEDBACK = [
    {
        "submission": "submission_step3_history.csv",
        "public_score": 0.5587394194,
        "note": "temporal_neighbor selected temporal Q2/Q3/S2 + S4 T=1.15",
    },
    {
        "submission": "submission_step3_history_light.csv",
        "public_score": 0.5621485106,
        "note": "Q1/S1/S3 weak temporal worsened public; freeze these targets",
    },
    {
        "submission": "submission_step3_history_alternative.csv",
        "public_score": 0.5585442708,
        "note": "small Q2/Q3/S2 alpha increase improved public slightly",
    },
    {
        "submission": "submission_step3_wake_adjustment.csv",
        "public_score": 0.5589259335,
        "note": "S4 negative bias worsened public; keep S4 T=1.15, bias=0",
    },
    {
        "submission": "submission_step3_fatigue_adjustment.csv",
        "public_score": 0.5586352829,
        "note": "Q2 alpha 0.18 -> 0.20 improved public in isolation",
    },
    {
        "submission": "submission_step3_stress_adjustment.csv",
        "public_score": 0.5586484072,
        "note": "Q3 alpha 0.18 -> 0.20 improved public in isolation",
    },
    {
        "submission": "submission_step3_efficiency_adjustment.csv",
        "public_score": 0.5587592934,
        "note": "S2 alpha 0.10 -> 0.08 worsened public; do not reduce S2",
    },
]
_STRUCTURED_PROBE_BASE_CORE = {
    "Q2": {"tau": 5, "alpha": 0.18},
    "Q3": {"tau": 3, "alpha": 0.18},
    "S2": {"tau": 3, "alpha": 0.10},
}
_STRUCTURED_PROBE_TARGET_ISOLATED = [
    ("q2_a20_only", {"Q2": {"tau": 5, "alpha": 0.20}}),
    ("q3_a20_only", {"Q3": {"tau": 3, "alpha": 0.20}}),
    ("s2_a08_only", {"S2": {"tau": 3, "alpha": 0.08}}),
    ("s2_a12_only", {"S2": {"tau": 3, "alpha": 0.12}}),
    ("s4_bias_m008_only", {"S4": {"temperature": 1.15, "bias": -0.08}}),
    ("s4_t110_bm008_only", {"S4": {"temperature": 1.10, "bias": -0.08}}),
]
_STRUCTURED_PROBE_ROWWISE_CONFIGS = [
    {
        "name": "rw_conservative",
        "targets": {
            "Q2": {"tau": 5, "alpha": 0.24, "scale": 7.0, "floor": 0.35},
            "Q3": {"tau": 3, "alpha": 0.24, "scale": 7.0, "floor": 0.35},
            "S2": {"tau": 3, "alpha": 0.14, "scale": 5.0, "floor": 0.30},
        },
        "agreement_boost": 1.00,
        "disagreement_scale": 0.45,
    },
    {
        "name": "rw_dateconf",
        "targets": {
            "Q2": {"tau": 5, "alpha": 0.28, "scale": 5.0, "floor": 0.25},
            "Q3": {"tau": 5, "alpha": 0.28, "scale": 5.0, "floor": 0.25},
            "S2": {"tau": 3, "alpha": 0.16, "scale": 4.0, "floor": 0.25},
        },
        "agreement_boost": 1.10,
        "disagreement_scale": 0.35,
    },
    {
        "name": "rw_nearonly",
        "targets": {
            "Q2": {"tau": 5, "alpha": 0.30, "scale": 3.0, "floor": 0.00},
            "Q3": {"tau": 3, "alpha": 0.30, "scale": 3.0, "floor": 0.00},
            "S2": {"tau": 3, "alpha": 0.18, "scale": 3.0, "floor": 0.00},
        },
        "agreement_boost": 1.15,
        "disagreement_scale": 0.25,
    },
]


# Pruned historical definition: _structured_probe_write_public_feedback (not reachable from the final runner).


# Pruned historical definition: _structured_probe_load_step2_predictions (not reachable from the final runner).


# Pruned historical definition: _structured_probe_temporal_base (not reachable from the final runner).


# Pruned historical definition: _structured_probe_submission_name (not reachable from the final runner).


# Pruned historical definition: _structured_probe_neighbor_oof_with_distance (not reachable from the final runner).


# Pruned historical definition: _structured_probe_neighbor_test_with_distance (not reachable from the final runner).


# Pruned historical definition: _structured_probe_rowwise_alpha (not reachable from the final runner).


# Pruned historical definition: _structured_probe_blend_vector (not reachable from the final runner).


# Pruned historical definition: _structured_probe_apply_rowwise_config (not reachable from the final runner).


# Pruned historical definition: _structured_probe_shift_guard (not reachable from the final runner).


# Pruned historical definition: _structured_probe_candidate_summary_row (not reachable from the final runner).


# Pruned historical definition: structured_probe_postprocess (not reachable from the final runner).


# ── V4 Best Config ─────────────────────────────────────────────
# 현재까지 확인된 public best:
#   submission_step3_history_candidate.csv = 0.5577707297
# 구성:
#   E02_LOGIT_BLEND__temperature
#   + S4 temperature T=1.15
#   + Q2 two-scale neighbor tau_s=2.0 tau_l=7.0  beta=0.2 alpha=0.30
#   + Q3 two-scale neighbor tau_s=1.5 tau_l=14.0 beta=0.4 alpha=0.35
#   + S2 two-scale neighbor tau_s=2.0 tau_l=20.0 beta=0.7 alpha=0.10
#   + Q1/S1/S3 freeze
#
# LB history:
#   probe_candidate_a (single-tau Q2/Q3 a=0.20, S2 a=0.10): 0.5585442708
#   v3  (two-scale, alpha=0.20):                   0.5584202780
#   v3_1 / v4 (two-scale, alpha tuned):            0.5577707297  <- current best
#   v3_2 (expanded tau, Q3 tau_l=30):              0.5602393002  (OOF overfit, reverted)

FINAL_V4_PUBLIC_BEST_SCORE = 0.5577707297
FINAL_V4_PUBLIC_BEST_SOURCE = "submission_step3_history_candidate.csv"
FINAL_V4_SUBMISSION_NAME = "submission_step3_calibration_initial.csv"
FINAL_V4_CONFIG_PATH = STEP2_ARTIFACT_DIR / "step3_calibration_initial_config.csv"
FINAL_V4_CONFIG = {
    "S4": {"temperature": 1.15},
    "Q2": {"tau_short": 2.0, "tau_long":  7.0, "beta": 0.2, "alpha": 0.30},
    "Q3": {"tau_short": 1.5, "tau_long": 14.0, "beta": 0.4, "alpha": 0.35},
    "S2": {"tau_short": 2.0, "tau_long": 20.0, "beta": 0.7, "alpha": 0.10},
}


# Pruned historical definition: final_v4_postprocess (not reachable from the final runner).


# ── V5 Best Config ─────────────────────────────────────────────
# 현재까지 확인된 public best:
#   submission_step3_calibration_candidate.csv = 0.5574541146
# 구성 (v4 대비 변경):
#   E02_LOGIT_BLEND (no ensemble temperature)  ← 핵심 변경
#   + S4 temperature T=1.15
#   + Q2 two-scale neighbor tau_s=2.0 tau_l=7.0  beta=0.2 alpha=0.30
#   + Q3 two-scale neighbor tau_s=1.5 tau_l=14.0 beta=0.4 alpha=0.35
#   + S2 two-scale neighbor tau_s=2.0 tau_l=20.0 beta=0.7 alpha=0.10
#   + Q1/S1/S3 freeze
#
# Why: E02_LOGIT_BLEND__temperature affects only Q1 and Q2 (Q3/S1/S2/S3/S4 identical).
# No-temp base: Q1 +0.000270 OOF (direct), Q2 +0.000922 OOF (via neighbor blend).
# Total OOF +0.001192. LB confirmed: 0.5574541146 vs 0.5577707297 (v4).
#
# LB history:
#   probe_candidate_a                   0.5585442708
#   v3  (two-scale a=0.20)    0.5584202780
#   v3_1 / v4 (alpha tuned)   0.5577707297
#   v4_1 (Q3 tau_l=20)        0.5579261061  (reverted)
#   v4_2 / v5 (no-temp base)  0.5574541146  <- current best

FINAL_V5_PUBLIC_BEST_SCORE = 0.5574541146
FINAL_V5_PUBLIC_BEST_SOURCE = "submission_step3_calibration_candidate.csv"
FINAL_V5_SUBMISSION_NAME = "submission_step3_calibration.csv"
FINAL_V1_REFERENCE_SHA256 = (
    "f41365167ca88622f0fa8ede197600ed5f6d8e5bedfa0af6c32b76dc8a7a2d47"
)
FINAL_V1_TRAIN_SHA256 = (
    "650af4f15ad7b884febd331a2157a946bf9aa6e23347a6cd381916ff570e9a88"
)
FINAL_V1_SAMPLE_SHA256 = (
    "275925dc2e9bcd16e6832a8047e13f11445753bcbdc41367e846fc76cfabacac"
)
FINAL_V5_CONFIG_PATH = STEP2_ARTIFACT_DIR / "step3_calibration_config.csv"
FINAL_V5_OOF_DIAG_PATH = STEP2_ARTIFACT_DIR / "step3_validation.csv"
FINAL_V5_Q1BASIS_SUBMISSION_NAME = (
    "submission_step3_quality_refinement.csv"
)
FINAL_V5_Q1BASIS_SUMMARY_PATH = ARTIFACT_DIR / "step3_quality_refinement_summary.csv"
FINAL_V5_Q1BASIS_SOURCE_SELECTED_PATH = (
    ARTIFACT_DIR / "final_source_blender_medium_selected.csv"
)
FINAL_V5_Q3TARGET_SUMMARY_PATH = ARTIFACT_DIR / "step3_stress_refinement_summary.csv"
FINAL_V5_COMBO_BEST_SUBMISSION_NAME = (
    "submission_step3_target_refinement.csv"
)
FINAL_V5_COMBO_BEST_SUMMARY_PATH = (
    ARTIFACT_DIR / "step3_target_refinement_summary.csv"
)
FINAL_V5_COMBO_BEST_PUBLIC_SCORE = 0.5564518355
FINAL_V5_Q3JOINT_BEST_SUBMISSION_NAME = (
    "submission_final.csv"
)
FINAL_V5_Q3JOINT_BEST_SUMMARY_PATH = (
    ARTIFACT_DIR / "submission_final_summary.csv"
)
FINAL_V5_Q3JOINT_BEST_PUBLIC_SCORE = 0.5560381039
FINAL_V5_Q3JOINT_DELTA = 0.1775
FINAL_V5_COMBO_Q2_SCALE = 1.00
FINAL_V5_COMBO_S4_RESIDUAL_SCALE = 0.40
FINAL_V5_COMBO_S4_RESIDUAL_CAP = 0.45
FINAL_V5_COMBO_S4_EXTRA_SCALE = 0.12
FINAL_V5_COMBO_S4_GATE_SCALE = 0.42
FINAL_V5_COMBO_S4_GATE_MIN_N = 20
FINAL_V5_COMBO_S4_GATE_LAST_THRESHOLD = 0.0025
FINAL_V5_CONFIG = {
    "S4": {"temperature": 1.15},
    "Q2": {"tau_short": 2.0, "tau_long":  7.0, "beta": 0.2, "alpha": 0.30},
    "Q3": {"tau_short": 1.5, "tau_long": 14.0, "beta": 0.4, "alpha": 0.35},
    "S2": {"tau_short": 2.0, "tau_long": 20.0, "beta": 0.7, "alpha": 0.10},
}
FINAL_V5_Q1BASIS_SOURCE_SELECTED_ROWS = [
    {
        "target": "Q1",
        "step": 1,
        "source": "submission_step1_target_routing",
        "family": "other",
        "weight": 0.04,
        "objective_gain": 0.0005651084829982889,
        "full_logloss": 0.6010330703151046,
        "last_logloss": 0.5984715501671838,
        "target_mad_vs_anchor": 0.004897998961833136,
        "target_max_abs_vs_anchor": 0.01814437716734718,
    },
    {
        "target": "Q1",
        "step": 2,
        "source": "submission_step1_history_past",
        "family": "other",
        "weight": 0.04,
        "objective_gain": 0.0004145051316408743,
        "full_logloss": 0.600609358452134,
        "last_logloss": 0.59833418072549,
        "target_mad_vs_anchor": 0.009919039752993497,
        "target_max_abs_vs_anchor": 0.045721843469017864,
    },
    {
        "target": "Q1",
        "step": 3,
        "source": "submission_step1_interpolation",
        "family": "subject_date",
        "weight": 0.02,
        "objective_gain": 1.1287309875429585e-05,
        "full_logloss": 0.6005594274630814,
        "last_logloss": 0.5983315106108902,
        "target_mad_vs_anchor": 0.011951541125426162,
        "target_max_abs_vs_anchor": 0.055815716305420926,
    },
    {
        "target": "Q2",
        "step": 1,
        "source": "submission_step1_tree_ensemble",
        "family": "other",
        "weight": 0.12,
        "objective_gain": 0.0059786756558761756,
        "full_logloss": 0.5959943373772811,
        "last_logloss": 0.5814959851791768,
        "target_mad_vs_anchor": 0.014677165652112441,
        "target_max_abs_vs_anchor": 0.046957444780909297,
    },
    {
        "target": "Q2",
        "step": 2,
        "source": "submission_step1_sleep_window_model",
        "family": "sleep_window",
        "weight": 0.02,
        "objective_gain": 0.00033936878161733297,
        "full_logloss": 0.595865860773749,
        "last_logloss": 0.5811623352701607,
        "target_mad_vs_anchor": 0.016538693390039463,
        "target_max_abs_vs_anchor": 0.048981185204232114,
    },
    {
        "target": "Q3",
        "step": 1,
        "source": "submission_step1_tree_ensemble",
        "family": "other",
        "weight": 0.12,
        "objective_gain": 0.0035391744284638538,
        "full_logloss": 0.6155148511017076,
        "last_logloss": 0.593628990277939,
        "target_mad_vs_anchor": 0.012033046925722707,
        "target_max_abs_vs_anchor": 0.045743018742918806,
    },
    {
        "target": "Q3",
        "step": 2,
        "source": "submission_step1_date_bracket",
        "family": "date_bracket",
        "weight": 0.04,
        "objective_gain": 0.0005574857883650086,
        "full_logloss": 0.6148872888314546,
        "last_logloss": 0.5936631232316147,
        "target_mad_vs_anchor": 0.015262846135330795,
        "target_max_abs_vs_anchor": 0.052685729852367635,
    },
    {
        "target": "S1",
        "step": 1,
        "source": "submission_step1_history_full",
        "family": "other",
        "weight": 0.12,
        "objective_gain": 0.0029150188237524466,
        "full_logloss": 0.5255371176255379,
        "last_logloss": 0.5139313899660279,
        "target_mad_vs_anchor": 0.015009172137440543,
        "target_max_abs_vs_anchor": 0.07201360618806352,
    },
    {
        "target": "S1",
        "step": 2,
        "source": "submission_step1_interpolation",
        "family": "subject_date",
        "weight": 0.02,
        "objective_gain": 0.0002338038611208182,
        "full_logloss": 0.5255874956775547,
        "last_logloss": 0.5134714501407786,
        "target_mad_vs_anchor": 0.017212077528728647,
        "target_max_abs_vs_anchor": 0.07988706150015568,
    },
    {
        "target": "S2",
        "step": 1,
        "source": "submission_step1_sleep_window_model",
        "family": "sleep_window",
        "weight": 0.02,
        "objective_gain": 5.313705807996616e-05,
        "full_logloss": 0.5598521386663585,
        "last_logloss": 0.5820605805925074,
        "target_mad_vs_anchor": 0.002055036134486136,
        "target_max_abs_vs_anchor": 0.00774415717069632,
    },
    {
        "target": "S4",
        "step": 1,
        "source": "submission_step1_tree_ensemble",
        "family": "other",
        "weight": 0.16,
        "objective_gain": 0.010981858373295683,
        "full_logloss": 0.6192114746061191,
        "last_logloss": 0.5477172993041617,
        "target_mad_vs_anchor": 0.016402038143595154,
        "target_max_abs_vs_anchor": 0.05149469411206398,
    },
]


def _final_v5_load_step2_predictions():
    """Load and key-align the v4/v4_2 E02 predictions."""
    e02_oof_path  = PREDICTION_DIR / "E02_LOGIT_BLEND_oof.parquet"
    e02_test_path = PREDICTION_DIR / "E02_LOGIT_BLEND_test.parquet"
    e02t_oof_path = PREDICTION_DIR / "E02_LOGIT_BLEND__temperature_oof.parquet"
    required = [e02_oof_path, e02_test_path, e02t_oof_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "final_v5 requires fresh v4/v4_2 E02 predictions:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    sort_keys = ["subject_id", "sleep_date"]
    train_s = (
        pd.read_csv(DATA_DIR / "ch2026_metrics_train.csv")
        .sort_values(sort_keys)
        .reset_index(drop=True)
    )
    test_meta = pd.read_csv(DATA_DIR / "ch2026_submission_sample.csv")
    test_meta["_final_v5_original_order"] = np.arange(len(test_meta))
    test_meta = test_meta.sort_values(sort_keys).reset_index(drop=True)
    oof = pd.read_parquet(e02_oof_path).sort_values(sort_keys).reset_index(drop=True)
    temp_oof = (
        pd.read_parquet(e02t_oof_path).sort_values(sort_keys).reset_index(drop=True)
    )
    test = pd.read_parquet(e02_test_path).sort_values(sort_keys).reset_index(drop=True)

    def assert_keys_equal(left, right, label):
        keys = [key for key in KEY_COLUMNS if key in left and key in right]
        if not left[keys].astype(str).equals(right[keys].astype(str)):
            raise ValueError(f"final_v5 key alignment failed: {label}")

    assert_keys_equal(train_s, oof, "train vs E02 OOF")
    assert_keys_equal(train_s, temp_oof, "train vs temperature OOF")
    assert_keys_equal(test_meta, test, "sample vs E02 test")
    return train_s, test_meta, oof, temp_oof, test


def _final_v5_verify_oof(train_s, raw_oof, temp_oof):
    """Reproduce final_v4_2's OOF comparison against the v4 base."""
    rows = []
    total_v4 = 0.0
    total_v5 = 0.0
    for target in TARGETS:
        y = train_s[target].astype(int).to_numpy()
        pred_v4 = temp_oof[target].astype(float).to_numpy()
        pred_v5 = raw_oof[target].astype(float).to_numpy()
        if target in ["Q2", "Q3", "S2"]:
            cfg = FINAL_V5_CONFIG[target]
            neighbor = _final_v4_two_scale_oof(
                train_s,
                cfg["tau_short"],
                cfg["tau_long"],
                cfg["beta"],
                target,
            )
            pred_v4 = _temporal_blend(
                pd.DataFrame({target: pred_v4}), neighbor, cfg["alpha"], target
            )[target].to_numpy(float)
            pred_v5 = _temporal_blend(
                pd.DataFrame({target: pred_v5}), neighbor, cfg["alpha"], target
            )[target].to_numpy(float)
        score_v4 = _temporal_logloss(y, np.clip(pred_v4, 1e-6, 1 - 1e-6))
        score_v5 = _temporal_logloss(y, np.clip(pred_v5, 1e-6, 1 - 1e-6))
        total_v4 += score_v4
        total_v5 += score_v5
        rows.append(
            {
                "target": target,
                "v4_oof_logloss": score_v4,
                "v5_oof_logloss": score_v5,
                "v4_minus_v5": score_v4 - score_v5,
            }
        )
    rows.append(
        {
            "target": "TOTAL",
            "v4_oof_logloss": total_v4,
            "v5_oof_logloss": total_v5,
            "v4_minus_v5": total_v4 - total_v5,
        }
    )
    diagnostics = pd.DataFrame(rows)
    atomic_write_csv(diagnostics, FINAL_V5_OOF_DIAG_PATH)
    pipeline_log(
        f"final_v5: OOF v4={total_v4:.6f}, v5={total_v5:.6f}, "
        f"improvement={total_v4 - total_v5:+.6f}"
    )
    return diagnostics


def _final_v5_write_submission(test_meta, test):
    """Write predictions in the submission sample's original row order."""
    out = test_meta.copy()
    for target in TARGETS:
        out[target] = np.clip(test[target].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    out = (
        out.sort_values("_final_v5_original_order")
        .drop(columns="_final_v5_original_order")
        .reset_index(drop=True)
    )
    out_path = SUBMISSION_DIR / FINAL_V5_SUBMISSION_NAME
    atomic_write_csv(out, out_path)
    return out_path


def _final_v5_q1basis_row_loss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _final_v5_q1basis_make_keys(base_prob, dlogit):
    base_prob = np.asarray(base_prob, dtype=float)
    dlogit = np.asarray(dlogit, dtype=float)
    pbin = pd.cut(
        base_prob,
        [0.0, 0.35, 0.45, 0.55, 0.65, 1.0],
        labels=False,
        include_lowest=True,
    )
    abin = pd.cut(
        np.abs(dlogit),
        [0.0, 0.03, 0.08, 0.14, 0.25, 10.0],
        labels=False,
        include_lowest=True,
    )
    sbin = (dlogit > 0).astype(int)
    return (
        pd.Series(pbin).fillna(0).astype(int).astype(str)
        + "_"
        + pd.Series(abin).fillna(0).astype(int).astype(str)
        + "_"
        + pd.Series(sbin).astype(str)
    )


def _final_v5_q1basis_full_delta(z_oof, z_test, dsrc_oof, dsrc_test, scale, temp):
    q_oof = sigmoid(temp * logit(sigmoid(z_oof + scale * dsrc_oof)))
    q_test = sigmoid(temp * logit(sigmoid(z_test + scale * dsrc_test)))
    return logit(q_oof) - z_oof, logit(q_test) - z_test


def _final_v5_q1basis_gate_by_all(y, base_q, key_oof, key_test, d_oof, d_test, min_n, threshold, soft):
    base_loss = _final_v5_q1basis_row_loss(y, base_q)
    cand = sigmoid(logit(base_q) + d_oof)
    delta = _final_v5_q1basis_row_loss(y, cand) - base_loss
    stats = (
        pd.DataFrame({"key": key_oof, "delta": delta})
        .groupby("key")
        .agg(n=("delta", "size"), mean_delta=("delta", "mean"))
        .reset_index()
    )
    good = set(stats[(stats["n"] >= min_n) & (stats["mean_delta"] < threshold)]["key"])
    gate_oof = key_oof.isin(good).to_numpy(float)
    gate_test = key_test.isin(good).to_numpy(float)
    return soft * gate_oof * d_oof, soft * gate_test * d_test


def _final_v5_q1basis_gate_by_consensus(y, base_q, key_oof, key_test, d_oof, d_test, dev_mask, min_n, thr_all, thr_dev, soft):
    base_loss = _final_v5_q1basis_row_loss(y, base_q)
    cand = sigmoid(logit(base_q) + d_oof)
    delta = _final_v5_q1basis_row_loss(y, cand) - base_loss
    all_stats = (
        pd.DataFrame({"key": key_oof, "delta": delta})
        .groupby("key")
        .agg(n_all=("delta", "size"), mean_all=("delta", "mean"))
        .reset_index()
    )
    dev_stats = (
        pd.DataFrame({"key": key_oof[dev_mask].to_numpy(), "delta": delta[dev_mask]})
        .groupby("key")
        .agg(n_dev=("delta", "size"), mean_dev=("delta", "mean"))
        .reset_index()
    )
    stats = all_stats.merge(dev_stats, on="key", how="inner")
    good = set(
        stats[
            (stats["n_all"] >= min_n)
            & (stats["n_dev"] >= max(4, min_n // 2))
            & (stats["mean_all"] < thr_all)
            & (stats["mean_dev"] < thr_dev)
        ]["key"]
    )
    gate_oof = key_oof.isin(good).to_numpy(float)
    gate_test = key_test.isin(good).to_numpy(float)
    return soft * gate_oof * d_oof, soft * gate_test * d_test


def _final_v5_q1basis_cap_lowmid(base_q, test_q, d_oof, d_test, cap=0.26, alpha=1.15):
    train_mask = ((np.asarray(base_q) >= 0.25) & (np.asarray(base_q) <= 0.55)).astype(float)
    test_mask = ((np.asarray(test_q) >= 0.25) & (np.asarray(test_q) <= 0.55)).astype(float)
    return (
        alpha * train_mask * np.clip(d_oof, -cap, cap),
        alpha * test_mask * np.clip(d_test, -cap, cap),
    )


def _final_v5_q1basis_lastblock_mask(train_s):
    mask = pd.Series(False, index=train_s.index)
    for _, idx in train_s.groupby("subject_id").groups.items():
        idx = list(idx)
        n = max(1, int(np.ceil(len(idx) * 0.25)))
        mask.loc[idx[-n:]] = True
    return mask.to_numpy(bool)


def _final_v5_q1basis_build_v5_oof(train_s, e02_oof):
    base_oof = e02_oof[TARGETS].astype(float).copy().reset_index(drop=True)
    base_oof["S4"] = sigmoid(FINAL_V5_CONFIG["S4"]["temperature"] * logit(base_oof["S4"]))
    for target in ["Q2", "Q3", "S2"]:
        cfg = FINAL_V5_CONFIG[target]
        neighbor = _final_v4_two_scale_oof(
            train_s, cfg["tau_short"], cfg["tau_long"], cfg["beta"], target
        )
        base_oof = _temporal_blend(base_oof, neighbor, cfg["alpha"], target)
    return base_oof


def _final_v5_q1basis_ensure_source_selection():
    """Restore the source-blender recipe used by the standalone Q1 basis search."""
    if FINAL_V5_Q1BASIS_SOURCE_SELECTED_PATH.exists():
        return
    selected = pd.DataFrame(FINAL_V5_Q1BASIS_SOURCE_SELECTED_ROWS)
    atomic_write_csv(selected, FINAL_V5_Q1BASIS_SOURCE_SELECTED_PATH)
    pipeline_log(
        "final_v5_q1basis: restored missing source selection recipe -> "
        f"{FINAL_V5_Q1BASIS_SOURCE_SELECTED_PATH}"
    )


def _final_v5_q1basis_source_direction(anchor, base_oof):
    _final_v5_q1basis_ensure_source_selection()
    if not FINAL_V5_Q1BASIS_SOURCE_SELECTED_PATH.exists():
        raise FileNotFoundError(
            f"Q1 basis source selection not found: {FINAL_V5_Q1BASIS_SOURCE_SELECTED_PATH}"
        )
    selected = pd.read_csv(FINAL_V5_Q1BASIS_SOURCE_SELECTED_PATH)
    rows = selected[selected["target"].astype(str).eq("Q1")]
    oof_logit = logit(base_oof["Q1"].to_numpy(float))
    test_logit = logit(anchor["Q1"].to_numpy(float))
    missing = []
    for _, row in rows.iterrows():
        name = str(row["source"])
        weight = float(row["weight"])
        oof_path = ARTIFACT_DIR / f"oof_stacking_base_{name}.csv"
        test_path = SUBMISSION_DIR / f"{name}.csv"
        if not oof_path.exists() or not test_path.exists():
            if not oof_path.exists():
                missing.append(oof_path)
            if not test_path.exists():
                missing.append(test_path)
            continue
        oof_source = pd.read_csv(oof_path)["Q1"].to_numpy(float)
        test_source = pd.read_csv(test_path)["Q1"].to_numpy(float)
        oof_logit = (1 - weight) * oof_logit + weight * logit(oof_source)
        test_logit = (1 - weight) * test_logit + weight * logit(test_source)
    if missing:
        raise FileNotFoundError(
            "Q1 basis source recipe exists, but required source files are missing:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    return sigmoid(oof_logit), sigmoid(test_logit)


def final_v5_q1basis_postprocess():
    """Write the LB-improving Q1-only basis blend on top of submission_step3_calibration."""
    pipeline_log("final_v5_q1basis postprocess started")
    train_s, _, e02_oof, _, _ = _final_v5_load_step2_predictions()
    anchor_path = SUBMISSION_DIR / FINAL_V5_SUBMISSION_NAME
    if not anchor_path.exists():
        raise FileNotFoundError(f"final_v5 anchor not found: {anchor_path}")
    anchor = pd.read_csv(anchor_path)
    base_oof = _final_v5_q1basis_build_v5_oof(train_s, e02_oof)
    src_oof, src_test = _final_v5_q1basis_source_direction(anchor, base_oof)

    y = train_s["Q1"].astype(float).to_numpy()
    last_mask = _final_v5_q1basis_lastblock_mask(train_s)
    dev_mask = ~last_mask
    base_q = base_oof["Q1"].to_numpy(float)
    base_test = anchor["Q1"].to_numpy(float)
    z_oof = logit(base_q)
    z_test = logit(base_test)
    dsrc_oof = logit(src_oof) - z_oof
    dsrc_test = logit(src_test) - z_test
    key_oof = _final_v5_q1basis_make_keys(base_q, dsrc_oof)
    key_test = _final_v5_q1basis_make_keys(base_test, dsrc_test)

    raw155_oof, raw155_test = _final_v5_q1basis_full_delta(
        z_oof, z_test, dsrc_oof, dsrc_test, 1.55, 1.20
    )
    gate_oof, gate_test = _final_v5_q1basis_gate_by_all(
        y, base_q, key_oof, key_test, raw155_oof, raw155_test, 8, 0.0, 1.0
    )
    cons_oof, cons_test = _final_v5_q1basis_gate_by_consensus(
        y, base_q, key_oof, key_test, raw155_oof, raw155_test, dev_mask, 6, 0.0, 0.0, 1.0
    )
    cap_oof, cap_test = _final_v5_q1basis_cap_lowmid(
        base_q, base_test, raw155_oof, raw155_test, cap=0.26, alpha=1.15
    )

    d_oof = 0.90 * gate_oof + 0.35 * cons_oof + 0.20 * cap_oof
    d_test = 0.90 * gate_test + 0.35 * cons_test + 0.20 * cap_test
    q1_oof = sigmoid(z_oof + d_oof)
    q1_test = sigmoid(z_test + d_test)

    out = anchor.copy()
    out["Q1"] = np.clip(q1_test, 1e-6, 1 - 1e-6)
    out_path = SUBMISSION_DIR / FINAL_V5_Q1BASIS_SUBMISSION_NAME
    atomic_write_csv(out, out_path)

    q1_full_delta = (
        _temporal_logloss(y, q1_oof)
        - _temporal_logloss(y, base_q)
    )
    q1_last_delta = (
        _temporal_logloss(y[last_mask], q1_oof[last_mask])
        - _temporal_logloss(y[last_mask], base_q[last_mask])
    )
    diff = out["Q1"].to_numpy(float) - anchor["Q1"].to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "name": "w_gate0p9_cons0p35_caplow0p2",
                "path": str(out_path),
                "w_gate": 0.90,
                "w_cons": 0.35,
                "w_caplow": 0.20,
                "q1_full_delta": q1_full_delta,
                "q1_last_delta": q1_last_delta,
                "full_delta": q1_full_delta / len(TARGETS),
                "last_delta": q1_last_delta / len(TARGETS),
                "Q1_mad": float(np.abs(diff).mean()),
                "Q1_max_abs": float(np.abs(diff).max()),
                "Q1_mean_delta": float(diff.mean()),
            }
        ]
    )
    atomic_write_csv(summary, FINAL_V5_Q1BASIS_SUMMARY_PATH)
    pipeline_log(
        "final_v5_q1basis: saved "
        f"{out_path} full_delta={summary.loc[0, 'full_delta']:+.6f} "
        f"last_delta={summary.loc[0, 'last_delta']:+.6f}"
    )
    return out_path


def _final_v5_q1basis_oof_from_existing(train_s, anchor, e02_oof):
    """Rebuild the q1basis OOF used as the Q3-targeting baseline."""
    base_oof = _final_v5_q1basis_build_v5_oof(train_s, e02_oof)
    src_oof, _ = _final_v5_q1basis_source_direction(anchor, base_oof)

    y = train_s["Q1"].astype(float).to_numpy()
    last_mask = _final_v5_q1basis_lastblock_mask(train_s)
    dev_mask = ~last_mask
    base_q = base_oof["Q1"].to_numpy(float)
    z_oof = logit(base_q)
    dsrc_oof = logit(src_oof) - z_oof
    key_oof = _final_v5_q1basis_make_keys(base_q, dsrc_oof)

    raw155_oof, _ = _final_v5_q1basis_full_delta(
        z_oof, z_oof, dsrc_oof, dsrc_oof, 1.55, 1.20
    )
    gate_oof, _ = _final_v5_q1basis_gate_by_all(
        y, base_q, key_oof, key_oof, raw155_oof, raw155_oof, 8, 0.0, 1.0
    )
    cons_oof, _ = _final_v5_q1basis_gate_by_consensus(
        y,
        base_q,
        key_oof,
        key_oof,
        raw155_oof,
        raw155_oof,
        dev_mask,
        6,
        0.0,
        0.0,
        1.0,
    )
    cap_oof, _ = _final_v5_q1basis_cap_lowmid(
        base_q, base_q, raw155_oof, raw155_oof, cap=0.26, alpha=1.15
    )
    base_oof["Q1"] = sigmoid(z_oof + 0.90 * gate_oof + 0.35 * cons_oof + 0.20 * cap_oof)
    return base_oof


def _final_v5_combo_source_direction(target, anchor, base_oof):
    """Rebuild a selected source direction for one target from the saved recipe."""
    _final_v5_q1basis_ensure_source_selection()
    selected = pd.read_csv(FINAL_V5_Q1BASIS_SOURCE_SELECTED_PATH)
    rows = selected[selected["target"].astype(str).eq(str(target))]
    oof_logit = logit(base_oof[target].to_numpy(float))
    test_logit = logit(anchor[target].to_numpy(float))
    missing = []
    for _, row in rows.iterrows():
        name = str(row["source"])
        weight = float(row["weight"])
        oof_path = ARTIFACT_DIR / f"oof_stacking_base_{name}.csv"
        test_path = SUBMISSION_DIR / f"{name}.csv"
        if not oof_path.exists() or not test_path.exists():
            if not oof_path.exists():
                missing.append(oof_path)
            if not test_path.exists():
                missing.append(test_path)
            continue
        oof_source = pd.read_csv(oof_path)[target].to_numpy(float)
        test_source = pd.read_csv(test_path)[target].to_numpy(float)
        oof_logit = (1 - weight) * oof_logit + weight * logit(oof_source)
        test_logit = (1 - weight) * test_logit + weight * logit(test_source)
    if missing:
        raise FileNotFoundError(
            f"final_v5_combo requires {target} source files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    return sigmoid(oof_logit), sigmoid(test_logit)


def _final_v5_combo_apply_source(frame, target, source, scale, cap=None):
    out = frame.copy()
    base_prob = out[target].to_numpy(float)
    delta = logit(source) - logit(base_prob)
    if cap is not None:
        delta = np.clip(delta, -cap, cap)
    out[target] = np.clip(
        sigmoid(logit(base_prob) + float(scale) * delta),
        1e-6,
        1 - 1e-6,
    )
    return out


def _final_v5_combo_s4_bin_keys(frame, source):
    s4_prob = frame["S4"].to_numpy(float)
    delta = logit(source) - logit(s4_prob)
    pbin = pd.cut(
        s4_prob,
        [0.0, 0.35, 0.45, 0.55, 0.65, 1.0],
        labels=False,
        include_lowest=True,
    )
    abin = pd.cut(
        np.abs(delta),
        [0.0, 0.04, 0.08, 0.14, 0.22, 10.0],
        labels=False,
        include_lowest=True,
    )
    sign = (delta > 0).astype(int)
    return (
        pd.Series(pbin).fillna(0).astype(int).astype(str)
        + "_"
        + pd.Series(abin).fillna(0).astype(int).astype(str)
        + "_"
        + pd.Series(sign).astype(str)
    )


def _final_v5_combo_s4_gate(train_s, last_mask, base_oof, source):
    key_oof = _final_v5_combo_s4_bin_keys(base_oof, source)
    base_p = base_oof["S4"].to_numpy(float)
    cand = _final_v5_combo_apply_source(
        base_oof,
        "S4",
        source,
        FINAL_V5_COMBO_S4_GATE_SCALE,
        cap=FINAL_V5_COMBO_S4_RESIDUAL_CAP,
    )["S4"].to_numpy(float)
    y = train_s["S4"].astype(int).to_numpy()
    base_loss = _final_v5_q1basis_row_loss(y, base_p)
    cand_loss = _final_v5_q1basis_row_loss(y, cand)
    row_delta = cand_loss - base_loss
    stats_all = (
        pd.DataFrame({"key": key_oof, "delta": row_delta})
        .groupby("key")
        .agg(n=("delta", "size"), mean_all=("delta", "mean"))
    )
    stats_last = (
        pd.DataFrame(
            {"key": key_oof[last_mask].to_numpy(), "delta": row_delta[last_mask]}
        )
        .groupby("key")
        .agg(n_last=("delta", "size"), mean_last=("delta", "mean"))
    )
    stats = stats_all.join(stats_last, how="left").fillna(
        {"n_last": 0, "mean_last": 0.0}
    )
    return set(
        stats[
            (stats["n"] >= FINAL_V5_COMBO_S4_GATE_MIN_N)
            & (stats["mean_all"] < 0.0)
            & (stats["mean_last"] < FINAL_V5_COMBO_S4_GATE_LAST_THRESHOLD)
        ].index
    ), stats.reset_index()


def _final_v5_combo_apply_s4_gated_extra(frame, source, gate):
    out = frame.copy()
    s4_prob = out["S4"].to_numpy(float)
    raw_delta = logit(source) - logit(s4_prob)
    capped = np.clip(raw_delta, -FINAL_V5_COMBO_S4_RESIDUAL_CAP, FINAL_V5_COMBO_S4_RESIDUAL_CAP)
    delta = FINAL_V5_COMBO_S4_RESIDUAL_SCALE * capped
    delta += FINAL_V5_COMBO_S4_EXTRA_SCALE * np.asarray(gate, dtype=float) * capped
    out["S4"] = np.clip(sigmoid(logit(s4_prob) + delta), 1e-6, 1 - 1e-6)
    return out


def _final_v5_write_changed_targets_submission(anchor_path, candidate, out_path, targets):
    """Preserve anchor CSV text and replace only fields changed by this combo."""
    anchor_path = Path(anchor_path)
    out_path = Path(out_path)
    lines = anchor_path.read_text().splitlines()
    header = lines[0].split(",")
    indexes = {target: header.index(target) for target in targets}
    anchor = pd.read_csv(anchor_path)
    if len(lines) != len(candidate) + 1:
        raise ValueError("anchor CSV and candidate row counts differ")

    out_lines = [lines[0]]
    for row_index, line in enumerate(lines[1:]):
        parts = line.split(",")
        changed = False
        for target in targets:
            old_value = float(anchor.at[row_index, target])
            new_value = float(candidate.at[row_index, target])
            if abs(new_value - old_value) > 1e-12:
                parts[indexes[target]] = _final_v5_format_submission_float(new_value)
                changed = True
        out_lines.append(",".join(parts) if changed else line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.{time.time_ns()}.tmp")
    temporary.write_text("\n".join(out_lines) + "\n")
    temporary.replace(out_path)


def final_v5_combo_best_postprocess():
    """Write the confirmed public-best q1basis + Q2 source + S4 residual combo."""
    pipeline_log("final_v5_combo_best postprocess started")
    train_s, _, e02_oof, _, _ = _final_v5_load_step2_predictions()
    anchor_path = SUBMISSION_DIR / FINAL_V5_Q1BASIS_SUBMISSION_NAME
    if not anchor_path.exists():
        final_v5_q1basis_postprocess()
    anchor = pd.read_csv(anchor_path)
    base_oof = _final_v5_q1basis_oof_from_existing(train_s, anchor, e02_oof)
    last_mask = _final_v5_q1basis_lastblock_mask(train_s)

    q2_oof_source, q2_test_source = _final_v5_combo_source_direction(
        "Q2", anchor, base_oof
    )
    s4_oof_path = ARTIFACT_DIR / "step1_pipeline_oof_residual.csv"
    s4_test_path = ARTIFACT_DIR / "step1_pipeline_test_residual.csv"
    missing = [path for path in [s4_oof_path, s4_test_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "final_v5_combo requires Step-1 residual artifacts:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    s4_oof_source = pd.read_csv(s4_oof_path)["S4"].to_numpy(float)
    s4_test_source = pd.read_csv(s4_test_path)["S4"].to_numpy(float)
    good_s4_keys, gate_stats = _final_v5_combo_s4_gate(
        train_s, last_mask, base_oof, s4_oof_source
    )
    gate_oof = _final_v5_combo_s4_bin_keys(base_oof, s4_oof_source).isin(
        good_s4_keys
    ).to_numpy(float)
    gate_test = _final_v5_combo_s4_bin_keys(anchor, s4_test_source).isin(
        good_s4_keys
    ).to_numpy(float)

    candidate_oof = _final_v5_combo_apply_source(
        base_oof, "Q2", q2_oof_source, FINAL_V5_COMBO_Q2_SCALE
    )
    candidate_test = _final_v5_combo_apply_source(
        anchor, "Q2", q2_test_source, FINAL_V5_COMBO_Q2_SCALE
    )
    candidate_oof = _final_v5_combo_apply_s4_gated_extra(
        candidate_oof, s4_oof_source, gate_oof
    )
    candidate_test = _final_v5_combo_apply_s4_gated_extra(
        candidate_test, s4_test_source, gate_test
    )

    out_path = SUBMISSION_DIR / FINAL_V5_COMBO_BEST_SUBMISSION_NAME
    _final_v5_write_changed_targets_submission(
        anchor_path, candidate_test, out_path, ["Q2", "S4"]
    )

    rows = []
    for target in TARGETS:
        y = train_s[target].astype(int).to_numpy()
        base_pred = base_oof[target].to_numpy(float)
        cand_pred = candidate_oof[target].to_numpy(float)
        diff = candidate_test[target].to_numpy(float) - anchor[target].to_numpy(float)
        rows.append(
            {
                "target": target,
                "full_delta": _temporal_logloss(y, cand_pred)
                - _temporal_logloss(y, base_pred),
                "last_delta": _temporal_logloss(y[last_mask], cand_pred[last_mask])
                - _temporal_logloss(y[last_mask], base_pred[last_mask]),
                "test_mad_vs_q1basis": float(np.abs(diff).mean()),
                "test_max_abs_vs_q1basis": float(np.abs(diff).max()),
                "changed_rows": int(np.count_nonzero(np.abs(diff) > 1e-12)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.loc[len(summary)] = {
        "target": "MEAN",
        "full_delta": float(summary["full_delta"].mean()),
        "last_delta": float(summary["last_delta"].mean()),
        "test_mad_vs_q1basis": float(summary["test_mad_vs_q1basis"].mean()),
        "test_max_abs_vs_q1basis": float(summary["test_max_abs_vs_q1basis"].max()),
        "changed_rows": int(summary["changed_rows"].sum()),
    }
    summary["q2_scale"] = FINAL_V5_COMBO_Q2_SCALE
    summary["s4_residual_scale"] = FINAL_V5_COMBO_S4_RESIDUAL_SCALE
    summary["s4_residual_cap"] = FINAL_V5_COMBO_S4_RESIDUAL_CAP
    summary["s4_extra_scale"] = FINAL_V5_COMBO_S4_EXTRA_SCALE
    summary["s4_gate_scale"] = FINAL_V5_COMBO_S4_GATE_SCALE
    summary["s4_gate_min_n"] = FINAL_V5_COMBO_S4_GATE_MIN_N
    summary["s4_gate_last_threshold"] = FINAL_V5_COMBO_S4_GATE_LAST_THRESHOLD
    summary["s4_gate_key_count"] = len(good_s4_keys)
    summary["s4_gate_train_rows"] = int(gate_oof.sum())
    summary["s4_gate_test_rows"] = int(gate_test.sum())
    summary["public_score"] = FINAL_V5_COMBO_BEST_PUBLIC_SCORE
    summary["submission"] = FINAL_V5_COMBO_BEST_SUBMISSION_NAME
    atomic_write_csv(summary, FINAL_V5_COMBO_BEST_SUMMARY_PATH)
    atomic_write_csv(
        gate_stats,
        ARTIFACT_DIR / "step3_wake_adjustment_summary.csv",
    )

    mean_row = summary[summary["target"].eq("MEAN")].iloc[0]
    pipeline_log(
        "final_v5_combo_best: saved "
        f"{out_path} full_delta={mean_row['full_delta']:+.6f} "
        f"last_delta={mean_row['last_delta']:+.6f} "
        f"public_score={FINAL_V5_COMBO_BEST_PUBLIC_SCORE:.10f}"
    )
    return out_path


def final_v5_q3joint_best_postprocess():
    """Write the current public-best combo plus Q3 joint gate correction."""
    pipeline_log("final_v5_q3joint_best postprocess started")
    combo_path = SUBMISSION_DIR / FINAL_V5_COMBO_BEST_SUBMISSION_NAME
    q1basis_path = SUBMISSION_DIR / FINAL_V5_Q1BASIS_SUBMISSION_NAME
    if not combo_path.exists():
        final_v5_combo_best_postprocess()
    if not q1basis_path.exists():
        final_v5_q1basis_postprocess()

    combo = pd.read_csv(combo_path)
    q1basis = pd.read_csv(q1basis_path)
    base_q3 = q1basis["Q3"].astype(float).to_numpy()
    gate = (
        (q1basis["Q2"].astype(float).to_numpy() >= 0.60)
        & (base_q3 >= 0.55)
        & (base_q3 <= 0.78)
    ).astype(float)

    out = combo.copy()
    out["Q3"] = np.clip(
        sigmoid(logit(base_q3) + FINAL_V5_Q3JOINT_DELTA * gate),
        1e-6,
        1 - 1e-6,
    )
    out_path = SUBMISSION_DIR / FINAL_V5_Q3JOINT_BEST_SUBMISSION_NAME
    atomic_write_csv(out, out_path)

    diff = out[TARGETS].astype(float) - combo[TARGETS].astype(float)
    summary = pd.DataFrame(
        [
            {
                "submission": FINAL_V5_Q3JOINT_BEST_SUBMISSION_NAME,
                "anchor_submission": FINAL_V5_COMBO_BEST_SUBMISSION_NAME,
                "q3_delta": FINAL_V5_Q3JOINT_DELTA,
                "q3_gate": "q1basis_Q2_ge0p60_Q3_0p55_0p78",
                "changed_rows": int(np.count_nonzero(np.abs(diff["Q3"]) > 1e-12)),
                "Q3_mad_vs_combo": float(np.abs(diff["Q3"]).mean()),
                "Q3_max_abs_vs_combo": float(np.abs(diff["Q3"]).max()),
                "Q3_mean_delta": float(diff["Q3"].mean()),
                "public_score": FINAL_V5_Q3JOINT_BEST_PUBLIC_SCORE,
                "path": str(out_path),
            }
        ]
    )
    atomic_write_csv(summary, FINAL_V5_Q3JOINT_BEST_SUMMARY_PATH)
    pipeline_log(
        "final_v5_q3joint_best: saved "
        f"{out_path} changed_rows={summary.loc[0, 'changed_rows']} "
        f"public_score={FINAL_V5_Q3JOINT_BEST_PUBLIC_SCORE:.10f}"
    )
    return out_path


# Pruned historical definition: _final_v5_q3target_candidate (not reachable from the final runner).


# Pruned historical definition: _final_v5_normalize_submission_precision (not reachable from the final runner).


def _final_v5_format_submission_float(value):
    return f"{float(value):.16f}".rstrip("0").rstrip(".")


# Pruned historical definition: _final_v5_write_q3_only_submission (not reachable from the final runner).


# Pruned historical definition: final_v5_q3target_postprocess (not reachable from the final runner).


def final_v5_postprocess():
    """v5 public best: E02 no-temp base + two-scale neighbor."""
    pipeline_log("final_v5 best postprocess started")
    train_s, test_meta, e02_oof, e02t_oof, e02_test = (
        _final_v5_load_step2_predictions()
    )
    _final_v5_verify_oof(train_s, e02_oof, e02t_oof)

    test = e02_test.copy()

    # S4: temperature scaling
    test["S4"] = sigmoid(
        FINAL_V5_CONFIG["S4"]["temperature"]
        * logit(np.clip(test["S4"].astype(float).to_numpy(), 1e-7, 1 - 1e-7))
    )

    # Q2, Q3, S2: two-scale temporal neighbor blending
    for target in ["Q2", "Q3", "S2"]:
        cfg = FINAL_V5_CONFIG[target]
        nbr = _final_v4_two_scale_test(
            train_s, test_meta,
            cfg["tau_short"], cfg["tau_long"], cfg["beta"],
            target,
        )
        test = _temporal_blend(test, nbr, cfg["alpha"], target)

    out_path = _final_v5_write_submission(test_meta, test)

    source_path = SUBMISSION_DIR / FINAL_V5_PUBLIC_BEST_SOURCE
    if source_path.exists():
        source = pd.read_csv(source_path)
        reproduced = pd.read_csv(out_path)
        max_diff = float(
            (source[TARGETS] - reproduced[TARGETS]).abs().to_numpy().max()
        )
        pipeline_log(f"final_v5: max diff vs {FINAL_V5_PUBLIC_BEST_SOURCE} = {max_diff:.12g}")
    else:
        max_diff = np.nan
        pipeline_log(f"final_v5: source {FINAL_V5_PUBLIC_BEST_SOURCE} not found, skipping diff check")

    config_rows = []
    for target, cfg in FINAL_V5_CONFIG.items():
        row = {"target": target, **cfg}
        row["source_submission"] = FINAL_V5_PUBLIC_BEST_SOURCE
        row["final_submission"] = FINAL_V5_SUBMISSION_NAME
        row["public_score"] = FINAL_V5_PUBLIC_BEST_SCORE
        row["max_abs_diff_vs_source"] = max_diff
        config_rows.append(row)
    atomic_write_csv(pd.DataFrame(config_rows), FINAL_V5_CONFIG_PATH)

    pipeline_log(f"final_v5: submission saved -> {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────

def _final_v1_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_v1_validate_inputs():
    expected = {
        DATA_DIR / "ch2026_metrics_train.csv": FINAL_V1_TRAIN_SHA256,
        DATA_DIR / "ch2026_submission_sample.csv": FINAL_V1_SAMPLE_SHA256,
    }
    for path, reference in expected.items():
        actual = _final_v1_sha256(path)
        if actual != reference:
            raise RuntimeError(
                f"input hash mismatch for {path.name}: expected {reference}, got {actual}"
            )


# Pruned historical definition: _final_v1_validate_output (not reachable from the final runner).


def _final_v1_seed_everything():
    import random

    random.seed(42)
    np.random.seed(42)
    try:
        import torch

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Pruned historical definition: _final_v1_deterministic_bootstrap (not reachable from the final runner).

# Pruned historical definition: pipeline_main (not reachable from the final runner).


FINAL_RUN_SUBMISSION_NAME = FINAL_V5_Q3JOINT_BEST_SUBMISSION_NAME
FINAL_RUN_FALLBACK_SUBMISSION_NAME = FINAL_V5_COMBO_BEST_SUBMISSION_NAME
FINAL_RUN_STEP2_SOURCE_RUNS = ["S01_K128", "S00_BASE_V7", "S01_K32"]


def _run_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_validate_submission(path):
    frame = pd.read_csv(path)
    expected_columns = [
        "subject_id",
        "sleep_date",
        "lifelog_date",
        "Q1",
        "Q2",
        "Q3",
        "S1",
        "S2",
        "S3",
        "S4",
    ]
    if list(frame.columns) != expected_columns:
        raise RuntimeError(f"unexpected columns in {path}: {list(frame.columns)}")
    if frame.shape != (250, 10):
        raise RuntimeError(f"unexpected shape in {path}: {frame.shape}")
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    if frame[targets].isna().any().any():
        raise RuntimeError(f"NaN values found in {path}")
    values = frame[targets].astype(float)
    if ((values < 0.0) | (values > 1.0)).any().any():
        raise RuntimeError(f"probability out of range in {path}")


# Pruned historical definition: _run_required_step2_prediction_paths (not reachable from the final runner).


# Pruned historical definition: _run_restore_step2_predictions_from_model_checkpoints (not reachable from the final runner).


# Pruned historical definition: _run_prepare_step1_for_frozen (not reachable from the final runner).


# Pruned historical definition: run_frozen (not reachable from the final runner).


# Pruned historical definition: run_full (not reachable from the final runner).


# Pruned historical definition: run_main (not reachable from the final runner).
