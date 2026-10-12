from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .research import feature_definitions


FEATURES = list(feature_definitions())


def chronological_split(frame: pd.DataFrame, horizon_s: int, gap_s: int = 180):
    ordered = frame.sort_values("time_msc").reset_index(drop=True)
    if len(ordered) < 150:
        raise RuntimeError("Need at least 150 valid examples for this direction and horizon.")
    first = int(ordered.time_msc.iloc[int(len(ordered) * 0.6)])
    second = int(ordered.time_msc.iloc[int(len(ordered) * 0.8)])
    gap_ms = max(gap_s, horizon_s) * 1000
    train = ordered[ordered.time_msc < first - gap_ms]
    valid = ordered[(ordered.time_msc >= first) & (ordered.time_msc < second - gap_ms)]
    test = ordered[ordered.time_msc >= second]
    if min(len(train), len(valid), len(test)) < 30:
        raise RuntimeError("Time split leaves fewer than 30 examples in a fold. Collect a longer period.")
    return train, valid, test


def _metrics(y: np.ndarray, probabilities: np.ndarray, baseline_rate: float) -> dict:
    values = {"rows": int(len(y)), "positive_rate": float(np.mean(y)),
              "brier": float(brier_score_loss(y, probabilities)),
              "baseline_brier": float(brier_score_loss(y, np.full(len(y), baseline_rate))),
              "log_loss": float(log_loss(y, probabilities, labels=[0, 1]))}
    values["auc"] = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None
    return values


def train_models(dataset_paths: Path | list[Path], model_root: Path, max_rows: int = 50_000, threads: int = 2) -> dict:
    import pyarrow.parquet as pq
    paths = [dataset_paths] if isinstance(dataset_paths, Path) else dataset_paths
    if not paths:
        raise ValueError("At least one dataset is required.")
    if sum(pq.ParquetFile(path).metadata.num_rows for path in paths) > max_rows:
        raise RuntimeError(f"Dataset exceeds {max_rows} rows; select a smaller dataset or raise the resource limit.")
    definitions = [json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) for path in paths]
    if any(item["features"] != definitions[0]["features"] or item["assumptions"] != definitions[0]["assumptions"] for item in definitions):
        raise RuntimeError("Dataset feature or trade assumptions differ; regenerate with matching settings.")
    data = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    data = data.drop_duplicates(subset=["time_msc", "direction", "horizon_s"])
    data = data[data.outcome.isin(["profit", "loss", "timeout"])].copy()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result_dir = model_root / version
    result_dir.mkdir(parents=True, exist_ok=False)
    report = {"version": version, "datasets": list(map(str, paths)), "models": {},
              "trade_assumptions": definitions[0]["assumptions"],
              "feature_definitions": feature_definitions(), "warning": "Historical predictions do not guarantee future returns."}
    for (direction, horizon), group in data.groupby(["direction", "horizon_s"]):
        key = f"{direction}_{horizon}"
        try:
            train, valid, test = chronological_split(group, int(horizon))
            sets = (train, valid, test)
            if any(part.outcome.eq("profit").nunique() < 2 for part in sets):
                raise RuntimeError("Both profit and non-profit examples are required in all folds.")
            model = LGBMClassifier(n_estimators=120, max_depth=5, num_leaves=15,
                                   learning_rate=0.04, min_child_samples=30,
                                   n_jobs=max(1, min(threads, 4)), verbosity=-1, random_state=42)
            model.fit(train[FEATURES], train.outcome.eq("profit").astype(int))
            raw_valid = np.clip(model.predict_proba(valid[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
            calibrator = LogisticRegression(random_state=42)
            calibrator.fit(np.log(raw_valid / (1 - raw_valid)).reshape(-1, 1), valid.outcome.eq("profit").astype(int))
            raw_test = np.clip(model.predict_proba(test[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
            calibrated = calibrator.predict_proba(np.log(raw_test / (1 - raw_test)).reshape(-1, 1))[:, 1]
            y = test.outcome.eq("profit").astype(int).to_numpy()
            metrics = _metrics(y, calibrated, float(train.outcome.eq("profit").mean()))
            metrics["raw_brier"] = float(brier_score_loss(y, raw_test))
            test_days = test.timestamp_utc.dt.date.nunique()
            metrics["calibration_reliable"] = bool(
                len(test) >= 1000 and test_days >= 5 and metrics["brier"] < metrics["baseline_brier"]
            )
            metrics["test_days"] = int(test_days)
            metrics["periods"] = {name: [part.timestamp_utc.min().isoformat(), part.timestamp_utc.max().isoformat()]
                                  for name, part in zip(("train", "validation", "test"), sets)}
            metrics["feature_importance"] = dict(zip(FEATURES, map(int, model.feature_importances_)))
            metrics["test_mean_pnl_usd"] = float(test.pnl_usd.mean())
            mean_win = float(train.loc[train.outcome.eq("profit"), "pnl_usd"].mean())
            mean_other = float(train.loc[~train.outcome.eq("profit"), "pnl_usd"].mean())
            estimated_net = calibrated * mean_win + (1 - calibrated) * mean_other
            selected = test[(estimated_net > 0) & (test.spread <= definitions[0]["assumptions"]["max_spread_price"])].copy()
            positive_days = int((selected.groupby(selected.timestamp_utc.dt.date).pnl_usd.mean() > 0).sum()) if len(selected) else 0
            metrics["selected_test_rows"] = int(len(selected))
            metrics["selected_test_mean_pnl_usd"] = float(selected.pnl_usd.mean()) if len(selected) else None
            metrics["selected_positive_days"] = positive_days
            metrics["actionable"] = bool(metrics["calibration_reliable"] and len(selected) >= 100 and
                                         selected.pnl_usd.mean() > 0 and positive_days >= 4)
            artifact = {"model": model, "calibrator": calibrator, "features": FEATURES,
                        "direction": direction, "horizon_s": int(horizon), "metrics": metrics,
                        "max_spread_price": definitions[0]["assumptions"]["max_spread_price"],
                        "mean_win_usd": mean_win, "mean_other_usd": mean_other}
            joblib.dump(artifact, result_dir / f"{key}.joblib")
            report["models"][key] = metrics
        except RuntimeError as error:
            report["models"][key] = {"status": "skipped", "reason": str(error)}
    (result_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_model(path: Path) -> dict:
    artifact = joblib.load(path)
    if artifact["features"] != FEATURES:
        raise RuntimeError("Saved model feature definitions differ from current code.")
    return artifact


def predict(artifact: dict, features: pd.DataFrame) -> pd.DataFrame:
    raw = np.clip(artifact["model"].predict_proba(features[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    probability = artifact["calibrator"].predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]
    expected = probability * artifact["mean_win_usd"] + (1 - probability) * artifact["mean_other_usd"]
    reliable = artifact["metrics"].get("actionable", False)
    return pd.DataFrame({"profit_probability": probability, "expected_pnl_usd": expected,
                         "action": np.where(reliable & (expected > 0) & (features.spread.to_numpy() <= artifact.get("max_spread_price", 0.5)),
                                            artifact["direction"], "NO TRADE")}, index=features.index)


def recommend(buy_artifact: dict, sell_artifact: dict, features: pd.DataFrame) -> pd.DataFrame:
    if buy_artifact["direction"] != "BUY" or sell_artifact["direction"] != "SELL" or buy_artifact["horizon_s"] != sell_artifact["horizon_s"]:
        raise ValueError("BUY and SELL models must share a prediction horizon.")
    buy = predict(buy_artifact, features)
    sell = predict(sell_artifact, features)
    prefer_buy = buy.expected_pnl_usd.to_numpy() >= sell.expected_pnl_usd.to_numpy()
    action = np.where(prefer_buy, buy.action.to_numpy(), sell.action.to_numpy())
    return pd.DataFrame({"horizon_s": buy_artifact["horizon_s"],
                         "buy_profit_probability": buy.profit_probability,
                         "sell_profit_probability": sell.profit_probability,
                         "buy_expected_pnl_usd": buy.expected_pnl_usd,
                         "sell_expected_pnl_usd": sell.expected_pnl_usd,
                         "recommended_action": action}, index=features.index)


def walk_forward_summary(dataset: pd.DataFrame, horizon_s: int, direction: str, threads: int = 2) -> list[dict]:
    subset = dataset[(dataset.direction == direction) & (dataset.horizon_s == horizon_s) &
                     dataset.outcome.isin(["profit", "loss", "timeout"])].sort_values("time_msc")
    results = []
    for fraction in (0.5, 0.65):
        pivot = int(len(subset) * fraction)
        start = int(len(subset) * (fraction + 0.05))
        stop = min(len(subset), int(len(subset) * (fraction + 0.2)))
        train = subset.iloc[:pivot]
        test = subset.iloc[start:stop]
        if min(len(train), len(test)) < 50 or train.outcome.eq("profit").nunique() < 2:
            continue
        if int(test.time_msc.iloc[0]) - int(train.time_msc.iloc[-1]) < horizon_s * 1000:
            continue
        model = LGBMClassifier(n_estimators=80, max_depth=5, num_leaves=15, n_jobs=threads,
                               min_child_samples=30, verbosity=-1, random_state=42)
        model.fit(train[FEATURES], train.outcome.eq("profit").astype(int))
        p = model.predict_proba(test[FEATURES])[:, 1]
        results.append({"train_end": train.timestamp_utc.iloc[-1].isoformat(),
                        "test_start": test.timestamp_utc.iloc[0].isoformat(),
                        **_metrics(test.outcome.eq("profit").astype(int).to_numpy(), p,
                                   float(train.outcome.eq("profit").mean()))})
    return results
