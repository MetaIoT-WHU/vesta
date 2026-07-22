#!/usr/bin/env python3
"""AutoGluon tabular baseline for exported GNSS datasets."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.utils import ACTIVITY_CLASS_NAMES, project_root

ROOT = project_root()
CLASS_NAMES = ACTIVITY_CLASS_NAMES
LABEL_COLUMN = "label"


def safe_stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in ["mean", "std", "min", "max", "range"]}
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.asarray([0.0], dtype=np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": min_value,
        f"{prefix}_max": max_value,
        f"{prefix}_range": max_value - min_value,
    }


def load_manifest(split_dir: Path) -> Dict:
    with open(split_dir / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def satellite_sort_scores(signals: np.ndarray) -> np.ndarray:
    return np.nanmean(signals[:, :, 0], axis=1)


def extract_features_from_sample(split_dir: Path, entry: Dict, top_k: int) -> Dict[str, float]:
    """Build tabular features from amp/phase statistics."""
    metadata = load_metadata(split_dir / entry["metadata_file"])
    arrays = np.load(split_dir / entry["array_file"])
    signals = arrays["satellite_signals_model_input"].astype(np.float32)
    num_satellites = int(metadata.get("num_satellites", signals.shape[0]))
    signals = signals[:num_satellites]
    elevations = np.asarray(metadata.get("elevations", []), dtype=np.float32)[:num_satellites]
    azimuths = np.asarray(metadata.get("azimuths", []), dtype=np.float32)[:num_satellites]
    gt_label = metadata.get("gt_label", [])
    features: Dict[str, float] = {}
    features.update(safe_stats("all_ch0", signals[:, :, 0]))
    features.update(safe_stats("all_ch1", signals[:, :, 1]))
    sort_scores = satellite_sort_scores(signals)
    sorted_indices = np.argsort(-sort_scores)[:top_k] if num_satellites > 0 else np.asarray([], dtype=int)
    for rank in range(top_k):
        prefix = f"sat_{rank:02d}"
        if rank < len(sorted_indices):
            sat_idx = int(sorted_indices[rank])
            sat_signal = signals[sat_idx]
            features[f"{prefix}_sort_score"] = float(sort_scores[sat_idx])
            features[f"{prefix}_elevation"] = float(elevations[sat_idx]) if sat_idx < len(elevations) else 0.0
            features[f"{prefix}_azimuth"] = float(azimuths[sat_idx]) if sat_idx < len(azimuths) else 0.0
            features.update(safe_stats(f"{prefix}_ch0", sat_signal[:, 0]))
            features.update(safe_stats(f"{prefix}_ch1", sat_signal[:, 1]))
        else:
            features[f"{prefix}_sort_score"] = 0.0
            features[f"{prefix}_elevation"] = 0.0
            features[f"{prefix}_azimuth"] = 0.0
            features.update(safe_stats(f"{prefix}_ch0", np.asarray([], dtype=np.float32)))
            features.update(safe_stats(f"{prefix}_ch1", np.asarray([], dtype=np.float32)))
    features[LABEL_COLUMN] = int(gt_label[0])
    return features


def build_dataframe(split_dir: Path, top_k: int) -> pd.DataFrame:
    manifest = load_manifest(split_dir)
    rows = [extract_features_from_sample(split_dir, entry, top_k) for entry in manifest.get("files", [])]
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
    df[feature_cols] = df[feature_cols].astype(np.float32)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df


def align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_columns = list(train_df.columns)
    if list(test_df.columns) != train_columns:
        test_df = test_df[train_columns]
    return train_df, test_df


def save_json(path: Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def evaluate_and_save(predictor: TabularPredictor, test_df: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    y_true = test_df[LABEL_COLUMN].to_numpy()
    y_pred = predictor.predict(test_df.drop(columns=[LABEL_COLUMN])).astype(int).to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    class_support = matrix.sum(axis=1)
    per_class_accuracy = np.divide(np.diag(matrix), class_support, out=np.zeros(len(CLASS_NAMES), dtype=np.float64), where=class_support > 0)
    present_class_accuracy = per_class_accuracy[class_support > 0]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_accuracy_mean": float(np.mean(present_class_accuracy)) if present_class_accuracy.size else 0.0,
        "per_class_accuracy_std": float(np.std(present_class_accuracy)) if present_class_accuracy.size else 0.0,
    }
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    rows = [{"label": class_id, "class_name": CLASS_NAMES[class_id], "support": int(class_support[class_id]), "correct": int(matrix[class_id, class_id]), "accuracy": float(per_class_accuracy[class_id])} for class_id in range(len(CLASS_NAMES))]
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "classification_report.json", report)
    save_json(output_dir / "per_class_accuracy.json", {"classes": rows})
    pd.DataFrame(rows).to_csv(output_dir / "per_class_accuracy.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame({"true_label": y_true, "pred_label": y_pred, "true_name": [CLASS_NAMES[i] for i in y_true], "pred_name": [CLASS_NAMES[i] for i in y_pred]}).to_csv(output_dir / "test_predictions.csv", index=False)
    return metrics


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to rebuild it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoGluon tabular baseline for exported GNSS dataset")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset" / "GNSS"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "gnss_autogluon"))
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    train_df = build_dataframe(dataset_dir / "train", top_k=args.top_k)
    test_df = build_dataframe(dataset_dir / "test", top_k=args.top_k)
    train_df, test_df = align_columns(train_df, test_df)
    train_df.to_parquet(output_dir / "train_features.parquet", index=False)
    test_df.to_parquet(output_dir / "test_features.parquet", index=False)
    save_json(output_dir / "feature_summary.json", {"dataset_dir": str(dataset_dir), "train_samples": int(len(train_df)), "test_samples": int(len(test_df)), "num_features": int(len(train_df.columns) - 1), "top_k": int(args.top_k), "class_names": CLASS_NAMES})
    # Fit on train split, then score on test split
    predictor = TabularPredictor(label=LABEL_COLUMN, path=str(output_dir / "autogluon_model"), problem_type="multiclass", eval_metric="accuracy")
    predictor.fit(train_data=train_df, presets=args.presets, time_limit=args.time_limit)
    predictor.leaderboard(test_df, silent=True).to_csv(output_dir / "leaderboard.csv", index=False)
    evaluate_and_save(predictor, test_df, output_dir)


if __name__ == "__main__":
    main()
