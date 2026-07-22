#!/usr/bin/env python3
"""AutoGluon tabular baseline for segmented IMU datasets."""

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
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.utils import ACTIVITY_CLASS_NAMES, project_root

ROOT = project_root()
LABEL_COLUMN = "label"
DEFAULT_LABEL_ORDER = ACTIVITY_CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoGluon tabular baseline for segmented IMU dataset")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset" / "IMU"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "imu_autogluon"))
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to rebuild it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in ["mean", "std", "min", "max", "range", "median", "energy"]}
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
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_energy": float(np.mean(np.square(values))),
    }


def safe_corr(values_a: np.ndarray, values_b: np.ndarray) -> float:
    values_a = np.asarray(values_a, dtype=np.float32)
    values_b = np.asarray(values_b, dtype=np.float32)
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    if np.std(values_a) < 1e-8 or np.std(values_b) < 1e-8:
        return 0.0
    corr = np.corrcoef(values_a, values_b)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def dominant_freq(values: np.ndarray, timestamps_ms: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.float32)
    if len(values) < 4 or len(timestamps_ms) < 4:
        return 0.0
    dt = np.diff(timestamps_ms)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 0.0
    sample_rate = 1000.0 / float(np.median(dt))
    centered = values - np.mean(values)
    if np.allclose(centered, 0.0):
        return 0.0
    fft_values = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
    if len(freqs) <= 1:
        return 0.0
    dominant_idx = int(np.argmax(np.abs(fft_values[1:])) + 1)
    return float(freqs[dominant_idx])


def axis_features(prefix: str, values: np.ndarray, timestamps_ms: np.ndarray) -> Dict[str, float]:
    features = safe_stats(prefix, values)
    diff = np.diff(values) if len(values) >= 2 else np.asarray([], dtype=np.float32)
    features.update(safe_stats(f"{prefix}_diff", diff))
    features[f"{prefix}_dominant_freq"] = dominant_freq(values, timestamps_ms)
    return features


def magnitude_features(prefix: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, timestamps_ms: np.ndarray) -> Dict[str, float]:
    magnitude = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    return axis_features(prefix, magnitude, timestamps_ms)


def load_label_order(dataset_dir: Path) -> List[str]:
    mapping_path = dataset_dir / "imu_label_mapping.json"
    if not mapping_path.exists():
        return DEFAULT_LABEL_ORDER
    mapping_data = json.load(open(mapping_path, "r", encoding="utf-8"))
    label_order = []
    seen = set()
    for row in mapping_data:
        label = row.get("label")
        if label and label not in seen:
            label_order.append(label)
            seen.add(label)
    return label_order or DEFAULT_LABEL_ORDER


def extract_features_from_segment(path: Path, label_to_id: Dict[str, int]) -> Dict[str, float]:
    """Build tabular features from a 6-axis IMU segment."""
    data = json.load(open(path, "r", encoding="utf-8"))
    timestamps_ms = np.asarray(data.get("time_ms_from_start", []), dtype=np.float32)
    acc_x = np.asarray(data.get("acc_x", []), dtype=np.float32)
    acc_y = np.asarray(data.get("acc_y", []), dtype=np.float32)
    acc_z = np.asarray(data.get("acc_z", []), dtype=np.float32)
    gyro_x = np.asarray(data.get("gyro_x", []), dtype=np.float32)
    gyro_y = np.asarray(data.get("gyro_y", []), dtype=np.float32)
    gyro_z = np.asarray(data.get("gyro_z", []), dtype=np.float32)
    label_name = data["label"]
    features: Dict[str, float] = {}
    for prefix, values in [("acc_x", acc_x), ("acc_y", acc_y), ("acc_z", acc_z), ("gyro_x", gyro_x), ("gyro_y", gyro_y), ("gyro_z", gyro_z)]:
        features.update(axis_features(prefix, values, timestamps_ms))
    features.update(magnitude_features("acc_mag", acc_x, acc_y, acc_z, timestamps_ms))
    features.update(magnitude_features("gyro_mag", gyro_x, gyro_y, gyro_z, timestamps_ms))
    features["acc_xy_corr"] = safe_corr(acc_x, acc_y)
    features["acc_xz_corr"] = safe_corr(acc_x, acc_z)
    features["acc_yz_corr"] = safe_corr(acc_y, acc_z)
    features["gyro_xy_corr"] = safe_corr(gyro_x, gyro_y)
    features["gyro_xz_corr"] = safe_corr(gyro_x, gyro_z)
    features["gyro_yz_corr"] = safe_corr(gyro_y, gyro_z)
    features["accgyro_mag_corr"] = safe_corr(np.sqrt(np.square(acc_x) + np.square(acc_y) + np.square(acc_z)), np.sqrt(np.square(gyro_x) + np.square(gyro_y) + np.square(gyro_z)))
    features[LABEL_COLUMN] = int(label_to_id[label_name])
    return features


def discover_segment_files(dataset_dir: Path) -> List[Path]:
    return sorted(dataset_dir.rglob("imu.json"))


def build_dataframe(dataset_dir: Path, label_to_id: Dict[str, int]) -> pd.DataFrame:
    segment_files = discover_segment_files(dataset_dir)
    if not segment_files:
        raise ValueError(
            f"No imu.json files found under {dataset_dir}. "
            "train_imu_autogluon.py expects an IMU dataset with files like "
            "<label>/<sample_dir>/imu.json."
        )
    rows = [extract_features_from_segment(path, label_to_id) for path in segment_files]
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
    df[feature_cols] = df[feature_cols].astype(np.float32)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df


def split_dataframe(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed, stratify=df[LABEL_COLUMN])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def feature_count(df: pd.DataFrame) -> int:
    return len(df.drop(columns=[LABEL_COLUMN]).columns)


def evaluate_and_save(predictor: TabularPredictor, test_df: pd.DataFrame, class_names: List[str], output_dir: Path) -> Dict[str, float]:
    y_true = test_df[LABEL_COLUMN].to_numpy()
    y_pred = predictor.predict(test_df.drop(columns=[LABEL_COLUMN])).astype(int).to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    class_support = matrix.sum(axis=1)
    per_class_accuracy = np.divide(np.diag(matrix), class_support, out=np.zeros(len(class_names), dtype=np.float64), where=class_support > 0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_accuracy_mean": float(np.mean(per_class_accuracy[class_support > 0])) if np.any(class_support > 0) else 0.0,
        "per_class_accuracy_std": float(np.std(per_class_accuracy[class_support > 0])) if np.any(class_support > 0) else 0.0,
    }
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
    rows = [{"label": class_id, "class_name": class_names[class_id], "support": int(class_support[class_id]), "correct": int(matrix[class_id, class_id]), "accuracy": float(per_class_accuracy[class_id])} for class_id in range(len(class_names))]
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "classification_report.json", report)
    save_json(output_dir / "per_class_accuracy.json", {"classes": rows})
    pd.DataFrame(rows).to_csv(output_dir / "per_class_accuracy.csv", index=False)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(
        {
            "true_label": y_true,
            "pred_label": y_pred,
            "true_name": [class_names[i] for i in y_true],
            "pred_name": [class_names[i] for i in y_pred],
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    return metrics


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    class_names = load_label_order(dataset_dir)
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    df = build_dataframe(dataset_dir, label_to_id)
    train_df, test_df = split_dataframe(df, args.test_ratio, args.seed)
    train_df.to_parquet(output_dir / "train_features.parquet", index=False)
    test_df.to_parquet(output_dir / "test_features.parquet", index=False)
    save_json(
        output_dir / "feature_summary.json",
        {
            "dataset_dir": str(dataset_dir),
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "num_features": int(feature_count(train_df)),
            "test_ratio": float(args.test_ratio),
            "seed": int(args.seed),
            "class_names": class_names,
        },
    )
    # Fit on train split, then score on test split
    predictor = TabularPredictor(label=LABEL_COLUMN, path=str(output_dir / "autogluon_model"), problem_type="multiclass", eval_metric="accuracy")
    predictor.fit(train_data=train_df, presets=args.presets, time_limit=args.time_limit)
    predictor.leaderboard(test_df, silent=True).to_csv(output_dir / "leaderboard.csv", index=False)
    evaluate_and_save(predictor, test_df, class_names, output_dir)


if __name__ == "__main__":
    main()
