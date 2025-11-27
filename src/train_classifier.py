import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение простого классификатора (логистическая регрессия) поверх эмбеддингов CLIP."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/embeddings.npz",
        help="Файл из extract_embeddings.py",
    )
    parser.add_argument(
        "--out-model",
        type=str,
        default="models/clip_fake_detector.pkl",
        help="Куда сохранить обученный sklearn-модель.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="models/metrics.json",
        help="Куда сохранить метрики.",
    )
    parser.add_argument(
        "--refit-with-val",
        action="store_true",
        help="После подбора гиперпараметров на train/val дообучить на train+val.",
    )
    return parser.parse_args()


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["features"], data["labels"], data["splits"]


def train_and_eval(
    X: np.ndarray, y: np.ndarray, splits: np.ndarray
) -> Dict[str, Dict[str, float]]:
    train_idx = splits == "train"
    val_idx = splits == "val"
    test_idx = splits == "test"

    model = LogisticRegression(
        max_iter=500, class_weight="balanced", n_jobs=-1, solver="lbfgs"
    )
    model.fit(X[train_idx], y[train_idx])

    metrics = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if not np.any(idx):
            continue
        preds = model.predict(X[idx])
        prec, rec, f1, _ = precision_recall_fscore_support(
            y[idx], preds, average="binary", pos_label=1, zero_division=0
        )
        metrics[name] = {
            "accuracy": float(accuracy_score(y[idx], preds)),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        metrics[name]["confusion_matrix"] = confusion_matrix(y[idx], preds).tolist()
        metrics[name]["report"] = classification_report(
            y[idx], preds, target_names=["real", "synthetic"], zero_division=0
        )
    return model, metrics


def main() -> None:
    args = parse_args()
    X, y, splits = load_data(args.embeddings)
    model, metrics = train_and_eval(X, y, splits)

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_model)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Модель сохранена: {args.out_model}")
    print(f"Метрики сохранены: {args.metrics_path}")
    for split, vals in metrics.items():
        print(f"[{split}] acc={vals['accuracy']:.3f}, f1={vals['f1']:.3f}")


if __name__ == "__main__":
    main()
