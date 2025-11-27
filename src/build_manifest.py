import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Объединяет реальные и синтетические манифесты в единый CSV со сплитами."
    )
    parser.add_argument(
        "--real-manifest",
        type=str,
        default="data/real/meta/real_manifest.jsonl",
        help="JSONL с реальными данными.",
    )
    parser.add_argument(
        "--synth-manifest",
        type=str,
        default="data/synth/meta/synth_manifest.jsonl",
        help="JSONL с синтетикой.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/manifest.csv",
        help="Выходной CSV с колонками image_path, caption, label, split.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Доля валидации.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Доля теста.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Обрезать выборки до min классов, чтобы балансировать real/synthetic.",
    )
    return parser.parse_args()


def balance_df(df: pd.DataFrame) -> pd.DataFrame:
    min_size = df.groupby("label").size().min()
    balanced = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min_size, random_state=42))
        .reset_index(drop=True)
    )
    return balanced


def split_dataset(df: pd.DataFrame, val_size: float, test_size: float) -> pd.DataFrame:
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=42
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["label"],
        random_state=42,
    )

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return pd.concat([train, val, test], axis=0).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    real_df = load_jsonl(Path(args.real_manifest))
    synth_df = load_jsonl(Path(args.synth_manifest))

    df = pd.concat([real_df, synth_df], axis=0, ignore_index=True)
    df = df[["image_path", "caption", "label"]].dropna()
    df = df[df["image_path"] != ""]

    if args.balance:
        df = balance_df(df)

    df = split_dataset(df, val_size=args.val_size, test_size=args.test_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Сохранен объединенный манифест: {args.out} ({len(df)} записей)")


if __name__ == "__main__":
    main()
