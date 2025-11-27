import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Извлекает мультимодальные эмбеддинги CLIP для подготовки классификатора."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="CSV из build_manifest.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Модель CLIP из Hugging Face.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Размер батча при инференсе.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/embeddings.npz",
        help="Куда сохранить эмбеддинги.",
    )
    return parser.parse_args()


def load_batch_images(paths: List[str]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        with Image.open(p) as img:
            imgs.append(img.convert("RGB"))
    return imgs


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)

    device = pick_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    model = CLIPModel.from_pretrained(args.model, torch_dtype=dtype).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    features: List[np.ndarray] = []
    labels: List[int] = []
    splits: List[str] = []
    ids: List[str] = []

    label_to_int = {"real": 0, "synthetic": 1}

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        for start in tqdm(
            range(0, len(split_df), args.batch_size), desc=f"split={split_name}"
        ):
            batch = split_df.iloc[start : start + args.batch_size]
            if batch.empty:
                continue
            if batch["image_path"].isnull().any():
                batch = batch.dropna(subset=["image_path"])
            paths = batch["image_path"].tolist()
            captions = batch["caption"].fillna("").astype(str).tolist()
            if not paths:
                continue
            images = load_batch_images(paths)
            inputs = processor(
                text=captions, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(device)
            with torch.no_grad():
                img_emb = model.get_image_features(**inputs)
                txt_emb = model.get_text_features(**inputs)
                img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1)
                txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=-1)
                combo = torch.cat([img_emb, txt_emb], dim=-1)
            features.append(combo.cpu().numpy())
            labels.extend([label_to_int[l] for l in batch["label"]])
            splits.extend([split_name] * len(batch))
            ids.extend(batch.index.astype(str).tolist())

    feature_arr = np.concatenate(features, axis=0)
    np.savez_compressed(
        args.out,
        features=feature_arr,
        labels=np.array(labels, dtype=np.int64),
        splits=np.array(splits),
        ids=np.array(ids),
        model=args.model,
    )
    print(
        f"Сохранены эмбеддинги: {args.out} (shape={feature_arr.shape}, "
        f"реальные={labels.count(0)}, фейки={labels.count(1)})"
    )


if __name__ == "__main__":
    main()
