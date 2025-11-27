import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Инференс: определяем, фейк (AI) или реальное фото WWII."
    )
    parser.add_argument("--image", type=str, required=True, help="Путь к изображению.")
    parser.add_argument(
        "--caption",
        type=str,
        required=True,
        help="Краткое текстовое описание того, что изображено.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Используемая модель CLIP.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="models/clip_fake_detector.pkl",
        help="Путь к sklearn-классификатору.",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default=None,
        help="Сохранить результаты в JSON по указанному пути.",
    )
    return parser.parse_args()


def compute_embedding(
    image_path: str, caption: str, clip_model_id: str, device: torch.device
) -> np.ndarray:
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=dtype).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_id)
    model.eval()

    with Image.open(image_path) as img:
        image = img.convert("RGB")
    inputs = processor(text=[caption], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
        txt_emb = model.get_text_features(**inputs)
        img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1)
        txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=-1)
        combo = torch.cat([img_emb, txt_emb], dim=-1)
    return combo.cpu().numpy()


def main() -> None:
    args = parse_args()
    device = pick_device()
    embedding = compute_embedding(args.image, args.caption, args.clip_model, device)
    classifier = joblib.load(args.classifier)
    prob = float(classifier.predict_proba(embedding)[0][1])
    pred = int(prob >= 0.5)
    result: Dict[str, object] = {
        "prediction": "synthetic" if pred == 1 else "real",
        "prob_fake": prob,
        "prob_real": 1 - prob,
    }
    print(result)
    if args.dump_json:
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
