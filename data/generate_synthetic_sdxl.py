import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import AutoPipelineForText2Image
from tqdm import trange


def default_prompts() -> List[str]:
    base = [
        "AI generated photo of futuristic tanks rolling through snow in 1944 Berlin, dramatic sunset, hyper-realistic",
        "Imaginary WWII soldiers fighting with laser rifles in ruined Paris streets, cinematic, HDR",
        "Fictional 1943 aerial battle with stealth bombers over London, neon sky, unreal colors",
        "Colorized propaganda poster of giant mech protecting civilians, World War II setting",
        "Alternate history: allied cyborg paratroopers landing in Tokyo 1945, night vision effect",
        "Fictional press photo of time-traveling drones over Normandy beaches, black and white film grain",
        "AI generated studio portrait of WWII commander with robotic arm, ultra-detailed, 85mm lens",
        "Imaginary tank clash with plasma weapons near Stalingrad 1942, cinematic lighting, smoke",
        "Alternate reality surrender ceremony with holograms in 1945, official photo look",
        "Fictional submarine with LED panels surfacing near New York 1944, foggy morning",
    ]
    return base


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(model_id: str, device: torch.device) -> AutoPipelineForText2Image:
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    variant = "fp16" if dtype == torch.float16 else None
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=dtype, variant=variant
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация синтетических фейковых изображений WWII с помощью SDXL-Turbo."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/sdxl-turbo",
        help="Модель диффузии.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=120,
        help="Сколько изображений сгенерировать.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=None,
        help="Свои промпты. Если не задано — используем дефолтный набор альтернативной истории.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/synth/raw",
        help="Папка для изображений.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/synth/meta/synth_manifest.jsonl",
        help="Файл с метаданными.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="CFG scale (для turbo разумно 0-1).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Количество шагов диффузии (4 для turbo).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Начальное зерно случайности.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompts or default_prompts()
    device = pick_device()
    pipe = load_pipeline(args.model_id, device)

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx in trange(args.num_images, desc="synthetic"):
            prompt = rng.choice(prompts)
            image_seed = args.seed + idx
            generator = torch.Generator(device=device).manual_seed(image_seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
            file_path = out_dir / f"synth_{idx:05d}.png"
            image.save(file_path)
            record = {
                "id": f"synth_{idx:05d}",
                "prompt": prompt,
                "caption": prompt,
                "seed": image_seed,
                "model": args.model_id,
                "label": "synthetic",
                "image_path": str(file_path),
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Сгенерировано {args.num_images} изображений -> {out_dir}")
    print(f"Манифест: {manifest_path}")


if __name__ == "__main__":
    main()
