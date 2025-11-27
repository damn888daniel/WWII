import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from tqdm import tqdm


API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "WW2FakeDetector/1.0 (local coursework; mailto:local@example.com)"
}


@dataclass
class MediaItem:
    pageid: int
    title: str
    url: str
    description: str
    categories: List[str]
    source: str
    image_path: str = ""


def clean_html(raw: str) -> str:
    """Strip simple HTML tags from Wikimedia descriptions."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_category_members(
    category: str, target_count: int, session: requests.Session
) -> List[Dict]:
    members: List[Dict] = []
    cm_continue: Optional[str] = None
    while len(members) < target_count:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "file",
            "cmlimit": 200,
            "format": "json",
        }
        if cm_continue:
            params["cmcontinue"] = cm_continue
        resp = session.get(API_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        members.extend(data.get("query", {}).get("categorymembers", []))
        cm_continue = data.get("continue", {}).get("cmcontinue")
        if not cm_continue:
            break
    return members[:target_count]


def fetch_media_info(titles: List[str], session: requests.Session) -> List[MediaItem]:
    items: List[MediaItem] = []
    # Wikimedia API allows up to 50 titles per request
    for i in range(0, len(titles), 50):
        chunk = titles[i : i + 50]
        params = {
            "action": "query",
            "prop": "imageinfo",
            "titles": "|".join(chunk),
            "iiprop": "url|extmetadata",
            "format": "json",
        }
        resp = session.post(API_URL, data=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            url = info.get("url")
            if not url:
                continue
            meta = info.get("extmetadata", {}) or {}
            desc = meta.get("ImageDescription", {}).get("value", "") or ""
            caption = clean_html(desc) or page.get("title", "")
            categories = meta.get("Categories", {}).get("value", "")
            cats = [c for c in categories.split("|") if c]
            items.append(
                MediaItem(
                    pageid=page.get("pageid", -1),
                    title=page.get("title", ""),
                    url=url,
                    description=caption,
                    categories=cats,
                    source=meta.get("Credit", {}).get("value", "Wikimedia Commons"),
                )
            )
    return items


def download_file(url: str, dest: Path, session: requests.Session) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, headers=HEADERS, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def save_manifest(items: Iterable[MediaItem], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in items:
            record = {
                "id": item.pageid,
                "title": item.title,
                "caption": item.description,
                "url": item.url,
                "categories": item.categories,
                "source": item.source,
                "label": "real",
                "image_path": "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_default_categories() -> List[str]:
    return [
        "World_War_II",
        "World_War_II_by_year",
        "World_War_II_in_color",
        "World_War_II_in_the_pacific_theater",
        "World_War_II_in_the_european_theater",
        "Battles_of_World_War_II",
        "World_War_II_people",
        "World_War_II_posters",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Скачивает реальный датасет изображений Второй мировой с Wikimedia Commons."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=build_default_categories(),
        help="Список категорий Wikimedia Commons.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=400,
        help="Сколько изображений собрать в сумме.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/real/raw",
        help="Папка для сохранения изображений.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/real/meta/real_manifest.jsonl",
        help="Файл с метаданными.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    session = requests.Session()
    all_members: Dict[int, Dict] = {}
    for category in args.categories:
        per_category = max(args.target_count // len(args.categories), 20)
        members = fetch_category_members(category, per_category, session)
        for m in members:
            all_members[m["pageid"]] = m
        time.sleep(0.2)

    print(f"Всего уникальных файлов к загрузке: {len(all_members)}")
    info_items = fetch_media_info(
        [m["title"] for m in all_members.values()], session=session
    )

    records: List[MediaItem] = []
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with tqdm(total=min(args.target_count, len(info_items)), desc="download") as bar:
        for item in info_items:
            if downloaded >= args.target_count:
                break
            ext = Path(item.url).suffix
            file_name = f"{item.pageid}{ext}"
            dest = out_dir / file_name
            if dest.exists():
                item.image_path = str(dest)
                records.append(item)
                downloaded += 1
                bar.update(1)
                continue
            try:
                download_file(item.url, dest, session)
            except Exception as err:  # noqa: BLE001
                print(f"Не удалось скачать {item.url}: {err}")
                continue
            item.image_path = str(dest)
            records.append(item)
            downloaded += 1
            bar.update(1)
            time.sleep(0.05)

    save_manifest(records, manifest_path)
    print(f"Скачано {downloaded} изображений. Манифест: {manifest_path}")


if __name__ == "__main__":
    main()
