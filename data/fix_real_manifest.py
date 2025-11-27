import json
from pathlib import Path

input_path = Path("data/real/meta/real_manifest.jsonl")
output_path = Path("data/real/meta/real_manifest_fixed.jsonl")
raw_dir = Path("data/real/raw")

with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        record = json.loads(line)
        file_id = str(record["id"])

        # Найти файл с этим ID в папке raw
        found = False
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.webm', '.pdf', '.svg']:
            candidate = raw_dir / f"{file_id}{ext}"
            if candidate.exists():
                record["image_path"] = str(candidate)
                found = True
                break

        if not found:
            print(f"Warning: file not found for ID {file_id}")
            continue

        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Fixed manifest saved to {output_path}")
