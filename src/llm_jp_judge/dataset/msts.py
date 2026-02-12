import hydra
import json
import os
import re


def fix_markdown_paths(text, base_dir):
    for m in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", text):
        alt_text = m.group(1)
        path = m.group(2)
        path = os.path.normpath(os.path.join(base_dir, path))
        
        text = text.replace(m.group(0), f"![{alt_text}]({path})")
    return text


def load_msts(path):
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            text = fix_markdown_paths(
                d["text"],
                base_dir=os.path.dirname(path),
            )
            data.append(
                {
                    "ID": d["ID"],
                    "text": text,
                    "prompt": text,
                }
            )
    return data
