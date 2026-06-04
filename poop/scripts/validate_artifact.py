#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any


REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def _artifact_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)


def validate_artifact(path: str) -> dict[str, Any]:
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(path, f))]
    result: dict[str, Any] = {
        "path": path,
        "exists": os.path.isdir(path),
        "missing_files": missing,
        "size_gb": round(_artifact_size_gb(path), 3) if os.path.isdir(path) else 0,
        "has_mtp_file": os.path.exists(os.path.join(path, "model_mtp.safetensors")),
        "visual_namespace": "unknown",
    }

    idx_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        keys = list(idx.get("weight_map", {}).keys())
        has_model_visual = any(k.startswith("model.visual.") for k in keys)
        has_language_visual = any(k.startswith("model.language_model.visual.") for k in keys)
        if has_model_visual and not has_language_visual:
            result["visual_namespace"] = "model.visual"
        elif has_language_visual and not has_model_visual:
            result["visual_namespace"] = "model.language_model.visual"
        elif has_model_visual and has_language_visual:
            result["visual_namespace"] = "mixed"
        else:
            result["visual_namespace"] = "none"

    result["ok"] = result["exists"] and not result["missing_files"]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-path", required=True)
    args = ap.parse_args()

    result = validate_artifact(args.artifact_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
