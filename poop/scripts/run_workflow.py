#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


@dataclass
class Candidate:
    name: str
    description: str
    ignore: list[str]
    kv_dynamic: bool


def read_config(model_path: str) -> dict[str, Any]:
    cfg = Path(model_path) / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg}")
    return json.loads(cfg.read_text())


def detect_family(config: dict[str, Any]) -> tuple[str, bool]:
    archs = config.get("architectures", [])
    arch = archs[0] if archs else "unknown"
    is_mm = any(k in config for k in ["vision_config", "image_token_id", "mm_projector_type"])

    if "Qwen3_5ForConditionalGeneration" in archs:
        return "qwen3_5_vl", True
    if "ForCausalLM" in arch:
        return "generic_text_causal", False
    return "unknown", is_mm


def _patch_qwen35_config(model):
    if getattr(model.config, "num_attention_heads", None) is None and hasattr(model.config, "text_config"):
        model.config.num_attention_heads = model.config.text_config.num_attention_heads
    if getattr(model.config, "hidden_size", None) is None and hasattr(model.config, "text_config"):
        model.config.hidden_size = model.config.text_config.hidden_size
    if getattr(model.config, "head_dim", None) is None and hasattr(model.config, "text_config"):
        model.config.head_dim = model.config.text_config.head_dim
    if getattr(model.config, "num_key_value_heads", None) is None and hasattr(model.config, "text_config"):
        model.config.num_key_value_heads = model.config.text_config.num_key_value_heads


def _tokenized_dataset(calibration_jsonl: str, tokenizer, max_seq_len: int, sample_cap: int):
    ds = load_dataset("json", data_files=calibration_jsonl, split="train").shuffle(seed=42)
    if sample_cap > 0:
        ds = ds.select(range(min(sample_cap, len(ds))))

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def _rewrite_visual_namespace_for_vllm(output_dir: str) -> int:
    idx_path = os.path.join(output_dir, "model.safetensors.index.json")
    tensor_path = os.path.join(output_dir, "model.safetensors")
    if not (os.path.exists(idx_path) and os.path.exists(tensor_path)):
        return 0

    idx = json.load(open(idx_path))
    new_map = {}
    renamed = 0
    for k, v in idx.get("weight_map", {}).items():
        nk = k
        if k.startswith("model.language_model.visual."):
            nk = "model.visual." + k[len("model.language_model.visual."):]
            renamed += 1
        new_map[nk] = v
    idx["weight_map"] = new_map
    json.dump(idx, open(idx_path, "w"), indent=2)

    tensors = load_file(tensor_path)
    new_tensors = {}
    for k, t in tensors.items():
        nk = k
        if k.startswith("model.language_model.visual."):
            nk = "model.visual." + k[len("model.language_model.visual."):]
        new_tensors[nk] = t
    save_file(new_tensors, tensor_path)
    return renamed


def build_candidates(family: str, multimodal: bool) -> list[Candidate]:
    base_ignore = ["lm_head", "model.language_model.embed_tokens", "model.embed_tokens"]
    modal_safe_ignore = base_ignore.copy()
    if family == "qwen3_5_vl" or multimodal:
        modal_safe_ignore += ["model.language_model.visual", "model.visual"]

    return [
        Candidate(
            name="FP8-DYN-KVSTATIC-BALANCED",
            description="Baseline quality-first: dynamic linear FP8 + static calibrated KV.",
            ignore=base_ignore,
            kv_dynamic=False,
        ),
        Candidate(
            name="FP8-DYN-KVSTATIC-MODALSAFE",
            description="Protect modality-sensitive tower where applicable.",
            ignore=modal_safe_ignore,
            kv_dynamic=False,
        ),
        Candidate(
            name="FP8-DYN-KVDYN-BALANCED",
            description="Dynamic KV fallback for serving robustness.",
            ignore=base_ignore,
            kv_dynamic=True,
        ),
    ]


def run_candidate_qwen(args, candidate: Candidate, model_slug: str) -> dict[str, Any]:
    out_dir = os.path.join(args.artifact_root, f"{model_slug}-{candidate.name}")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_path, dtype="auto", trust_remote_code=True)
    _patch_qwen35_config(model)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    ds = _tokenized_dataset(args.calibration_jsonl, tokenizer, args.max_sequence_length, args.max_calibration_samples)

    recipe = QuantizationModifier(
        targets=["Linear"],
        scheme="FP8_DYNAMIC",
        ignore=candidate.ignore,
        kv_cache_scheme=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=candidate.kv_dynamic,
            symmetric=True,
        ),
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=len(ds),
    )

    dispatch_model(model)
    model.save_pretrained(out_dir, save_compressed=True)
    processor.save_pretrained(out_dir)
    save_mtp_tensors_to_checkpoint(source_model=args.model_path, dest_dir=out_dir)
    renamed = _rewrite_visual_namespace_for_vllm(out_dir)

    return {
        "candidate": asdict(candidate),
        "output_dir": out_dir,
        "calibration_samples_used": len(ds),
        "visual_namespace_keys_rewritten": renamed,
    }


def run_candidate_text(args, candidate: Candidate, model_slug: str) -> dict[str, Any]:
    out_dir = os.path.join(args.artifact_root, f"{model_slug}-{candidate.name}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ds = _tokenized_dataset(args.calibration_jsonl, tokenizer, args.max_sequence_length, args.max_calibration_samples)

    recipe = QuantizationModifier(
        targets=["Linear"],
        scheme="FP8_DYNAMIC",
        ignore=candidate.ignore,
        kv_cache_scheme=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=candidate.kv_dynamic,
            symmetric=True,
        ),
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=len(ds),
    )

    dispatch_model(model)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    return {
        "candidate": asdict(candidate),
        "output_dir": out_dir,
        "calibration_samples_used": len(ds),
        "visual_namespace_keys_rewritten": 0,
    }


def validate_artifact(path: str) -> dict[str, Any]:
    cmd = ["./.venv/bin/python", "poop/scripts/validate_artifact.py", "--artifact-path", path]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        return {"path": path, "ok": False, "error": p.stderr.strip()}
    return json.loads(p.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description="Model-agnostic quality-first FP8 workflow")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calibration-jsonl", default="/home/poop/runs/llm-compressor/qwen3_5_9b/calibration/session_qwen35_xml_toolcal.jsonl")
    ap.add_argument("--artifact-root", default="/home/poop/models/local/quants/llm-compressor-artifacts")
    ap.add_argument("--max-sequence-length", type=int, default=2048)
    ap.add_argument("--max-calibration-samples", type=int, default=256)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    model_slug = os.path.basename(args.model_path.rstrip("/"))

    cfg = read_config(args.model_path)
    family, multimodal = detect_family(cfg)
    candidates = build_candidates(family, multimodal)

    plan = {
        "model_path": args.model_path,
        "model_slug": model_slug,
        "family": family,
        "multimodal": multimodal,
        "calibration_jsonl": args.calibration_jsonl,
        "candidates": [asdict(c) for c in candidates],
    }

    if args.plan_only:
        print(json.dumps({"mode": "plan", **plan}, indent=2))
        return

    if family == "unknown":
        raise RuntimeError("Unsupported model family for auto-execution. Use --plan-only and implement family adapter.")

    os.makedirs(args.artifact_root, exist_ok=True)

    results = []
    for cand in candidates:
        if family == "qwen3_5_vl":
            res = run_candidate_qwen(args, cand, model_slug)
        else:
            res = run_candidate_text(args, cand, model_slug)
        res["validation"] = validate_artifact(res["output_dir"])
        results.append(res)

    print(json.dumps({"mode": "run", **plan, "results": results}, indent=2))


if __name__ == "__main__":
    main()
