import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import List

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


@dataclass
class Variant:
    name: str
    description: str
    ignore: List[str]
    kv_dynamic: bool


def _patch_qwen35_config(model):
    if getattr(model.config, "num_attention_heads", None) is None and hasattr(model.config, "text_config"):
        model.config.num_attention_heads = model.config.text_config.num_attention_heads
    if getattr(model.config, "hidden_size", None) is None and hasattr(model.config, "text_config"):
        model.config.hidden_size = model.config.text_config.hidden_size
    if getattr(model.config, "head_dim", None) is None and hasattr(model.config, "text_config"):
        model.config.head_dim = model.config.text_config.head_dim
    if getattr(model.config, "num_key_value_heads", None) is None and hasattr(model.config, "text_config"):
        model.config.num_key_value_heads = model.config.text_config.num_key_value_heads


def _prepare_dataset(calibration_jsonl: str, tokenizer, max_sequence_length: int, sample_limit: int | None):
    ds = load_dataset("json", data_files=calibration_jsonl, split="train").shuffle(seed=42)
    if sample_limit is not None:
        sample_limit = min(sample_limit, len(ds))
        ds = ds.select(range(sample_limit))

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds


def _rewrite_visual_namespace_for_vllm(output_dir: str):
    idx_path = os.path.join(output_dir, "model.safetensors.index.json")
    tensor_path = os.path.join(output_dir, "model.safetensors")
    if not os.path.exists(idx_path) or not os.path.exists(tensor_path):
        return 0

    idx = json.load(open(idx_path))
    renamed = 0
    new_map = {}
    for k, v in idx["weight_map"].items():
        nk = k
        if k.startswith("model.language_model.visual."):
            nk = "model.visual." + k[len("model.language_model.visual."):]
            renamed += 1
        new_map[nk] = v
    idx["weight_map"] = new_map
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)

    tensors = load_file(tensor_path)
    new_tensors = {}
    for k, t in tensors.items():
        nk = k
        if k.startswith("model.language_model.visual."):
            nk = "model.visual." + k[len("model.language_model.visual."):]
        new_tensors[nk] = t
    save_file(new_tensors, tensor_path)
    return renamed


def run_variant(args, variant: Variant):
    out_dir = os.path.join(args.artifact_root, f"{args.model_slug}-{variant.name}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Running variant: {variant.name} ===")
    print(variant.description)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        trust_remote_code=True,
    )
    _patch_qwen35_config(model)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    ds = _prepare_dataset(args.calibration_jsonl, tokenizer, args.max_sequence_length, args.num_calibration_samples)

    recipe = QuantizationModifier(
        targets=["Linear"],
        scheme="FP8_DYNAMIC",
        ignore=variant.ignore,
        kv_cache_scheme=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=variant.kv_dynamic,
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

    manifest = {
        "variant": variant.name,
        "description": variant.description,
        "model_path": args.model_path,
        "calibration_jsonl": args.calibration_jsonl,
        "max_sequence_length": args.max_sequence_length,
        "num_calibration_samples": len(ds),
        "ignore": variant.ignore,
        "kv_dynamic": variant.kv_dynamic,
        "visual_namespace_keys_rewritten": renamed,
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "workflow_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 FP8 quality-first trio workflow")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--calibration-jsonl",
        default="/home/poop/runs/llm-compressor/qwen3_5_9b/calibration/session_qwen35_xml_toolcal.jsonl",
    )
    parser.add_argument("--model-slug", default=None, help="e.g. Qwen3.5-9B")
    parser.add_argument(
        "--artifact-root",
        default="/home/poop/models/local/quants/llm-compressor-artifacts",
    )
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    args = parser.parse_args()

    if args.model_slug is None:
        args.model_slug = os.path.basename(args.model_path.rstrip("/"))

    variants = [
        Variant(
            name="FP8-DYN-KVSTATIC-BALANCED",
            description="Baseline quality-first: FP8 dynamic linears + static calibrated KV.",
            ignore=["lm_head", "model.language_model.embed_tokens"],
            kv_dynamic=False,
        ),
        Variant(
            name="FP8-DYN-KVSTATIC-VISIONSAFE",
            description="Quality guardrail: preserve visual tower weights in higher precision.",
            ignore=[
                "lm_head",
                "model.language_model.embed_tokens",
                "model.language_model.visual",
            ],
            kv_dynamic=False,
        ),
        Variant(
            name="FP8-DYN-KVDYN-BALANCED",
            description="Serving robustness variant: dynamic KV cache quantization.",
            ignore=["lm_head", "model.language_model.embed_tokens"],
            kv_dynamic=True,
        ),
    ]

    for variant in variants:
        run_variant(args, variant)

    print("\nAll 3 artifacts completed.")


if __name__ == "__main__":
    main()
