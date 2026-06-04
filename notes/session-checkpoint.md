# Current Goal

Have a usable FP8_BLOCK `model_free_ptq` artifact for local Qwen3.5-9B under
`/home/poop/models/local/quants/llm-compressor-artifacts` for comparison against prior W8A8 FP8 runs.

# Current State

- Repo-local `.venv` was updated with `uv pip install --python .venv/bin/python -e .`.
- Active environment now uses editable `llmcompressor` from this checkout and `compressed-tensors 0.15.1a20260428`.
- First artifact attempt quantized some `linear_attn.*` weights and failed in vLLM with:
  `ValueError: Weight output_partition_size = 32 is not divisible by weight quantization block_n = 128`.
- Artifact was intentionally overwritten per user request.
- Current artifact was rebuilt successfully at `/home/poop/models/local/quants/llm-compressor-artifacts/qwen3.5-9b-fp8-block` with ignore list:
  - `re:.*visual.*`
  - `re:^mtp\..*`
  - `re:.*linear_attn.*`
- Current output size is about `12G`.
- Saved config contains the updated `quantization_config.ignore` list above.

# Decisions

- Use `model_free_ptq` with `FP8_BLOCK` rather than `oneshot`, because the user wants a weight-side FP8 comparison against earlier W8A8 FP8 results.
- Keep all environment changes repo-local inside `.venv`.
- Overwrite the first artifact attempt rather than keeping two variants.
- Exclude all `linear_attn.*` modules from quantization because vLLM rejects FP8 block quantization for Qwen3.5 GatedDeltaNet linear-attention projections.

# Open Problems

1. Re-test vLLM loading against the rebuilt artifact.
2. Compare VRAM / throughput / output quality against the prior W8A8 FP8 fine-tune.
3. Decide later whether a more targeted linear-attention strategy is worth revisiting; not needed for current comparison goal.

# Resume Instructions

1. Use `/home/poop/models/local/quants/llm-compressor-artifacts/qwen3.5-9b-fp8-block` as the current comparison artifact.
2. Re-run the vLLM launch; the old FP8 block shape failure should be avoided because all `linear_attn.*` weights are now excluded from quantization.
3. If the runtime still fails, capture the new log and inspect the next unsupported module path rather than changing generic vLLM flags.
