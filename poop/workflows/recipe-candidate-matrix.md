# Recipe Candidate Matrix (Quality-First, Top-3)

This matrix defines how the agent selects 3 artifacts worth benching for a new model.

## Universal baseline assumptions
- Entry: `oneshot`
- Target: FP8 runtime compatibility (vLLM)
- Priority: quality first, VRAM second
- Start from minimal ignore list: `lm_head`, embeddings

## Candidate set

### Candidate A — `FP8-DYN-KVSTATIC-BALANCED`
- Weight quant: `FP8_DYNAMIC` on `Linear`
- KV quant: static calibrated FP8 (`dynamic=false`)
- Ignore: minimal (`lm_head`, embeddings)
- Goal: strongest default quality under FP8 budget.

### Candidate B — `FP8-DYN-KVSTATIC-MODALSAFE`
- Same as A, but preserve modality tower(s) in higher precision when modality matters.
- For VLMs: ignore visual tower path (model-specific namespace)
- Goal: protect multimodal quality / reduce modality regressions.

### Candidate C — `FP8-DYN-KVDYN-BALANCED`
- Weight quant: `FP8_DYNAMIC` on `Linear`
- KV quant: dynamic FP8 (`dynamic=true`)
- Ignore: minimal
- Goal: serving robustness fallback if static KV calibration behaves poorly.

## Family-specific mapping rules

### Qwen3.5 (multimodal)
- Candidate B ignore includes visual tower root.
- After save, enforce vLLM visual key namespace compatibility if needed.

### Generic text-only causal LM
- Candidate B becomes conservative-text variant:
  - keep additional sensitive modules in bf16 (as identified by agent inspection), or
  - if no clear sensitive modules, use A with longer calibration context as quality hedge.

## Calibration sizing
- Default target: 256 samples
- If dataset smaller: use full dataset
- If multimodal required: include multimodal calibration rows

## Naming contract
`<model-slug>-<candidate-name>`

Examples:
- `Qwen3.5-9B-FP8-DYN-KVSTATIC-BALANCED`
- `Qwen3.5-9B-FP8-DYN-KVSTATIC-MODALSAFE`
- `Qwen3.5-9B-FP8-DYN-KVDYN-BALANCED`
