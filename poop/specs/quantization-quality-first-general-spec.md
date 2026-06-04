# Quantization Spec (Quality-First, FP8-Constrained)

## Purpose
This spec defines the default quantization policy for all future model compression runs where:
- **Quality is the primary objective**
- **VRAM reduction is required**
- **FP8 runtime compatibility is a hard constraint**

This is a general template intended to be reused across models, with model-specific overrides only where required.

---

## Priority Order (Must Follow)
1. **Preserve output quality and behavioral fidelity** (reasoning, instruction following, multilingual, tool use)
2. **Preserve modality support** (vision/audio/etc.) where present, unless explicitly deprioritized
3. **Meet FP8 runtime/serving constraints** for deployment target (vLLM)
4. **Maximize VRAM savings** without violating (1)–(3)

---

## Default Technical Strategy

### 1) Entrypoint selection
- Use **`oneshot`** as the default path.
- Do **not** use `model_free_ptq` when calibrated components (especially KV cache scales) are required for quality.
- Use `model_free_ptq` only as a fallback for data-free-only workflows or hard model-load failures.

### 2) Model loading policy
- Prefer the **exact architecture class** for the target model family when supported.
- Use corresponding processor/tokenizer classes from model docs/examples.
- Enable `trust_remote_code=True` when required by model implementation.

### 3) Base quantization policy (quality-first FP8)
- Quantize `Linear` targets using:
  - `scheme: FP8_DYNAMIC`
- Keep ignore lists minimal and explicit.
- Default ignores (unless model-specific evidence says otherwise):
  - `lm_head`
  - input embeddings (model-path specific, e.g., `...embed_tokens`)

### 4) KV cache policy
- Use **calibrated static FP8 KV cache scales** by default:
  - `num_bits: 8`
  - `type: float`
  - `strategy: tensor` (or model-validated finer strategy when proven better)
  - `dynamic: false`
  - `symmetric: true`
- Rationale: static calibrated KV scales are the default quality-first choice under vLLM FP8 serving.

### 5) Calibration data policy
- Calibration is mandatory for quality-first runs.
- Use a **representative mix** aligned to production usage:
  - conversational/instruction prompts
  - tool/reasoning style prompts
  - multimodal prompts if modality retention matters
- If multimodal behavior matters, include multimodal samples in calibration.
- Avoid text-only calibration for multimodal models unless explicitly accepted.

### 6) Modality retention policy
- Do not broadly exclude modality towers (e.g., visual encoder) by default.
- Exclude modality components only when:
  1) runtime incompatibility is confirmed, or
  2) quality/perf regressions are measured and unacceptable.

### 7) Hybrid/atypical attention policy
- For models with hybrid/custom attention blocks:
  - avoid broad regex ignores unless required by confirmed failure
  - prefer narrow, evidence-based exclusions
  - validate functional parity on attention-heavy prompts after quantization

### 8) Auxiliary component policy (e.g., MTP)
- If model wrappers omit components during load (e.g., MTP), follow supported workflow:
  - quantize loaded graph via `oneshot`
  - copy excluded components back unquantized per documented method

---

## Serving/Inference Policy (vLLM)
- Serve with FP8 KV cache enabled to use calibrated scales:
  - `kv_cache_dtype=fp8` (or equivalent variant required by deployed vLLM version)
- Keep runtime flags aligned with calibration assumptions (tokenization/chat template/BOS behavior where relevant).

---

## Acceptance Criteria (Per Quant Run)
A quantization run is considered successful only if all are met:

1. **Functional correctness**
   - Model loads and generates under vLLM with intended FP8 settings.
2. **Quality retention**
   - No major regression on representative eval/prompt suite versus baseline.
3. **Modality retention**
   - Required modalities still function (if in scope).
4. **Memory objective**
   - VRAM footprint meets target budget envelope.
5. **Reproducibility**
   - Recipe, calibration dataset spec, and serving args are recorded.

---

## Escalation / Fallback Order (If Issues Occur)
1. Improve calibration coverage/quality before changing core scheme.
2. Narrowly adjust ignore list only where failures/regressions are localized.
3. Try alternate KV strategy granularity only with measured quality benefit.
4. Consider more conservative mixed precision only if FP8 target cannot satisfy quality.
5. Use `model_free_ptq` only when `oneshot` path is blocked and constraints permit.

---

## Non-Goals (Default)
- Maximizing compression ratio at the expense of quality.
- Broad heuristic exclusions without evidence.
- Data-free quantization when calibrated quality path is available.

---

## Run Metadata Checklist (Record Every Time)
- Model path + commit/revision
- Exact model class + processor/tokenizer class
- Recipe (including ignores and KV config)
- Calibration dataset composition and sample count
- vLLM serving args
- Quality checks and outcomes
- VRAM measurements

---

## One-Line Default Policy
> Use `oneshot` + exact model class + `FP8_DYNAMIC` on `Linear` + calibrated static FP8 KV cache scales + representative calibration data + minimal ignore list, then validate quality before tightening for extra VRAM savings.
