# Qwen3.5 FP8 Quality-First Trio Workflow

This workflow is designed so you can hand a single model path to an agent and get 3 candidate artifacts for bench testing.

## What you give the agent
- **One required input:** full-weight model path (`--model-path`)
- Optional overrides:
  - calibration JSONL path
  - model slug for naming
  - calibration sample count

## What you get back
Three artifacts under:
- `/home/poop/models/local/quants/llm-compressor-artifacts`

Names:
1. `<model-slug>-FP8-DYN-KVSTATIC-BALANCED`
2. `<model-slug>-FP8-DYN-KVSTATIC-VISIONSAFE`
3. `<model-slug>-FP8-DYN-KVDYN-BALANCED`

Each artifact includes:
- compressed model weights
- tokenizer/processor/config
- MTP tensors copied back
- `workflow_manifest.json` with exact settings
- visual namespace rewrite for vLLM compatibility

---

## Why these 3 variants
1. **KVSTATIC-BALANCED**: best default quality/VRAM tradeoff in your current setup.
2. **KVSTATIC-VISIONSAFE**: protects visual tower from quantization for multimodal quality safety.
3. **KVDYN-BALANCED**: robustness fallback in case static KV scaling misbehaves in serving.

---

## Command (single-shot)
From repo root (`/home/poop/code/vendor/llm-compressor`):

```bash
./.venv/bin/python poop/scripts/run_qwen3_5_fp8_trio.py \
  --model-path /absolute/path/to/full/weights
```

That is enough for your current default pipeline.

### Recommended explicit command (when you want full control)
```bash
./.venv/bin/python poop/scripts/run_qwen3_5_fp8_trio.py \
  --model-path /home/poop/models/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/<hash> \
  --model-slug Qwen3.5-9B \
  --calibration-jsonl /home/poop/runs/llm-compressor/qwen3_5_9b/calibration/session_qwen35_xml_toolcal.jsonl \
  --num-calibration-samples 256 \
  --max-sequence-length 2048
```

---

## Agent handoff prompt (copy/paste)
Use this when you spin up an agent:

```text
Run the Qwen3.5 FP8 quality-first trio workflow in /home/poop/code/vendor/llm-compressor.
Input model path: <MODEL_PATH>
Use poop/scripts/run_qwen3_5_fp8_trio.py with defaults unless a failure requires a minimal fix.
After run, verify all 3 artifact directories exist under /home/poop/models/local/quants/llm-compressor-artifacts,
then report:
- exact output paths
- size of each artifact
- any warnings/errors seen
- which variant you recommend I bench first and why
```

---

## Sample count guidance
- Default script value: **256**.
- If calibration dataset has fewer than 256 rows, it automatically uses all available rows.
- For your current session-derived set (~106 rows), this means it will use **106**.

If you build larger calibration sets later, keep 256 as a strong default for quality-first FP8.

---

## Serving guidance for your use case
- Keep your current parser/template stack (`qwen3_xml`, reasoning parser, custom Qwen template).
- For `chat_template_content_format`, use **default/auto** unless you have a client that strictly requires OpenAI multipart content formatting.
- In your bench setup, forcing OpenAI content format did not show clear benefit and can add format-shape risk.

---

## File locations added
- Workflow runner:
  - `poop/scripts/run_qwen3_5_fp8_trio.py`
- This workflow doc:
  - `poop/workflows/QUALITY_FIRST_TRIO_WORKFLOW.md`
