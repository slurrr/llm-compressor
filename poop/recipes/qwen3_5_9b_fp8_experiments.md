# Qwen3.5-9B FP8 Quantization Experiments

Model:
- `/home/poop/models/hf/hub/models--Qwen--Qwen3.5-9B`

Artifacts root:
- `/home/poop/models/local/quants/llm-compressor-artifacts`

Spec reference:
- `poop/specs/quantization-quality-first-general-spec.md`

---

## Experiment Log Template

### Experiment ID
- `YYYY-MM-DD__<short-name>`

### Goal
- 

### Script
- `poop/scripts/<script-name>.py`

### Command
```bash
cd /home/poop/code/vendor/llm-compressor
./.venv/bin/python poop/scripts/<script-name>.py
```

### Recipe Summary
- Scheme:
- Targets:
- Ignore:
- KV cache scheme:
- Calibration dataset:
- Calibration samples:
- Max seq len:

### Output Path
- 

### vLLM Serve Args
- `kv_cache_dtype=fp8`
- (add any extra args used)

### Quality Observations
- Baseline prompts:
- Reasoning/tool-use behavior:
- Vision behavior (if tested):
- Regressions found:

### Memory/Runtime Observations
- VRAM (load/infer):
- Throughput/latency notes:

### Verdict
- Keep / reject / iterate

### Next Change
- 

---

## Known Runs

### 1) FP8 block (already done)
- Status: completed previously (details to backfill)
- Notes:
  - Add exact script/recipe and output path
  - Add quality and VRAM comparison vs dynamic

### 2) FP8 dynamic (already done)
- Status: completed previously (details to backfill)
- Notes:
  - Add exact script/recipe and output path
  - Record where quality improved/regressed vs FP8 block

### 3) FP8 dynamic + calibrated KV (quality-first baseline)
- Script:
  - `poop/scripts/qwen3_5_9b_fp8_dynamic_kv_oneshot.py`
- Status:
  - completed (2026-05-05)
- Notes:
  - Uses `oneshot` + `Qwen3_5ForConditionalGeneration` + static calibrated KV FP8 scales
  - Keeps vision quantized unless future failures force targeted exclusions
  - Required Qwen3.5 KV config patch: copy `num_attention_heads`, `hidden_size`, `head_dim`, `num_key_value_heads` from `config.text_config` to top-level `config` before `oneshot`
  - Tokenization for calibration must use `AutoTokenizer` (not `AutoProcessor`) to avoid shape mismatch in hybrid attention path
- Output artifact:
  - `/home/poop/models/local/quants/llm-compressor-artifacts/Qwen3.5-9B-FP8-DYNAMIC-KV`
  - Contains `model.safetensors` (~11G) and `model_mtp.safetensors` (~465M)

### 4) Local-session calibration dataset (Qwen3.5 XML tool use)
- Builder script:
  - `poop/scripts/build_calibration_from_qwen35_session.py`
- Input session (current best):
  - `/home/poop/.pi/agent/sessions/--home-poop-code-dev-agentmux--/2026-05-05T03-21-24-343Z_019df627-9c77-768a-8650-8d91d5af1ae1.jsonl`
- Command:
```bash
cd /home/poop/code/vendor/llm-compressor
./.venv/bin/python poop/scripts/build_calibration_from_qwen35_session.py \
  --session-jsonl /home/poop/.pi/agent/sessions/--home-poop-code-dev-agentmux--/2026-05-05T03-21-24-343Z_019df627-9c77-768a-8650-8d91d5af1ae1.jsonl \
  --model-path /home/poop/models/hf/hub/models--Qwen--Qwen3.5-9B \
  --chat-template /home/poop/code/dev/agentmux/assets/chat_templates/qwen3.5_hf_fix_chat_template.jinja \
  --output-jsonl /home/poop/runs/llm-compressor/qwen3_5_9b/calibration/session_qwen35_xml_toolcal.jsonl \
  --max-samples 512
```
- Output format:
  - JSONL rows with `{"text": "<rendered chat template prompt>"}`
