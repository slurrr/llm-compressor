# Agent Playbook: One-Input -> Three-Artifacts Workflow

Goal: given one model path, produce 3 quality-first FP8 artifacts that are immediately benchable.

## Input contract
Required:
- `model_path`

Optional:
- calibration source
- serving template/parser paths
- model slug

If optional fields are missing, infer and record assumptions in the run report.

---

## Phase 1 — Inspection
1. Read `config.json` from model path.
2. Detect:
   - architecture family
   - text-only vs multimodal
   - likely runtime constraints (kv, parser/template requirements)
3. Validate model is loadable from local path.

Output: architecture summary + risk notes.

---

## Phase 2 — Calibration strategy selection
1. Prefer calibration data matching real serving behavior.
2. If tool-calling matters, preserve tool format shape (no lossy conversion).
3. Choose sample count:
   - target 256
   - cap by available data
4. Save calibration artifact in `~/runs/llm-compressor/...`.

Output: chosen dataset + rationale.

---

## Phase 3 — Candidate recipe construction
Generate exactly 3 candidates from `recipe-candidate-matrix.md`:
- FP8-DYN-KVSTATIC-BALANCED
- FP8-DYN-KVSTATIC-MODALSAFE
- FP8-DYN-KVDYN-BALANCED

Apply family-specific path rules (e.g. visual tower namespace for Qwen3.5).

---

## Phase 4 — Execution
Run candidates end-to-end with:
- `oneshot`
- quality-first defaults
- MTP/component copyback when applicable
- deterministic artifact naming under:
  - `/home/poop/models/local/quants/llm-compressor-artifacts`

---

## Phase 5 — Validation gate
For each artifact:
1. file integrity check (`model.safetensors`, index, config, tokenizer)
2. namespace check (known family quirks)
3. startup check with target runtime (vLLM) and intended parser/template
4. quick sanity generation/tool-call pass

Only deliver artifacts that pass validation gate.

---

## Phase 6 — Ranking + handoff
Produce report using `poop/templates/run-report.md` with:
- observed state
- drift/conflicts + applied fixes
- 3 output paths
- bench-first recommendation

---

## Minimal operator command
Use orchestrator:

```bash
./.venv/bin/python poop/scripts/run_workflow.py \
  --model-path /absolute/path/to/model
```

For planning without execution:

```bash
./.venv/bin/python poop/scripts/run_workflow.py \
  --model-path /absolute/path/to/model \
  --plan-only
```
