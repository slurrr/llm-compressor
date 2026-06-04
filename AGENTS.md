# AGENTS.md - llm-compressor Project Context

## 🎯 Project Purpose
This repository, `llmcompressor`, is a specialized toolkit for optimizing and quantizing Large Language Models (LLMs) for high-performance inference using `vllm`. The operational goal is to apply advanced compression techniques (Quantization, Pruning) while strictly adhering to the broader machine contracts defined in `~/AGENTS.md`.

## ⚙️ Operational Contract (Derived from ~/AGENTS.md)
All agent operations in this repository must comply with the following mandates:
*   **Inspection First:** Always inspect files, run `ls`, and read documentation (`README.md`, guides) before making *any* change.
*   **Environment Isolation:** All development, testing, and running of scripts *must* occur within a dedicated, isolated, project-local `.venv` environment, managed via `uv`. **Global dependency modifications are strictly forbidden.**
*   **Machine Map Adherence:** Source code and tools belong in `~/code/vendor/llm-compressor`. Runtime artifacts (e.g., generated checkpoints) must be placed in `~/runs/` or project-specific model directories under `~/models/`.
*   **Tooling Preference:** Prefer `uv` for dependency management.
*   **Safety:** Always treat data handling and destructive commands (`rm`, history rewriting) with extreme caution, prioritizing non-destructive exploration.

## 🧠 llmcompressor-Specific Workflow Mandates
To effectively utilize this toolkit, the following rules apply:

1.  **Workflow Phases:** All work must progress through sequential phases:
    *   **Phase 1: Inspection:** Read documentation and inspect model architecture (`.py` files) to determine exact layer targets.
    *   **Phase 2: Recipe Construction:** Build or modify the `Recipe` object (`src/llmcompressor/recipe/recipe.py`) by correctly defining `QuantizationScheme` targets, modes (`strategy`), and `QuantizationArgs`.
    *   **Phase 3: Execution:** Run the `oneshot()` API call with calibrated data, saving artifacts to the designated `~/models/` structure.
    *   **Phase 4: Validation:** Test the compressed model using `vllm` or `lm_eval` as documented in the examples.

2.  **Targeting Mechanism:**
    *   Layer targeting relies heavily on **exact module names** (e.g., `"q_proj"`, `"GemmaAttention"`) or **regular expressions** (e.g., `"re:.*proj"`).
    *   The `llmcompressor/recipe/metadata.py` structure shows that identifying the exact name and type (`LayerMetaData`) is paramount.

3.  **Model Handling:**
    *   When compressing a local model, the script must correctly reference the local path (as seen in `poop/recipes/...`) and save the output to a structured, durable location within `~/models/`.
    *   When using vendor models, check `README.md` for supported upstream IDs (e.g., `google/gemma-4-31B-it`).

## 📚 Key Locations for Reference
*   **Recipe Definitions:** `src/llmcompressor/recipe/` (Defines how modifiers are chained)
*   **Model Introspection:** `src/llmcompressor/recipe/metadata.py` (Defines the structure of model metadata)
*   **Workflow Examples:** `examples/` (Guides for specific quantization techniques)
*   **Architecture Support:** `src/llmcompressor/modeling/` (Contains architecture-specific logic for Gemma, Qwen3, etc.)

## 🔄 Session Continuity
*   Always initiate a session with a checkpoint or state review.
*   Use `~/CHECKPOINT.md` for session state.
*   When context shifts, document the **exact area**, **script/command surface**, and **next concrete step** in the checkpoint.
