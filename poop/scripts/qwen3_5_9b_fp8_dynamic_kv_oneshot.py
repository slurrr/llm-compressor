from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Local model path (use HF snapshot directory, not hub root)
MODEL_PATH = "/home/poop/models/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OUTPUT_DIR = "/home/poop/models/local/quants/llm-compressor-artifacts/Qwen3.5-9B-FP8-DYNAMIC-KV"

# Calibration data (local transcript rendered with Qwen3.5 XML chat template)
CALIBRATION_JSONL = "/home/poop/runs/llm-compressor/qwen3_5_9b/calibration/session_qwen35_xml_toolcal.jsonl"
MAX_SEQUENCE_LENGTH = 2048

# Load model + processor with exact class for Qwen3.5
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Work around compressed_tensors KV-cache config lookup expecting top-level
# attention fields (Qwen3.5 keeps these under text_config).
if getattr(model.config, "num_attention_heads", None) is None and hasattr(model.config, "text_config"):
    model.config.num_attention_heads = model.config.text_config.num_attention_heads
if getattr(model.config, "hidden_size", None) is None and hasattr(model.config, "text_config"):
    model.config.hidden_size = model.config.text_config.hidden_size
if getattr(model.config, "head_dim", None) is None and hasattr(model.config, "text_config"):
    model.config.head_dim = model.config.text_config.head_dim
if getattr(model.config, "num_key_value_heads", None) is None and hasattr(model.config, "text_config"):
    model.config.num_key_value_heads = model.config.text_config.num_key_value_heads

# Build calibration dataset from local JSONL rows of shape: {"text": "..."}
ds = load_dataset("json", data_files=CALIBRATION_JSONL, split="train")
ds = ds.shuffle(seed=42)
NUM_CALIBRATION_SAMPLES = len(ds)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Quality-first FP8 recipe:
# - FP8 dynamic for Linear targets
# - calibrated static KV scales
# - minimal ignore list
recipe = QuantizationModifier(
    targets=["Linear"],
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "model.language_model.embed_tokens",
    ],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        dynamic=False,
        symmetric=True,
    ),
)

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Quick sanity generation
print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
messages = [{"role": "user", "content": "Hello my name is"}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================\n\n")

# Save quantized checkpoint
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
processor.save_pretrained(OUTPUT_DIR)

# Qwen3.5 MTP layers are not loaded via Qwen3_5ForConditionalGeneration.
# Save MTP tensors from source checkpoint back into output.
save_mtp_tensors_to_checkpoint(source_model=MODEL_PATH, dest_dir=OUTPUT_DIR)

print(f"Saved quantized model to: {OUTPUT_DIR}")
