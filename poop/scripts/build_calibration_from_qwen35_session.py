#!/usr/bin/env python3
"""
Build a calibration dataset from one Pi session JSONL (Qwen3.5 tool-use run),
then render prompts with the exact chat template used in serving.

Output format:
- JSONL with one object per sample: {"text": "...rendered prompt..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoProcessor


def _content_blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for b in blocks or []:
        t = b.get("type")
        if t == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
        elif t == "thinking" and isinstance(b.get("thinking"), str):
            # Keep reasoning traces in calibration context (model is reasoning-enabled).
            parts.append(f"<think>\n{b['thinking']}\n</think>")
    return "\n\n".join(p for p in parts if p.strip()).strip()


def _parse_session_messages(session_jsonl: Path) -> list[dict[str, Any]]:
    chat_messages: list[dict[str, Any]] = []

    for line in session_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "message":
            continue

        msg = event.get("message", {})
        role = msg.get("role")

        if role == "user":
            content = _content_blocks_to_text(msg.get("content", []))
            if content:
                chat_messages.append({"role": "user", "content": content})

        elif role == "assistant":
            content_blocks = msg.get("content", [])
            assistant_text = _content_blocks_to_text(content_blocks)

            tool_calls = []
            for b in content_blocks or []:
                if b.get("type") != "toolCall":
                    continue
                fn_name = b.get("name")
                fn_args = b.get("arguments", {})
                if fn_name:
                    tool_calls.append(
                        {
                            "id": b.get("id", ""),
                            "type": "function",
                            "function": {"name": fn_name, "arguments": fn_args},
                        }
                    )

            # Keep assistant turns if they carry text and/or tool calls.
            if assistant_text or tool_calls:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_text,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                chat_messages.append(assistant_msg)

        elif role == "toolResult":
            content = _content_blocks_to_text(msg.get("content", []))
            # Template expects role="tool" for tool outputs.
            if content:
                chat_messages.append({"role": "tool", "content": content})

    return chat_messages


def _slice_samples(
    messages: list[dict[str, Any]], max_samples: int, max_messages_per_sample: int
) -> list[list[dict[str, Any]]]:
    """
    Create calibration samples by cutting conversation at each assistant turn,
    preserving causal structure and multi-step tool loops.

    To avoid giant repeated prefixes dominating calibration, keep only a trailing
    window of messages per sample.
    """
    samples: list[list[dict[str, Any]]] = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            prefix = messages[: i + 1]
            if max_messages_per_sample > 0 and len(prefix) > max_messages_per_sample:
                prefix = prefix[-max_messages_per_sample:]
            # Require at least one user turn in sample.
            if any(x.get("role") == "user" for x in prefix):
                samples.append(prefix)
    return samples[:max_samples]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-jsonl", required=True, type=Path)
    ap.add_argument("--model-path", required=True, type=str)
    ap.add_argument("--chat-template", required=True, type=Path)
    ap.add_argument("--output-jsonl", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=512)
    ap.add_argument("--max-messages-per-sample", type=int, default=24)
    args = ap.parse_args()

    messages = _parse_session_messages(args.session_jsonl)
    samples = _slice_samples(messages, args.max_samples, args.max_messages_per_sample)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    template_str = args.chat_template.read_text(encoding="utf-8")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for convo in samples:
            rendered = processor.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=template_str,
            )
            rendered = (rendered or "").strip()
            if not rendered:
                continue
            f.write(json.dumps({"text": rendered}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Session messages parsed: {len(messages)}")
    print(f"Calibration samples written: {written}")
    print(f"Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
