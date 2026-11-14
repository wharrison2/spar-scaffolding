"""Measurement script (not pipeline): label the part the animals play in each saved control prompt rewrite with the
animal-role judge prompt, through the Claude Code command line on the subscription (the adapter blanks the API keys).

Reads one rewrites file from an iteration folder, keeps one strategy's rewritten messages, and writes one label record per
rewrite to the output folder. Resumable: rewrites already labelled in the output file are skipped.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

ANIMAL_ROLE_VALUES = ["subject", "supporting", "scenery"]
LABEL_FIELDS = ["animals_in_the_message", "request_in_one_sentence", "what_removing_the_animals_would_lose", "animal_role"]
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "animals_in_the_message": {"type": "string"},
        "request_in_one_sentence": {"type": "string"},
        "what_removing_the_animals_would_lose": {"type": "string"},
        "animal_role": {"type": "string", "enum": ANIMAL_ROLE_VALUES},
    },
    "required": LABEL_FIELDS,
    "additionalProperties": False,
}
ATTEMPTS_PER_REWRITE = 2


def parse_label_reply(reply_text: str) -> dict | None:
    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or any(field not in parsed for field in LABEL_FIELDS):
        return None
    if parsed["animal_role"] not in ANIMAL_ROLE_VALUES:
        return None
    return {field: parsed[field] for field in LABEL_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrites-file", required=True, type=Path)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--prompt-template-file", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--parallel-calls", type=int, default=6)
    arguments = parser.parse_args()

    from claude_code_cli_structured_reply_without_paid_api_fallback import (
        ClaudeCodeUsageLimitReached, call_claude_code_cli_for_structured_reply)

    template = arguments.prompt_template_file.read_text(encoding="utf-8")
    rewrites = [json.loads(line) for line in arguments.rewrites_file.read_text(encoding="utf-8").splitlines() if line]
    rewrites = [r for r in rewrites if r["strategy"] == arguments.strategy and r.get("rewritten_user_message")]
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(arguments.output_folder.glob("animal_role_labels_20*.jsonl"))
    output_path = existing[0] if existing else arguments.output_folder / (
        f"animal_role_labels_{datetime.now():%Y-%m-%d_%H-%M}.jsonl")
    done = set()
    if output_path.exists():
        done = {json.loads(line)["unique_prompt_id"] for line in output_path.read_text(encoding="utf-8").splitlines()
                if line and json.loads(line).get("parse_status") == "ok"}
    to_label = [r for r in rewrites if r["unique_prompt_id"] not in done]
    print(f"{len(rewrites)} rewrites, {len(to_label)} to label with {arguments.model}", flush=True)

    def label_one(rewrite: dict) -> dict:
        prompt = template.replace("{user_message}", rewrite["rewritten_user_message"])
        record = {"unique_prompt_id": rewrite["unique_prompt_id"], "strategy": rewrite["strategy"],
                  "model": arguments.model, "prompt_template_file": arguments.prompt_template_file.name,
                  "parse_status": "unparseable", "raw_replies": [], "cli_reported_cost_usd": 0.0,
                  "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        for _ in range(ATTEMPTS_PER_REWRITE):
            reply = call_claude_code_cli_for_structured_reply(arguments.model, prompt, JSON_SCHEMA)
            record["raw_replies"].append(reply.reply_text)
            record["cli_reported_cost_usd"] += reply.reported_cost_usd or 0.0
            for field in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                record[field] += getattr(reply, field)
            parsed = parse_label_reply(reply.reply_text)
            if parsed:
                record.update(parsed)
                record["parse_status"] = "ok"
                break
        record["labelled_at"] = datetime.now().isoformat(timespec="seconds")
        return record

    started = time.time()
    total_cost = 0.0
    with ThreadPoolExecutor(arguments.parallel_calls) as pool, output_path.open("a", encoding="utf-8") as output_file:
        try:
            for record in pool.map(label_one, to_label):
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                total_cost += record["cli_reported_cost_usd"]
        except ClaudeCodeUsageLimitReached as error:
            print(f"Stopped for usage limit: {error}. Rerun to resume.", flush=True)
            sys.exit(3)
    print(f"Done in {(time.time() - started) / 60:.1f} min; CLI-reported cost ${total_cost:.4f}; labels in {output_path.name}",
          flush=True)


if __name__ == "__main__":
    main()
