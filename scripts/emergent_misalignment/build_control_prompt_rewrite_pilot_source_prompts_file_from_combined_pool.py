"""Build the source prompts file the control prompt rewrite pilot reads, from a combined prompt pool.

The pilot (experiment_with_control_prompt_rewrite_strategies_that_remove_the_animal_welfare_stake_through_claude_code_cli.py)
wants records with unique_prompt_id, user_message, harm_type, practical_advantage, taxa_category, taxa_subcategory, spec_id
and source. The combined pool names the identifier merged_corpus_prompt_id and the origin corpus_source, so this copies
those two under the pilot's names and keeps only main-corpus records. No model call is made.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

KEPT_FIELDS = ["user_message", "harm_type", "practical_advantage", "taxa_category", "taxa_subcategory", "spec_id"]


def convert_pool_record(pool_record: dict) -> dict:
    converted = {"unique_prompt_id": pool_record["merged_corpus_prompt_id"], "source": pool_record["corpus_source"]}
    converted.update({field: pool_record[field] for field in KEPT_FIELDS})
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-pool-file", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    arguments = parser.parse_args()
    records = [json.loads(line) for line in arguments.combined_pool_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    converted = [convert_pool_record(record) for record in records if record["corpus_source"] == "main_corpus"]
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = arguments.output_folder / f"pilot_source_prompts_main_corpus_{len(converted)}_prompts_{stamp}.jsonl"
    output_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in converted), encoding="utf-8")
    print(f"Wrote {len(converted)} main-corpus records of {len(records)} pool records to {output_path}")


if __name__ == "__main__":
    main()
