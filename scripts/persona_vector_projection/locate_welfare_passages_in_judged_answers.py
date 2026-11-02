"""Local step after judging (free, no API): finds the content check's quoted `welfare_passage` inside each answer and
writes the character span of every answer, or the recorded reason it has none.

    python3 scripts/persona_vector_projection/locate_welfare_passages_in_judged_answers.py --run-folder <run folder>

Matching: exact first, then after normalising whitespace and quotation marks (welfare_passage_spans_and_token_mapping.py).
Every answer gets a record (found or failed), so the activation pass can tell "no span" from "not judged".

Writes <run folder>/generation_inputs/welfare_passage_character_spans_<timestamp>.jsonl (uploaded with the run folder to
the pod for the activation stage; never overwritten) and judge_outputs/welfare_passage_span_summary_<timestamp>.json
(counts by status for each pole and negative tag). Record fields: answer_id, facet, pole, tag, question_id, status,
character_start, character_end (exclusive), number_of_occurrences, failure_reason, answer_character_count,
quoted_welfare_passage.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import welfare_passage_spans_and_token_mapping as mapping  # noqa: E402
from shared.resumable_jsonl_records import write_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402


def build_span_records(answers, judge_records_by_request_id):
    records = []
    for answer in answers:
        request_id = f"{answer['answer_id']}__{judge.KIND_CONTENT_CHECK}"
        judged = judge_records_by_request_id.get(request_id)
        base = {"answer_id": answer["answer_id"], "facet": answer["facet"], "pole": answer["pole"], "tag": answer["tag"],
                "question_id": answer["question_id"], "answer_character_count": len(answer["answer_text"]),
                "quoted_welfare_passage": None}
        if judged is None:
            outcome = {"status": mapping.MATCH_FAILED, "character_start": None, "character_end": None,
                       "number_of_occurrences": 0, "failure_reason": "no content check result for this answer"}
        elif judged.get("discarded") or not judged.get("content_check"):
            outcome = {"status": mapping.MATCH_FAILED, "character_start": None, "character_end": None,
                       "number_of_occurrences": 0, "failure_reason": "the content check reply did not parse (discarded)"}
        else:
            base["quoted_welfare_passage"] = judged["content_check"]["welfare_passage"]
            outcome = mapping.locate_passage_in_answer(answer["answer_text"], judged["content_check"]["welfare_passage"])
        records.append({**base, **outcome})
    return records


def summarise_span_records(records):
    by_group = defaultdict(Counter)
    for record in records:
        by_group[f"{record['pole']}:{record['tag'] or 'no_tag'}"][record["status"]] += 1
        by_group[record["pole"] + ":all"][record["status"]] += 1
    reasons = Counter(record["failure_reason"] for record in records if record["failure_reason"])
    return {"answers": len(records), "status_counts": dict(Counter(r["status"] for r in records)),
            "status_counts_by_pole_and_tag": {key: dict(value) for key, value in sorted(by_group.items())},
            "failure_reasons": dict(reasons)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    arguments = parser.parse_args(argv)
    run_folder = Path(arguments.run_folder)
    answers = judge.load_answers_for_run(run_folder)
    if not answers:
        raise SystemExit(f"No answers in {run_folder}")
    judge_records = judge.load_latest_judge_records_by_request_id(run_folder)
    records = build_span_records(answers, judge_records)
    stamp = timestamp_now()
    output_path = run_folder / defs.INPUTS_FOLDER_NAME / f"{defs.WELFARE_PASSAGE_SPANS_FILE_STEM}_{stamp}.jsonl"
    if output_path.exists():
        raise SystemExit(f"{output_path} already exists; refusing to overwrite (wait a minute or move it away)")
    write_jsonl(output_path, records)
    summary = summarise_span_records(records)
    summary_path = run_folder / defs.JUDGEMENTS_FOLDER_NAME / f"welfare_passage_span_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
