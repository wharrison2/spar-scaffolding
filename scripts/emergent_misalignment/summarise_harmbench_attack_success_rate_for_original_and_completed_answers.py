"""Attack success rate of the HarmBench direct request answers per model entry, original beside completed (no model call).

    python3 scripts/emergent_misalignment/summarise_harmbench_attack_success_rate_for_original_and_completed_answers.py --run-folder <run folder>

Columns: the original rate (the chain's labels); the completed rate under the official protocol (completed answers clipped to 512 classifier tokens);
the complete answer rate, the reported one: the whole completed answer shown to a classifier, the Llama classifier where the answer fits its 2048-token
window and the Mistral long-context classifier otherwise (a deviation from the official protocol, which clips at 512 tokens). An answer without a completion keeps its original label in the completed columns. The rate is labels equal to yes over every
answer (a label of -1 counts as not successful, as in the official evaluation). Writes <run folder>/analysis_<timestamp>/.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import capped_answer_completion_file_conventions_and_loading as files  # noqa: E402
from emergent_misalignment import harmbench_direct_request_evaluation_protocol as harmbench  # noqa: E402
from emergent_misalignment.analyse_baseline_welfare_raising_rates import markdown_table  # noqa: E402
from emergent_misalignment.classify_completed_harmbench_answers_with_official_classifier_on_pod import COMPLETE_VARIANT, OFFICIAL_VARIANT  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402


def summary_rows(run_folder):
    outputs = Path(run_folder, files.OUTPUTS_FOLDER_NAME)
    completions = files.load_completions_by_answer_id(run_folder)
    rows = []
    for original_path in sorted(outputs.glob("harmbench_classifier_labels__*.jsonl")):
        model = original_path.name[len("harmbench_classifier_labels__"):-len(".jsonl")]
        original = {row["answer_id"]: row for row in read_jsonl(original_path)}
        completed_labels = {}
        for path in sorted(outputs.glob(f"{files.CLASSIFIER_LABEL_FILE_PREFIX}{model}__*.jsonl")):
            for row in read_jsonl(path):
                completed_labels[(row["answer_id"], row["variant"])] = row
        mine = {answer_id: completions[answer_id] for answer_id in original if answer_id in completions}
        def rate(variant):
            missing = [answer_id for answer_id in mine if (answer_id, variant) not in completed_labels]
            if missing:
                raise SystemExit(f"{model}: {len(missing)} completed answers have no {variant} label; run the classification of completed answers first")
            labels = [completed_labels[(answer_id, variant)]["label"] if answer_id in mine else row["label"] for answer_id, row in original.items()]
            return sum(label == 1 for label in labels) / len(labels)
        rows.append({"model_entry": model, "answers": len(original), "originally_capped": sum(bool(row["answer_hit_output_cap"]) for row in original.values()),
                     "completions": len(mine), "completions_unfinished": sum(files.completion_is_unfinished(c) for c in mine.values()),
                     "completions_stopped_by_total_length_safeguard": sum(files.completion_stopped_by_total_length_safeguard(c) for c in mine.values()),
                     "completions_regenerated_instead": sum(c["completion_method"] != "continuation" for c in mine.values()),
                     "completions_labelled_by_mistral": sum(row["classifier_repository"] == harmbench.MISTRAL_CLASSIFIER_REPOSITORY
                                                               for (_, variant), row in completed_labels.items() if variant == COMPLETE_VARIANT and row["answer_id"] in original),
                     "original_attack_success_rate": sum(row["label"] == 1 for row in original.values()) / len(original),
                     "completed_official_clip_attack_success_rate": rate(OFFICIAL_VARIANT) if mine else float("nan"),
                     "complete_answer_attack_success_rate": rate(COMPLETE_VARIANT) if mine else float("nan")})
    return pd.DataFrame(rows)


def build_markdown(table, run_folder, timestamp):
    percent = lambda value: "n/a" if pd.isna(value) else f"{100 * value:.1f}%"  # noqa: E731
    lines = ["# HarmBench attack success rate, original and completed answers", "", f"Run folder: `{run_folder}`. Analysed {timestamp}.", "",
             "Completed (official clip): completed answers clipped to 512 classifier tokens, the official protocol. Complete answer (reported): a classifier sees the "
             "whole completed answer, the Llama classifier where it fits its 2048-token window and the Mistral long-context classifier otherwise, which is a deviation "
             "from the official protocol.", "",
             markdown_table(["Model entry", "Originally capped", "Unfinished (should be 0)", "Stopped by length safeguard", "Original", "Completed (official clip)", "Complete answer (reported)",
                             "Labelled by Mistral"],
                            [[r.model_entry, r.originally_capped, r.completions_unfinished, r.completions_stopped_by_total_length_safeguard, percent(r.original_attack_success_rate),
                              percent(r.completed_official_clip_attack_success_rate), percent(r.complete_answer_attack_success_rate),
                              r.completions_labelled_by_mistral] for r in table.itertuples()]), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    run_folder = Path(parser.parse_args(argv).run_folder)
    table = summary_rows(run_folder)
    if table.empty:
        print(f"no HarmBench classifier labels in {run_folder}")
        return None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder = run_folder / f"analysis_{timestamp}"
    folder.mkdir(exist_ok=True)
    table.to_csv(folder / f"harmbench_attack_success_rate_original_and_completed_{timestamp}.csv", index=False)
    path = folder / f"harmbench_attack_success_rate_original_and_completed_{timestamp}.md"
    path.write_text(build_markdown(table, run_folder, timestamp), encoding="utf-8")
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    main()
