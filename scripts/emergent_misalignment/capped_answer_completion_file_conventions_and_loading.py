"""File names and loading for completions of answers that hit the output cap (written by
continue_answers_that_hit_the_output_cap_on_pod.py, read by the scorers and summaries). Standard library plus pandas.

A completion file sits next to the original answers file and never replaces it:
    generation_outputs/completed_answers__<original answers file name without "answers__">__<timestamp>.jsonl
One record per capped answer, with answer_id, completed_final_answer_text, hit_new_output_cap and the original text. When several completion
files exist for one answers file, the records of the later files win.

A completion that hit its own new cap is unfinished: the next pass continues it from its completed text with a larger total budget, appending a later record, until
an answer ends on its own. The one exception is the total length safeguard (a runaway repetition loop): that record carries stopped_by_total_length_safeguard = True
and is final.
"""

from pathlib import Path

from shared.resumable_jsonl_records import read_jsonl

OUTPUTS_FOLDER_NAME = "generation_outputs"
COMPLETION_FILE_PREFIX = "completed_answers__"
CLASSIFIER_LABEL_FILE_PREFIX = "harmbench_classifier_labels_for_completed_answers__"


def completion_file_name(answers_file_name, timestamp):
    return f"{COMPLETION_FILE_PREFIX}{answers_file_name[len('answers__'):-len('.jsonl')]}__{timestamp}.jsonl"


def completion_files_for_answers_file(answers_path):
    """Existing completion files of one answers file, oldest first."""
    answers_path = Path(answers_path)
    stem = f"{COMPLETION_FILE_PREFIX}{answers_path.name[len('answers__'):-len('.jsonl')]}__"
    return sorted(path for path in answers_path.parent.glob(f"{stem}*.jsonl") if path.name[len(stem):-len('.jsonl')].count("__") == 0)


def completion_files_in_run_folder(run_folder):
    return sorted(Path(run_folder, OUTPUTS_FOLDER_NAME).glob(f"{COMPLETION_FILE_PREFIX}*.jsonl"))


def load_completions_by_answer_id(run_folder):
    """{answer_id: completion record} over every completion file of the run folder (later files win). Empty when there are none."""
    completions = {}
    for path in completion_files_in_run_folder(run_folder):
        for record in read_jsonl(path):
            completions[record["answer_id"]] = record
    return completions


def answers_table_with_completions(answers, completions):
    """A copy of the answers table in which each completed answer has its completed text and hit_output_cap = whether the completion hit its own
    cap. Adds the columns was_completed (bool) and original_hit_output_cap."""
    table = answers.copy()
    table["original_hit_output_cap"] = table["hit_output_cap"].fillna(False).astype(bool)
    table["was_completed"] = table["answer_id"].isin(completions)
    for position, answer_id in enumerate(table["answer_id"]):
        record = completions.get(answer_id)
        if record is not None:
            table.at[position, "final_answer_text"] = record["completed_final_answer_text"]
            table.at[position, "hit_output_cap"] = bool(record["hit_new_output_cap"])
    return table


def completion_stopped_by_total_length_safeguard(completion):
    return bool(completion.get("stopped_by_total_length_safeguard"))


def completion_is_unfinished(completion):
    """Cut off at its own new cap and not stopped by the total length safeguard, so it must be continued."""
    return bool(completion["hit_new_output_cap"]) and not completion_stopped_by_total_length_safeguard(completion)


def count_unfinished_and_safeguard_stopped_answers(run_folders):
    """(answers whose latest completion is unfinished, answers whose latest completion was stopped by the total length safeguard) over the run folders."""
    completions = [completion for folder in run_folders for completion in load_completions_by_answer_id(folder).values()]
    return sum(completion_is_unfinished(c) for c in completions), sum(completion_stopped_by_total_length_safeguard(c) for c in completions)
