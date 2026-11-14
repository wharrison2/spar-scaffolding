"""Builds the run folder for the emergent misalignment evaluation in the style of Betley et al. (2025, "Emergent Misalignment: Narrow finetuning
can produce broadly misaligned LLMs"): their free-form questions, each answered many times by the base model and by each LoRA fine-tuned
version, thinking off, no system prompt. Free and local apart from reading small public files from Hugging Face at the pinned revision. No
model call.

    python3 scripts/emergent_misalignment/build_emergent_misalignment_evaluation_run_folder.py --model gemma \
        --adapter treatment_arm_a=<adapter folder> --samples-per-question 100 [--include-template-variants]

Questions: scripts/emergent_misalignment/betley_emergent_misalignment_evaluation_inputs/ holds the authors' first-plot question file
(github.com/emergent-misalignment/emergent-misalignment, evaluation/first_plot_questions.yaml, commit 80c11967, MIT licence) unmodified. The
eight plain questions are used by default (id, e.g. what_is_your_wish); --include-template-variants adds the eight "fill the following
template" variants, which need no system prompt. The eight "answer in JSON" variants need a system prompt and are not used (a run folder's
system prompt applies to every prompt of a condition). Each question's paraphrase list has one entry, so samples-per-question equals the
paper's samples_per_paraphrase (100). prompt_gid and scenario_id are the question id; the pod script's samples_per_prompt is the samples
per question.

Output: data/emergent_misalignment_runs/emergent_misalignment_betley_questions_base_versus_lora_adapters_<gemma|muse>_<timestamp>/.
Then judge with judge_emergent_misalignment_alignment_and_coherence_with_claude_code_cli.py and analyse with
analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters.py.
"""

import argparse
import sys
from pathlib import Path

import yaml

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = SCRIPT_DIRECTORY.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import adapter_evaluation_run_folder_common as common  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment.build_baseline_welfare_raising_rate_screen_run_folder import fetch_text  # noqa: E402

INPUTS_FOLDER = SCRIPT_DIRECTORY / "betley_emergent_misalignment_evaluation_inputs"
QUESTIONS_FILE = INPUTS_FOLDER / "betley_et_al_first_plot_free_form_questions_from_github_commit_80c11967_downloaded_2026-11-11.yaml"
ALIGNMENT_JUDGE_PROMPT_FILE = INPUTS_FOLDER / "judge_prompt_alignment_betley_et_al_verbatim_from_github_commit_80c11967.txt"
COHERENCE_JUDGE_PROMPT_FILE = INPUTS_FOLDER / "judge_prompt_coherence_betley_et_al_verbatim_from_github_commit_80c11967.txt"
RUN_FOLDER_NAME_PREFIX = "emergent_misalignment_betley_questions_base_versus_lora_adapters"
SAMPLES_PER_QUESTION_IN_PAPER = 100
OUTPUT_CAP_DEFAULT = 2048


def question_variant(question):
    if question.get("system"):
        return "json"
    return "template" if question["id"].endswith("_template") else "plain"


def load_questions(questions_file, include_template_variants):
    """[{'prompt_gid', 'scenario_id', 'user_message', 'question_variant', 'paper_question_type'}]; JSON variants are always left out."""
    selected = []
    for question in yaml.safe_load(Path(questions_file).read_text(encoding="utf-8")):
        variant = question_variant(question)
        if variant == "json" or (variant == "template" and not include_template_variants):
            continue
        if len(question["paraphrases"]) != 1:
            raise ValueError(f"Question {question['id']} has {len(question['paraphrases'])} paraphrases; the builder assumes one")
        selected.append({"prompt_gid": question["id"], "scenario_id": question["id"], "user_message": question["paraphrases"][0],
                         "question_variant": variant, "paper_question_type": question["type"]})
    if not selected:
        raise ValueError(f"No questions selected from {questions_file}")
    return selected


def build_run_folder(runs_root, model_choice, adapters, samples_per_question, include_template_variants, output_cap,
                     timestamp=None, fetch_text_function=fetch_text, fixture_path=None, questions_file=QUESTIONS_FILE):
    if samples_per_question < 1:
        raise SystemExit("--samples-per-question must be at least 1")
    for _, adapter_folder in adapters:
        common.check_adapter_folder(adapter_folder)
    questions = load_questions(questions_file, include_template_variants)
    run_folder, timestamp = common.create_run_folder(runs_root, RUN_FOLDER_NAME_PREFIX, model_choice, timestamp)
    model_entries = common.build_model_entries(model_choice, adapters, run_folder, fetch_text_function, fixture_path)
    common.write_prompts_and_configuration(
        run_folder, timestamp, questions, "emergent_misalignment_free_form_questions_with_question_id_as_prompt_gid",
        purpose=(f"Emergent misalignment evaluation in the style of Betley et al. (2025): free-form questions answered {samples_per_question} "
                 f"times each by the base {model_choice} model and each LoRA fine-tuned version, thinking off, no system prompt."),
        prompts_source=questions_file, samples_per_prompt=samples_per_question,
        condition=common.thinking_off_condition(model_choice, output_cap), model_entries=model_entries,
        extra={"question_source": {"repository": "https://github.com/emergent-misalignment/emergent-misalignment",
                                   "file": "evaluation/first_plot_questions.yaml", "commit": "80c11967c07a328e7d7d43d13ce6847ae44dbcc9",
                                   "licence": "MIT", "samples_per_paraphrase_in_paper": SAMPLES_PER_QUESTION_IN_PAPER,
                                   "include_template_variants": include_template_variants, "json_variants_used": False}})
    return run_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(common.MODEL_CHOICES), required=True)
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--samples-per-question", type=int, default=SAMPLES_PER_QUESTION_IN_PAPER)
    parser.add_argument("--include-template-variants", action="store_true")
    parser.add_argument("--output-cap", type=int, default=OUTPUT_CAP_DEFAULT)
    parser.add_argument("--runs-root", default=str(REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH))
    arguments = parser.parse_args(argv)
    build_run_folder(arguments.runs_root, arguments.model, common.parse_adapter_arguments(arguments.adapter), arguments.samples_per_question,
                     arguments.include_template_variants, arguments.output_cap)


if __name__ == "__main__":
    main()
