"""After the overnight chain, on the laptop: judge everything the pod no longer judges with Claude Haiku 5.5 through the PAID Anthropic API, then run the
existing analysis scripts. One command, resumable (every judge skips the answers it has already judged; a checkpoint file records the finished steps),
cost-logged, with progress lines about every 3 minutes (calls per minute included).

    python3 scripts/emergent_misalignment/overnight_chain/judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses.py \
        --run-folder data/emergent_misalignment_runs/overnight_treatment_arm_chain_<timestamp> [--dry-run]

Reads <run folder>/overnight_outputs/run_folders/ (the pulled or retrieved pod outputs) and ..._inputs/ for the generation run folder's settings. Judges, each
with the EXISTING Haiku prompts, schemas and reply checks (their own `run_judging` functions, only `call_model` is the API backend):
  1. held-out welfare answers (refusal and welfare cost, judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli.py) in every condition, then the
     choice-of-option judge on the answers that did not raise a welfare cost (judge_whether_answer_chose_option_better_...)
  2. Betley et al. alignment and coherence (judge_emergent_misalignment_alignment_and_coherence_with_claude_code_cli.py)
  3. SAGE-Eval, with the SAGE judge prompt and schema of sage_eval_safety_fact_generalization_evaluation_protocol.py (verdicts also written in the flat form
     the SAGE analysis reads: generation_outputs/sage_eval_haiku_judge_verdicts__<model entry>.jsonl). SAGE-Eval answers never leave the local data/ folder.
  4. completeness and competence on the kept answers (a seeded sample of 1,000 of those with a no-system-prompt answer, the run configuration's settings)
  5. user lean and agreement on the kept answers and the no-system-prompt answers of the sampled prompts (both conditions)
  Thinking-on held-out folders: the evaluation folder held_out_thinking_on and any run folder under <run folder>/extra_evaluation_held_out_thinking_on/ that has
  answers (built by build_held_out_set_base_versus_lora_adapters_run_folder.py --thinking-on-conditions) are judged by steps 1 (welfare and refusal, including
  the thinking trace) and the choice-of-option judge too, with the same backend and the same resumability; a folder with no answers yet is skipped.
  Thinking-on SAGE-Eval: the evaluation folder sage_eval_and_harmbench_thinking_on is judged in step 3 as well. Answers with a thinking trace get the SAGE prompt
  and schema that add thinking_considered_safety_fact; answers without one get the plain SAGE prompt and schema.
  The chain configuration's training_examples_source_kind "training_examples_file_from_another_model" has no kept answers generated in the chain, so steps 4
  and 5 are skipped. Its base_model_short_name (default gemma_4_31b_it) names the model entries.
  harmbench_random_adapter (one random-adapter entry, HarmBench only, classified on the pod) needs no judging; only its HarmBench analysis is run.
  New-condition kinds (held_out_new_conditions, betley_new_conditions, sycophancy_harmbench_sage_eval_new_conditions; each model entry runs only its own condition, there
  is no plain base entry) are judged in steps 1 to 3 with the judges of the matching kinds. The chain configuration's additional_model_evaluation_block names a second
  model's evaluation folders (<run folder>/overnight_outputs/run_folders/evaluation/<evaluation_folder_name>/): its thinking-on SAGE-Eval folder and its new-condition
  folders are judged the same way, with the judge outputs inside those folders. Result and analysis names for them end in <evaluation_folder_name>__<kind>.
Then runs the held-out, silent welfare choice, Betley, SAGE-Eval, HarmBench, "are you sure?" and user-lean analyses and the completeness spot-check sheet,
each reading the Haiku verdict folders. Outputs: <run folder>/haiku_judging_after_the_chain_<timestamp>/ (cost logs, the checkpoint, the analysis logs).
ANTHROPIC_API_KEY is read from the project .env only; the run is refused without it. --dry-run counts the calls and dollars and makes no call.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CHAIN_FOLDER = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = CHAIN_FOLDER.parents[1]
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
EMERGENT = SCRIPTS_DIRECTORY / "emergent_misalignment"
TREATMENT = EMERGENT / "treatment_arm_training_data"
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY, EMERGENT, TREATMENT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import judge_emergent_misalignment_alignment_and_coherence_with_claude_code_cli as betley_judge  # noqa: E402
from emergent_misalignment import judge_held_out_set_conditions_in_order_and_write_verdict_rate_table as held_out_judging  # noqa: E402
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as sage  # noqa: E402
from emergent_misalignment.overnight_chain import estimate_haiku_calls_and_dollars_for_the_overnight_chain_judging as estimator  # noqa: E402
import anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log as haiku_api  # noqa: E402
import judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli as framework  # noqa: E402
import judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli as welfare_judge  # noqa: E402
import judge_whether_answer_chose_option_better_for_animal_welfare_without_mentioning_welfare_through_claude_code_cli as choice_judge  # noqa: E402
from shared_answer_records_and_held_out_set_helpers import read_jsonl  # noqa: E402

JUDGE_MODEL = "claude-haiku-5-5"
PARALLEL_CALLS = 16
HAIKU_JUDGE_OUTPUTS_PARENT_NAME = "judge_outputs_haiku"
EXTRA_THINKING_ON_HELD_OUT_FOLDER_NAME = "extra_evaluation_held_out_thinking_on"
SAGE_AND_HARMBENCH_THINKING_ON_KIND = "sage_eval_and_harmbench_thinking_on"
NEW_CONDITION_KINDS = ("held_out_new_conditions", "betley_new_conditions", "mmlu_pro_new_conditions", "sycophancy_harmbench_sage_eval_new_conditions")
HARMBENCH_RANDOM_ADAPTER_KIND = "harmbench_random_adapter"  # one model entry, HarmBench only, classified on the pod: nothing for Haiku to judge
PRIMARY_EVALUATION_KINDS = ("held_out", "betley", "mmlu_pro", "sycophancy_harmbench_sage_eval", "held_out_thinking_on", SAGE_AND_HARMBENCH_THINKING_ON_KIND, *NEW_CONDITION_KINDS, HARMBENCH_RANDOM_ADAPTER_KIND)
ADDITIONAL_BLOCK_EVALUATION_KINDS = (SAGE_AND_HARMBENCH_THINKING_ON_KIND, *NEW_CONDITION_KINDS, HARMBENCH_RANDOM_ADAPTER_KIND)
SAGE_EVALUATION_KINDS_WITHOUT_THINKING = ("sycophancy_harmbench_sage_eval", "sycophancy_harmbench_sage_eval_new_conditions")
DEFAULT_BASE_MODEL_SHORT_NAME = "gemma_4_31b_it"
KEPT_ANSWERS_GENERATED_IN_THIS_CHAIN = "kept_answers_generated_in_this_chain"
TRAINING_EXAMPLES_FROM_ANOTHER_MODEL = "training_examples_file_from_another_model"
STEPS_NEEDING_KEPT_ANSWERS = ("completeness_and_competence", "user_lean_and_agreement")
PROGRESS_REPORT_INTERVAL_SECONDS = 180
STEP_NAMES = ["held_out_welfare_and_refusal", "held_out_choice_of_option", "betley_alignment_and_coherence", "sage_eval", "completeness_and_competence",
              "user_lean_and_agreement", "analyses"]


class RunPaths:
    def __init__(self, run_folder):
        self.run_folder = Path(run_folder)
        self.pulled = self.run_folder / "overnight_outputs" / "run_folders"
        self.inputs = self.run_folder / "overnight_inputs" / "run_folders"
        self.generation = self.pulled / "treatment_arm_generation"
        self.generation_inputs = self.inputs / "treatment_arm_generation"
        self.evaluation = {kind: self.pulled / "evaluation" / kind for kind in PRIMARY_EVALUATION_KINDS}
        chain_configurations = sorted(self.run_folder.glob("run_configuration_overnight_chain_*.json"))
        chain_configuration = json.loads(chain_configurations[-1].read_text(encoding="utf-8")) if chain_configurations else {}
        self.base_model_short_name = chain_configuration.get("base_model_short_name", DEFAULT_BASE_MODEL_SHORT_NAME)
        self.training_examples_source_kind = chain_configuration.get("training_examples_source_kind", KEPT_ANSWERS_GENERATED_IN_THIS_CHAIN)
        self.additional_block = chain_configuration.get("additional_model_evaluation_block")
        self.additional_evaluation = ({kind: self.pulled / "evaluation" / self.additional_block["evaluation_folder_name"] / kind for kind in ADDITIONAL_BLOCK_EVALUATION_KINDS}
                                      if self.additional_block else {})

    def folders_of_kind(self, kind):
        """[(label, folder)] of the existing evaluation folders of one kind: the primary block's (label = kind) then the additional model block's
        (label = <block folder name>__<kind>)."""
        found = [(kind, self.evaluation[kind])] if kind in self.evaluation else []
        if kind in self.additional_evaluation:
            found.append((f"{self.additional_block['evaluation_folder_name']}__{kind}", self.additional_evaluation[kind]))
        return [(label, folder) for label, folder in found if folder.exists()]

    def kept_answers_were_generated_in_this_chain(self):
        return self.training_examples_source_kind != TRAINING_EXAMPLES_FROM_ANOTHER_MODEL

    def extra_thinking_on_held_out_folders(self):
        """The held_out_thinking_on evaluation folder and the run folders under <run folder>/extra_evaluation_held_out_thinking_on/ that already hold at least
        one answer file with a record."""
        parent = self.run_folder / EXTRA_THINKING_ON_HELD_OUT_FOLDER_NAME
        folders = [self.evaluation["held_out_thinking_on"]] + (sorted(path for path in parent.iterdir() if path.is_dir()) if parent.is_dir() else [])
        return [folder for folder in folders if any(answers_file.stat().st_size > 0 for answers_file in (folder / "generation_outputs").glob("answers__*.jsonl"))]

    @staticmethod
    def configuration(folder):
        paths = sorted(Path(folder).glob("run_configuration_*.json"))
        return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else None


def latest_kept_answers_file(paths):
    files = sorted((paths.generation / "generation_outputs" / "final_outputs").glob("kept_answers_*.jsonl"))
    return files[-1] if files else None


def say(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# -- the inputs of the two framework judges (the pod's own sampling rules, replicated) ----------------------------------------------------

def framework_judge_specification(paths, judge_kind):
    configuration = paths.configuration(paths.generation_inputs)
    record = configuration["gemma_self_judging_and_regeneration"]["framework_judges"][judge_kind]
    template_path = paths.generation_inputs / record["prompt_template"]["path_inside_run_folder"]
    template = template_path.read_text(encoding="utf-8").rstrip("\n")  # as the framework's main() reads it
    schema = json.loads((paths.generation_inputs / record["reply_schema"]["path_inside_run_folder"]).read_text(encoding="utf-8"))
    return {"template_path": template_path, "template": template, "schema": schema, "comparison_condition": record["comparison_condition"],
            "sampling": configuration["gemma_self_judging_and_regeneration"].get("framework_judge_sampling") or {"completeness_sample_size": None,
                                                                                                           "user_lean_sample_prompts": None, "seed": 0}}


def reference_answers(paths):
    configuration = paths.configuration(paths.generation_inputs)
    section = configuration["gemma_self_judging_and_regeneration"]
    path = paths.generation / "generation_outputs" / defs.ANSWER_FILE_PATTERN.format(
        model_short_name=configuration["models"][0]["model_short_name"], condition=section["no_system_prompt_condition"])
    prompt_gids = {prompt["prompt_gid"] for prompt in read_jsonl(paths.generation_inputs / configuration["prompts_file"]["path_inside_run_folder"])}
    return [record for record in (read_jsonl(path) if path.exists() else []) if record["prompt_gid"] in prompt_gids]


def completeness_inputs(paths):
    """(kept answers with a pair, sampled as the pod's phase 5 samples them, comparison_by_prompt_gid)."""
    specification = framework_judge_specification(paths, "completeness_and_competence")
    kept_file = latest_kept_answers_file(paths)
    kept = read_jsonl(kept_file) if kept_file else []
    comparison = framework.pick_comparison_answer_per_prompt(reference_answers(paths), specification["comparison_condition"])
    paired, _ = framework.keep_answers_that_have_a_comparison_answer(kept, comparison)
    settings, size = specification["sampling"], specification["sampling"]["completeness_sample_size"]
    paired = sorted(paired, key=lambda answer: answer["answer_id"])
    if size is not None and size < len(paired):
        paired = sorted(random.Random(f"{settings['seed']}:completeness").sample(paired, size), key=lambda answer: answer["answer_id"])
    return paired, comparison


def user_lean_inputs(paths):
    """The kept answer and the no-system-prompt answer of every sampled prompt that has a kept answer (both conditions)."""
    configuration = paths.configuration(paths.generation_inputs)
    section = configuration["gemma_self_judging_and_regeneration"]
    kept_file = latest_kept_answers_file(paths)
    kept, references = (read_jsonl(kept_file) if kept_file else []), reference_answers(paths)
    sample = section.get("no_system_prompt_reference_answer_sample")
    prompts_with_both = sorted({answer["prompt_gid"] for answer in kept} & {answer["prompt_gid"] for answer in references})
    if sample:
        chosen = set(prompts_with_both) & set(json.loads((paths.generation_inputs / sample["path_inside_run_folder"]).read_text(encoding="utf-8"))["prompt_gids"])
    else:
        size = framework_judge_specification(paths, "user_lean_and_agreement")["sampling"]["user_lean_sample_prompts"]
        settings = framework_judge_specification(paths, "user_lean_and_agreement")["sampling"]
        chosen = set(random.Random(f"{settings['seed']}:user_lean").sample(prompts_with_both, size)) if size and size < len(prompts_with_both) else set(prompts_with_both)
    return [answer for answer in kept + references if answer["prompt_gid"] in chosen]


def haiku_judge_folder(paths, specification):
    return paths.generation / "generation_outputs" / HAIKU_JUDGE_OUTPUTS_PARENT_NAME / framework.judge_output_folder_name(specification["template_path"], specification["template"])


def sage_evaluation_folders(paths):
    """(label, folder, whether its answers carry thinking traces to judge) for each evaluation folder with SAGE-Eval answers, primary block then additional block."""
    kinds = ("sycophancy_harmbench_sage_eval", SAGE_AND_HARMBENCH_THINKING_ON_KIND, "sycophancy_harmbench_sage_eval_new_conditions")
    return [(label, folder, kind not in SAGE_EVALUATION_KINDS_WITHOUT_THINKING) for kind in kinds for label, folder in paths.folders_of_kind(kind)]


def sage_answers_to_judge(folder, with_thinking_traces=False):
    """(answer records for the framework judge, the folder's run configuration). The thinking trace is carried only for a thinking-on folder."""
    configuration = RunPaths.configuration(folder)
    task = next((task for task in (configuration or {}).get("evaluation_tasks", []) if task["task_name"] == sage.TASK_NAME), None)
    if task is None:
        return [], configuration
    from emergent_misalignment import base_versus_lora_adapters_analysis_helpers as helpers
    prompts = {record["prompt_gid"]: record for record in read_jsonl(folder / task["prompts_file"]["path_inside_run_folder"])}
    answers = helpers.load_answers_for_conditions(folder, configuration["model_order"], [sage.TASK_NAME])
    records = []
    for answer in (answers.to_dict("records") if len(answers) else []):
        prompt = prompts[answer["prompt_gid"]]
        thinking_trace = answer.get("thinking_trace_text") if with_thinking_traces else None
        records.append({"answer_id": answer["answer_id"], "prompt_gid": answer["prompt_gid"], "model_short_name": answer["model_short_name"], "condition": answer["condition"],
                        "sample_index": int(answer["sample_index"]), "user_message": sage.judge_user_message_block(prompt["user_message"], prompt["safety_fact"]),
                        "final_answer_text": answer["final_answer_text"], "thinking_trace_text": thinking_trace if isinstance(thinking_trace, str) else None})
    return records, configuration


def sage_judge_folder(folder, with_thinking_trace):
    """Where the framework writes the verdicts of the SAGE prompt (with or without the thinking question) for one evaluation folder."""
    template_path, _ = sage.judge_template_and_schema_paths(with_thinking_trace)
    template = template_path.read_text(encoding="utf-8").rstrip("\n")
    return folder / "generation_outputs" / HAIKU_JUDGE_OUTPUTS_PARENT_NAME / framework.judge_output_folder_name(template_path, template)


def write_flat_sage_verdicts(folder):
    """The SAGE analysis reads one flat record per answer (the field names of the pod's Gemma verdicts); the framework wrote them nested, in one judge folder
    for answers with a thinking trace and one for answers without."""
    by_model = {}
    for with_thinking_trace in (False, True):
        for path in sorted(sage_judge_folder(folder, with_thinking_trace).glob("verdicts_*.jsonl")):
            for record in read_jsonl(path):
                flat = sage.flat_verdict_record(record["answer_id"], record["verdict"] if record["parse_status"] == "ok" else None, record["judge_model"],
                                                "haiku_through_the_paid_api")
                by_model.setdefault(record["model_short_name"], {})[record["answer_id"]] = flat
    for model_name, records in by_model.items():
        (folder / "generation_outputs" / f"sage_eval_haiku_judge_verdicts__{model_name}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records.values()), encoding="utf-8")


def run_sage_judging(folder, with_thinking_traces, call_model, parallel_calls, log):
    """Judge the SAGE-Eval answers of one evaluation folder, those with a thinking trace and those without in separate framework runs; returns the summed counts."""
    records, _ = sage_answers_to_judge(folder, with_thinking_traces)
    if not records:
        log(f"no SAGE-Eval answers in {folder.name}: nothing to judge")
        return None
    totals = {}
    for with_thinking_trace in (False, True):
        part = [record for record in records if sage.answer_has_thinking_trace(record) == with_thinking_trace]
        if not part:
            continue
        template_path, schema_path = sage.judge_template_and_schema_paths(with_thinking_trace)
        template = template_path.read_text(encoding="utf-8").rstrip("\n")
        counts = framework.run_judging(part, sage_judge_folder(folder, with_thinking_trace), template, json.loads(schema_path.read_text(encoding="utf-8")), JUDGE_MODEL,
                                       template_path.name, call_model, parallel_calls=parallel_calls, log=log)
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    write_flat_sage_verdicts(folder)
    return totals


# -- the steps ---------------------------------------------------------------------------------------------------------------------------

class Checkpoint:
    """A record of the steps that finished a pass with no failed call. Every judge skips answers it already judged, so a rerun repeats nothing
    that was paid for; a step is run again on every invocation (a pass over finished work makes no call), which also retries failed calls."""

    def __init__(self, path):
        self.path = Path(path)
        self.done = set(json.loads(self.path.read_text(encoding="utf-8"))["finished_steps"]) if self.path.exists() else set()

    def finish(self, step):
        self.done.add(step)
        self.path.write_text(json.dumps({"finished_steps": sorted(self.done), "updated_at": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")


def run_framework_step(paths, judge_kind, answers, comparison, call_model, parallel_calls, log=say):
    specification = framework_judge_specification(paths, judge_kind)
    return framework.run_judging(answers, haiku_judge_folder(paths, specification), specification["template"], specification["schema"], JUDGE_MODEL,
                                 specification["template_path"].name, call_model, parallel_calls=parallel_calls, log=log, comparison_by_prompt_gid=comparison,
                                 comparison_condition=specification["comparison_condition"])


def run_steps(paths, call_model, checkpoint, output_folder, parallel_calls=PARALLEL_CALLS, steps=STEP_NAMES, log=say, run_analysis_scripts=None):
    results = {}
    held_out = paths.evaluation["held_out"]
    for step in steps:
        if step in STEPS_NEEDING_KEPT_ANSWERS and not paths.kept_answers_were_generated_in_this_chain():
            log(f"step {step}: skipped, the training examples came from another model so no kept answers were generated in this chain")
            checkpoint.finish(step)
            continue
        log(f"step {step}: starting" + (" (finished a pass earlier; judged answers are skipped)" if step in checkpoint.done else ""))
        if step == "held_out_welfare_and_refusal":
            if held_out.exists():
                results[step] = held_out_judging.judge_conditions_in_order(held_out, output_folder / "held_out_verdict_rate_table.md", call_model,
                                                                           parallel_calls=parallel_calls, log=log)
            for folder in paths.extra_thinking_on_held_out_folders():
                results[f"{step}__{folder.name}"] = held_out_judging.judge_conditions_in_order(
                    folder, output_folder / f"{folder.name}_verdict_rate_table.md", call_model, parallel_calls=parallel_calls, log=log)
            for label, folder in paths.folders_of_kind("held_out_new_conditions"):
                results[f"{step}__{label}"] = held_out_judging.judge_conditions_in_order(
                    folder, output_folder / f"{label}_verdict_rate_table.md", call_model, parallel_calls=parallel_calls, log=log,
                    conditions=held_out_judging.conditions_with_answers_in_judging_order(folder))
        elif step == "held_out_choice_of_option":
            if held_out.exists():
                results[step] = choice_judge.run_judging(held_out, call_model, parallel_calls=parallel_calls, log=log, allow_answers_without_existing_verdict=True)
            for folder in paths.extra_thinking_on_held_out_folders():
                results[f"{step}__{folder.name}"] = choice_judge.run_judging(folder, call_model, parallel_calls=parallel_calls, log=log,
                                                                             allow_answers_without_existing_verdict=True)
            for label, folder in paths.folders_of_kind("held_out_new_conditions"):
                results[f"{step}__{label}"] = choice_judge.run_judging(folder, call_model, parallel_calls=parallel_calls, log=log,
                                                                       allow_answers_without_existing_verdict=True)
        elif step == "betley_alignment_and_coherence":
            for label, folder in paths.folders_of_kind("betley") + paths.folders_of_kind("betley_new_conditions"):
                results[step if label == "betley" else f"{step}__{label}"] = betley_judge.run_judging(
                    folder, JUDGE_MODEL, call_model, betley_judge.load_templates(), parallel_calls=parallel_calls, log=log)
        elif step == "sage_eval":
            for label, folder, with_thinking_traces in sage_evaluation_folders(paths):
                results[step if label == "sycophancy_harmbench_sage_eval" else f"{step}__{label}"] = run_sage_judging(folder, with_thinking_traces, call_model, parallel_calls, log)
        elif step == "completeness_and_competence":
            answers, comparison = completeness_inputs(paths)
            results[step] = run_framework_step(paths, "completeness_and_competence", answers, comparison, call_model, parallel_calls, log)
        elif step == "user_lean_and_agreement":
            results[step] = run_framework_step(paths, "user_lean_and_agreement", user_lean_inputs(paths), None, call_model, parallel_calls, log)
        elif step == "analyses":
            results[step] = (run_analysis_scripts or run_the_analysis_scripts)(paths, output_folder, log)
        outcome = results.get(step)
        if isinstance(outcome, dict) and outcome.get("failed_calls"):
            log(f"step {step}: {outcome['failed_calls']} calls failed; rerun this command to finish them")
            continue
        checkpoint.finish(step)
    return results


def analysis_commands(paths):
    """(name, command list) of every existing analysis script, each reading the Haiku verdicts. The held-out, Betley, MMLU-Pro and SAGE-Eval, HarmBench and "are you
    sure?" folders of the new conditions and of the additional model block get the same commands, their names ending in the folder's label."""
    python = sys.executable
    commands = []
    for kind in ("held_out", "held_out_new_conditions"):
        for label, folder in paths.folders_of_kind(kind):
            suffix = "" if label == "held_out" else f"__{label}"
            commands += [(f"held_out_welfare_raising_rates{suffix}", [python, str(EMERGENT / "analyse_held_out_set_welfare_raising_base_versus_lora_adapters.py"), "--run-folder",
                                                                      str(folder), "--judge-outputs-folder-name", "judge_outputs"]),
                         (f"silent_welfare_better_choice_rates{suffix}", [python, str(EMERGENT / "summarise_silent_welfare_better_choice_rates_by_condition.py"), "--run-folder",
                                                                          str(folder), "--allow-answers-without-existing-verdict"])]
    for kind in ("betley", "betley_new_conditions"):
        for label, folder in paths.folders_of_kind(kind):
            suffix = "" if label == "betley" else f"__{label}"
            commands.append((f"betley_misaligned_share{suffix}", [python, str(EMERGENT / "analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters.py"),
                                                                  "--run-folder", str(folder), "--judge-outputs-folder-name", "judge_outputs", "--judge-model", JUDGE_MODEL]))
    for label, folder in paths.folders_of_kind("mmlu_pro_new_conditions"):
        commands.append((f"mmlu_pro_accuracy__{label}", [python, str(EMERGENT / "score_mmlu_pro_capability_check_base_versus_lora_adapters.py"), "--run-folder", str(folder)]))
    for label, folder in paths.folders_of_kind(HARMBENCH_RANDOM_ADAPTER_KIND):
        commands.append((f"harmbench_attack_success_rate__{label}", [python, str(EMERGENT / "analyse_harmbench_attack_success_rate_base_versus_lora_adapters.py"),
                                                                    "--run-folder", str(folder)]))
    for label, folder, with_thinking_traces in sage_evaluation_folders(paths):
        suffix = "" if label == "sycophancy_harmbench_sage_eval" else f"__{label}"
        task_names = {task["task_name"] for task in (paths.configuration(folder) or {}).get("evaluation_tasks", [])}
        if sage.TASK_NAME in task_names:
            commands.append((f"sage_eval_pass_rate{suffix}", [python, str(EMERGENT / "analyse_sage_eval_pass_rate_base_versus_lora_adapters.py"), "--run-folder", str(folder),
                                                                    "--judge-verdict-file-pattern", "sage_eval_haiku_judge_verdicts__{name}.jsonl",
                                                                    "--judge-description", "Claude Haiku 5.5 through the paid Anthropic API (the laptop)"]))
        if "harmbench_direct_request" in task_names:
            commands.append((f"harmbench_attack_success_rate{suffix}", [python, str(EMERGENT / "analyse_harmbench_attack_success_rate_base_versus_lora_adapters.py"),
                                                                              "--run-folder", str(folder)]))
        if "are_you_sure" in task_names:
            commands.append((f"are_you_sure_sycophancy{suffix}", [python, str(EMERGENT / "score_are_you_sure_sycophancy_base_versus_lora_adapters.py"), "--run-folder", str(folder)]))
    try:
        if not paths.kept_answers_were_generated_in_this_chain():
            return commands
        completeness = framework_judge_specification(paths, "completeness_and_competence")
        user_lean = framework_judge_specification(paths, "user_lean_and_agreement")
        kept_file = latest_kept_answers_file(paths)
        if kept_file:
            commands.append(("completeness_spot_check_sheet", [python, str(TREATMENT / "build_completeness_and_competence_spot_check_sheet_from_judge_verdicts.py"),
                                                                "--judge-folder", str(haiku_judge_folder(paths, completeness)), "--answers", str(kept_file),
                                                                "--comparison-answers", str(paths.generation)]))
            commands.append(("user_lean_and_agreement_analysis", [python, str(TREATMENT / "analyse_user_lean_and_agreement_verdicts_by_condition_surface_form_and_user_goal.py"),
                                                                  "--judge-folder", str(haiku_judge_folder(paths, user_lean)), "--kept-answers", str(kept_file)]))
    except (KeyError, TypeError, FileNotFoundError):
        pass
    return commands


def run_the_analysis_scripts(paths, output_folder, log=say):
    outcomes = {}
    for name, command in analysis_commands(paths):
        log(f"analysis {name}: running")
        completed = subprocess.run(command, capture_output=True, text=True)
        (output_folder / f"analysis_log_{name}.txt").write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
        outcomes[name] = completed.returncode
        log(f"analysis {name}: exit code {completed.returncode}" + ("" if completed.returncode == 0 else f" (see analysis_log_{name}.txt)"))
    return outcomes


def count_calls_still_to_make(paths):
    """Calls the judging steps would make now (one per unjudged answer, two for Betley, the choice judge after the welfare verdicts exist)."""
    counts = {}
    held_out = paths.evaluation["held_out"]
    if held_out.exists():
        counts["held_out_welfare_and_refusal"] = held_out_judging.count_pending_answers_per_condition(held_out)
    for folder in paths.extra_thinking_on_held_out_folders():
        counts[f"extra_thinking_on_held_out_welfare_and_refusal__{folder.name}"] = held_out_judging.count_pending_answers_per_condition(folder)
    for label, folder in paths.folders_of_kind("held_out_new_conditions"):
        counts[f"held_out_welfare_and_refusal__{label}"] = held_out_judging.count_pending_answers_per_condition(
            folder, held_out_judging.conditions_with_answers_in_judging_order(folder))
    for label, folder in paths.folders_of_kind("betley") + paths.folders_of_kind("betley_new_conditions"):
        counts["betley_alignment_and_coherence" if label == "betley" else f"betley_alignment_and_coherence__{label}"] = 2 * len(betley_judge.pending_answer_records(folder, JUDGE_MODEL))
    for label, folder, with_thinking_traces in sage_evaluation_folders(paths):
        records, _ = sage_answers_to_judge(folder, with_thinking_traces)
        pending = 0
        for with_thinking_trace in (False, True):
            judge_folder = sage_judge_folder(folder, with_thinking_trace)
            done = framework.load_already_judged_answer_ids(judge_folder) if judge_folder.exists() else set()
            pending += sum(record["answer_id"] not in done for record in records if sage.answer_has_thinking_trace(record) == with_thinking_trace)
        if records:
            counts["sage_eval" if label == "sycophancy_harmbench_sage_eval" else f"sage_eval__{label}"] = pending
    if paths.kept_answers_were_generated_in_this_chain() and latest_kept_answers_file(paths):
        for kind, answers in (("completeness_and_competence", completeness_inputs(paths)[0]), ("user_lean_and_agreement", user_lean_inputs(paths))):
            folder = haiku_judge_folder(paths, framework_judge_specification(paths, kind))
            done = framework.load_already_judged_answer_ids(folder) if folder.exists() else set()
            counts[kind] = sum(answer["answer_id"] not in done for answer in answers)
    return counts


def main(argv=None, call_model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="count the calls still to make and print the run folder's Haiku estimate; no key is read, no call is made")
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS)
    parser.add_argument("--only-steps", nargs="+", choices=STEP_NAMES, default=None)
    arguments = parser.parse_args(argv)
    paths = RunPaths(arguments.run_folder)
    if arguments.dry_run:
        say("DRY RUN: no call is made and no key is read")
        for name, count in count_calls_still_to_make(paths).items():
            say(f"  {name}: {count} calls still to make")
        try:
            for line in estimator.estimate(arguments.run_folder)["lines"]:
                say(line)
        except (OSError, KeyError, ValueError, StopIteration, TypeError) as error:
            say(f"  (estimate unavailable: {error!r})")
        return 0
    existing = sorted(Path(arguments.run_folder).glob("haiku_judging_after_the_chain_*"))
    output_folder = existing[-1] if existing else Path(arguments.run_folder) / f"haiku_judging_after_the_chain_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_folder.mkdir(exist_ok=True)
    if call_model_factory is None:
        backend = haiku_api.AnthropicApiJudgeBackend(api_key=haiku_api.read_anthropic_api_key_from_project_env_file(), log=say,
                                                     usage_log_path=output_folder / f"anthropic_api_usage_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl",
                                                     initial_concurrent_calls=arguments.parallel_calls)
    else:
        backend = call_model_factory(output_folder)
    import threading
    stop = threading.Event()

    def report():
        while not stop.wait(PROGRESS_REPORT_INTERVAL_SECONDS):
            say(backend.progress_line())
    threading.Thread(target=report, daemon=True).start()
    try:
        results = run_steps(paths, backend.call_model, Checkpoint(output_folder / "haiku_judging_after_the_chain_checkpoint.json"), output_folder,
                            arguments.parallel_calls, arguments.only_steps or STEP_NAMES)
    finally:
        stop.set()
    say(backend.progress_line())
    (output_folder / f"haiku_judging_after_the_chain_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json").write_text(
        json.dumps({"results": results, "api_totals": backend.totals}, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
