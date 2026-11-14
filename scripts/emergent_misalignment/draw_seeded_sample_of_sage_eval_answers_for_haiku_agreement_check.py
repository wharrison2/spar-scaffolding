"""Draws a seeded sample of the SAGE-Eval answers Gemma judged on the pod and writes them as an answers file the Claude Code CLI judging framework can judge with Haiku
(local, free, no model call; it prints the command and runs nothing).

    python3 scripts/emergent_misalignment/draw_seeded_sample_of_sage_eval_answers_for_haiku_agreement_check.py --run-folder <run folder after the pod run was pulled> \
        [--sample-size 150] [--seed 0]

The sample is stratified by Gemma's two verdict fields (raised the hazard or gave the safer alternative, refused or failed to help), so the rarer combinations are not drowned
out: equal allocation per stratum, a stratum with fewer answers than its share gives all it has and the shortfall is shared out round-robin. The manifest records every stratum's
population and sample count, so agreement can be reweighted to the population. Every answer of the sample is a record of the framework's input shape (answer_id, prompt_gid,
model_short_name, condition, sample_index, final_answer_text) whose user_message holds the request AND the safety fact as one block (the framework's template has no placeholder for
the fact; sage_eval_safety_fact_generalization_evaluation_protocol.judge_user_message_block builds it, identically to Gemma's prompt), so the same template text judges both.

Output, in <run folder>/haiku_sample_check_inputs/ (names end with the timestamp):
    haiku_sample_check_sage_eval_answers_to_judge_<stamp>.jsonl
    haiku_sample_check_sage_eval_manifest_<stamp>.json
Haiku's verdicts should go to <run folder>/judge_outputs_haiku_sample_check/ (the printed command does that); then run
analyse_sage_eval_gemma_versus_haiku_judge_agreement.py.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import base_versus_lora_adapters_analysis_helpers as helpers  # noqa: E402
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as protocol  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402

FRAMEWORK_SCRIPT = ("scripts/emergent_misalignment/treatment_arm_training_data/"
                    "judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli.py")
HAIKU_MODEL = "claude-haiku-5-5"
HAIKU_OUTPUTS_FOLDER_NAME = "judge_outputs_haiku_sample_check"
HAIKU_INPUTS_FOLDER_NAME = "haiku_sample_check_inputs"


def stratum_name(verdict_record):
    return f"raised_{str(verdict_record[protocol.RAISED_FIELD]).lower()}__refused_{str(verdict_record[protocol.REFUSED_FIELD]).lower()}"


def draw_stratified_sample(gemma_verdicts_by_answer_id, sample_size, seed):
    """({stratum: [answer ids]}, {stratum: population count})."""
    strata = {}
    for answer_id in sorted(gemma_verdicts_by_answer_id):
        strata.setdefault(stratum_name(gemma_verdicts_by_answer_id[answer_id]), []).append(answer_id)
    generator = random.Random(seed)
    shuffled = {name: generator.sample(ids, len(ids)) for name, ids in sorted(strata.items())}
    allocation = {name: 0 for name in shuffled}
    remaining = min(sample_size, sum(len(ids) for ids in shuffled.values()))
    while remaining > 0:
        progressed = False
        for name in shuffled:
            if remaining > 0 and allocation[name] < len(shuffled[name]):
                allocation[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return {name: shuffled[name][:allocation[name]] for name in shuffled}, {name: len(ids) for name, ids in strata.items()}


def read_gemma_verdicts(run_folder, model_order):
    verdicts = {}
    for name in model_order:
        path = Path(run_folder) / "generation_outputs" / f"sage_eval_gemma_judge_verdicts__{name}.jsonl"
        if path.exists():
            for record in read_jsonl(path):
                if record["parse_status"] == "ok":
                    verdicts[record["answer_id"]] = record
    return verdicts


def build_answers_to_judge(run_folder, configuration, sampled_answer_ids):
    task = next(task for task in configuration["evaluation_tasks"] if task["task_name"] == protocol.TASK_NAME)
    prompts = {record["prompt_gid"]: record for record in read_jsonl(Path(run_folder) / task["prompts_file"]["path_inside_run_folder"])}
    answers = helpers.load_answers_for_conditions(run_folder, configuration["model_order"], [protocol.TASK_NAME]).set_index("answer_id")
    records = []
    for answer_id in sampled_answer_ids:
        answer = answers.loc[answer_id]
        prompt = prompts[answer["prompt_gid"]]
        records.append({"answer_id": answer_id, "prompt_gid": answer["prompt_gid"], "model_short_name": answer["model_short_name"], "condition": answer["condition"],
                        "sample_index": int(answer["sample_index"]), "user_message": protocol.judge_user_message_block(prompt["user_message"], prompt["safety_fact"]),
                        "final_answer_text": answer["final_answer_text"], "thinking_trace_text": None})
    return records


def build_command(answers_file, output_parent):
    return " ".join(["python3", FRAMEWORK_SCRIPT, "--answers", str(answers_file), "--judge-prompt-file", str(protocol.JUDGE_PROMPT_TEMPLATE_PATH),
                     "--reply-schema-file", str(protocol.JUDGE_REPLY_SCHEMA_PATH), "--judge-model", HAIKU_MODEL, "--output-parent-folder", str(output_parent)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    run_folder = arguments.run_folder
    configuration = helpers.load_latest_run_configuration(run_folder)
    verdicts = read_gemma_verdicts(run_folder, configuration["model_order"])
    if not verdicts:
        raise SystemExit(f"No readable Gemma verdicts under {run_folder}/generation_outputs; pull the pod outputs first")
    sampled, population = draw_stratified_sample(verdicts, arguments.sample_size, arguments.seed)
    sampled_ids = [answer_id for ids in sampled.values() for answer_id in ids]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = run_folder / HAIKU_INPUTS_FOLDER_NAME
    folder.mkdir(parents=True, exist_ok=True)
    answers_file = folder / f"haiku_sample_check_sage_eval_answers_to_judge_{stamp}.jsonl"
    write_jsonl(answers_file, build_answers_to_judge(run_folder, configuration, sampled_ids))
    output_parent = run_folder / HAIKU_OUTPUTS_FOLDER_NAME
    command = build_command(answers_file, output_parent)
    manifest = {"seed": arguments.seed, "sample_size": len(sampled_ids), "requested_sample_size": arguments.sample_size, "created_at": datetime.now().isoformat(timespec="seconds"),
                "strata_population_counts": population, "strata_sample_counts": {name: len(ids) for name, ids in sampled.items()},
                "answers_file": answers_file.name, "haiku_output_parent_folder": str(output_parent), "haiku_model": HAIKU_MODEL,
                "judge_prompt_template": str(protocol.JUDGE_PROMPT_TEMPLATE_PATH), "judge_reply_schema": str(protocol.JUDGE_REPLY_SCHEMA_PATH), "command": command}
    (folder / f"haiku_sample_check_sage_eval_manifest_{stamp}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Sampled {len(sampled_ids)} answers; strata {manifest['strata_sample_counts']} of populations {population}")
    print("Ready-to-run command (uses the Claude subscription through the CLI; not run by this script):\n" + command)


if __name__ == "__main__":
    main()
