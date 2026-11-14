"""Draw seeded samples of the answers the Gemma completeness and user-lean judges judged on the pod, and write them as answers files the
Claude Code CLI judging framework can judge directly with Haiku (local, free, no model call; it prints the commands, it runs none).

    python3 scripts/emergent_misalignment/treatment_arm_training_data/draw_seeded_samples_of_gemma_judged_answers_for_haiku_agreement_check.py \
        --run-folder <treatment arm run folder after phases 5 and 6 were pulled> [--completeness-sample-size 150] \
        [--user-lean-sample-size 150] [--seed 0]

Completeness: half the sample flagged by Gemma on either field (left_out_part_of_the_request or clearly_worse_on_everything_else), half
not flagged. Flagged answers are over-sampled on purpose (they are the minority), so raw agreement with Haiku is not the population rate;
the manifest records each stratum's population and sample counts so the agreement analysis can reweight.
User lean: stratified by Gemma's user_lean value and by condition (kept system-prompted answers and no-system-prompt answers), equal
allocation per stratum, a short stratum taking all it has and the shortfall going to the others in turn.

Output, in <run folder>/haiku_sample_check_inputs/ (names end with the timestamp):
    haiku_sample_check_completeness_kept_answers_to_judge_<stamp>.jsonl
    haiku_sample_check_completeness_paired_no_system_prompt_reference_answers_<stamp>.jsonl   (the comparison answers Gemma saw)
    haiku_sample_check_user_lean_answers_to_judge_<stamp>.jsonl
    haiku_sample_check_manifest_<stamp>.json
The population of each stratum is counted among the answers Gemma judged (the pod may have judged only a seeded sample of the kept answers or
prompts, see the run configuration's framework_judge_sampling); because that sample is random, reweighting to those counts estimates the
population rates.
Haiku's verdicts should go to <run folder>/judge_outputs_haiku_sample_check/ (the printed commands do that).
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

THIS_FOLDER = Path(__file__).resolve().parent
if str(THIS_FOLDER) not in sys.path:
    sys.path.insert(0, str(THIS_FOLDER))

from shared_answer_records_and_held_out_set_helpers import load_answer_records, read_jsonl, write_jsonl  # noqa: E402

LEFT_OUT_FIELD = "left_out_part_of_the_request"
WORSE_FIELD = "clearly_worse_on_everything_else"
GEMMA_JUDGE_OUTPUTS_FOLDER_NAME = "judge_outputs_gemma_on_pod"
HAIKU_SAMPLE_OUTPUTS_FOLDER_NAME = "judge_outputs_haiku_sample_check"
HAIKU_SAMPLE_INPUTS_FOLDER_NAME = "haiku_sample_check_inputs"
FRAMEWORK_SCRIPT = "scripts/emergent_misalignment/treatment_arm_training_data/judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli.py"
PROMPTS_FOLDER_RELATIVE = "scripts/emergent_misalignment/treatment_arm_training_data/prompts"
HAIKU_MODEL = "claude-haiku-5-5"


def read_run_configuration(run_folder):
    return json.loads(sorted(Path(run_folder).glob("run_configuration_*.json"))[-1].read_text(encoding="utf-8"))


def find_gemma_judge_folder(run_folder, framework_judge_record):
    stem = Path(framework_judge_record["prompt_template"]["path_inside_run_folder"]).stem
    matches = sorted((Path(run_folder) / "generation_outputs" / GEMMA_JUDGE_OUTPUTS_FOLDER_NAME).glob(f"{stem}__sha256_*"))
    if not matches:
        raise SystemExit(f"No Gemma judge output folder for {stem} under {run_folder}/generation_outputs/{GEMMA_JUDGE_OUTPUTS_FOLDER_NAME}")
    return matches[-1]


def read_ok_verdict_records(judge_folder):
    """answer_id -> verdict record with parse_status ok (the latest record wins)."""
    records = {}
    for path in sorted(Path(judge_folder).glob("verdicts_*.jsonl")):
        for record in read_jsonl(path):
            if record["parse_status"] == "ok":
                records[record["answer_id"]] = record
    return records


def gemma_flagged(verdict):
    return verdict[LEFT_OUT_FIELD] == "yes" or verdict[WORSE_FIELD] == "yes"


def draw_completeness_sample(verdict_records, sample_size, seed):
    """({stratum: [answer ids]}, {stratum: population count}). Half flagged on either field, half not; a stratum with fewer answers than
    its half gives all it has."""
    ordered = sorted(verdict_records)
    strata = {"flagged": [i for i in ordered if gemma_flagged(verdict_records[i]["verdict"])],
              "unflagged": [i for i in ordered if not gemma_flagged(verdict_records[i]["verdict"])]}
    generator = random.Random(seed)
    half = {"flagged": math.ceil(sample_size / 2), "unflagged": sample_size // 2}
    sampled = {name: generator.sample(ids, min(half[name], len(ids))) for name, ids in strata.items()}
    return sampled, {name: len(ids) for name, ids in strata.items()}


def draw_user_lean_sample(verdict_records, sample_size, seed):
    """({stratum "lean|condition": [answer ids]}, {stratum: population count}), equal allocation with the shortfall of small strata
    shared out round-robin among strata that still have answers."""
    strata = {}
    for answer_id in sorted(verdict_records):
        record = verdict_records[answer_id]
        strata.setdefault(f"{record['verdict']['user_lean']}|{record['condition']}", []).append(answer_id)
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
    return ({name: shuffled[name][:allocation[name]] for name in shuffled}, {name: len(ids) for name, ids in strata.items()})


def build_command(answers_file, template_path, schema_path, output_parent, comparison_file=None, comparison_condition=None):
    parts = ["python3", FRAMEWORK_SCRIPT, "--answers", str(answers_file), "--judge-prompt-file", str(template_path),
             "--reply-schema-file", str(schema_path), "--judge-model", HAIKU_MODEL, "--output-parent-folder", str(output_parent)]
    if comparison_file:
        parts += ["--comparison-answers", str(comparison_file), "--comparison-condition", comparison_condition]
    return " ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--completeness-sample-size", type=int, default=150)
    parser.add_argument("--user-lean-sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    run_folder = arguments.run_folder
    configuration = read_run_configuration(run_folder)
    judges = configuration["gemma_self_judging_and_regeneration"]["framework_judges"]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = run_folder / HAIKU_SAMPLE_INPUTS_FOLDER_NAME
    output_folder.mkdir(parents=True, exist_ok=True)
    haiku_output_parent = run_folder / HAIKU_SAMPLE_OUTPUTS_FOLDER_NAME
    kept_file = sorted((run_folder / "generation_outputs" / "final_outputs").glob("kept_answers_*.jsonl"))[-1]
    kept_answers = {record["answer_id"]: record for record in load_answer_records([kept_file])}
    all_answers = {record["answer_id"]: record for record in load_answer_records([run_folder])}
    manifest = {"gemma_judge_sampling_in_the_run_configuration": configuration["gemma_self_judging_and_regeneration"].get("framework_judge_sampling"),
                "population_note": ("strata populations are counts among the answers Gemma judged on the pod, which are a seeded random sample "
                                    "of the kept answers (or all of them) when the run configuration says so"),
                "seed": arguments.seed, "created_at": datetime.now().isoformat(timespec="seconds"), "kept_answers_file": kept_file.name,
                "haiku_output_parent_folder": str(haiku_output_parent)}
    commands = []

    if "completeness_and_competence" in judges:
        judge_record = judges["completeness_and_competence"]
        gemma_folder = find_gemma_judge_folder(run_folder, judge_record)
        verdicts = read_ok_verdict_records(gemma_folder)
        sampled, population = draw_completeness_sample(verdicts, arguments.completeness_sample_size, arguments.seed)
        ids = [answer_id for name in sampled for answer_id in sampled[name]]
        answers_path = output_folder / f"haiku_sample_check_completeness_kept_answers_to_judge_{stamp}.jsonl"
        reference_path = output_folder / f"haiku_sample_check_completeness_paired_no_system_prompt_reference_answers_{stamp}.jsonl"
        write_jsonl(answers_path, [kept_answers[i] for i in ids])
        write_jsonl(reference_path, [all_answers[verdicts[i]["comparison_answer_id"]] for i in ids])
        manifest["completeness_and_competence"] = {
            "gemma_judge_folder": str(gemma_folder), "answers_file": answers_path.name, "paired_reference_answers_file": reference_path.name,
            "sampled_answer_ids_by_stratum": sampled, "population_count_by_stratum": population,
            "sample_count_by_stratum": {name: len(chosen) for name, chosen in sampled.items()},
            "stratum_definition": "flagged: Gemma said yes to left_out_part_of_the_request or clearly_worse_on_everything_else"}
        commands.append(build_command(answers_path, f"{PROMPTS_FOLDER_RELATIVE}/{Path(judge_record['prompt_template']['path_inside_run_folder']).name}",
                                      f"{PROMPTS_FOLDER_RELATIVE}/{Path(judge_record['reply_schema']['path_inside_run_folder']).name}",
                                      haiku_output_parent, reference_path, judge_record["comparison_condition"]))
    judge_record = judges["user_lean_and_agreement"]
    gemma_folder = find_gemma_judge_folder(run_folder, judge_record)
    verdicts = read_ok_verdict_records(gemma_folder)
    sampled, population = draw_user_lean_sample(verdicts, arguments.user_lean_sample_size, arguments.seed)
    ids = sorted(answer_id for chosen in sampled.values() for answer_id in chosen)
    answers_path = output_folder / f"haiku_sample_check_user_lean_answers_to_judge_{stamp}.jsonl"
    write_jsonl(answers_path, [all_answers[i] for i in ids])
    manifest["user_lean_and_agreement"] = {
        "gemma_judge_folder": str(gemma_folder), "answers_file": answers_path.name, "sampled_answer_ids_by_stratum": sampled,
        "population_count_by_stratum": population, "sample_count_by_stratum": {name: len(chosen) for name, chosen in sampled.items()},
        "stratum_definition": "Gemma's user_lean value and the answer's condition, written lean|condition"}
    commands.append(build_command(answers_path, f"{PROMPTS_FOLDER_RELATIVE}/{Path(judge_record['prompt_template']['path_inside_run_folder']).name}",
                                  f"{PROMPTS_FOLDER_RELATIVE}/{Path(judge_record['reply_schema']['path_inside_run_folder']).name}",
                                  haiku_output_parent))
    manifest["commands"] = commands
    manifest_path = output_folder / f"haiku_sample_check_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote the samples and {manifest_path}")
    for key in ("completeness_and_competence", "user_lean_and_agreement"):
        if key in manifest:
            print(f"{key}: sample counts {manifest[key]['sample_count_by_stratum']} of populations {manifest[key]['population_count_by_stratum']}")
    print("\nRun Haiku on each sample (add --dry-run first to see the rendered prompt and the call count; subscription quota, not a paid API):")
    for command in commands:
        print("\n" + command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
