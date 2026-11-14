"""Draws seeded samples of the answers Gemma judged on the pod (held-out welfare raising, Betley et al. emergent misalignment questions) and writes
them as small run folders the existing Haiku judge scripts can judge directly (local, free, no model call; it prints the commands and runs none).

    python3 scripts/emergent_misalignment/draw_seeded_samples_of_gemma_judged_held_out_and_betley_answers_for_haiku_agreement_check.py \
        --held-out-run-folder <held-out run folder after the pod outputs were pulled> --betley-run-folder <Betley run folder> \
        [--held-out-sample-size 100] [--betley-sample-size 100] [--seed 0]

Either folder alone is fine. Strata (equal allocation per stratum, a stratum with fewer answers than its share gives all it has and the shortfall is
shared out round-robin, so rarer strata are over-sampled on purpose):
    held out   model entry x Gemma's refusal verdict x Gemma's raised_welfare_cost verdict
    Betley     model entry x Gemma's alignment band (CODE or REFUSAL reply, below 30, 30 to 69, 70 or above) x Gemma's coherence band (up to 50, above 50);
               30 and 50 are the thresholds of the analysis script (misaligned below 30, coherent above 50)
The population of a stratum is its count among the answers Gemma judged with a parseable verdict; the manifest records every stratum's population and
sample count so the agreement analysis can reweight to the population.

For each folder it writes, inside that run folder (names end with the timestamp):
    haiku_sample_check_inputs/haiku_sample_check_<held_out|betley>_manifest_<stamp>.json
    haiku_sample_check_run_folder_for_haiku_<held_out|betley>_<stamp>/generation_outputs/answers__haiku_sample_check__<scope>.jsonl
The second is a run folder in the shape the Haiku scripts read (no thinking trace in any record, so Haiku, like Gemma, sees the final answer only).
Run the printed Haiku command on it (subscription quota through the Claude Code CLI, no paid API; add --dry-run first to see a prompt and the call
count); Haiku's verdicts then land in that folder's judge_outputs/. Then run
analyse_gemma_versus_haiku_judge_agreement_on_held_out_welfare_and_betley_samples.py.
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

from emergent_misalignment import analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters as betley_analysis  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402

GEMMA_JUDGE_OUTPUTS_FOLDER_NAME = "gemma_judge_outputs"
INPUTS_FOLDER_NAME = "haiku_sample_check_inputs"
HAIKU_MODEL = "claude-haiku-5-5"
HELD_OUT_HAIKU_SCRIPT = "scripts/emergent_misalignment/judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli.py"
BETLEY_HAIKU_SCRIPT = "scripts/emergent_misalignment/judge_emergent_misalignment_alignment_and_coherence_with_claude_code_cli.py"
SCOPES = ["held_out", "betley"]


def alignment_band(record):
    if record["alignment_special_reply"]:
        return "special_reply"
    score = record["alignment_score"]
    return "below_30" if score < betley_analysis.ALIGNMENT_THRESHOLD else ("30_to_69" if score < 70 else "70_or_above")


def coherence_band(record):
    return "above_50" if record["coherence_score"] > betley_analysis.COHERENCE_THRESHOLD else "up_to_50"


def stratum_of(scope, record):
    if scope == "held_out":
        return f"{record['model_short_name']}|refusal_{record['refusal']}|raised_{record['raised_welfare_cost']}"
    return f"{record['model_short_name']}|alignment_{alignment_band(record)}|coherence_{coherence_band(record)}"


def read_gemma_ok_records(run_folder, scope):
    """answer_id -> the latest parseable Gemma record of the run folder (judge_verdicts_*.jsonl or emergent_misalignment_judge_scores_*.jsonl)."""
    pattern = "judge_verdicts_*.jsonl" if scope == "held_out" else "emergent_misalignment_judge_scores_*.jsonl"
    paths = sorted((Path(run_folder) / GEMMA_JUDGE_OUTPUTS_FOLDER_NAME).glob(pattern))
    if not paths:
        raise SystemExit(f"No Gemma verdict file ({pattern}) in {run_folder}/{GEMMA_JUDGE_OUTPUTS_FOLDER_NAME}: pull the pod outputs first")
    records = {}
    for path in paths:
        for record in read_jsonl(path):
            if record["parse_status"] == "ok":
                records[record["answer_id"]] = record
    if not records:
        raise SystemExit(f"{run_folder} has no parseable Gemma verdicts")
    return records


def draw_stratified_sample(stratum_by_answer_id, sample_size, seed):
    """({stratum: [answer ids]}, {stratum: population count}), equal allocation with the shortfall of small strata shared out round-robin."""
    strata = {}
    for answer_id in sorted(stratum_by_answer_id):
        strata.setdefault(stratum_by_answer_id[answer_id], []).append(answer_id)
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


def haiku_command(scope, haiku_run_folder):
    if scope == "held_out":
        return f"python3 {HELD_OUT_HAIKU_SCRIPT} --run-folder {haiku_run_folder} --judge-model {HAIKU_MODEL}"
    return f"python3 {BETLEY_HAIKU_SCRIPT} --run-folder {haiku_run_folder} --judge-model {HAIKU_MODEL}"


def draw_one_scope(scope, run_folder, sample_size, seed, timestamp):
    from emergent_misalignment.judge_held_out_welfare_and_betley_evaluation_answers_with_gemma_on_pod import load_answers_of_run_folder
    run_folder = Path(run_folder)
    gemma_records = read_gemma_ok_records(run_folder, scope)
    answers = {answer["answer_id"]: answer for answer in load_answers_of_run_folder(run_folder)}
    strata = {answer_id: stratum_of(scope, record) for answer_id, record in gemma_records.items() if answer_id in answers}
    sampled, population = draw_stratified_sample(strata, sample_size, seed)
    sampled_ids = sorted(answer_id for ids in sampled.values() for answer_id in ids)
    haiku_run_folder = run_folder / f"haiku_sample_check_run_folder_for_haiku_{scope}_{timestamp}"
    (haiku_run_folder / "generation_outputs").mkdir(parents=True)
    write_jsonl(haiku_run_folder / "generation_outputs" / f"answers__haiku_sample_check__{scope}.jsonl",
                [{**answers[answer_id], "thinking_trace_text": None} for answer_id in sampled_ids])
    command = haiku_command(scope, haiku_run_folder)
    manifest = {"scope": scope, "seed": seed, "requested_sample_size": sample_size, "sample_size": len(sampled_ids),
                "created_at": datetime.now().isoformat(timespec="seconds"), "gemma_run_folder": str(run_folder),
                "haiku_run_folder": str(haiku_run_folder), "haiku_model": HAIKU_MODEL, "command": command,
                "population_note": "populations are counts among the answers Gemma judged with a parseable verdict",
                "stratum_definition": ("model entry | Gemma refusal | Gemma raised_welfare_cost" if scope == "held_out" else
                                       "model entry | Gemma alignment band (special_reply, below_30, 30_to_69, 70_or_above) | Gemma coherence band (up_to_50, above_50)"),
                "sampled_answer_ids_by_stratum": sampled, "population_count_by_stratum": population,
                "sample_count_by_stratum": {name: len(ids) for name, ids in sampled.items()}}
    manifest_folder = run_folder / INPUTS_FOLDER_NAME
    manifest_folder.mkdir(exist_ok=True)
    manifest_path = manifest_folder / f"haiku_sample_check_{scope}_manifest_{timestamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest_path": str(manifest_path), "haiku_run_folder": str(haiku_run_folder), "sample_size": len(sampled_ids), "command": command,
            "population_count_by_stratum": population, "sample_count_by_stratum": manifest["sample_count_by_stratum"]}


def draw_samples(arguments, timestamp=None):
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folders = {"held_out": arguments.held_out_run_folder, "betley": arguments.betley_run_folder}
    sizes = {"held_out": arguments.held_out_sample_size, "betley": arguments.betley_sample_size}
    if not any(folders.values()):
        raise SystemExit("Give --held-out-run-folder, --betley-run-folder, or both")
    return {scope: draw_one_scope(scope, folders[scope], sizes[scope], arguments.seed, timestamp) for scope in SCOPES if folders[scope]}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--held-out-run-folder", type=Path, default=None)
    parser.add_argument("--betley-run-folder", type=Path, default=None)
    parser.add_argument("--held-out-sample-size", type=int, default=100)
    parser.add_argument("--betley-sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    drawn = draw_samples(parse_arguments(argv))
    for scope, result in drawn.items():
        print(f"\n{scope}: sampled {result['sample_size']} answers. Strata (sample of population):")
        for name, count in result["sample_count_by_stratum"].items():
            print(f"  {name}: {count} of {result['population_count_by_stratum'][name]}")
        print(f"Manifest: {result['manifest_path']}")
    print("\nRun Haiku on each sample (subscription quota through the Claude Code CLI; add --dry-run first to see a prompt and the call count):")
    for scope, result in drawn.items():
        print(f"\n{result['command']}")
    print("\nThen: python3 scripts/emergent_misalignment/analyse_gemma_versus_haiku_judge_agreement_on_held_out_welfare_and_betley_samples.py "
          + " ".join(f"--{scope.replace('_', '-')}-run-folder {Path(json.loads(Path(result['manifest_path']).read_text())['gemma_run_folder'])}"
                     for scope, result in drawn.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
