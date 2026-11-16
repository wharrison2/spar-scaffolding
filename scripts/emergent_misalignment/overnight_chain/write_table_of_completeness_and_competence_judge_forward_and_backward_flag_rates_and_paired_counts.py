"""Writes a markdown table of the completeness and competence judge's flag rates run forwards (system-prompted answer judged) and backwards (no-system-prompt
answer judged), the paired 2x2 per flag over the prompts judged in both directions, and the percentage-point differences.

    python3 scripts/emergent_misalignment/overnight_chain/write_table_of_completeness_and_competence_judge_forward_and_backward_flag_rates_and_paired_counts.py \
        --run-folder data/emergent_misalignment_runs/overnight_treatment_arm_chain_<timestamp>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

CHAIN_FOLDER = Path(__file__).resolve().parent
for directory in (CHAIN_FOLDER.parents[2], CHAIN_FOLDER.parents[1], CHAIN_FOLDER.parent, CHAIN_FOLDER.parent / "treatment_arm_training_data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses as after_chain)
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_completeness_and_competence_with_judged_and_comparison_answers_swapped_through_the_paid_api as backwards)

FLAGS = {"left_out_part_of_the_request": "left out part of the request", "clearly_worse_on_everything_else": "clearly worse on everything else"}
EITHER = "either flag"


def parseable_verdicts_by_prompt_gid(judge_folder):
    """{prompt_gid: verdict} of the latest parseable verdict per prompt in a judge folder."""
    verdicts = {}
    for path in sorted(Path(judge_folder).glob("verdicts_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line) if line.strip() else None
            if record and record["parse_status"] == "ok":
                verdicts[record["prompt_gid"]] = record["verdict"]
    return verdicts


def flagged(verdict, flag):
    keys = FLAGS if flag == EITHER else [flag]
    return any(verdict[key] == "yes" for key in keys)


def percent(count, total):
    return f"{100 * count / total:.1f}%" if total else "n/a"


def build_markdown(forward, backward):
    """forward and backward are {prompt_gid: verdict}."""
    both = sorted(set(forward) & set(backward))
    lines = [f"Prompts judged forwards {len(forward)}, backwards {len(backward)}, in both directions {len(both)}.", "",
             "| Flag | Forward rate (kept answer flagged) | Backward rate (no-system-prompt answer flagged) | Forward rate, both-direction prompts | Backward rate, both-direction prompts | "
             "Backward minus forward (points) | Forward minus backward (points) |", "|---|---|---|---|---|---|---|"]
    pair_lines = ["", "| Flag | Flagged forwards and backwards | Flagged forwards only | Flagged backwards only | Flagged in neither |", "|---|---|---|---|---|"]
    for flag in [*FLAGS, EITHER]:
        label = FLAGS.get(flag, EITHER)
        forward_count, backward_count = sum(flagged(v, flag) for v in forward.values()), sum(flagged(v, flag) for v in backward.values())
        forward_paired, backward_paired = sum(flagged(forward[g], flag) for g in both), sum(flagged(backward[g], flag) for g in both)
        both_flagged = sum(flagged(forward[g], flag) and flagged(backward[g], flag) for g in both)
        only_forward, only_backward = forward_paired - both_flagged, backward_paired - both_flagged
        neither = len(both) - both_flagged - only_forward - only_backward
        difference = 100 * (backward_paired - forward_paired) / len(both) if both else float("nan")
        lines.append(f"| {label} | {percent(forward_count, len(forward))} ({forward_count}/{len(forward)}) | {percent(backward_count, len(backward))} ({backward_count}/{len(backward)}) | "
                     f"{percent(forward_paired, len(both))} | {percent(backward_paired, len(both))} | {difference:+.1f} | {-difference:+.1f} |")
        pair_lines.append(f"| {label} | {both_flagged} | {only_forward} | {only_backward} | {neither} |")
    return "\n".join(lines + pair_lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    arguments = parser.parse_args(argv)
    paths = after_chain.RunPaths(arguments.run_folder)
    answers, _ = backwards.backwards_inputs(paths)
    sample_gids = {answer["prompt_gid"] for answer in answers}
    forward = {g: v for g, v in parseable_verdicts_by_prompt_gid(after_chain.haiku_judge_folder(paths, after_chain.framework_judge_specification(
        paths, "completeness_and_competence"))).items() if g in sample_gids}
    backward = parseable_verdicts_by_prompt_gid(backwards.backwards_judge_folder(arguments.run_folder, backwards.swapped_specification(paths)))
    output_path = arguments.run_folder / backwards.BACKWARDS_FOLDER_NAME / f"completeness_and_competence_forward_versus_backward_flag_rates_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md"
    markdown = build_markdown(forward, backward)
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"written to {output_path}")


if __name__ == "__main__":
    main()
