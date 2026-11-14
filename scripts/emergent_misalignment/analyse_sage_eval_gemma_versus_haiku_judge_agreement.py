"""Gemma-versus-Haiku agreement on the SAGE-Eval sample check (local, free, no model call).

    python3 scripts/emergent_misalignment/analyse_sage_eval_gemma_versus_haiku_judge_agreement.py --run-folder <run folder>

Reads the manifest and answers file in <run folder>/haiku_sample_check_inputs/, Gemma's verdicts in generation_outputs/ and Haiku's verdicts in
<run folder>/judge_outputs_haiku_sample_check/. For each of the two fields (raised_hazard_or_gave_safer_alternative, refused_or_failed_to_help): a confusion table (Haiku in rows,
Gemma in columns), percent agreement and Cohen's kappa over the answers both judges gave a readable verdict for, Gemma's precision and recall against Haiku's true, and the same
figures reweighted to the population (each stratum weighted by its population count over its sample count, because the sample over-represents the rarer strata). Also the
agreement on the pass decision, on the paper-style pass (raised or refused) and, per model entry, the pass rate each judge gives on the sample. Writes
<run folder>/gemma_versus_haiku_judge_agreement_on_sage_eval_sample_<timestamp>.md.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import base_versus_lora_adapters_analysis_helpers as helpers  # noqa: E402
from emergent_misalignment import draw_seeded_sample_of_sage_eval_answers_for_haiku_agreement_check as drawing  # noqa: E402
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as protocol  # noqa: E402
from emergent_misalignment.analyse_baseline_welfare_raising_rates import markdown_table  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

FIELDS = [protocol.RAISED_FIELD, protocol.REFUSED_FIELD]


def read_haiku_verdicts(haiku_parent_folder):
    verdicts = {}
    for path in sorted(Path(haiku_parent_folder).glob("*/verdicts_*.jsonl")):
        for record in read_jsonl(path):
            if record["parse_status"] == "ok":
                verdicts[record["answer_id"]] = record["verdict"]
    return verdicts


def weighted_confusion(pairs, weights):
    """{(haiku, gemma): weight} over boolean pairs."""
    table = {(h, g): 0.0 for h in (True, False) for g in (True, False)}
    for (haiku_value, gemma_value), weight in zip(pairs, weights):
        table[(haiku_value, gemma_value)] += weight
    return table


def agreement_figures(table):
    """(percent agreement, Cohen's kappa, Gemma precision against Haiku true, Gemma recall against Haiku true), each None when undefined."""
    total = sum(table.values())
    if not total:
        return None, None, None, None
    observed = (table[(True, True)] + table[(False, False)]) / total
    haiku_true, gemma_true = table[(True, True)] + table[(True, False)], table[(True, True)] + table[(False, True)]
    expected = (haiku_true * gemma_true + (total - haiku_true) * (total - gemma_true)) / total ** 2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    precision = table[(True, True)] / gemma_true if gemma_true else None
    recall = table[(True, True)] / haiku_true if haiku_true else None
    return observed, kappa, precision, recall


def number(value, percent=False):
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%" if percent else f"{value:.2f}"


def build_report(run_folder, manifest, gemma, haiku, stamp):
    answer_ids = [answer_id for answer_id in gemma if answer_id in haiku and answer_id in manifest["answer_stratum"]]
    strata = [manifest["answer_stratum"][answer_id] for answer_id in answer_ids]
    population, sampled = manifest["strata_population_counts"], manifest["strata_sample_counts"]
    unit = {name: 1.0 for name in population}
    reweight = {name: population[name] / sampled[name] for name in sampled if sampled[name]}
    decisions = {"pass (raised the hazard or gave the safer alternative)": lambda v: v[protocol.RAISED_FIELD],
                 "paper-style pass (raised, or refused)": lambda v: v[protocol.RAISED_FIELD] or v[protocol.REFUSED_FIELD]}
    lines = ["# SAGE-Eval: Gemma versus Haiku judge agreement on the sample check", "", f"Run folder: `{run_folder}`", "",
             f"Seed {manifest['seed']}, {manifest['sample_size']} answers sampled, {len(answer_ids)} with a readable verdict from both judges ({manifest['sample_size'] - len(answer_ids)} "
             f"without). Strata (Gemma's verdicts): population {population}, sample {sampled}. The sample over-represents the rarer strata, so raw agreement is not the population "
             f"rate; the reweighted rows are. Analysed {stamp}.", ""]
    rows = []
    for label, extractor in [(field, (lambda v, field=field: v[field])) for field in FIELDS] + list(decisions.items()):
        pairs = [(extractor(haiku[a]), extractor(gemma[a])) for a in answer_ids]
        for weighting_name, weights in (("sample", [unit[s] for s in strata]), ("reweighted to population", [reweight.get(s, 1.0) for s in strata])):
            table = weighted_confusion(pairs, weights)
            observed, kappa, precision, recall = agreement_figures(table)
            rows.append([label, weighting_name, f"{table[(True, True)]:.1f} / {table[(True, False)]:.1f} / {table[(False, True)]:.1f} / {table[(False, False)]:.1f}",
                         number(observed, True), number(kappa), number(precision, True), number(recall, True)])
    lines += [markdown_table(["Field or decision", "Weighting", "Haiku true & Gemma true / Haiku true & Gemma false / Haiku false & Gemma true / both false", "Agreement", "Cohen's kappa",
                              "Gemma precision against Haiku true", "Gemma recall against Haiku true"], rows), ""]
    models = sorted({manifest["answer_model"][a] for a in answer_ids})
    rows = []
    for model in models:
        chosen = [a for a in answer_ids if manifest["answer_model"][a] == model]
        for label, extractor in decisions.items():
            rows.append([model, label, len(chosen), number(sum(extractor(gemma[a]) for a in chosen) / len(chosen), True),
                         number(sum(extractor(haiku[a]) for a in chosen) / len(chosen), True)])
    lines += ["## Pass rate each judge gives on the sample, per model entry (unweighted sample, not a population rate)", "",
              markdown_table(["Model entry", "Decision", "Answers", "Gemma", "Haiku"], rows), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run_folder = arguments.run_folder
    configuration = helpers.load_latest_run_configuration(run_folder)
    manifest_path = sorted((run_folder / drawing.HAIKU_INPUTS_FOLDER_NAME).glob("haiku_sample_check_sage_eval_manifest_*.json"))[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_records = read_jsonl(run_folder / drawing.HAIKU_INPUTS_FOLDER_NAME / manifest["answers_file"])
    gemma_all = drawing.read_gemma_verdicts(run_folder, configuration["model_order"])
    manifest["answer_stratum"] = {record["answer_id"]: drawing.stratum_name(gemma_all[record["answer_id"]]) for record in sample_records}
    manifest["answer_model"] = {record["answer_id"]: record["model_short_name"] for record in sample_records}
    haiku = read_haiku_verdicts(manifest["haiku_output_parent_folder"] if Path(manifest["haiku_output_parent_folder"]).exists() else run_folder / drawing.HAIKU_OUTPUTS_FOLDER_NAME)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report = build_report(run_folder, manifest, gemma_all, haiku, stamp)
    path = run_folder / f"gemma_versus_haiku_judge_agreement_on_sage_eval_sample_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
