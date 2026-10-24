"""Synthetic run folder generator and smoke test for analyze_preachiness_scores.py.

Builds a fake run folder (documents, the real 62-condition structure, per-token log-probabilities with planted
effects) and, when run as a script, writes it into data/log_probability_persona_check_runs/small_sample_checks/
and runs the full analysis on it. No GPU, no API calls.

Usage, from the repository root:
    python scripts/log_probability_persona_check/smoke_test_analyze_preachiness_scores.py
"""

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = SCRIPT_DIRECTORY.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (SCRIPTS_DIRECTORY, SCRIPT_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from shared.run_folder_management import timestamp_now  # noqa: E402

CONTROL_PERSONA_NAMES = ["animal_husbandry_expertise", "veterinary_clinic_manager", "zoo_tour_guide",
                         "wildlife_photographer", "pet_shop_assistant"]
TONE_VALUES = ["neutral or journalistic", "supportive, like an animal advocate", "skeptical or displeased"]
LANGUAGES = ["English", "Hindi", "Mandarin"]
SPEC_VARIABLE_NAMES = ["document_type", "culture", "tone", "resolution", "tradeoff", "centrality", "domain",
                       "sentient_category", "decision_scale", "naming", "tech_savvy", "reasoning_featured"]
SMALL_SAMPLE_CHECKS_DIRECTORY = REPOSITORY_ROOT / "data" / "log_probability_persona_check_runs" / "small_sample_checks"


def build_condition_records():
    """The real 62-condition structure: no_system_prompt, generic, 30 preachy, 25 control (5 personas x 5), 5 welfare."""
    records = [
        {"condition_id": "no_system_prompt", "condition_group": "no_system_prompt", "persona_name": "no_system_prompt",
         "rewording_index": None, "system_prompt": None},
        {"condition_id": "generic", "condition_group": "generic", "persona_name": "generic", "rewording_index": None,
         "system_prompt": "You are an AI assistant."},
    ]
    for index in range(30):
        records.append({"condition_id": f"preachy__rewording_{index:02d}", "condition_group": "preachy",
                        "persona_name": "preachy", "rewording_index": index, "system_prompt": f"preachy {index}"})
    for persona_name in CONTROL_PERSONA_NAMES:
        for index in range(5):
            records.append({"condition_id": f"{persona_name}__rewording_{index:02d}", "condition_group": "control",
                            "persona_name": persona_name, "rewording_index": index,
                            "system_prompt": f"{persona_name} {index}"})
    for index in range(5):
        records.append({"condition_id": f"proportionate_welfare__rewording_{index:02d}",
                        "condition_group": "proportionate_welfare", "persona_name": "proportionate_welfare",
                        "rewording_index": index, "system_prompt": f"welfare {index}"})
    return records


def write_jsonl_lines(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_synthetic_run_folder(run_folder, seed=0, tokens_per_document=200, planted_shift_per_token=0.3,
                               documents_per_effect_kind=4, planted_span_length=32, noise_standard_deviation=0.05,
                               include_rewrites=True, condition_ids_dropped_for_document=None, rewrites_per_direction=2):
    """Writes a fake run folder and returns a dict describing the planted effects.

    Original documents come in three kinds: 'elevated' (preachy log-probabilities raised by planted_shift_per_token
    over a planted span of tokens), 'suppressed' (lowered by the same amount) and 'null' (no effect). Every preachy
    rewording's shift is scaled by a factor between 0.9 and 1.1. If include_rewrites, two preachier rewrites
    (planted shift added to a null source) and two plainer rewrites (planted shift subtracted) are added
    (rewrites_per_direction of each instead of two; needs documents_per_effect_kind of at least twice that).
    condition_ids_dropped_for_document: optional {document_id: [condition ids]} to remove rows for that document
    (to test mismatch detection)."""
    run_folder = Path(run_folder)
    generator = np.random.default_rng(seed)
    conditions = build_condition_records()
    condition_ids = [condition["condition_id"] for condition in conditions]
    preachy_rows = [i for i, c in enumerate(conditions) if c["condition_group"] == "preachy"]
    (run_folder / "scoring_inputs").mkdir(parents=True, exist_ok=True)
    output_directory = run_folder / "scoring_outputs" / "per_token_log_probabilities"
    output_directory.mkdir(parents=True, exist_ok=True)

    document_plans = []  # (document_id, kind, source_document_id, rewrite_direction, signed_shift, span_start)
    kind_counter = 0
    for kind, sign in (("elevated", 1.0), ("suppressed", -1.0), ("null", 0.0)):
        for index in range(documents_per_effect_kind):
            document_plans.append((f"matrix_{kind_counter:06d}", kind, f"matrix_{kind_counter:06d}", None,
                                   sign * planted_shift_per_token, 20 + 10 * index + 3 * kind_counter))
            kind_counter += 1
    null_document_ids = [plan[0] for plan in document_plans if plan[1] == "null"]
    if include_rewrites:
        for index, (direction, sign) in enumerate([("preachier", 1.0)] * rewrites_per_direction
                                                    + [("plainer", -1.0)] * rewrites_per_direction):
            source_id = null_document_ids[index]
            document_plans.append((f"{source_id}__{direction}", f"rewrite_{direction}", source_id, direction,
                                   sign * planted_shift_per_token, 50))

    documents = []
    planted = {"documents": {}, "condition_ids": condition_ids, "planted_span_length": planted_span_length,
               "planted_shift_per_token": planted_shift_per_token}
    for number, (document_id, kind, source_id, direction, signed_shift, span_start) in enumerate(document_plans):
        language = LANGUAGES[number % len(LANGUAGES)]
        variables = {name: f"{name}_value_{number % 3}" for name in SPEC_VARIABLE_NAMES}
        variables["tone"] = TONE_VALUES[number % len(TONE_VALUES)]
        variables["culture"] = f"{language} speaking community"
        documents.append({"document_id": document_id, "source_document_id": source_id, "rewrite_direction": direction,
                          "content": " ".join(f"tok{t}" for t in range(tokens_per_document)), "language": language,
                          "variables": variables, "generation_prompt": "Write a document from a world where ..."})
        baseline = generator.normal(-2.0, 0.5, size=tokens_per_document)
        log_probabilities = np.empty((len(conditions), tokens_per_document))
        shift_mask = np.zeros(tokens_per_document)
        shift_mask[span_start:span_start + planted_span_length] = 1.0
        for row, condition in enumerate(conditions):
            noise = generator.normal(0, noise_standard_deviation, size=tokens_per_document)
            log_probabilities[row] = baseline + (0 if condition["condition_id"] == "generic" else noise)
            if row in preachy_rows:
                log_probabilities[row] += signed_shift * generator.uniform(0.9, 1.1) * shift_mask
        planted["documents"][document_id] = {"kind": kind, "span_start": span_start, "signed_shift": signed_shift,
                                             "source_document_id": source_id, "rewrite_direction": direction}
        stored_condition_ids = list(condition_ids)
        dropped = (condition_ids_dropped_for_document or {}).get(document_id, [])
        keep = [i for i, cid in enumerate(condition_ids) if cid not in dropped]
        stored_condition_ids = [condition_ids[i] for i in keep]
        np.savez_compressed(
            output_directory / f"{document_id}.npz",
            document_token_ids=np.arange(tokens_per_document, dtype=np.int32),
            document_token_strings=np.array([f"tok{t} " for t in range(tokens_per_document)]),
            condition_ids=np.array(stored_condition_ids),
            log_probabilities=log_probabilities[keep].astype(np.float32))
    write_jsonl_lines(run_folder / "scoring_inputs" / "documents.jsonl", documents)
    write_jsonl_lines(run_folder / "scoring_inputs" / "conditions.jsonl", conditions)
    return planted


def run_smoke_test():
    from analyze_preachiness_scores import run_analysis
    timestamp = timestamp_now()
    run_folder = SMALL_SAMPLE_CHECKS_DIRECTORY / f"analyze_preachiness_scores_synthetic_run_{timestamp}"
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists; wait a minute and rerun")
    build_synthetic_run_folder(run_folder)
    analysis_folder = run_analysis(run_folder, timestamp=timestamp)
    expected = ["per_document_scores", "per_document_significance_tests", "extreme_spans", "rewording_agreement",
                "scores_by_spec_variable", "positive_control_check", "analysis_notes"]
    for stem in expected:
        assert list(analysis_folder.glob(f"{stem}_*")), f"missing output {stem}"
    assert len(list((analysis_folder / "figures").glob("*.png"))) == 5, "expected 5 figures"
    print(f"Smoke test passed. Output folder: {run_folder}")
    return run_folder


if __name__ == "__main__":
    run_smoke_test()
