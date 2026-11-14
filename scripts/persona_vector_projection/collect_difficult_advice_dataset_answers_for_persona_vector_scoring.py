"""Collect the difficult-advice (DAD) model-per-stage experiment answers into one machine-readable document set for
persona vector scoring, and attach the pairwise judge verdicts to the answers they concern.

Reads only (never modifies) dad-model-experiment-2026-11-01/. Makes no model calls and launches nothing. Cheap to
rerun: the whole job takes a few seconds, and a run folder that already holds its completion marker is left
untouched unless --force is passed (then every output is rewritten deterministically).

Outputs, in data/persona_vector_projection_runs/difficult_advice_dataset_answers_for_persona_vector_scoring_<timestamp>/:
    difficult_advice_answers_<timestamp>.jsonl                  one record per answer text
    difficult_advice_judgement_verdicts_per_answer_<timestamp>.jsonl   the verdicts, per answer
    difficult_advice_prompts_<timestamp>.jsonl                  the prompts with their scenario cards
    retally_of_win_rates_against_experiment_reports_<timestamp>.json   re-tally versus the experiment's reports
    README_<timestamp>.md                                       counts, field meanings, shared prompts, oddities

Which texts each judgement compared (from the experiment's code/judge.py, `comparisons`):
    stage2   arm = the arm's stage-2 answer;                     reference = Opus 5.5 stage-2 answer
    stage3a  arm = the arm's rewrite of the Opus stage-2 answer; reference = Opus 5.5's rewrite of its own answer
    stage3b  arm = Opus 5.5's rewrite of the arm's stage-2 answer; reference = Opus 5.5's rewrite of its own answer
    controls arm = plain-model baseline answer / planted-flaw edit of the Opus answer / repeated Opus stage-2 answer /
             GPT or Argon stage-2 answer (self-preference sample); reference = Opus 5.5 stage-2 answer
The tally rules (an order flip counts as a tie; win rate counts a tie as half) are copied from judge.py's `outcome`
and `pair_report`, and key_of (md5 of the user message, first 12 hex characters) from `key_of`.

Run from the repository root:
    python3 scripts/persona_vector_projection/collect_difficult_advice_dataset_answers_for_persona_vector_scoring.py
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from shared.run_folder_management import choose_run_folder  # noqa: E402

REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
DEFAULT_SOURCE_FOLDER = REPOSITORY_ROOT / "dad-model-experiment-2026-11-01"
DEFAULT_OUTPUT_PARENT = REPOSITORY_ROOT / "data" / "persona_vector_projection_runs"
RUN_NAME = "difficult_advice_dataset_answers_for_persona_vector_scoring"

REFERENCE_MODEL = "opus55"
MODELS_WITH_FULL_STAGE_2 = ("opus55", "sonnet55", "gpt-sol", "argon")
NON_REFERENCE_MODELS = ("sonnet55", "gpt-sol", "argon")
PILOT_MODELS = ("opus55", "sonnet55", "gpt-sol", "argon", "gem31pro")
MODEL_DISPLAY_NAMES = {
    "opus55": "Claude Opus 5.5", "sonnet55": "Claude Sonnet 5.5", "gpt-sol": "GPT-6.1 Sol",
    "argon": "Gemini 4 Argon", "gem31pro": "Gemini 3.1 Pro preview",
}
JUDGEMENT_AXES = {"overall": "winner", "welfare": "welfare_winner", "delivery": "delivery_winner"}
FAILURE_CHECKS = ("too_little", "too_much", "inconsistent", "caves")
REFUSAL_OPENING_PATTERN = re.compile(
    r"^\W*(i can'?t|i cannot|i won'?t|i'm not able to|i am not able to|i'm unable to|sorry, but i)", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------------
# Small helpers (key_of and the tally rules mirror judge.py)
# ---------------------------------------------------------------------------------------------------

def key_of(user_message):
    return hashlib.md5(user_message.encode()).hexdigest()[:12]


def word_count(text):
    return len((text or "").split())


def read_jsonl_with_line_numbers(path):
    """[(1-based line number, row)] for every non-blank line."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                rows.append((line_number, json.loads(line)))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_run_folder(source_folder, run_suffix):
    """The single runs/*_local_expC-<run_suffix> folder (the experiment's own naming)."""
    matches = sorted((Path(source_folder) / "runs").glob(f"*_local_expC-{run_suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one run folder for '{run_suffix}', found {matches}")
    return matches[0]


def outcome_for_arm(winner, arm_side):
    """judge.py `outcome`: 'tie', 'arm' or 'ref' for one reading, seen from the arm's side."""
    return "tie" if winner == "tie" else "arm" if winner == arm_side else "ref"


def combine_two_orders(outcome_arm_first_or_a, outcome_other_order):
    """Both orders agree: that outcome. Otherwise (an order flip) a tie, as judge.py does."""
    if outcome_arm_first_or_a == outcome_other_order:
        return outcome_arm_first_or_a, False
    return "tie", True


# ---------------------------------------------------------------------------------------------------
# Answer documents
# ---------------------------------------------------------------------------------------------------

def make_document_id(dataset_group, stage, writer_model, prompt_gid, source_model=None, planted_flaw=None):
    group_part = "" if dataset_group == "main_fifty_prompts" else "pilot__"
    if stage == "stage_2":
        middle = f"stage_2__{writer_model}"
    elif stage == "stage_2_repeat":
        middle = f"stage_2_repeat__{writer_model}"
    elif stage == "stage_3a_rewrite":
        middle = f"stage_3a_rewrite__{writer_model}_rewriting_{source_model}"
    elif stage == "stage_3b_rewrite":
        middle = f"stage_3b_rewrite__{writer_model}_rewriting_{source_model}"
    elif stage == "baseline":
        middle = f"baseline__{writer_model}"
    elif stage == "planted_flaw_control":
        middle = f"planted_flaw_control__{planted_flaw}__{writer_model}_edited"
    else:
        raise ValueError(stage)
    return f"difficult_advice__{group_part}{middle}__{prompt_gid}"


def relative_to_source(source_folder, path):
    return str(Path(path).relative_to(Path(source_folder).parent))


def make_answer_record(source_folder, dataset_group, stage, writer_model, prompt_gid, user_message, answer_text,
                       source_path, source_line, source_field, source_document=None, planted_flaw=None,
                       extra_fields=None):
    record = {
        "document_id": make_document_id(dataset_group, stage, writer_model, prompt_gid,
                                        source_model=source_document["writer_model"] if source_document else None,
                                        planted_flaw=planted_flaw),
        "dataset_group": dataset_group,
        "prompt_gid": prompt_gid,
        "judgement_key": key_of(user_message),
        "stage": stage,
        "writer_model": writer_model,
        "writer_model_display_name": MODEL_DISPLAY_NAMES[writer_model],
        "rewritten_source_model": source_document["writer_model"] if source_document else None,
        "rewritten_source_document_id": source_document["document_id"] if source_document else None,
        "planted_flaw": planted_flaw,
        "user_message": user_message,
        "answer_text": answer_text,
        "word_count": word_count(answer_text),
        "character_count": len(answer_text),
        "source_file": relative_to_source(source_folder, source_path),
        "source_line": source_line,
        "source_field": source_field,
    }
    record.update(extra_fields or {})
    return record


def load_stage_2_documents(source_folder, run_suffix, writer_model, stage, dataset_group, issues):
    path = find_run_folder(source_folder, run_suffix) / "step2" / "responses.jsonl"
    documents = {}
    for line_number, row in read_jsonl_with_line_numbers(path):
        if not row.get("assistant_response"):
            issues.append(f"empty stage-2 answer in {path.name} line {line_number} ({row.get('prompt_gid')})")
            continue
        if row["prompt_gid"] in documents:
            issues.append(f"duplicate prompt_gid {row['prompt_gid']} in {path} line {line_number}")
        documents[row["prompt_gid"]] = make_answer_record(
            source_folder, dataset_group, stage, writer_model, row["prompt_gid"], row["user_message"],
            row["assistant_response"], path, line_number, "assistant_response",
            extra_fields={"response_gid": row["response_gid"], "sample_index": row.get("sample_index")})
    return documents


def load_rewrite_documents(source_folder, run_suffix, rewriter_model, stage, dataset_group, source_documents_by_prompt,
                           issues):
    """Rewrites from step3/rewrites.jsonl. Each rewrite is linked to the stage-2 answer it rewrote and the link is
    verified: the rewrite's `draft_response` must equal that answer's text and its response_gid that answer's."""
    path = find_run_folder(source_folder, run_suffix) / "step3" / "rewrites.jsonl"
    documents = {}
    for line_number, row in read_jsonl_with_line_numbers(path):
        prompt_gid = row["prompt_gid"]
        if not row.get("rewritten_response"):
            issues.append(f"empty rewrite in {path.name} line {line_number} ({prompt_gid})")
            continue
        source_document = source_documents_by_prompt.get(prompt_gid)
        if source_document is None:
            raise ValueError(f"rewrite in {path} line {line_number} has no source answer for {prompt_gid}")
        if source_document["answer_text"] != row["draft_response"]:
            raise ValueError(f"rewrite in {path} line {line_number}: draft_response differs from the text of "
                             f"{source_document['document_id']}")
        if source_document.get("response_gid") != row["response_gid"]:
            raise ValueError(f"rewrite in {path} line {line_number}: response_gid {row['response_gid']} differs "
                             f"from {source_document['document_id']}")
        if row["user_message"] != source_document["user_message"]:
            raise ValueError(f"rewrite in {path} line {line_number}: user message differs from its source answer")
        if prompt_gid in documents:
            issues.append(f"duplicate prompt_gid {prompt_gid} in {path} line {line_number}")
        if row["rewritten_response"] == row["draft_response"]:
            issues.append(f"rewrite identical to its source answer (the rewriter returned the text unchanged): "
                          f"{relative_to_source(source_folder, path)} line {line_number} ({prompt_gid})")
        documents[prompt_gid] = make_answer_record(
            source_folder, dataset_group, stage, rewriter_model, prompt_gid, row["user_message"],
            row["rewritten_response"], path, line_number, "rewritten_response", source_document=source_document,
            extra_fields={"response_gid": row["response_gid"], "example_gid": row["example_gid"],
                          "rewrite_record_id": row["record_id"]})
    return documents


def load_baseline_documents(source_folder, run_suffix, dataset_group, issues):
    path = find_run_folder(source_folder, run_suffix) / "baseline" / "baseline_responses.jsonl"
    documents = {}
    for line_number, row in read_jsonl_with_line_numbers(path):
        if not row.get("baseline_response"):
            issues.append(f"empty baseline answer in {path} line {line_number} ({row.get('prompt_gid')})")
            continue
        documents[row["prompt_gid"]] = make_answer_record(
            source_folder, dataset_group, "baseline", REFERENCE_MODEL, row["prompt_gid"], row["user_message"],
            row["baseline_response"], path, line_number, "baseline_response",
            extra_fields={"plain_response_gid": row["plain_gid"], "baseline_model_identifier": row.get("model")})
    return documents


def baseline_texts_by_prompt_for_run(source_folder, run_suffix):
    path = find_run_folder(source_folder, run_suffix) / "baseline" / "baseline_responses.jsonl"
    return {row["prompt_gid"]: (row["user_message"], row["baseline_response"])
            for _, row in read_jsonl_with_line_numbers(path)}


def collect_baseline_documents_verifying_identical_copies(source_folder, issues):
    """Every main run holds a copy of the same plain baseline answers; take the one in the s2-opus55 run and verify
    the copies in every other main run are identical."""
    reference_run_suffix = f"s2-{REFERENCE_MODEL}"
    documents = load_baseline_documents(source_folder, reference_run_suffix, "main_fifty_prompts", issues)
    reference_texts = baseline_texts_by_prompt_for_run(source_folder, reference_run_suffix)
    compared = []
    for run_folder in sorted((Path(source_folder) / "runs").glob("*_local_expC-*")):
        suffix = run_folder.name.split("_local_expC-")[1]
        if not re.match(r"s[23][ab]?-", suffix) or suffix == reference_run_suffix:
            continue
        copy_path = run_folder / "baseline" / "baseline_responses.jsonl"
        if not copy_path.exists():
            continue
        if baseline_texts_by_prompt_for_run(source_folder, suffix) != reference_texts:
            raise ValueError(f"baseline answers in {suffix} differ from those in {reference_run_suffix}")
        compared.append(suffix)
    return documents, compared


def collect_planted_flaw_documents(source_folder, reference_documents, issues):
    path = Path(source_folder) / "judging" / "controls" / "planted_answers.jsonl"
    by_key = {document["judgement_key"]: document for document in reference_documents.values()}
    documents = {}
    for line_number, row in read_jsonl_with_line_numbers(path):
        reference_document = by_key[row["key"]]
        record = make_answer_record(
            source_folder, "main_fifty_prompts", "planted_flaw_control", REFERENCE_MODEL,
            reference_document["prompt_gid"], reference_document["user_message"], row["text"], path, line_number,
            "text", planted_flaw=row["flaw"],
            extra_fields={"edited_from_document_id": reference_document["document_id"]})
        documents[(row["flaw"], row["key"])] = record
    return documents


def collect_pilot_documents(source_folder, issues):
    """Pilot runs: every model ran the whole pipeline on its own five prompts (a different prompt set per model, not
    shared with the main fifty), so pilot answers have no judgements and no cross-model prompt pairs."""
    documents = []
    for model in PILOT_MODELS:
        suffix = f"pilot-{model}"
        stage_2 = load_stage_2_documents(source_folder, suffix, model, "stage_2", "pilot_prompts", issues)
        # a pilot rewrite rewrites that pilot run's own stage-2 answer
        rewrites = load_rewrite_documents(source_folder, suffix, model, "stage_3a_rewrite", "pilot_prompts",
                                          stage_2, issues)
        baselines = load_baseline_documents(source_folder, suffix, "pilot_prompts", issues)
        for group in (stage_2, rewrites, baselines):
            documents.extend(group.values())
    return documents


# ---------------------------------------------------------------------------------------------------
# Prompts and scenario cards
# ---------------------------------------------------------------------------------------------------

def collect_prompts(source_folder, dataset_group, run_suffix, expected_prompt_gids, issues):
    run_folder = find_run_folder(source_folder, run_suffix)
    dilemmas = {row["prompt_gid"]: row for _, row in read_jsonl_with_line_numbers(run_folder / "step1" / "dilemmas.jsonl")}
    deals = {row["scenario_id"]: row for _, row in read_jsonl_with_line_numbers(run_folder / "step1" / "scenario_deals.jsonl")}
    scenarios = {row["scenario_id"]: row for _, row in read_jsonl_with_line_numbers(run_folder / "step1" / "scenarios.jsonl")}
    if set(dilemmas) != set(expected_prompt_gids):
        raise ValueError(f"{run_suffix}: dilemma prompt ids differ from the answered prompt ids")
    prompts = []
    for prompt_gid in sorted(dilemmas):
        dilemma = dilemmas[prompt_gid]
        scenario_id = dilemma["scenario_id"]
        deal = deals[scenario_id]
        if deal["scenario_gid"] != dilemma["scenario_gid"]:
            raise ValueError(f"scenario deal {scenario_id} has scenario_gid {deal['scenario_gid']}, dilemma has "
                             f"{dilemma['scenario_gid']}")
        prompts.append({
            "dataset_group": dataset_group, "prompt_gid": prompt_gid, "judgement_key": key_of(dilemma["user_message"]),
            "user_message": dilemma["user_message"], "user_message_word_count": word_count(dilemma["user_message"]),
            "scenario_id": scenario_id, "scenario_gid": dilemma["scenario_gid"],
            "archetype": deal.get("archetype"), "user_attitude": deal.get("user_attitude"),
            "visibility": deal.get("visibility"), "welfare_magnitude": deal.get("welfare_magnitude"),
            "user_stakes": deal.get("user_stakes"), "taxa_category": deal.get("taxa_category"),
            "taxa_subcategory": deal.get("taxa_subcategory"), "conflict": deal.get("conflict"),
            "deliverable": deal.get("deliverable"), "cultural_setting": deal.get("cultural_setting"),
            "scenario_deal": deal,
            "scenario_description": scenarios[scenario_id].get("scenario_description"),
            "scenario_cards_in_prompt_record": dilemma.get("scenario_cards"),
            "source_file": relative_to_source(source_folder, run_folder / "step1" / "scenario_deals.jsonl"),
        })
    return prompts


def verify_prompts_shared_across_runs(source_folder, reference_prompts, issues):
    """The 50 user messages must be identical, per prompt_gid, in every main run's step-2 answers and dilemmas."""
    expected = {prompt["prompt_gid"]: prompt["user_message"] for prompt in reference_prompts}
    checked = []
    for run_folder in sorted((Path(source_folder) / "runs").glob("*_local_expC-*")):
        suffix = run_folder.name.split("_local_expC-")[1]
        if not re.match(r"s[23][ab]?-", suffix):
            continue
        for relative in ("step2/responses.jsonl", "step3/rewrites.jsonl", "step1/dilemmas.jsonl"):
            path = run_folder / relative
            if not path.exists():
                continue
            for line_number, row in read_jsonl_with_line_numbers(path):
                if expected.get(row["prompt_gid"]) != row["user_message"]:
                    raise ValueError(f"{path} line {line_number}: prompt {row['prompt_gid']} differs from the "
                                     "shared fifty prompts")
        checked.append(suffix)
    return checked


# ---------------------------------------------------------------------------------------------------
# Judgements
# ---------------------------------------------------------------------------------------------------

def build_comparison_specifications(documents_by_kind):
    """[(task, comparison label, {prompt_gid: arm document}, {prompt_gid: reference document})], as in judge.py."""
    stage_2 = documents_by_kind["stage_2"]
    rewrites_3a = documents_by_kind["stage_3a_rewrite"]
    rewrites_3b = documents_by_kind["stage_3b_rewrite"]
    reference_stage_2 = stage_2[REFERENCE_MODEL]
    reference_rewrites = rewrites_3a[REFERENCE_MODEL]
    specifications = []
    for model in NON_REFERENCE_MODELS:
        specifications.append(("stage2", model, stage_2[model], reference_stage_2))
    for model in NON_REFERENCE_MODELS:
        specifications.append(("stage3a", model, rewrites_3a[model], reference_rewrites))
    for model in NON_REFERENCE_MODELS:
        specifications.append(("stage3b", f"{model} (rewritten by opus55)", rewrites_3b[model], reference_rewrites))
    controls = documents_by_kind["controls"]
    specifications.append(("controls", "control: plain model", controls["plain_model"], reference_stage_2))
    for flaw, documents in sorted(controls["planted"].items()):
        specifications.append(("controls", f"control: planted {flaw}", documents, reference_stage_2))
    specifications.append(("controls", "control: reference repeated", controls["reference_repeated"], reference_stage_2))
    for model in ("gpt-sol", "argon"):
        specifications.append(("controls", f"self-pref: {model}", stage_2[model], reference_stage_2))
    return specifications


def attach_judgements_to_comparisons(source_folder, specifications, issues):
    """Returns (verdict_rows, readings). Each reading ties one judgement row to the arm and reference documents it
    compared. The arm and reference word counts the judge recorded must equal the word counts of the attached texts,
    which catches an arm/reference swap or a wrong-prompt attachment."""
    readings = []
    judged_labels_by_task = defaultdict(set)
    for task, label, arm_by_prompt, reference_by_prompt in specifications:
        judged_labels_by_task[task].add(label)
    for task in ("stage2", "stage3a", "stage3b", "controls"):
        path = Path(source_folder) / "judging" / task / "c1" / "judgements.jsonl"
        specification_by_label = {label: (arm, reference) for t, label, arm, reference in specifications if t == task}
        latest_by_identity = {}
        for line_number, row in read_jsonl_with_line_numbers(path):
            if not row.get("winner"):
                issues.append(f"unusable judgement ignored in {task} line {line_number} ({row.get('failed')})")
                continue
            identity = (row["label"], row["key"], row["judge"], row["arm_side"])
            if identity in latest_by_identity:
                issues.append(f"duplicate judgement {identity} in {task} line {line_number}; the later one is used")
            latest_by_identity[identity] = (line_number, row)
        for identity, (line_number, row) in latest_by_identity.items():
            if row["label"] not in specification_by_label:
                raise ValueError(f"{path} line {line_number}: unknown comparison label {row['label']}")
            arm_by_prompt, reference_by_prompt = specification_by_label[row["label"]]
            arm_by_key = {document["judgement_key"]: document for document in arm_by_prompt.values()}
            reference_by_key = {document["judgement_key"]: document for document in reference_by_prompt.values()}
            arm_document, reference_document = arm_by_key[row["key"]], reference_by_key[row["key"]]
            if arm_document["prompt_gid"] != reference_document["prompt_gid"]:
                raise ValueError(f"{path} line {line_number}: arm and reference answers are for different prompts")
            if (arm_document["word_count"], reference_document["word_count"]) != (row["arm_words"], row["ref_words"]):
                raise ValueError(
                    f"{path} line {line_number}: the judge recorded {row['arm_words']}/{row['ref_words']} words for "
                    f"arm/reference but the attached answers have {arm_document['word_count']}/"
                    f"{reference_document['word_count']} (arm and reference swapped or wrong text attached)")
            readings.append({"task": task, "comparison_label": row["label"], "judge_family": row["judge"],
                             "arm_side": row["arm_side"], "row": row, "source_line": line_number,
                             "arm_document_id": arm_document["document_id"],
                             "reference_document_id": reference_document["document_id"]})
    return readings


def perspective_outcome(arm_outcome, role):
    """Win/loss/tie for the answer itself, given the arm-relative outcome ('arm', 'ref' or 'tie')."""
    if arm_outcome == "tie":
        return "tie"
    if role == "arm":
        return "win" if arm_outcome == "arm" else "loss"
    return "win" if arm_outcome == "ref" else "loss"


def shares_of_flagged_checks(check_lists):
    """{check: share of readings flagging it} over a list of per-reading {check: bool} dictionaries."""
    shares = {}
    for check in FAILURE_CHECKS:
        values = [checks[check] for checks in check_lists if isinstance(checks.get(check), bool)]
        shares[check] = (sum(values) / len(values)) if values else None
    return shares


def build_verdict_records(answer_records, readings):
    """One record per answer: every comparison it took part in, as the arm or as the reference."""
    grouped = defaultdict(list)    # (document_id, task, label, role) -> readings
    for reading in readings:
        grouped[(reading["arm_document_id"], reading["task"], reading["comparison_label"], "arm")].append(reading)
        grouped[(reading["reference_document_id"], reading["task"], reading["comparison_label"], "reference")].append(reading)
    comparisons_by_document = defaultdict(list)
    for (document_id, task, label, role), group in sorted(grouped.items()):
        counterpart_id_field = "reference_document_id" if role == "arm" else "arm_document_id"
        counterpart_ids = {reading[counterpart_id_field] for reading in group}
        if len(counterpart_ids) != 1:
            # the reference answer meets a different counterpart per prompt only through different labels
            raise ValueError(f"{document_id} {task} {label}: several counterparts {counterpart_ids}")
        own_check_lists, counterpart_check_lists, readings_out = [], [], []
        by_judge = defaultdict(dict)
        for reading in sorted(group, key=lambda item: (item["judge_family"], item["arm_side"])):
            row = reading["row"]
            counterpart_side = "B" if row["arm_side"] == "A" else "A"
            own_side = row["arm_side"] if role == "arm" else counterpart_side
            checks = row.get("checks") or {}
            own_checks = checks.get(own_side) or {}
            counterpart_checks = checks.get("B" if own_side == "A" else "A") or {}
            own_check_lists.append(own_checks)
            counterpart_check_lists.append(counterpart_checks)
            arm_relative = {axis: outcome_for_arm(row[field], row["arm_side"]) for axis, field in JUDGEMENT_AXES.items()}
            outcomes = {axis: perspective_outcome(arm_relative[axis], role) for axis in JUDGEMENT_AXES}
            reading_record = {"judge_family": reading["judge_family"],
                              "answer_shown_position": "first" if own_side == "A" else "second",
                              "outcomes": outcomes, "own_failure_checks": own_checks,
                              "counterpart_failure_checks": counterpart_checks,
                              "judgement_source_line": reading["source_line"]}
            readings_out.append(reading_record)
            by_judge[reading["judge_family"]][own_side] = arm_relative
        judge_summaries = {}
        for judge_family, sides in sorted(by_judge.items()):
            summary = {"both_orders_available": len(sides) == 2}
            for axis in JUDGEMENT_AXES:
                if len(sides) == 2:
                    combined, flipped = combine_two_orders(sides["A"][axis], sides["B"][axis])
                    summary[axis] = {"outcome": perspective_outcome(combined, role), "order_flipped": flipped}
                else:
                    summary[axis] = None
            judge_summaries[judge_family] = summary
        comparisons_by_document[document_id].append({
            "task": task, "comparison_label": label, "role": role,
            "counterpart_document_id": counterpart_ids.pop(),
            "outcomes_by_judge_family": judge_summaries,
            "readings": readings_out,
            "readings_counted": len(readings_out),
            "own_failure_check_share_of_readings": shares_of_flagged_checks(own_check_lists),
            "counterpart_failure_check_share_of_readings": shares_of_flagged_checks(counterpart_check_lists),
        })
    return [{"document_id": answer["document_id"], "prompt_gid": answer["prompt_gid"],
             "judgement_key": answer["judgement_key"], "stage": answer["stage"],
             "writer_model": answer["writer_model"],
             "comparisons": comparisons_by_document.get(answer["document_id"], [])}
            for answer in answer_records]


# ---------------------------------------------------------------------------------------------------
# Re-tally from the per-answer verdict records, compared with the experiment's reports
# ---------------------------------------------------------------------------------------------------

def retally_win_rates(verdict_records):
    """{(task, label, judge, axis): {n, win_percent, arm, ref, tie, flipped}} from the arm-role comparisons only. Uses only
    the per-answer verdict file's content, so it also tests that file. Mirrors judge.py pair_report."""
    tallies = defaultdict(lambda: {"scores": [], "arm": 0, "ref": 0, "tie": 0, "flipped": 0})
    for record in verdict_records:
        for comparison in record["comparisons"]:
            if comparison["role"] != "arm":
                continue
            by_judge = defaultdict(dict)
            for reading in comparison["readings"]:
                by_judge[reading["judge_family"]][reading["answer_shown_position"]] = reading["outcomes"]
            for judge_family, positions in by_judge.items():
                if len(positions) < 2:
                    continue
                for axis in JUDGEMENT_AXES:
                    first, second = positions["first"][axis], positions["second"][axis]
                    tally = tallies[(comparison["task"], comparison["comparison_label"], judge_family, axis)]
                    final = first if first == second else "tie"
                    tally["flipped"] += first != second
                    tally[{"win": "arm", "loss": "ref", "tie": "tie"}[final]] += 1
                    tally["scores"].append({"win": 1.0, "loss": 0.0, "tie": 0.5}[final])
    return {key: {"n": len(tally["scores"]),
                  "win_percent": f"{100 * sum(tally['scores']) / len(tally['scores']):.0f}",
                  "arm": tally["arm"], "ref": tally["ref"], "tie": tally["tie"], "flipped": tally["flipped"]}
            for key, tally in tallies.items()}


def pooled_check_shares_per_label(verdict_records):
    """{(task, label, check): (arm share, reference share)} pooled over readings and judges, mirroring pair_report."""
    arm_values, reference_values = defaultdict(list), defaultdict(list)
    for record in verdict_records:
        for comparison in record["comparisons"]:
            if comparison["role"] != "arm":
                continue
            for reading in comparison["readings"]:
                for check in FAILURE_CHECKS:
                    if isinstance(reading["own_failure_checks"].get(check), bool):
                        arm_values[(comparison["task"], comparison["comparison_label"], check)].append(reading["own_failure_checks"][check])
                    if isinstance(reading["counterpart_failure_checks"].get(check), bool):
                        reference_values[(comparison["task"], comparison["comparison_label"], check)].append(
                            reading["counterpart_failure_checks"][check])

    def percent(values):
        return f"{100 * sum(values) / len(values):.0f}%" if values else "—"
    return {key: (percent(arm_values[key]), percent(reference_values[key])) for key in arm_values}


WIN_RATE_LINE = re.compile(r"^\s+(.+?)\s+(gpt|gemini|claude)\s+(\d+)\s+(\d+)%\s+\S+\s+(\d+)/(\d+)/(\d+)\s+(\d+)\s*$")
CHECK_LINE = re.compile(r"^\s+(.+?)\s+((?:(?:\d+%|—) \| (?:\d+%|—)\s+){4})$")


def parse_experiment_report(report_text):
    """({(label, judge, axis): reported numbers}, {(label, check): (arm share, reference share)}) from a report."""
    win_rates, check_shares, axis = {}, {}, None
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped in ("overall:", "welfare:", "delivery:"):
            axis = stripped[:-1]
            continue
        if stripped.startswith("failure-mode checks"):
            axis = "checks"
            continue
        if stripped.startswith("length"):
            axis = None
        if axis in JUDGEMENT_AXES:
            match = WIN_RATE_LINE.match(line)
            if match:
                label, judge_family, count, win, arm, ref, tie, flipped = match.groups()
                win_rates[(label, judge_family, axis)] = {
                    "n": int(count), "win_percent": win, "arm": int(arm), "ref": int(ref), "tie": int(tie),
                    "flipped": int(flipped)}
        elif axis == "checks":
            match = CHECK_LINE.match(line)
            if match and not line.strip().startswith("arm "):
                cells = re.findall(r"(\d+%|—) \| (\d+%|—)", match.group(2))
                for check, cell in zip(FAILURE_CHECKS, cells):
                    check_shares[(match.group(1), check)] = cell
    return win_rates, check_shares


def compare_retally_with_reports(verdict_records, source_folder):
    retallied_win_rates = retally_win_rates(verdict_records)
    retallied_checks = pooled_check_shares_per_label(verdict_records)
    comparison = {"report_files": {}, "win_rate_rows_compared": 0, "win_rate_rows_matching": 0,
                  "check_share_cells_compared": 0, "check_share_cells_matching": 0, "mismatches": [],
                  "tally_rows_not_found_in_any_report": []}
    seen_win_rate_keys = set()
    for report_name, task in (("03_judge_controls.txt", "controls"), ("04_stage2.txt", "stage2"),
                              ("05_stage3a.txt", "stage3a"), ("06_stage3b.txt", "stage3b")):
        report_path = Path(source_folder) / "reports" / report_name
        reported_win_rates, reported_checks = parse_experiment_report(report_path.read_text(encoding="utf-8"))
        comparison["report_files"][report_name] = {"win_rate_rows": len(reported_win_rates),
                                                   "check_share_cells": len(reported_checks)}
        reported_win_rates = {(task,) + key: value for key, value in reported_win_rates.items()}
        reported_checks = {(task,) + key: value for key, value in reported_checks.items()}
        for key, reported in reported_win_rates.items():
            seen_win_rate_keys.add(key)
            comparison["win_rate_rows_compared"] += 1
            ours = retallied_win_rates.get(key)
            if ours == reported:
                comparison["win_rate_rows_matching"] += 1
            else:
                comparison["mismatches"].append({"report": report_name, "key": list(key), "reported": reported,
                                                 "retallied": ours})
        for key, reported in reported_checks.items():
            comparison["check_share_cells_compared"] += 1
            if retallied_checks.get(key) == reported:
                comparison["check_share_cells_matching"] += 1
            else:
                comparison["mismatches"].append({"report": report_name, "key": list(key), "reported": list(reported),
                                                 "retallied": list(retallied_checks.get(key, ()))})
    comparison["tally_rows_not_found_in_any_report"] = [list(key) for key in retallied_win_rates
                                                        if key not in seen_win_rate_keys]
    return comparison


# ---------------------------------------------------------------------------------------------------
# Oddities and the README
# ---------------------------------------------------------------------------------------------------

def find_oddities(answer_records, issues):
    notes = list(issues)
    for record in answer_records:
        # "I can't confirm that claim" openings are ordinary answers, so only short ones count as refusal candidates
        if REFUSAL_OPENING_PATTERN.match(record["answer_text"]) and record["word_count"] < 200:
            notes.append(f"possible refusal (refusal-style opening and under 200 words): {record['document_id']}")
        if record["word_count"] < 60:
            notes.append(f"very short answer ({record['word_count']} words): {record['document_id']}")
    for group in ("pilot_prompts",):
        answered = {r["prompt_gid"] for r in answer_records if r["dataset_group"] == group and r["stage"] == "stage_2"}
        with_baseline = {r["prompt_gid"] for r in answer_records if r["dataset_group"] == group and r["stage"] == "baseline"}
        for prompt_gid in sorted(answered - with_baseline):
            notes.append(f"{group}: prompt {prompt_gid} has a stage-2 answer but no baseline answer")
    identical = defaultdict(list)
    for record in answer_records:
        if record["dataset_group"] == "main_fifty_prompts":
            identical[record["answer_text"]].append(record["document_id"])
    for text, document_ids in identical.items():
        if len(document_ids) > 1:
            notes.append(f"identical answer text shared by {document_ids}")
    return notes


def write_readme(path, timestamp, answer_records, verdict_records, prompts, retally, oddities, checked_runs,
                 baseline_copies_compared):
    counts = Counter((record["dataset_group"], record["stage"], record["writer_model"],
                      record["rewritten_source_model"]) for record in answer_records)
    lines = [f"# Difficult-advice answers for persona vector scoring ({timestamp})", "",
             "Built by `scripts/persona_vector_projection/collect_difficult_advice_dataset_answers_for_persona_vector_scoring.py` "
             "from `dad-model-experiment-2026-11-01/` (read only). No model was called.", "",
             f"Total answer records: {len(answer_records)}. Prompts: {len(prompts)}. "
             f"Answer records with at least one judgement: {sum(bool(v['comparisons']) for v in verdict_records)}.", "",
             "## Counts", "",
             "| dataset group | stage | model that wrote it | model whose answer was rewritten | answers |", "|---|---|---|---|---|"]
    for key, count in sorted(counts.items(), key=lambda item: tuple(str(part) for part in item[0])):
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3] or ''} | {count} |")
    lines += ["", "## Files", "",
              f"- `difficult_advice_answers_{timestamp}.jsonl`: one record per answer. Fields: `document_id` (stable, "
              "`difficult_advice__[pilot__]<stage>__<writer>[_rewriting_<source>]__<prompt_gid>`), `dataset_group` "
              "(`main_fifty_prompts` or `pilot_prompts`), `prompt_gid`, `judgement_key` (md5 of the user message, first 12 "
              "characters, the key the judgements use), `stage`, `writer_model` and its display name, "
              "`rewritten_source_model` and `rewritten_source_document_id` (rewrites only: the answer that was rewritten), "
              "`planted_flaw` (planted controls only), `user_message`, `answer_text`, `word_count` (whitespace split, as "
              "the judge counted), `character_count`, `source_file`, `source_line`, `source_field`, plus identifiers from "
              "the source row (`response_gid`, `example_gid`, ...).",
              f"- `difficult_advice_judgement_verdicts_per_answer_{timestamp}.jsonl`: one record per answer, with a "
              "`comparisons` list. Each comparison names the `task` (stage2, stage3a, stage3b, controls), the "
              "`comparison_label` the experiment used, the answer's `role` (`arm` or `reference`), the "
              "`counterpart_document_id`, `outcomes_by_judge_family` (per judge family: win, loss or tie from this "
              "answer's side on overall, welfare and delivery; an order flip is a tie and `order_flipped` says so), "
              "`readings` (every individual reading: judge family, whether this answer was shown first or second, outcomes, "
              "failure checks for this answer and its counterpart), and the share of readings flagging each failure check "
              "(`own_` and `counterpart_`, pooled over judge families and both orders, as judge.py's report does).",
              f"- `difficult_advice_prompts_{timestamp}.jsonl`: the prompts with scenario card fields (archetype, user "
              "attitude, visibility, welfare magnitude, stakes, taxa, deliverable, cultural setting), the full scenario "
              "deal, and the scenario description.",
              f"- `retally_of_win_rates_against_experiment_reports_{timestamp}.json`: the re-tally from the verdict file "
              "against the numbers printed in the experiment's reports.", "",
              "## Which texts were compared (and so which answers share a prompt)", "",
              "All 50 main prompts are shared: for one `prompt_gid` there is one answer from each of the four models at "
              "stage 2 (opus55, sonnet55, gpt-sol, argon), one Opus 5.5 rewrite of its own stage-2 answer and one rewrite of "
              "it by each other model (stage 3a), one Opus 5.5 rewrite of each other model's stage-2 answer (stage 3b), and "
              "one plain baseline answer. Twenty of the prompts also have a repeated Opus 5.5 stage-2 answer "
              "(`stage_2_repeat`).", "",
              "- stage2: arm = the arm's stage-2 answer; reference = Opus 5.5 stage-2 answer.",
              "- stage3a: arm = the arm's rewrite of the Opus stage-2 answer; reference = Opus 5.5's rewrite of that answer.",
              "- stage3b: arm = Opus 5.5's rewrite of the arm's stage-2 answer; reference = Opus 5.5's rewrite of its own "
              "stage-2 answer. Note that stage-3b rewrites of different arms differ in source answer, so the same "
              "reference rewrite faces three counterparts.",
              "- controls: plain baseline (20 sampled prompts), planted-flaw edits of the Opus answer (5 per flaw), the "
              "repeated Opus answer (20), and 15 GPT and 15 Argon stage-2 answers judged by all three families, each "
              "against the Opus 5.5 stage-2 answer. Planted-flaw texts are included as `planted_flaw_control` records "
              "(edited by Opus 5.5, known flaw in `planted_flaw`), usable as a positive control.",
              "- Every rewrite is linked to the stage-2 answer it rewrote and the link is verified in code: the rewrite's "
              "`draft_response` equals that answer's text. Every judgement is attached after checking that the word counts "
              "the judge recorded for arm and reference equal those of the attached texts.",
              "- The pilot runs (`pilot_prompts`, 5 prompts for each of five models, 25 different prompts) are included as "
              "a separate group: stage-2 answer, rewrite of that run's own answer, baseline. They have no judgements, and "
              "no two pilot models answered the same prompt.", "",
              "## Re-tally against the experiment's reports", "",
              f"Win-rate rows matching: {retally['win_rate_rows_matching']} of {retally['win_rate_rows_compared']} "
              f"(n, win percent, arm/reference/tie counts, flipped). Failure-check cells matching: "
              f"{retally['check_share_cells_matching']} of {retally['check_share_cells_compared']}. "
              f"Mismatches: {len(retally['mismatches'])}.", "",
              f"Checked that the user message of each prompt is identical across the runs {', '.join(checked_runs)}, and "
              f"that the baseline answers are identical in {len(baseline_copies_compared)} other runs.", "",
              "## Oddities", ""]
    lines += [f"- {note}" for note in oddities] or ["- none found"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def collect_everything(source_folder):
    """Returns (answer_records, verdict_records, prompts, retally, oddities, checked_runs, baseline_copies_compared)."""
    issues = []
    stage_2 = {model: load_stage_2_documents(source_folder, f"s2-{model}", model, "stage_2", "main_fifty_prompts", issues)
               for model in MODELS_WITH_FULL_STAGE_2}
    repeat = load_stage_2_documents(source_folder, f"s2-{REFERENCE_MODEL}-repeat", REFERENCE_MODEL, "stage_2_repeat",
                                    "main_fifty_prompts", issues)
    rewrites_3a = {model: load_rewrite_documents(source_folder, f"s3a-{model}", model, "stage_3a_rewrite",
                                                 "main_fifty_prompts", stage_2[REFERENCE_MODEL], issues)
                   for model in MODELS_WITH_FULL_STAGE_2}
    rewrites_3b = {model: load_rewrite_documents(source_folder, f"s3b-{model}-by-opus55", REFERENCE_MODEL,
                                                 "stage_3b_rewrite", "main_fifty_prompts", stage_2[model], issues)
                   for model in NON_REFERENCE_MODELS}
    baseline, baseline_copies_compared = collect_baseline_documents_verifying_identical_copies(source_folder, issues)
    planted = collect_planted_flaw_documents(source_folder, stage_2[REFERENCE_MODEL], issues)
    planted_by_flaw = defaultdict(dict)
    for (flaw, _key), document in planted.items():
        planted_by_flaw[flaw][document["prompt_gid"]] = document
    documents_by_kind = {
        "stage_2": stage_2, "stage_3a_rewrite": rewrites_3a, "stage_3b_rewrite": rewrites_3b,
        "controls": {"plain_model": baseline, "planted": dict(planted_by_flaw), "reference_repeated": repeat}}
    answer_records = []
    for group in (list(stage_2.values()) + [repeat] + list(rewrites_3a.values()) + list(rewrites_3b.values())
                  + [baseline] + list(planted_by_flaw.values())):
        answer_records.extend(group.values())
    answer_records.sort(key=lambda record: (record["stage"], record["writer_model"],
                                            record["rewritten_source_model"] or "", record["planted_flaw"] or "",
                                            record["prompt_gid"]))
    answer_records.extend(collect_pilot_documents(source_folder, issues))
    document_ids = [record["document_id"] for record in answer_records]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("document ids are not unique: " + str([d for d, c in Counter(document_ids).items() if c > 1]))
    readings = attach_judgements_to_comparisons(source_folder, build_comparison_specifications(documents_by_kind),
                                                issues)
    verdict_records = build_verdict_records(answer_records, readings)
    main_prompt_gids = sorted(stage_2[REFERENCE_MODEL])
    prompts = collect_prompts(source_folder, "main_fifty_prompts", f"s2-{REFERENCE_MODEL}", main_prompt_gids, issues)
    checked_runs = verify_prompts_shared_across_runs(source_folder, prompts, issues)
    for model in PILOT_MODELS:
        pilot_gids = sorted(record["prompt_gid"] for record in answer_records
                            if record["dataset_group"] == "pilot_prompts" and record["stage"] == "stage_2"
                            and record["writer_model"] == model)
        prompts.extend(collect_prompts(source_folder, "pilot_prompts", f"pilot-{model}", pilot_gids, issues))
    retally = compare_retally_with_reports(verdict_records, source_folder)
    oddities = find_oddities(answer_records, issues)
    return answer_records, verdict_records, prompts, retally, oddities, checked_runs, baseline_copies_compared


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-folder", default=str(DEFAULT_SOURCE_FOLDER))
    parser.add_argument("--output-parent", default=str(DEFAULT_OUTPUT_PARENT))
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--resume-folder", default=None,
                        help="an existing run folder; skipped if it holds its completion marker, unless --force")
    parser.add_argument("--force", action="store_true", help="rewrite every output in --resume-folder")
    arguments = parser.parse_args()

    run_folder, is_resume = choose_run_folder(arguments.output_parent, arguments.run_name, arguments.resume_folder)
    timestamp = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})$", run_folder.name).group(1)
    completion_marker = run_folder / f"collection_complete_{timestamp}.json"
    if is_resume and completion_marker.exists() and not arguments.force:
        print(f"{run_folder} is already complete; pass --force to rewrite it")
        return

    (answer_records, verdict_records, prompts, retally, oddities, checked_runs,
     baseline_copies_compared) = collect_everything(arguments.source_folder)

    write_jsonl(run_folder / f"difficult_advice_answers_{timestamp}.jsonl", answer_records)
    write_jsonl(run_folder / f"difficult_advice_judgement_verdicts_per_answer_{timestamp}.jsonl", verdict_records)
    write_jsonl(run_folder / f"difficult_advice_prompts_{timestamp}.jsonl", prompts)
    (run_folder / f"retally_of_win_rates_against_experiment_reports_{timestamp}.json").write_text(
        json.dumps(retally, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(run_folder / f"README_{timestamp}.md", timestamp, answer_records, verdict_records, prompts, retally,
                 oddities, checked_runs, baseline_copies_compared)
    completion_marker.write_text(json.dumps({"answers": len(answer_records), "prompts": len(prompts)}), encoding="utf-8")
    print(f"wrote {run_folder}")
    print(f"answers {len(answer_records)}; prompts {len(prompts)}; "
          f"win-rate rows matching reports {retally['win_rate_rows_matching']}/{retally['win_rate_rows_compared']}; "
          f"check cells matching {retally['check_share_cells_matching']}/{retally['check_share_cells_compared']}; "
          f"mismatches {len(retally['mismatches'])}; oddities {len(oddities)}")


if __name__ == "__main__":
    main()
