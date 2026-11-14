"""Measure template, framing and structure homogeneity of all accepted prompts (no AI API calls, no network).

Reuses the document-frequency phrase finder from
find_phrases_repeated_across_generated_prompts_by_document_frequency.py through import (not edited).
Run: python3 -I scripts/emergent_misalignment/measure_template_and_framing_homogeneity_of_all_accepted_prompts.py
"""
import csv, json, re, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY)); sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))
import find_phrases_repeated_across_generated_prompts_by_document_frequency as phrase_finder
from shared.phrase_repetition_text_processing import normalise_text, tokenise_into_words

RUN_FOLDER = Path("/Users/williamharrison/Desktop/emergent_alignment/data/emergent_misalignment_runs/"
                  "prompt_similarity_problem_check_all_accepted_prompts_2026-11-11_13-30")
INPUT_FILE = RUN_FOLDER / "all_accepted_prompts_and_held_out_set_combined_2026-11-11_13-30.jsonl"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_FOLDER = RUN_FOLDER / f"template_and_framing_homogeneity_{TIMESTAMP}"

ELEMENT_PATTERNS = {
    "legal_or_lawful": r"\b(legal|legally|lawful|lawfully|within the law|compliant|compliance|permitted|licensed|regulat\w+)\b",
    "industry_standard_or_everyone_does": r"\b(industry[- ]standard|standard practice|standard (method|procedure|protocol|way)|industry norm|what everyone does|everyone (else )?does|everybody does|common practice|normal practice|usual practice|the norm|routine practice|how it'?s (always )?done|the industry)\b",
    "asks_what_would_you_do": r"\b(what would you do|in my (place|shoes|position)|would you back|which would you (pick|choose|go with|ship|take)|what would you (pick|choose|recommend|ship|go with)|which (one )?would you|what'?s your (call|pick|view|recommendation))\b",
    "deadline": r"\b(deadline|by (monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|end of)|(monday|tuesday|wednesday|thursday|friday)\b.{0,25}\b(freeze|due|hard stop|deadline|sign.?off|launch)|freezes|due (on|by|in)|hard stop|sign-?off (is|on)|next week|this week|by the end of)\b",
    "bonus_job_contract": r"\b(bonus|my job|my role|promotion|performance review|renewal|contract|headcount|my boss|my manager|my career|commission|laid off|redundan\w+)\b",
    "ai_system_framing": r"\b(our (ai|agent|model|bot|assistant)|the (agent|procurement agent|ai agent|scheduler)|ai[- ](driven|governed|powered|system)|system prompt|ruleset|rule set|config(uration)?|automated system|autonomous|an? (agent|ai)\b)",
    "asks_for_critique": r"\b(weak spots?|flaws?|what'?s weak|blind spots?|poke holes|tell me if i'?m wrong|stress[- ]test|push back|sanity[- ]check|gaps? in (my|the)|what am i missing|am i missing)\b",
    "asks_for_draft_or_deliverable": r"\b(draft|write (up|me|the|a)|put together|outline|prepare|produce|generate|build me|compose)\b",
}
COMPILED = {k: re.compile(v) for k, v in ELEMENT_PATTERNS.items()}
QUESTION_CLOSER = re.compile(r"\b(would you|which (one |option |route |way )?(would|should|do)|what (would|should|do) you|do you think|should i|should we|can you|could you|what do you)\b")
CHOICE_CLOSER = re.compile(r"\b(would you (do|pick|choose|go|back|ship|take|recommend)|in my (place|shoes)|which (one )?(would|should)|what would you)\b")
INTENDED_PRECEDENT_MARKERS = ("precedent as justification", "reform as resolution")


def load():
    rows = [json.loads(line) for line in open(INPUT_FILE, encoding="utf-8")]
    rows = [r for r in rows if r["source"] != "held_out_set"]
    for r in rows:
        text = r["user_message"]; r["norm"] = normalise_text(text)
        r["group"] = r["source"] if r["source"] == "main_corpus" else r["pass_label"]
        r["share_pass"] = "main_corpus" if r["source"] == "main_corpus" else f"share_{int(r['contributor_share_number']):03d}_{r['pass_label']}"
        r["elements"] = [k for k, p in COMPILED.items() if p.search(r["norm"])]
        words = text.split(); r["words"] = len(words)
        r["first3"] = " ".join(w.lower().strip(",.") for w in words[:3])
        r["first_word"] = words[0].lower().strip(",.!") if words else ""
        paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        r["paragraphs"] = len(paragraphs); r["questions"] = text.count("?")
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]
        r["last_sentence"] = sentences[-1] if sentences else ""
        r["ends_with_question"] = r["last_sentence"].endswith("?")
        r["closer_is_question_to_model"] = bool(r["ends_with_question"] and QUESTION_CLOSER.search(r["last_sentence"].lower()))
        r["closer_is_choice"] = bool(CHOICE_CLOSER.search(" ".join(sentences[-2:]).lower()))
        r["fmt"] = str(r.get("failure_mode_target"))
        r["intended_precedent"] = any(m in r["fmt"] for m in INTENDED_PRECEDENT_MARKERS)
    return rows


def pct(n, d): return round(100 * n / d, 1) if d else 0.0


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    rows = load(); N = len(rows); out = {"prompt_count": N}
    # 1 phrases
    for r in rows: r["pass_label"] = "all"; r["document_key"] = r["unique_prompt_id"]; r["tokens"] = tokenise_into_words(r["user_message"]); r["normalised_text"] = r["norm"]
    freq = phrase_finder.find_surviving_phrases(rows, 3, 8, 40, 0.9)
    phrase_rows = phrase_finder.describe_all_phrases(freq, rows, {}, ["all"])
    with open(OUTPUT_FOLDER / f"repeated_phrases_all_prompts_{TIMESTAMP}.csv", "w", newline="", encoding="utf-8") as f:
        fields = [k for k in phrase_rows[0] if k not in ("appears_in_generation_prompt", "generation_prompt_files_containing_phrase")] if phrase_rows else []
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(phrase_rows)
    out["top_phrases"] = [{k: p[k] for k in ("phrase", "document_frequency", "share_of_prompts", "distinct_spec_ids", "distinct_taxa_categories", "distinct_harm_types")} for p in phrase_rows[:40]]
    # 2 elements
    out["element_rates_overall"] = {k: pct(sum(k in r["elements"] for r in rows), N) for k in ELEMENT_PATTERNS}
    by_group = defaultdict(list)
    for r in rows: by_group[r["share_pass"]].append(r)
    per_group = {g: {k: pct(sum(k in r["elements"] for r in rs), len(rs)) for k in ELEMENT_PATTERNS} | {"n": len(rs)} for g, rs in sorted(by_group.items())}
    out["element_rates_per_share_pass"] = per_group
    by_pass = defaultdict(list)
    for r in rows: by_pass[r["group"]].append(r)
    out["element_rates_per_pass"] = {g: {k: pct(sum(k in r["elements"] for r in rs), len(rs)) for k in ELEMENT_PATTERNS} | {"n": len(rs)} for g, rs in by_pass.items()}
    with open(OUTPUT_FOLDER / f"element_rates_per_share_and_pass_{TIMESTAMP}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["share_pass", "n"] + list(ELEMENT_PATTERNS))
        for g, v in per_group.items(): w.writerow([g, v["n"]] + [v[k] for k in ELEMENT_PATTERNS])
    # spread: coefficient across share_pass
    import statistics
    out["element_spread_across_share_passes_min_max_sd"] = {k: [min(v[k] for v in per_group.values()), max(v[k] for v in per_group.values()), round(statistics.pstdev([v[k] for v in per_group.values()]), 1)] for k in ELEMENT_PATTERNS}
    # legal/standard crosstab
    legal_std = lambda r: "legal_or_lawful" in r["elements"] or "industry_standard_or_everyone_does" in r["elements"]
    tabs = {}
    for label, pred in (("legal_or_lawful", lambda r: "legal_or_lawful" in r["elements"]), ("industry_standard", lambda r: "industry_standard_or_everyone_does" in r["elements"]), ("either", legal_std)):
        tabs[label] = {"intended_precedent_spec": [sum(pred(r) for r in rows if r["intended_precedent"]), sum(r["intended_precedent"] for r in rows)],
                       "other_specs_with_failure_mode": [sum(pred(r) for r in rows if not r["intended_precedent"] and r["fmt"] != "None"), sum(1 for r in rows if not r["intended_precedent"] and r["fmt"] != "None")],
                       "no_failure_mode_target": [sum(pred(r) for r in rows if r["fmt"] == "None"), sum(1 for r in rows if r["fmt"] == "None")]}
    out["legal_standard_crosstab_hits_over_total"] = tabs
    out["legal_standard_by_failure_mode_target"] = {k[:70]: [sum(legal_std(r) for r in rs), len(rs), pct(sum(legal_std(r) for r in rs), len(rs))] for k, rs in sorted(((fm, [r for r in rows if r["fmt"][:70] == fm]) for fm in {r["fmt"][:70] for r in rows}), key=lambda x: -len(x[1]))}
    by_harm = defaultdict(list)
    for r in rows: by_harm[r["harm_type"]].append(r)
    out["legal_standard_by_harm_type_top"] = sorted(([h[:60], len(rs), pct(sum(legal_std(r) for r in rs), len(rs))] for h, rs in by_harm.items() if len(rs) >= 40), key=lambda x: -x[2])[:6] + sorted(([h[:60], len(rs), pct(sum(legal_std(r) for r in rs), len(rs))] for h, rs in by_harm.items() if len(rs) >= 40), key=lambda x: x[2])[:4]
    # 3 structure
    out["first_word_top"] = [(w, c, pct(c, N)) for w, c in Counter(r["first_word"] for r in rows).most_common(12)]
    out["first_three_words_top"] = [(w, c, pct(c, N)) for w, c in Counter(r["first3"] for r in rows).most_common(15)]
    out["distinct_first3"] = len({r["first3"] for r in rows})
    out["opening_move_distribution"] = Counter(r["opening_move"] for r in rows).most_common(10)
    out["closing_move_distribution"] = Counter(r["closing_move"] for r in rows).most_common(10)
    out["ends_with_question_pct"] = pct(sum(r["ends_with_question"] for r in rows), N)
    out["closer_question_to_model_pct"] = pct(sum(r["closer_is_question_to_model"] for r in rows), N)
    out["closer_choice_type_pct_last_two_sentences"] = pct(sum(r["closer_is_choice"] for r in rows), N)
    normalise_closer = lambda s: " ".join(re.findall(r"[a-z']+", s.lower())[:4])
    out["last_sentence_first4words_top"] = [(w, c, pct(c, N)) for w, c in Counter(normalise_closer(r["last_sentence"]) for r in rows).most_common(12)]
    wc = sorted(r["words"] for r in rows)
    q = lambda p: wc[int(p * (N - 1))]
    out["word_count"] = {"min": wc[0], "p10": q(.1), "p25": q(.25), "median": q(.5), "p75": q(.75), "p90": q(.9), "max": wc[-1], "mean": round(sum(wc) / N, 1)}
    out["word_count_bins"] = Counter(min(w // 50 * 50, 600) for w in wc)
    out["length_class_top"] = Counter(r["length_class"] for r in rows).most_common(8)
    out["paragraph_count_distribution"] = sorted(Counter(min(r["paragraphs"], 6) for r in rows).items())
    out["question_mark_count_distribution"] = sorted(Counter(min(r["questions"], 5) for r in rows).items())
    out["persona_distinct"] = len({r["persona"] for r in rows})
    # 4 template score
    score = Counter(len(r["elements"]) for r in rows)
    out["elements_per_prompt_distribution"] = sorted(score.items())
    out["prompts_with_3_or_more_elements_pct"] = pct(sum(v for k, v in score.items() if k >= 3), N)
    core = ["legal_or_lawful", "industry_standard_or_everyone_does", "asks_what_would_you_do", "deadline", "bonus_job_contract", "asks_for_critique"]
    cs = Counter(sum(k in r["elements"] for k in core) for r in rows)
    out["core_elements_per_prompt_distribution_excluding_ai_and_draft"] = sorted(cs.items())
    out["prompts_with_3_or_more_core_elements_pct"] = pct(sum(v for k, v in cs.items() if k >= 3), N)
    both = lambda r: legal_std(r) and "asks_what_would_you_do" in r["elements"]
    out["legal_standard_and_asks_what_would_you_do_pct"] = pct(sum(both(r) for r in rows), N)
    # drift: share_pass flags
    flags = []
    for k in ELEMENT_PATTERNS:
        vals = [(g, v[k]) for g, v in per_group.items()]; mean = out["element_rates_overall"][k]
        flags += [(k, g, v, mean) for g, v in vals if abs(v - mean) > max(10, 0.5 * mean) and v > 0]
    out["share_pass_outliers_vs_overall"] = flags
    # 5 random sample
    import random
    random.Random(20261111).shuffle(rows)
    sample = [{"id": r["unique_prompt_id"], "share_pass": r["share_pass"], "elements": r["elements"], "text": r["user_message"]} for r in rows[:20]]
    (OUTPUT_FOLDER / f"twenty_random_prompts_in_full_{TIMESTAMP}.json").write_text(json.dumps(sample, indent=1, ensure_ascii=False))
    (OUTPUT_FOLDER / f"template_and_framing_homogeneity_summary_{TIMESTAMP}.json").write_text(json.dumps(out, indent=1, default=str, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("element_rates_per_share_pass", "top_phrases")}, indent=1, default=str, ensure_ascii=False))
    print("TOP PHRASES"); [print(p) for p in out["top_phrases"][:30]]
    print(OUTPUT_FOLDER)

main()
