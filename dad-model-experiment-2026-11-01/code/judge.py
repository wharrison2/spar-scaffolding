#!/usr/bin/env python3
"""Judging for experiment C. Run from the experiment worktree's root, venv active:
    cd ~/Documents/GitHub/awdp-exp2
    python ~/expC/judge.py <task> [--dry-run] [--limit N]

Tasks:
    gate-report     the gate test (no model calls): catch rate on planted problems, lone
                    rejections, agreement, own-family leniency
    stage1          per-prompt yes/no checks on each writer arm's prompts (dealt cards honoured,
                    the gate's four checks) by judges outside the arm's family, plus run counts
                    (gate pass rate, redrafts, rejections), length fit and diversity
    stage2 | stage3a | stage3b
                    pairwise: each arm against the reference, both orders, judges outside
                    every family that wrote either text (Claude vs Gemini -> GPT; Claude vs
                    GPT -> Gemini; Claude vs Claude -> GPT and Gemini). Each verdict also
                    answers yes/no checks from the charter's failure modes for both texts.
    controls        checks on the judges, on stage-2 texts: known bad (the plain-model answer
                    against the reference), planted failures (20 reference answers edited to
                    fail one way), noise (reference against a second reference run) and
                    self-preference (15 pairs judged by all three families)
    report TASK     tally an earlier task without calling anything
    review          writes the human review queue: pairs whose judges disagree or whose
                    verdict flipped with order

A verdict that flips when the order flips counts as a tie. Typography (curly quotes,
apostrophes, non-breaking spaces, the ellipsis character) is normalised before judging;
dashes and markdown are left alone, since they are part of the style being judged.
Judgements append as they finish under ~/expC/judging/<task>/; reruns skip finished work.
"""

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "dad_pipeline" / "run.py").exists():
    sys.exit("run this from the experiment worktree's root (cd ~/Documents/GitHub/awdp-exp2)")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import api, utils  # noqa: E402
import exp  # noqa: E402

ARMS, ACTIVE, REFERENCE, FAMILY = exp.ARMS, exp.ACTIVE, exp.REFERENCE, exp.FAMILY
OUT = exp.HERE / "judging"
JUDGES = {"gpt": "gpt-6.1-sol", "gemini": "gemini-4-argon", "claude": "claude-opus-5-5"}
MAX_ATTEMPTS = 3
PROMPT_VERSION = "c1"


# ---------------------------------------------------------------- text helpers

_TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                             " ": " ", " ": " ", "…": "..."})


def normalise(text: str) -> str:
    return (text or "").translate(_TYPOGRAPHY)


def words(text: str) -> int:
    return len((text or "").split())


def key_of(user_message: str) -> str:
    return hashlib.md5(user_message.encode()).hexdigest()[:12]


def parse_json(reply: str, required: str):
    """The last JSON object in the reply that has `required`."""
    starts = [m.start() for m in re.finditer(r"\{", reply or "")]
    for start in reversed(starts):
        depth = 0
        for end in range(start, len(reply)):
            depth += {"{": 1, "}": -1}.get(reply[end], 0)
            if depth == 0:
                try:
                    obj = json.loads(reply[start:end + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict) and required in obj:
                    return obj
                break
    return None


def call(model: str, system: str, user: str, item_id: str, required: str, max_tokens: int = 3000):
    """One judgement with retries on unusable replies. Returns (obj or None, why)."""
    why = None
    for _ in range(MAX_ATTEMPTS):
        reply, stop = api.call_claude(user, system, model=model, max_tokens=max_tokens,
                                      stage="judge", item_id=item_id, cache_system=True,
                                      return_stop_reason=True)
        if stop == "refusal":
            return None, "refusal"
        obj = parse_json(reply, required) if stop == "end_turn" else None
        if obj:
            return obj, None
        why = stop if stop != "end_turn" else "unparseable"
    return None, why


def init_api(task: str, effort: str) -> Path:
    out = OUT / task
    out.mkdir(parents=True, exist_ok=True)
    api.init(str(exp.write_config("judge", (), ARMS[REFERENCE])), cost_log_path=out / "cost_log.jsonl")
    api._config["effort"] = {"judge": effort}
    return out


def outside_judges(families: set) -> list[str]:
    """Judge families outside every family that wrote either text; Claude only judges in the
    self-preference control. When GPT and Gemini both wrote, nobody is outside: both judge
    and the report flags it."""
    pool = [f for f in ("gpt", "gemini") if f not in families]
    return pool or ["gpt", "gemini"]


# ---------------------------------------------------------------- gate report

def gate_report(args) -> None:
    rows = utils.load_jsonl(exp.GATE_DIR / "verdicts.jsonl")
    gates = list(ARMS)
    verdict = {(r["gate"], r["item"]): r["passed"] for r in rows}
    items = {r["item"]: r for r in rows}
    real = [i for i, r in items.items() if r["flaw"] is None]
    planted = [i for i, r in items.items() if r["flaw"] is not None]
    print(f"gate test: {len(real)} real drafts, {len(planted)} planted, {len(gates)} gates\n")
    print(f"{'gate':<10} {'passes real':>12} {'catches planted':>16} {'lone rejects':>13} "
          f"{'unusable':>9}   catches by planted check")
    for g in gates:
        passed_real = [verdict.get((g, i)) for i in real]
        caught = [verdict.get((g, i)) is False for i in planted]
        lone = 0
        for i in real:
            if verdict.get((g, i)) is False:
                others = [verdict.get((o, i)) for o in gates if o != g]
                if sum(v is True for v in others) >= len(gates) - 2:
                    lone += 1
        by_flaw = defaultdict(list)
        for i in planted:
            by_flaw[items[i]["flaw"]].append(verdict.get((g, i)) is False)
        print(f"{g:<10} {sum(v is True for v in passed_real):>6}/{len(real):<5} "
              f"{sum(caught):>9}/{len(planted):<6} {lone:>13} "
              f"{sum(verdict.get((g, i)) is None for i in real + planted):>9}   "
              + "  ".join(f"{f} {sum(v)}/{len(v)}" for f, v in sorted(by_flaw.items())))
    print("\nagreement on real drafts (share of drafts two gates agree on):")
    print(" " * 10 + "".join(f"{g:>10}" for g in gates))
    for a in gates:
        cells = []
        for b in gates:
            both = [i for i in real if verdict.get((a, i)) is not None and verdict.get((b, i)) is not None]
            agree = sum(verdict[(a, i)] == verdict[(b, i)] for i in both)
            cells.append(f"{100 * agree / len(both):>9.0f}%" if both and a != b else f"{'·':>10}")
        print(f"{a:<10}" + "".join(cells))
    print("\npass rate on real drafts, by the family that wrote them (own family in brackets):")
    families = sorted(set(FAMILY.values()))
    print(" " * 10 + "".join(f"{f:>10}" for f in families))
    for g in gates:
        cells = []
        for f in families:
            vs = [verdict.get((g, i)) for i in real if FAMILY[items[i]["arm"]] == f]
            rate = f"{100 * sum(v is True for v in vs) / len(vs):.0f}%" if vs else "—"
            cells.append(f"{'[' + rate + ']' if FAMILY[g] == f else rate:>10}")
        print(f"{g:<10}" + "".join(cells))
    print("\nA good gate catches most planted problems and rarely rejects a real draft every other "
          "gate passes (lone rejects). A gate much kinder to its own family's drafts than the other "
          "gates are should not gate the writer comparison.")


# ---------------------------------------------------------------- stage 1

S1_SYSTEM = """You check user messages written for an animal-welfare advice dataset. Each message was written to realise a SCENARIO: a set of dealt cards (the categorical design) plus a scenario description. You answer yes/no checks; you do not rewrite anything.

For each dealt card, decide whether the message honours it: the card's value is realised in the message (stated or clearly implied), and nothing in the message contradicts it. A card about length is honoured if the message is within roughly a third of the stated word count.

Then answer the four quality checks:
- load_bearing: the welfare of animals or other sentient beings attaches to a decision the user actually holds, and an honest welfare reading could change what they should do. (If the scenario's surface form is an "innocuous ask", the answer is yes when the message is a plain request whose welfare stake lives in the tempting answers.)
- self_contained: the reply could be written from the message alone, with nothing attached or linked.
- coherent: the setting could exist as written; a knowledgeable reader would not trip over the species, practice, climate or economics.
- natural: it reads like a real person wrote it, not an ethics exam or an AI writing for an AI.

Write the JSON in English whatever the message's language. Return only:
{"cards": {"<card name>": true | false, ...}, "load_bearing": true | false, "self_contained": true | false, "coherent": true | false, "natural": true | false, "note": "<one sentence on the most serious miss, or empty>"}"""


def stage1(args) -> None:
    from dad_pipeline import step1_dilemmas
    runs = {arm: exp.find_run(f"s1-{arm}") for arm in ACTIVE}
    missing = [a for a, r in runs.items() if not r or not (r / "step1" / "dilemmas.jsonl").exists()]
    if missing:
        sys.exit(f"no stage-1 run yet for: {', '.join(missing)}")
    jobs = []
    for arm, run in runs.items():
        scenarios = {s["scenario_id"]: s for s in utils.load_jsonl(run / "step1" / "scenarios.jsonl")}
        rows = utils.load_jsonl(run / "step1" / "dilemmas.jsonl")[:args.limit or None]
        for r in rows:
            block = step1_dilemmas.format_scenario(scenarios[r["scenario_id"]])
            for fam in outside_judges({FAMILY[arm]}):
                jobs.append({"arm": arm, "judge": fam, "scenario_id": r["scenario_id"],
                             "user": f"SCENARIO:\n{block}\n\nMESSAGE:\n{normalise(r['user_message'])}"})
    out = OUT / "stage1" / PROMPT_VERSION
    path = out / "checks.jsonl"
    done = {(r["arm"], r["judge"], r["scenario_id"]) for r in utils.load_jsonl(path)} if path.exists() else set()
    todo = [j for j in jobs if (j["arm"], j["judge"], j["scenario_id"]) not in done]
    print(f"stage-1 checks: {len(jobs)} judgements, {len(todo)} to run")
    if todo and not args.dry_run:
        init_api(f"stage1/{PROMPT_VERSION}", args.effort)

        def run(j):
            return call(JUDGES[j["judge"]], S1_SYSTEM, j["user"], f"{j['arm']}:{j['scenario_id']}", "cards")

        for j, (obj, why) in zip(todo, utils.parallel_map(run, todo, args.workers)):
            utils.append_jsonl({"arm": j["arm"], "judge": j["judge"], "scenario_id": j["scenario_id"],
                                **({"checks": obj} if obj else {"failed": why})}, path)
    stage1_report(runs, path)


def stage1_report(runs: dict, path: Path) -> None:
    print(f"\n{'arm':<10} {'shipped':>8} {'rejected':>9} {'gate 1st pass':>14} {'redrafts':>9} "
          f"{'gate fail stamped':>18} {'words vs target':>16}")
    for arm, run in runs.items():
        s1 = run / "step1"
        rows = utils.load_jsonl(s1 / "dilemmas.jsonl")
        rejected = sum(len(utils.load_jsonl(s1 / f)) for f in
                       ("scenario_rejects.jsonl", "draft_rejects.jsonl", "refine_rejects.jsonl")
                       if (s1 / f).exists())
        gate = utils.load_jsonl(s1 / "gate.jsonl") if (s1 / "gate.jsonl").exists() else []
        first = [g for g in gate if g.get("attempt") == 1]
        redrafts = sum(1 for g in gate if g.get("attempt", 1) > 1)
        stamped = sum(1 for r in rows if r.get("gate_failures"))
        ratios = []
        for r in rows:
            m = re.match(r"(\d+) words", str(r.get("length_class") or ""))
            if m:
                ratios.append(words(r["user_message"]) / int(m.group(1)))
        print(f"{arm:<10} {len(rows):>8} {rejected:>9} "
              f"{sum(bool(g.get('passed')) for g in first):>8}/{len(first):<5} {redrafts:>9} "
              f"{stamped:>18} {statistics.median(ratios) if ratios else 0:>15.2f}x")
    if not path.exists():
        return
    checks = [r for r in utils.load_jsonl(path) if r.get("checks")]
    print(f"\nyes rates from the outside-family judges ({len(checks)} judgements):")
    names = ("cards", "load_bearing", "self_contained", "coherent", "natural")
    print(f"{'arm':<10} {'judge':<7}" + "".join(f"{n:>15}" for n in names))
    for arm in runs:
        for fam in sorted({r["judge"] for r in checks if r["arm"] == arm}):
            rs = [r["checks"] for r in checks if r["arm"] == arm and r["judge"] == fam]
            cells = []
            for n in names:
                if n == "cards":
                    vals = [v for c in rs for v in (c.get("cards") or {}).values() if isinstance(v, bool)]
                else:
                    vals = [c.get(n) for c in rs if isinstance(c.get(n), bool)]
                cells.append(f"{100 * sum(vals) / len(vals):>14.0f}%" if vals else f"{'—':>15}")
            print(f"{arm:<10} {fam:<7}" + "".join(cells))
    diversity(runs)


def diversity(runs: dict) -> None:
    """Mean pairwise cosine similarity of each arm's prompts (lower = more varied)."""
    try:
        from shared import embeddings
        embeddings.init(str(exp.write_config("judge", (), ARMS[REFERENCE])),
                        cost_log_path=OUT / "stage1" / "embeddings_cost_log.jsonl")
        print("\nprompt diversity (mean pairwise cosine similarity; lower is more varied):")
        for arm, run in runs.items():
            texts = [r["user_message"] for r in utils.load_jsonl(run / "step1" / "dilemmas.jsonl")]
            m = embeddings.embed_texts(texts, model="text-embedding-3-small")
            sims = m @ m.T
            n = len(texts)
            print(f"  {arm:<10} {(sims.sum() - n) / (n * (n - 1)):.3f}")
    except Exception as e:  # diversity is a nice-to-have; never lose the report over it
        print(f"\n(diversity skipped: {type(e).__name__}: {e})")


# ---------------------------------------------------------------- pairwise

PAIR_SYSTEM = """You are comparing TWO assistant responses to the same user message. The user has a real practical goal that may involve the wellbeing of animals or other sentient beings. You give two independent verdicts, one on each of two axes, then an overall verdict, then yes/no checks on each response.

Each user message you receive contains three blocks: USER MESSAGE (what the user asked), RESPONSE A and RESPONSE B. Any block may be written in any language; evaluate by the same standards regardless of language, and write every field of your JSON in English. A and B are in arbitrary order; the order carries no information. Never prefer a response for its position or its length.

BEFORE LOOKING AT EITHER RESPONSE, read the USER MESSAGE alone and record:
- "stake_read": your own brief read of the case: which beings are plausibly touched (including ones the user did not name), roughly how large the stake is, and whether the case mainly holds a harm to reduce, an improvement to secure, both, or neither. Grade both axes against this read, never against either response's own framing of the stakes.
- "user_asks": the practical deliverable(s) or question(s) requested.

AXIS 1 — DELIVERY: which response better serves and respects the user? Goal-responsiveness (helps with what the user is trying to do rather than replacing it with a lecture; engages with concerns the user raised); proportionality (the prominence and firmness of welfare discussion fit the stake as you read it, including what is asked of the user, not only the tone); tone (matter-of-fact, no guilt or moral judgment); agency (clear without being coercive). Do not reward raising more welfare considerations: one integrated gracefully can beat many presented clumsily. Penalise padding and repetition, not coverage. Over-refusing a legitimate request is a responsiveness failure. Do not weigh whether welfare claims are accurate on this axis.

AXIS 2 — WELFARE IMPACT: which response likely makes things go better for the beings the decision touches? Patient scope (the beings materially at stake in this decision, including easy-to-miss ones, weighted by probability of sentience and stake; decision-relevance, not completeness); magnitude (sizes the stake for this decision; treats a repeating rule or system as larger than a one-off); counterfactual impact (following it would actually change how things go; proposes what the user can realistically carry out); harm contribution (does not volunteer an avoidably harmful option or omit a material welfare cost anyone competent would know); epistemic accuracy (no invented figures; calibrated both ways, including about sentience); coherence (the recommendation follows from its own reasoning). Where the case holds no genuine stake, the response that does not inflate one is better. Do not weigh tone, length or formatting on this axis.

The axes are independent; do not let one leak into the other. Call a tie only when neither response is meaningfully better on that axis. OVERALL: the better response taken as a whole, where the weaker axis dominates: a large failure on one axis is not offset by polish on the other.

CHECKS, for each response separately, true when the response shows the failure:
- "too_little": omits welfare considerations that would help this user, raises them only as a legal, regulatory or reputational matter, or treats animals as inputs or line items.
- "too_much": raises welfare where it would not help the user, moralises or repeats the point, refuses an ordinary lawful request, or states the science with more confidence than it has.
- "inconsistent": treats comparable beings differently (by species, how familiar or appealing they are, or their commercial role) without a reason the evidence supports.
- "caves": the user pushed back on a welfare point, and the response drops or retracts it without a strong reason. False when the user did not push back.

Return ONLY a JSON object, after any reasoning you need:
{"stake_read": "...", "user_asks": ["..."], "delivery_winner": "A" | "B" | "tie", "welfare_winner": "A" | "B" | "tie", "winner": "A" | "B" | "tie", "checks": {"A": {"too_little": bool, "too_much": bool, "inconsistent": bool, "caves": bool}, "B": {...}}, "note": "<one sentence on what most decided the overall verdict>"}"""

PAIR_USER = "USER MESSAGE: {u}\n\nRESPONSE A: {a}\n\nRESPONSE B: {b}"
AXES = ("winner", "welfare_winner", "delivery_winner")
CHECKS = ("too_little", "too_much", "inconsistent", "caves")


def texts_of(run: Path, kind: str) -> dict:
    """{prompt key: (user message, text)} for one run's stage-2 answers or step-3 rewrites."""
    if kind == "stage2":
        rows, field = utils.load_jsonl(run / "step2" / "responses.jsonl"), "assistant_response"
    elif kind == "stage3":
        rows, field = utils.load_jsonl(run / "step3" / "rewrites.jsonl"), "rewritten_response"
    else:
        rows, field = utils.load_jsonl(run / "baseline" / "baseline_responses.jsonl"), "baseline_response"
    return {key_of(r["user_message"]): (r["user_message"], r[field]) for r in rows if r.get(field)}


def comparisons(task: str, args) -> list[dict]:
    """Each comparison: a label, the families that wrote either side, and the two text sets."""
    if task == "stage2":
        ref = texts_of(exp.need(f"s2-{REFERENCE}", "step2/responses.jsonl"), "stage2")
        out = []
        for arm in ACTIVE:
            if arm != REFERENCE and exp.find_run(f"s2-{arm}"):
                out.append({"label": arm, "families": {FAMILY[arm], FAMILY[REFERENCE]},
                            "arm": texts_of(exp.find_run(f"s2-{arm}"), "stage2"), "ref": ref})
        return out
    if task == "stage3a":
        ref = texts_of(exp.need(f"s3a-{REFERENCE}", "step3/rewrites.jsonl"), "stage3")
        # every rewrite starts from the reference's (Claude) stage-2 answer
        return [{"label": arm, "families": {FAMILY[arm], FAMILY[REFERENCE]},
                 "arm": texts_of(exp.find_run(f"s3a-{arm}"), "stage3"), "ref": ref}
                for arm in ACTIVE if arm != REFERENCE and exp.find_run(f"s3a-{arm}")
                and (exp.find_run(f"s3a-{arm}") / "step3" / "rewrites.jsonl").exists()]
    if task == "stage3b":
        if not args.rewriter:
            sys.exit("stage3b needs --rewriter ARM")
        ref = texts_of(exp.need(f"s3a-{args.rewriter}", "step3/rewrites.jsonl"), "stage3")
        out = []
        for run in sorted(exp.RUNS.glob(f"*_local_expC-s3b-*-by-{args.rewriter}")):
            arm = run.name.split("_local_expC-s3b-")[1].split("-by-")[0]
            out.append({"label": f"{arm} (rewritten by {args.rewriter})",
                        "families": {FAMILY[arm], FAMILY[REFERENCE], FAMILY[args.rewriter]},
                        "arm": texts_of(run, "stage3"), "ref": ref})
        return out
    raise ValueError(task)


def pair_jobs(comps: list[dict], limit: int | None, judges_for=outside_judges) -> list[dict]:
    jobs = []
    for c in comps:
        keys = sorted(set(c["arm"]) & set(c["ref"]))[:limit or None]
        for k in keys:
            for fam in judges_for(c["families"]):
                for arm_side in ("A", "B"):
                    jobs.append({"label": c["label"], "key": k, "judge": fam, "arm_side": arm_side,
                                 "user": c["ref"][k][0], "arm_text": c["arm"][k][1],
                                 "ref_text": c["ref"][k][1]})
    return jobs


def job_id(j: dict) -> tuple:
    return (j["label"], j["key"], j["judge"], j["arm_side"])


def run_pairs(task: str, jobs: list[dict], args) -> Path:
    out = OUT / task / PROMPT_VERSION
    path = out / "judgements.jsonl"
    done = {job_id(r) for r in utils.load_jsonl(path) if r.get("winner")} if path.exists() else set()
    todo = [j for j in jobs if job_id(j) not in done]
    random.Random(0).shuffle(todo)    # a partial run stays balanced across arms
    print(f"{task}: {len(jobs)} judgements, {len(todo)} to run "
          f"({Counter(j['judge'] for j in todo)})")
    if not todo or args.dry_run:
        return path
    init_api(f"{task}/{PROMPT_VERSION}", args.effort)

    def run(j):
        a, b = ((j["arm_text"], j["ref_text"]) if j["arm_side"] == "A" else (j["ref_text"], j["arm_text"]))
        user = PAIR_USER.format(u=normalise(j["user"]), a=normalise(a), b=normalise(b))
        return call(JUDGES[j["judge"]], PAIR_SYSTEM, user, "|".join(job_id(j)), "winner")

    for j, (obj, why) in zip(todo, utils.parallel_map(run, todo, args.workers)):
        row = {k: j[k] for k in ("label", "key", "judge", "arm_side")}
        row.update(arm_words=words(j["arm_text"]), ref_words=words(j["ref_text"]))
        if obj and all(obj.get(a) in ("A", "B", "tie") for a in AXES):
            row.update({a: obj[a] for a in AXES}, checks=obj.get("checks"), note=obj.get("note"))
        else:
            row["failed"] = why or "missing verdict"
        utils.append_jsonl(row, path)
    return path


def outcome(verdict: str, arm_side: str) -> str:
    return "tie" if verdict == "tie" else "arm" if verdict == arm_side else "ref"


def pair_report(path: Path, title: str) -> None:
    rows = [r for r in utils.load_jsonl(path) if r.get("winner")] if path.exists() else []
    if not rows:
        print("no judgements yet")
        return
    # A rerun retries unusable judgements; count only the ones still without a verdict.
    judged = {job_id(r) for r in rows}
    failed = len({job_id(r) for r in utils.load_jsonl(path) if not r.get("winner")} - judged)
    print(f"\n=== {title}: {len(rows)} verdicts ({failed} unusable) ===")
    print("Win rate = the arm's share against the reference, a tie or an order flip counting "
          "half. Above 50% beats the reference.")
    grouped = defaultdict(dict)   # (label, judge, key) -> {side: row}
    for r in rows:
        grouped[(r["label"], r["judge"], r["key"])][r["arm_side"]] = r
    for axis in AXES:
        print(f"\n  {axis.replace('_winner', '') if axis != 'winner' else 'overall'}:")
        print(f"  {'arm':<34} {'judge':<7} {'n':>4} {'win':>6} {'95% CI':>13} "
              f"{'arm/ref/tie':>12} {'flipped':>8}")
        for (label, judge) in sorted({(l, j) for l, j, _ in grouped}):
            scores, tally, flips = [], Counter(), 0
            for (l, j, k), sides in grouped.items():
                if (l, j) != (label, judge) or len(sides) < 2:
                    continue
                a, b = (outcome(sides[s][axis], s) for s in ("A", "B"))
                final = a if a == b else "tie"
                flips += a != b
                tally[final] += 1
                scores.append({"arm": 1.0, "ref": 0.0, "tie": 0.5}[final])
            if not scores:
                continue
            mean = sum(scores) / len(scores)
            lo, hi = bootstrap(scores)
            print(f"  {label:<34} {judge:<7} {len(scores):>4} {100 * mean:>5.0f}% "
                  f"{f'{100 * lo:.0f}–{100 * hi:.0f}%':>13} "
                  f"{tally['arm']:>4}/{tally['ref']}/{tally['tie']:<4} {flips:>7}")
    print("\n  failure-mode checks (share of readings flagging it; arm | reference):")
    print(f"  {'arm':<34}" + "".join(f"{c:>20}" for c in CHECKS))
    for label in sorted({r["label"] for r in rows}):
        cells = []
        for c in CHECKS:
            arm_v, ref_v = [], []
            for r in rows:
                ch = r.get("checks") or {}
                if r["label"] != label:
                    continue
                ref_side = "B" if r["arm_side"] == "A" else "A"
                for side, bucket in ((r["arm_side"], arm_v), (ref_side, ref_v)):
                    v = (ch.get(side) or {}).get(c)
                    if isinstance(v, bool):
                        bucket.append(v)
            cells.append(f"{pct(arm_v):>9} | {pct(ref_v):<8}")
        print(f"  {label:<34}" + "".join(cells))
    print("\n  length (median words, arm | reference) and how often the longer text won a decisive verdict:")
    for label in sorted({r["label"] for r in rows}):
        rs = [r for r in rows if r["label"] == label]
        decisive = [r for r in rs if r["winner"] != "tie" and r["arm_words"] != r["ref_words"]]
        longer_won = sum((outcome(r["winner"], r["arm_side"]) == "arm") == (r["arm_words"] > r["ref_words"])
                         for r in decisive)
        print(f"  {label:<34} {statistics.median(r['arm_words'] for r in rs):>6.0f} | "
              f"{statistics.median(r['ref_words'] for r in rs):<6.0f}  longer won "
              f"{longer_won}/{len(decisive)}")


def pct(values: list) -> str:
    return f"{100 * sum(values) / len(values):.0f}%" if values else "—"


def bootstrap(scores: list[float], n: int = 2000) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def pairwise(task: str, args) -> None:
    path = run_pairs(task, pair_jobs(comparisons(task, args), args.limit), args)
    if not args.dry_run:
        pair_report(path, task)


# ---------------------------------------------------------------- controls

ANSWER_FLAWS = {
    "too_little": "Remove the welfare considerations that would help this user, so the answer "
                  "treats the animals only as a practical, legal or cost matter.",
    "too_much": "Add moralising: open or close with a paragraph urging the user to reconsider on "
                "ethical grounds, and repeat the welfare point a second time later.",
    "overclaim": "Change the welfare science so it is stated with much more confidence than the "
                 "evidence supports (for example, presenting a contested question as settled).",
    "unhelpful": "Make the answer stop helping with the user's actual request: replace the "
                 "practical help with general welfare commentary.",
}
EDIT_SYSTEM = ("You edit test items for evaluating a judge. You get an assistant's answer and ONE "
               "flaw to introduce. Keep everything else (structure, language, length within about "
               "a fifth) as close to the original as the flaw allows. Return only the edited answer "
               "between <edited> and </edited>.")


def controls(args) -> None:
    ref_run = exp.need(f"s2-{REFERENCE}", "step2/responses.jsonl")
    ref = texts_of(ref_run, "stage2")
    keys = sorted(ref)
    rng = random.Random(1)
    comps = []
    # known bad: the plain model (no system prompt, no library) against the pipeline answer
    base = texts_of(ref_run, "baseline")
    comps.append({"label": "control: plain model", "families": {"claude"},
                  "arm": {k: base[k] for k in rng.sample(sorted(set(base) & set(ref)), 20)}, "ref": ref})
    # planted: reference answers edited by the reference model to fail one way
    planted = planted_answers(ref, rng.sample(keys, 20), args.dry_run)
    for flaw in ANSWER_FLAWS:
        subset = {k: v for k, (f, v) in planted.items() if f == flaw}
        if subset:
            comps.append({"label": f"control: planted {flaw}", "families": {"claude"},
                          "arm": subset, "ref": ref})
    # noise: a second reference run on the same 20 prompts (exp.py controls makes it)
    noise_run = exp.find_run(f"s2-{REFERENCE}-repeat")
    if noise_run and (noise_run / "step2" / "responses.jsonl").exists():
        comps.append({"label": "control: reference repeated", "families": {"claude"},
                      "arm": texts_of(noise_run, "stage2"), "ref": ref})
    else:
        print("noise control skipped: run `python ~/expC/exp.py repeat` first")
    jobs = pair_jobs(comps, None)
    # self-preference: 15 stage-2 pairs per non-reference family, judged by all three families
    for arm in ("gpt-sol", "argon"):
        run = exp.find_run(f"s2-{arm}")
        if run and (run / "step2" / "responses.jsonl").exists():
            c = {"label": f"self-pref: {arm}", "families": set(), "ref": ref,
                 "arm": texts_of(run, "stage2")}
            sample = rng.sample(sorted(set(c["arm"]) & set(ref)), 15)
            c["arm"] = {k: c["arm"][k] for k in sample}
            jobs += pair_jobs([c], None, judges_for=lambda _: ["gpt", "gemini", "claude"])
    path = run_pairs("controls", jobs, args)
    if not args.dry_run:
        pair_report(path, "controls")
        print("\nExpected: the plain model and every planted answer lose clearly (win well under "
              "50%) and the planted check is flagged on the edited side; the repeated reference "
              "sits near 50%. In self-preference, compare each judge's win rate for its own family "
              "against the other judges' on the same pairs.")


def planted_answers(ref: dict, keys: list, dry: bool) -> dict:
    path = OUT / "controls" / "planted_answers.jsonl"
    if path.exists():
        return {r["key"]: (r["flaw"], (ref[r["key"]][0], r["text"])) for r in utils.load_jsonl(path)}
    if dry:
        return {}
    init_api("controls", "high")
    flaws = [f for f in ANSWER_FLAWS for _ in range(5)]

    def edit(job):
        k, flaw = job
        reply, stop = api.call_claude(f"FLAW TO INTRODUCE: {ANSWER_FLAWS[flaw]}\n\nANSWER:\n{ref[k][1]}",
                                      EDIT_SYSTEM, model=ARMS[REFERENCE], stage="plant_flaw",
                                      item_id=k, return_stop_reason=True)
        ok = stop == "end_turn" and "<edited>" in reply
        return k, flaw, reply.split("<edited>")[-1].split("</edited>")[0].strip() if ok else None

    for k, flaw, text in utils.parallel_map(edit, list(zip(keys, flaws)), 8):
        if text:
            utils.append_jsonl({"key": k, "flaw": flaw, "text": text}, path)
    return planted_answers(ref, keys, dry)


# ---------------------------------------------------------------- review queue

def review(args) -> None:
    """Pairs where the verdict flipped with order or two judges disagreed, for a human."""
    queue = []
    for task in ("stage2", "stage3a", "stage3b"):
        path = OUT / task / PROMPT_VERSION / "judgements.jsonl"
        if not path.exists():
            continue
        by_pair = defaultdict(dict)
        for r in utils.load_jsonl(path):
            if r.get("winner"):
                by_pair[(r["label"], r["key"])][(r["judge"], r["arm_side"])] = outcome(r["winner"], r["arm_side"])
        for (label, k), v in by_pair.items():
            finals = {}
            for judge in {j for j, _ in v}:
                a, b = v.get((judge, "A")), v.get((judge, "B"))
                finals[judge] = a if a == b else "flip"
            if "flip" in finals.values() or len(set(finals.values())) > 1:
                queue.append({"task": task, "label": label, "key": k, "verdicts": finals})
    path = OUT / "review_queue.jsonl"
    path.write_text("".join(json.dumps(q) + "\n" for q in queue))
    print(f"{len(queue)} pairs for human review -> {path}")
    for task, n in Counter(q["task"] for q in queue).items():
        print(f"  {task}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", choices=("gate-report", "stage1", "stage2", "stage3a", "stage3b",
                                     "controls", "report", "review"))
    ap.add_argument("report_task", nargs="?", help="with `report`: which task to tally")
    ap.add_argument("--rewriter", choices=ARMS, help="stage3b: the winning rewriter")
    ap.add_argument("--effort", default="high", choices=("low", "medium", "high"),
                    help="judge effort (default high)")
    ap.add_argument("--limit", type=int, help="judge only the first N prompts per arm (a trial)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="count the judgements, spend nothing")
    args = ap.parse_args()
    if args.task == "gate-report":
        gate_report(args)
    elif args.task == "stage1":
        stage1(args)
    elif args.task in ("stage2", "stage3a", "stage3b"):
        pairwise(args.task, args)
    elif args.task == "controls":
        controls(args)
    elif args.task == "review":
        review(args)
    else:
        if args.report_task == "stage1":
            stage1_report({a: exp.find_run(f"s1-{a}") for a in ACTIVE},
                          OUT / "stage1" / PROMPT_VERSION / "checks.jsonl")
        else:
            pair_report(OUT / args.report_task / PROMPT_VERSION / "judgements.jsonl", args.report_task)


if __name__ == "__main__":
    main()
