#!/usr/bin/env python3
"""Experiment C: which model at which DAD stage. Animal profile, five arms, every
model at effort high. Stages are varied one at a time against the Opus 5.5
reference; everything else is held at the reference.

Run from the experiment worktree's root with the venv active:
    cd ~/Documents/GitHub/awdp-exp2
    python ~/expC/exp.py <phase> [--dry-run]

Phases, in order (each is resumable: rerunning skips finished work for free):
    pilot       5 examples through each arm at every stage, at effort high (token
                counts for the budget, ~$10)
    gate-pool   20 ungated stage-1 drafts per writer arm (100), plus 20 with a planted
                failure of one of the gate's four checks
    gate-test   every model gates the same ~120 drafts; then: gate-report
    stage1      the writer comparison: 50 scenarios per arm (1a plan, 1b draft, 1d refine
                on the arm, 1c on --gate, the winner of gate-report)
    stage2      the reference's stage-1 prompts -> shared baseline + 5 answer arms
    stage3a     the reference's stage-2 answers rewritten by all 5 models
    stage3b     --rewriter MODEL --arms A B: the winning rewriter on the top two
                stage-2 arms' answers
    repeat      the reference's stage 2 again on 20 prompts (judge.py's noise control)
    status      per-run progress and spend

Runs live in outputs/dad/runs/<date>_local_expC-<phase>-<arm>/ (gitignored);
configs are written to ~/expC/configs/. Judging is ~/expC/judge.py.
"""

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path.cwd()
if not (REPO / "shared" / "gemini_api.py").exists():  # only the experiment worktree has it
    sys.exit("run this from the experiment worktree's root (cd ~/Documents/GitHub/awdp-exp2)")
sys.path.insert(0, str(REPO))
from shared import api, utils  # noqa: E402

RUNS = REPO / "outputs" / "dad" / "runs"
HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
GATE_DIR = HERE / "gate"

ARMS = {
    "opus55": "claude-opus-5-5",
    "sonnet55": "claude-sonnet-5-5",
    "gpt-sol": "gpt-6.1-sol",
    "argon": "gemini-4-argon",
    "gem31pro": "gemini-3.1-pro-preview",
}
REFERENCE = "opus55"
FAMILY = {"opus55": "claude", "sonnet55": "claude", "gpt-sol": "gpt",
          "argon": "gemini", "gem31pro": "gemini"}
# Arms left out from stage 1 onward. gem31pro: its 250-requests-a-day Tier 1 limit cannot
# finish the experiment in one day (2026-10-31). The gate test ran with all five arms.
DROPPED = {"gem31pro"}
ACTIVE = {arm: model for arm, model in ARMS.items() if arm not in DROPPED}

STAGE1 = ("scenario_model", "prompt_draft_model", "prompt_refine_model")
GATE = ("prompt_gate_model",)
STAGE2 = ("response_scope_model", "response_select_model", "response_draft_model")
STAGE3 = ("constitution_rewrite_model",)

N_COMPARE = 50          # prompts per comparison
N_POOL = 20             # ungated drafts per writer arm for the gate test
POOL_SEED, STAGE1_SEED = 101, 202
# 8000 (the largest stage cap, a retry) + this stays under the Anthropic SDK's
# non-streaming ceiling (~21.3k max_tokens).
THINKING_HEADROOM = 13000


# ---------------------------------------------------------------- configs

def base_config() -> dict:
    """Main's config with every stage on the reference at effort high. Refusal
    fallbacks are off: a refusal is part of an arm's record, and a fallback
    would hand the item to a model from another arm."""
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    cfg["backend"] = "api"
    cfg["effort"] = "high"
    cfg["thinking_headroom"] = THINKING_HEADROOM
    d = cfg["dad"]
    d["profile"] = "animal"
    for knob in STAGE1 + GATE + STAGE2 + STAGE3:
        d[knob] = ARMS[REFERENCE]
    d["scenario_refusal_fallback_model"] = None
    d["scope_refusal_fallback_model"] = None
    d["baseline"] = {"enabled": True, "model": ARMS[REFERENCE]}
    d["evals"] = {"auto": False}
    d["dilemmas"].update(count=N_COMPARE, gate=True, refine=True)
    return cfg


def write_config(name: str, knobs: tuple, model: str, **dilemmas) -> Path:
    cfg = base_config()
    for knob in knobs:
        cfg["dad"][knob] = model
    cfg["dad"]["dilemmas"].update(dilemmas)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    path = CONFIGS / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# ---------------------------------------------------------------- runs

def find_run(label: str) -> Path | None:
    found = sorted(RUNS.glob(f"*_local_expC-{label}"))
    return found[-1] if found else None


def run_pipeline(label: str, config: Path, step: int, stop_after: int, dry: bool) -> None:
    """Start the run, or resume it if it exists (finished work is skipped for free)."""
    cmd = [sys.executable, "dad_pipeline/run.py", "--config", str(config),
           "--step", str(step), "--stop-after", str(stop_after)]
    existing = find_run(label)
    cmd += (["--resume", "--run-id", existing.name] if existing
            else ["--label", f"local_expC-{label}"])
    print(f"\n>>> {label}: {' '.join(cmd[1:])}")
    if dry:
        return
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        sys.exit(f"{label} exited {result.returncode}; fix and rerun this phase (it resumes)")


def copy_run(source: Path, label: str, parts: tuple, dry: bool) -> None:
    """A new run dir holding `parts` of `source` (the shared inputs), unless it exists."""
    if find_run(label):
        return
    target = source.with_name(source.name.split("_local_expC-")[0] + f"_local_expC-{label}")
    print(f"  copy {source.name} [{', '.join(parts)}] -> {target.name}")
    if dry:
        return
    target.mkdir(parents=True)
    for part in ("inputs", "run_manifest.json") + parts:
        src = source / part
        if src.is_dir():
            shutil.copytree(src, target / part)
        elif src.exists():
            shutil.copy2(src, target / part)


def need(label: str, path: str) -> Path:
    run = find_run(label)
    if not run or not (run / path).exists():
        sys.exit(f"needs {label} with {path} — run the earlier phase first")
    return run


# ---------------------------------------------------------------- phases

def pilot(args) -> None:
    for arm in ARMS:
        cfg = write_config(f"pilot-{arm}", STAGE1 + GATE + STAGE2 + STAGE3, ARMS[arm],
                           count=5, scenario_seed=7)
        run_pipeline(f"pilot-{arm}", cfg, 1, 3, args.dry_run)
    if not args.dry_run:
        status(args, prefix="pilot")


def gate_pool(args) -> None:
    for arm, model in ARMS.items():
        cfg = write_config(f"pool-{arm}", STAGE1, model, count=N_POOL,
                           scenario_seed=POOL_SEED, gate=False, refine=False)
        run_pipeline(f"pool-{arm}", cfg, 1, 1, args.dry_run)
    if not args.dry_run:
        plant_flaws()


FLAWS = {
    "decorative": "Check 1 (welfare load-bearing): rewrite it so the animals are still present "
                  "in the scene but nothing the user can decide touches their welfare: the "
                  "user's actual question is about something else, and no honest reading of the "
                  "animals' welfare could change what they should do.",
    "not_self_contained": "Check 2 (self-contained): rewrite it so the user's ask depends on "
                          "material that is not in the message, such as asking the reader to "
                          "look over an attached draft, a spreadsheet or a photo they mention "
                          "but do not include.",
    "incoherent": "Check 3 (cohesive scene): change one or two concrete details so the "
                  "setting no longer coheres in a way a knowledgeable reader would trip over (a "
                  "species, practice, climate or economy that could not coexist with the rest).",
    "synthetic": "Check 4 (reads like a real person): rewrite it in a polished, symmetrical "
                 "register that announces the dilemma like an ethics exam ('I face a moral "
                 "dilemma between X and Y... on one hand... on the other hand...').",
}
PLANT_SYSTEM = ("You edit test items for evaluating a quality gate. You will get a user message "
                "written for a dataset and ONE flaw to introduce. Keep everything else about the "
                "message (topic, length, language, voice) as close to the original as the flaw "
                "allows; introduce that flaw clearly and only that flaw. Return only the edited "
                "message between <edited> and </edited>.")


def plant_flaws() -> None:
    """20 planted drafts: 5 per gate check, drawn evenly from the five writer arms."""
    out = GATE_DIR / "planted.jsonl"
    if out.exists():
        print(f"  planted drafts already in {out}")
        return
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    api.init(str(CONFIGS / f"pool-{REFERENCE}.yaml"), cost_log_path=GATE_DIR / "cost_log.jsonl")
    rng = random.Random(0)
    picks = []
    for arm in ARMS:
        rows = utils.load_jsonl(find_run(f"pool-{arm}") / "step1" / "dilemmas.jsonl")
        picks += [(arm, r) for r in rng.sample(rows, 4)]
    rng.shuffle(picks)
    flaws = [f for f in FLAWS for _ in range(5)]

    def edit(job):
        (arm, row), flaw = job
        reply, stop = api.call_claude(
            f"FLAW TO INTRODUCE: {FLAWS[flaw]}\n\nMESSAGE:\n{row['user_message']}",
            PLANT_SYSTEM, model=ARMS[REFERENCE], stage="plant_flaw",
            item_id=row["scenario_id"], return_stop_reason=True)
        text = reply.split("<edited>")[-1].split("</edited>")[0].strip() if "<edited>" in reply else None
        return arm, row, flaw, text if stop == "end_turn" else None

    for arm, row, flaw, text in utils.parallel_map(edit, list(zip(picks, flaws)), 8):
        if text:
            utils.append_jsonl({"arm": arm, "scenario_id": row["scenario_id"], "flaw": flaw,
                                "original": row["user_message"], "prompt": text}, out)
        else:
            print(f"  plant failed for {row['scenario_id']} ({flaw}); skipped")
    print(f"  planted drafts: {out}  — read a few before gate-test")


def gate_test(args) -> None:
    from dad_pipeline import step1_dilemmas
    items = []
    for arm in ARMS:
        run = need(f"pool-{arm}", "step1/dilemmas.jsonl")
        scenarios = {s["scenario_id"]: s for s in utils.load_jsonl(run / "step1" / "scenarios.jsonl")}
        for r in utils.load_jsonl(run / "step1" / "dilemmas.jsonl"):
            items.append({"item": f"{arm}:{r['scenario_id']}", "arm": arm, "flaw": None,
                          "scenario": scenarios[r["scenario_id"]], "prompt": r["user_message"]})
        for p in utils.load_jsonl(GATE_DIR / "planted.jsonl"):
            if p["arm"] == arm:
                items.append({"item": f"{arm}:{p['scenario_id']}:{p['flaw']}", "arm": arm,
                              "flaw": p["flaw"], "scenario": scenarios[p["scenario_id"]],
                              "prompt": p["prompt"]})
    prompts_dir = find_run(f"pool-{REFERENCE}") / "inputs" / "prompts"
    out = GATE_DIR / "verdicts.jsonl"
    done = {(v["gate"], v["item"]) for v in (utils.load_jsonl(out) if out.exists() else [])}
    jobs = [(gate, it) for gate in ARMS for it in items if (gate, it["item"]) not in done]
    print(f"gate test: {len(items)} drafts x {len(ARMS)} gates, {len(jobs)} verdicts to run")
    if args.dry_run or not jobs:
        return
    api.init(str(CONFIGS / f"pool-{REFERENCE}.yaml"), cost_log_path=GATE_DIR / "cost_log.jsonl")

    def judge(job):
        gate, it = job
        return step1_dilemmas.gate_draft(it["scenario"], {"prompt": it["prompt"]}, prompts_dir,
                                         model=ARMS[gate])

    for (gate, it), (passed, failures, _) in zip(jobs, utils.parallel_map(judge, jobs, 8)):
        utils.append_jsonl({"gate": gate, "item": it["item"], "arm": it["arm"], "flaw": it["flaw"],
                            "passed": passed, "failures": failures}, out)
    print(f"verdicts: {out}\nnext: python ~/expC/judge.py gate-report")


def stage1(args) -> None:
    if not args.gate:
        sys.exit("stage1 needs --gate ARM (the winner of: python ~/expC/judge.py gate-report)")
    for arm, model in ACTIVE.items():
        cfg = write_config(f"s1-{arm}", STAGE1, model, scenario_seed=STAGE1_SEED)
        cfg_data = yaml.safe_load(cfg.read_text())
        cfg_data["dad"]["prompt_gate_model"] = ARMS[args.gate]
        cfg.write_text(yaml.safe_dump(cfg_data, sort_keys=False))
        run_pipeline(f"s1-{arm}", cfg, 1, 1, args.dry_run)


def stage2(args) -> None:
    source = need(f"s1-{REFERENCE}", "step1/dilemmas.jsonl") if not args.dry_run else find_run(f"s1-{REFERENCE}")
    # The reference arm's own stage-2 run also generates the shared baseline (first takes).
    copy_run(source, f"s2-{REFERENCE}", ("step1",), args.dry_run) if source else None
    cfg = write_config(f"s2-{REFERENCE}", (), ARMS[REFERENCE])
    run_pipeline(f"s2-{REFERENCE}", cfg, 2, 2, args.dry_run)
    shared = find_run(f"s2-{REFERENCE}")
    for arm, model in ACTIVE.items():
        if arm == REFERENCE:
            continue
        if shared:
            copy_run(shared, f"s2-{arm}", ("step1", "baseline"), args.dry_run)
        cfg = write_config(f"s2-{arm}", STAGE2, model)
        run_pipeline(f"s2-{arm}", cfg, 2, 2, args.dry_run)


def stage3a(args) -> None:
    source = need(f"s2-{REFERENCE}", "step2/responses.jsonl") if not args.dry_run else find_run(f"s2-{REFERENCE}")
    for arm, model in ACTIVE.items():
        if source:
            copy_run(source, f"s3a-{arm}", ("step1", "baseline", "step2"), args.dry_run)
        cfg = write_config(f"s3a-{arm}", STAGE3, ARMS[arm])
        run_pipeline(f"s3a-{arm}", cfg, 3, 3, args.dry_run)


def stage3b(args) -> None:
    if not args.rewriter or not args.arms:
        sys.exit("stage3b needs --rewriter ARM --arms ARM ARM (from judge.py's stage-2 and 3a reports)")
    for arm in args.arms:
        if arm == REFERENCE:
            print(f"  {arm}: its rewrite by {args.rewriter} already exists as s3a-{args.rewriter}")
            continue
        source = need(f"s2-{arm}", "step2/responses.jsonl")
        label = f"s3b-{arm}-by-{args.rewriter}"
        copy_run(source, label, ("step1", "baseline", "step2"), args.dry_run)
        cfg = write_config(label, STAGE3, ARMS[args.rewriter])
        run_pipeline(label, cfg, 3, 3, args.dry_run)


def repeat(args) -> None:
    """The noise control: the reference's stage 2 again on 20 of the same prompts, so
    judge.py controls can compare the reference with itself."""
    source = need(f"s2-{REFERENCE}", "step2/responses.jsonl")
    label = f"s2-{REFERENCE}-repeat"
    if not find_run(label) and not args.dry_run:
        copy_run(source, label, ("step1", "baseline"), args.dry_run)
        run = find_run(label)
        rows = utils.load_jsonl(run / "step1" / "dilemmas.jsonl")[:20]
        (run / "step1" / "dilemmas.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    run_pipeline(label, write_config(label, (), ARMS[REFERENCE]), 2, 2, args.dry_run)


def status(args, prefix: str = "") -> None:
    total = 0.0
    for run in sorted(RUNS.glob(f"*_local_expC-{prefix}*")):
        rows = utils.load_jsonl(run / "cost_log.jsonl") if (run / "cost_log.jsonl").exists() else []
        cost = sum(r.get("cost_usd", 0) for r in rows)
        total += cost
        counts = {name: len(utils.load_jsonl(run / f)) for name, f in
                  (("prompts", "step1/dilemmas.jsonl"), ("answers", "step2/responses.jsonl"),
                   ("rewrites", "step3/rewrites.jsonl")) if (run / f).exists()}
        print(f"  {run.name.split('_local_expC-')[1]:<28} ${cost:7.2f}  "
              + "  ".join(f"{k} {v}" for k, v in counts.items()))
        by_stage = {}
        for r in rows:
            s = by_stage.setdefault((r.get("stage"), r["model"]), [0, 0, 0, 0.0])
            s[0] += 1; s[1] += r["input_tokens"]; s[2] += r["output_tokens"]; s[3] += r["cost_usd"]
        if args.verbose:
            for (stage, model), (n, i, o, c) in sorted(by_stage.items(), key=str):
                print(f"      {stage:<22} {model:<24} {n:4d} calls  in {i // n:6d}  out {o // n:6d}"
                      f"  ${c / n:.3f}/call")
    for extra in (GATE_DIR / "cost_log.jsonl", HERE / "judging"):
        files = [extra] if extra.is_file() else list(extra.rglob("cost_log.jsonl")) if extra.exists() else []
        for f in files:
            cost = sum(r.get("cost_usd", 0) for r in utils.load_jsonl(f))
            total += cost
            print(f"  {str(f.relative_to(HERE)):<28} ${cost:7.2f}")
    print(f"  {'total':<28} ${total:7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("phase", choices=("pilot", "gate-pool", "gate-test", "stage1", "stage2",
                                      "stage3a", "stage3b", "repeat", "status"))
    ap.add_argument("--dry-run", action="store_true", help="print what would run, spend nothing")
    ap.add_argument("--gate", choices=ARMS, help="stage1: the gate every writer arm uses")
    ap.add_argument("--rewriter", choices=ARMS, help="stage3b: the winning rewriter")
    ap.add_argument("--arms", nargs="+", choices=ARMS, help="stage3b: the top two stage-2 arms")
    ap.add_argument("-v", "--verbose", action="store_true", help="status: per-stage token use")
    args = ap.parse_args()
    {"pilot": pilot, "gate-pool": gate_pool, "gate-test": gate_test, "stage1": stage1,
     "stage2": stage2, "stage3a": stage3a, "stage3b": stage3b, "repeat": repeat,
     "status": status}[args.phase](args)


if __name__ == "__main__":
    main()
