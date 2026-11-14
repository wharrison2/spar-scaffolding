"""Offline tests for run_second_pass_over_main_corpus_batches_1_to_5.py. No model call is made: the scenario dealer is
stubbed (and, in one test, run offline on a real spec set)."""

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
import run_second_pass_over_main_corpus_batches_1_to_5 as second_pass


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_fake_main_corpus(folder: Path, spec_ids_per_batch: dict[int, list[str]], seed_base=20280000) -> Path:
    spec_rows = [{"spec_id": f"spec_{number:05d}", "taxa_category": "farmed animals" if number % 2 else "wild animals"}
                 for number in range(1, 21)]
    write_jsonl(folder / "fixed_spec_set" / "balanced.jsonl", spec_rows)
    (folder / second_pass.CORPUS_SETTINGS_FILE_NAME).write_text(
        json.dumps({"scenario_seed_base": seed_base, "spec_set_file": "fixed_spec_set/balanced.jsonl"}), encoding="utf-8")
    ledger = []
    for batch_number, spec_ids in spec_ids_per_batch.items():
        batch_folder_text = f"batches/batch_{batch_number:03d}_seed_{seed_base + batch_number}_2026-11-07_01-59"
        write_jsonl(folder / batch_folder_text / f"spec_outcomes_for_batch_{batch_number:03d}_2026-11-07_02-23.jsonl",
                    [{"batch_number": batch_number, "spec_id": spec_id} for spec_id in spec_ids])
        ledger.append({"batch_number": batch_number, "batch_folder": batch_folder_text, "scenario_seed": seed_base + batch_number})
    write_jsonl(folder / second_pass.LEDGER_FILE_NAME, ledger)
    return folder


class StubExhausted(Exception):
    def __init__(self, category):
        self.category = category


class StubDealer:
    """Deals the first deal_count spec ids of the spec file; raises when deal_count exceeds the limit."""
    DEFAULT_VARIABLES = Path("unused")
    SpecPoolExhausted = StubExhausted

    def __init__(self, largest_dealable_count):
        self.largest_dealable_count = largest_dealable_count

    def deal_scenarios(self, count, random_generator, variables, spec_set_path, plan_path):
        if count > self.largest_dealable_count:
            raise StubExhausted("farmed animals")
        rows = [json.loads(line) for line in Path(spec_set_path).read_text().splitlines()]
        return [{"spec_id": row["spec_id"]} for row in rows[:count]]


@pytest.fixture
def main_corpus(tmp_path):
    return make_fake_main_corpus(tmp_path / "main", {1: ["spec_00001", "spec_00002"], 2: ["spec_00002", "spec_00003"],
                                                      3: ["spec_00010"]})


def test_collect_spec_ids_distinct_and_limited_to_batch_range(main_corpus):
    spec_ids, batches = second_pass.collect_spec_ids_of_main_batches(main_corpus, 1, 2)
    assert spec_ids == ["spec_00001", "spec_00002", "spec_00003"] and batches == [1, 2]
    all_ids, all_batches = second_pass.collect_spec_ids_of_main_batches(main_corpus, 1, 5)
    assert "spec_00010" in all_ids and all_batches == [1, 2, 3]


def test_collect_refuses_a_range_with_no_ledgered_batch(main_corpus):
    with pytest.raises(SystemExit):
        second_pass.collect_spec_ids_of_main_batches(main_corpus, 7, 9)


def test_prepare_corpus_folder_writes_settings_and_only_the_wanted_spec_rows_and_is_idempotent(main_corpus, tmp_path):
    corpus = tmp_path / "second_pass"
    spec_ids = ["spec_00001", "spec_00003"]
    main_spec_set = second_pass.newest_main_spec_set_path(main_corpus)
    spec_file = second_pass.prepare_second_pass_corpus_folder(corpus, spec_ids, main_spec_set, 20280500, "2026-11-12_10-00")
    assert spec_file.name == "fixed_spec_set_2_specs_main_corpus_batches_1_to_5_second_pass_2026-11-12_10-00.jsonl"
    assert [row["spec_id"] for row in second_pass.read_jsonl(spec_file)] == spec_ids
    settings = json.loads((corpus / second_pass.CORPUS_SETTINGS_FILE_NAME).read_text())
    assert settings["scenario_seed_base"] == 20280500 and settings["spec_set_file"] == f"fixed_spec_set/{spec_file.name}"
    # rerun at another time: same file kept, nothing new written
    again = second_pass.prepare_second_pass_corpus_folder(corpus, spec_ids, main_spec_set, 20280500, "2030-01-01_00-00")
    assert again == spec_file and len(list((corpus / "fixed_spec_set").iterdir())) == 1


def test_prepare_corpus_folder_refuses_mismatched_settings(main_corpus, tmp_path):
    corpus = tmp_path / "second_pass"
    main_spec_set = second_pass.newest_main_spec_set_path(main_corpus)
    second_pass.prepare_second_pass_corpus_folder(corpus, ["spec_00001"], main_spec_set, 20280500, "2026-11-12_10-00")
    with pytest.raises(SystemExit, match="seed base"):
        second_pass.prepare_second_pass_corpus_folder(corpus, ["spec_00001"], main_spec_set, 20280600, "x")
    with pytest.raises(SystemExit, match="does not hold exactly"):
        second_pass.prepare_second_pass_corpus_folder(corpus, ["spec_00001", "spec_00002"], main_spec_set, 20280500, "x")


def test_prepare_refuses_when_a_spec_is_missing_from_the_main_spec_set(main_corpus, tmp_path):
    with pytest.raises(SystemExit, match="Only"):
        second_pass.prepare_second_pass_corpus_folder(tmp_path / "c", ["spec_99999"],
                                                      second_pass.newest_main_spec_set_path(main_corpus), 20280500)


@pytest.mark.parametrize("seed_base", [20280000, 20280002, 20280003, 20281000, 20285000, 20289999])
def test_seed_collisions_are_refused(main_corpus, seed_base):
    # 20280000 -> batch seed 20280001 (main batch 1); 20280002 -> 20280003 (main batch 3); 20280003 is itself a main seed;
    # 20281000+ is the contributor range (the fake ledger holds batches 1 to 3)
    with pytest.raises(SystemExit):
        second_pass.refuse_seed_base_that_could_repeat_a_seed(seed_base, main_corpus, 1, 5)


def test_default_seed_base_is_accepted(main_corpus):
    second_pass.refuse_seed_base_that_could_repeat_a_seed(20280500, main_corpus, 1, 5)


def test_batch_seed_in_contributor_range_is_refused_even_when_base_is_below(main_corpus):
    with pytest.raises(SystemExit, match="contributor share seed range"):
        second_pass.refuse_seed_base_that_could_repeat_a_seed(20280999, main_corpus, 1, 5)


def test_safe_deal_count_search_and_unused_specs_with_stub_dealer(main_corpus, tmp_path):
    corpus = tmp_path / "second_pass"
    spec_ids = [f"spec_{number:05d}" for number in range(1, 11)]
    spec_file = second_pass.prepare_second_pass_corpus_folder(
        corpus, spec_ids, second_pass.newest_main_spec_set_path(main_corpus), 20280500, "s")
    dealer = StubDealer(largest_dealable_count=7)
    assert second_pass.find_largest_safe_deal_count(spec_file, 10, 20280501, dealer) == (7, "farmed animals")
    plan = second_pass.plan_deal(spec_file, spec_ids, 20280501, dealer)
    assert plan["safe_deal_count"] == 7 and plan["specs_unused_at_safe_deal_count"] == 3
    assert plan["unused_spec_ids"] == ["spec_00008", "spec_00009", "spec_00010"]
    assert plan["category_exhausted_when_dealing_every_spec"] == "farmed animals"
    assert sum(plan["unused_specs_by_category"].values()) == 3
    assert second_pass.find_largest_safe_deal_count(spec_file, 10, 1, StubDealer(10)) == (10, None)


def test_plan_refuses_a_deal_that_leaves_the_collected_specs(main_corpus, tmp_path):
    corpus = tmp_path / "second_pass"
    spec_file = second_pass.prepare_second_pass_corpus_folder(
        corpus, ["spec_00001", "spec_00002"], second_pass.newest_main_spec_set_path(main_corpus), 20280500, "s")

    class WrongDealer(StubDealer):
        def deal_scenarios(self, *arguments):
            return [{"spec_id": "spec_00099"}]

    with pytest.raises(SystemExit, match="outside"):
        second_pass.plan_deal(spec_file, ["spec_00001", "spec_00002"], 20280501, WrongDealer(5))


def test_command_for_fresh_and_resume_and_dry_run(tmp_path):
    corpus = tmp_path / "corpus"
    fresh = second_pass.batch_command(corpus, 123, None)
    assert fresh[1:] == [str(second_pass.BATCH_SCRIPT), "--corpus-folder", str(corpus), "--deals", "123"]
    resume = second_pass.batch_command(corpus, 123, corpus / "batches" / "batch_001_x")
    assert resume[-2:] == ["--resume-batch-folder", str(corpus / "batches" / "batch_001_x")]
    assert second_pass.batch_command(corpus, 5, None, dry_run=True)[-1] == "--dry-run"


def test_state_not_started_unfinished_and_complete(tmp_path):
    corpus = tmp_path / "corpus"
    assert second_pass.state_of_second_pass_corpus(corpus) == ("not_started", None)
    batch_folder = corpus / "batches" / "batch_001_seed_20280501_2026-11-12_10-00"
    batch_folder.mkdir(parents=True)
    assert second_pass.state_of_second_pass_corpus(corpus) == ("unfinished", batch_folder)
    write_jsonl(corpus / second_pass.LEDGER_FILE_NAME, [{"batch_folder": f"batches/{batch_folder.name}"}])
    assert second_pass.state_of_second_pass_corpus(corpus) == ("complete", None)


def test_deal_count_record_is_written_once(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    plan = {"safe_deal_count": 5}
    first = second_pass.write_deal_count_record(corpus, plan, 20280500, [1, 2])
    assert json.loads(first.read_text())["batch_scenario_seed"] == 20280501
    assert second_pass.write_deal_count_record(corpus, {"safe_deal_count": 9}, 20280500, [1]) == first
