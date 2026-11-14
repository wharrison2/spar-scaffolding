"""Offline tests of the spending-ceiling monitor: a fake subprocess and a fake cost log. No model calls, no keys."""

import json
import subprocess

import cost_ceiling_monitor as monitor


class FakeProcess:
    def __init__(self, ignores_terminate=False):
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.ignores_terminate = ignores_terminate

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignores_terminate:
            self.returncode = -15

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def write_costs(path, costs):
    path.write_text("\n".join(json.dumps({"cost_usd": cost}) for cost in costs) + "\n", encoding="utf-8")


def test_stops_at_ceiling_minus_eight_most_expensive_calls(tmp_path):
    log = tmp_path / "cost_log.jsonl"
    write_costs(log, [1.0] * 5 + [4.0])           # total 9, most expensive 4 -> threshold 20 - 32 < 0: stops at once
    process = FakeProcess()
    result = monitor.watch_pipeline_process(process, lambda: [log], 20.0, sleep=lambda seconds: None)
    assert process.terminated and result["stopped_by_cost_ceiling"]
    assert result["margin_usd"] == 32.0 and result["process_ended_by"] == "terminated"


def test_keeps_running_below_the_threshold_and_stops_when_crossed(tmp_path):
    log = tmp_path / "cost_log.jsonl"
    write_costs(log, [0.5] * 10)                   # total 5; threshold 75 - 8*0.5 = 71
    process = FakeProcess()
    reads = []

    def sleep(seconds):
        reads.append(seconds)
        if len(reads) == 2:
            write_costs(log, [0.5] * 143)          # total 71.5 >= 71
        if len(reads) == 5:
            process.returncode = 0                 # would end by itself if never stopped

    result = monitor.watch_pipeline_process(process, lambda: [log], 75.0, sleep=sleep)
    assert result["stopped_by_cost_ceiling"] and len(reads) == 2
    assert result["cost_when_stop_requested_usd"] == 71.5


def test_no_ceiling_never_stops_and_a_missing_log_is_fine(tmp_path):
    process = FakeProcess()
    ticks = []

    def sleep(seconds):
        ticks.append(1)
        if len(ticks) == 3:
            process.returncode = 0

    result = monitor.watch_pipeline_process(process, lambda: [tmp_path / "absent.jsonl"], None, sleep=sleep)
    assert not process.terminated and not result["stopped_by_cost_ceiling"]


def test_kills_a_process_that_ignores_terminate(tmp_path):
    log = tmp_path / "cost_log.jsonl"
    write_costs(log, [100.0])
    process = FakeProcess(ignores_terminate=True)
    result = monitor.watch_pipeline_process(process, lambda: [log], 75.0, sleep=lambda seconds: None,
                                            seconds_to_wait_after_terminate=0)
    assert process.killed and result["process_ended_by"] == "killed"


def test_a_half_written_row_is_skipped(tmp_path):
    log = tmp_path / "cost_log.jsonl"
    log.write_text('{"cost_usd": 2.0}\n{"cost_us', encoding="utf-8")
    assert monitor.read_cost_log_totals([log]) == (2.0, 2.0)
