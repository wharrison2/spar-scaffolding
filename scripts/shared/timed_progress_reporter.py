"""Progress reports with timings, emitted at most every ~3 minutes and once at the end."""

import threading
import time
from datetime import datetime

PROGRESS_REPORT_INTERVAL_SECONDS = 180


def format_duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


class TimedProgressReporter:
    """Counts finished tasks and prints/logs elapsed time, rate and remaining-time estimate.

    `status_suffix_function` returns extra text appended to each line (for example cost so far).
    `after_report_function` is called after each report (for example to write a usage log file).
    """

    def __init__(self, log_path, total_new_tasks, already_done_count=0, status_suffix_function=None,
                 after_report_function=None, report_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS):
        self.log_path = log_path
        self.total_new_tasks = total_new_tasks
        self.already_done_count = already_done_count
        self.status_suffix_function = status_suffix_function
        self.after_report_function = after_report_function
        self.report_interval_seconds = report_interval_seconds
        self.completed_new_tasks = 0
        self.failed_new_tasks = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_report_time = self.start_time

    def note_task_finished(self, failed):
        with self.lock:
            if failed:
                self.failed_new_tasks += 1
            else:
                self.completed_new_tasks += 1
            if time.time() - self.last_report_time >= self.report_interval_seconds:
                self._report(final=False)

    def finish(self):
        with self.lock:
            self._report(final=True)

    def _report(self, final):
        now = time.time()
        self.last_report_time = now
        elapsed = now - self.start_time
        done_new = self.completed_new_tasks + self.failed_new_tasks
        rate = done_new / elapsed if elapsed > 0 else 0.0
        remaining = (self.total_new_tasks - done_new) / rate if rate > 0 else float("nan")
        suffix = f"; {self.status_suffix_function()}" if self.status_suffix_function else ""
        line = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{'FINAL ' if final else ''}done {done_new}/{self.total_new_tasks} this session "
                f"({self.already_done_count} done before resume, {self.failed_new_tasks} failed); "
                f"elapsed {format_duration(elapsed)}; rate {rate * 60:.1f} calls/min; "
                f"estimated remaining {format_duration(remaining) if remaining == remaining else 'unknown'}"
                f"{suffix}")
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.after_report_function:
            self.after_report_function()
