"""Print a summary of the spec status of an emergent misalignment prompt corpus. Makes no model call.

Derives every spec's status from the corpus's batch folders and ledger (never from a stored table): counts by status and
failure stage, counts by category, and how many never-dealt specs remain per category. --write-table also writes a new
spec status table into <corpus folder>/spec_status_tables/.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import DEFAULT_CORPUS_FOLDER, spec_set_path_for_corpus  # noqa: E402
from spec_outcomes_and_status_table_for_prompt_corpus import (  # noqa: E402
    build_spec_status_table, completed_batches, summarise_status_table, write_spec_status_table)

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER)
    parser.add_argument("--write-table", action="store_true")
    arguments = parser.parse_args(argv)
    spec_set_path = spec_set_path_for_corpus(arguments.corpus_folder)
    if spec_set_path is None:
        print("This corpus has no spec set, so it has no spec status.", file=sys.stderr)
        return 1
    spec_rows = [json.loads(line) for line in spec_set_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    batches = completed_batches(arguments.corpus_folder)
    summary = summarise_status_table(build_spec_status_table(arguments.corpus_folder, spec_rows))
    print(f"Corpus: {arguments.corpus_folder}\nSpec set: {spec_set_path.name} ({summary['specs']} specs)\n"
          f"Completed batches: {[number for number, _ in batches]}")
    print("Specs by status:")
    for status, count in summary["by_status"].items():
        print(f"  {status}: {count}")
    print("Failed specs by latest failure stage:")
    for stage, count in sorted(summary["failed_by_latest_failure_stage"].items()):
        print(f"  {stage}: {count}")
    if not summary["failed_by_latest_failure_stage"]:
        print("  (none)")
    print("By category (never dealt remaining / accepted / failed):")
    for category, counts in summary["by_category"].items():
        print(f"  {category[:60]}: never dealt {counts['never_dealt']}, accepted {counts['accepted']}, "
              f"failed {counts['failed']}")
    if arguments.write_table:
        path = write_spec_status_table(arguments.corpus_folder, spec_rows, batches[-1][0] if batches else 0,
                                       datetime.now().strftime(TIMESTAMP_FORMAT))
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
