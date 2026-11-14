"""Copies the prebuilt run folders that the launcher uploaded to the chain's inputs folder into the outputs folder (the network volume
side that is pulled back), without ever overwriting a file that already exists, so a relaunch on the same volume keeps earlier answers.
Standard library only.

    python3 .../copy_prebuilt_inputs_into_outputs_without_overwriting_existing_files.py --source <folder> --destination <folder>
"""

import argparse
import shutil
import sys
from pathlib import Path


def copy_without_overwriting(source, destination):
    copied = skipped = 0
    for path in sorted(Path(source).rglob("*")):
        target = Path(destination) / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif target.exists():
            skipped += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied += 1
    return copied, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if not arguments.source.is_dir():
        raise SystemExit(f"{arguments.source} does not exist")
    copied, skipped = copy_without_overwriting(arguments.source, arguments.destination)
    print(f"copied {copied} files, left {skipped} existing files untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
