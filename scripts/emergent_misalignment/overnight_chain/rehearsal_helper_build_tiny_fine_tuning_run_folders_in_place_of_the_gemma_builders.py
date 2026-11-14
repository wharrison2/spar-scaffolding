"""REHEARSAL ONLY: builds the smoke and full fine-tuning run folders (fixed names, as the real wrapper does) for the tiny Qwen3-0.6B using the
fine-tuning rehearsal's own run folder builder, because the real builders pin Gemma 4 31B. The training examples are that rehearsal's fixed
24 examples, not the chain's kept answers. Arguments: --output-folder <fine tuning folder>.
"""

import argparse
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import rehearse_lora_fine_tuning_locally_with_tiny_model as tiny  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", required=True, type=Path)
    arguments, _ = parser.parse_known_args(argv)
    for name in ("fine_tuning_smoke_run", "fine_tuning_full_run"):
        folder = arguments.output_folder / "fine_tuning_run_folders" / name
        if not list(folder.glob("run_configuration_*.json")):
            tiny.build_rehearsal_run_folder(folder, "2026-11-13_00-00")
    return 0


if __name__ == "__main__":
    sys.exit(main())
