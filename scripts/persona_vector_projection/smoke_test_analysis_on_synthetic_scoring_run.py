"""Smoke test: builds a synthetic scoring-pod run (known ground truth) in
data/persona_vector_projection_runs/small_sample_checks/ and runs the full analysis on it. No model, no API.

    python3 scripts/persona_vector_projection/smoke_test_analysis_on_synthetic_scoring_run.py
"""

import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import analyze_persona_vector_projection_scores as analysis  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import synthetic_scoring_run_for_analysis_tests as synthetic  # noqa: E402
from shared.run_folder_management import choose_run_folder  # noqa: E402


def main():
    folder, _ = choose_run_folder(defs.SMALL_SAMPLE_CHECKS_ROOT, "analysis_smoke_test_on_synthetic_scoring_run", None)
    built = synthetic.build_synthetic_scoring_run(folder / "synthetic_scoring_run", seed=0)
    output = analysis.main(["--scoring-run-folder", str(folder / "synthetic_scoring_run"), "--corpus-path",
                            str(built["corpus_path"]), "--analysis-name", "smoke_test"])
    print(f"Smoke test finished; outputs in {output}")


if __name__ == "__main__":
    main()
