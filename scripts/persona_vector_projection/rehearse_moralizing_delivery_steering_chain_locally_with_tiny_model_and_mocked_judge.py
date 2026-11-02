"""Alias for rehearse_content_controlled_facet_steering_chain_locally_with_tiny_model_and_mocked_judge.py (moved there when the steering chain became facet-neutral). This module IS that module, so the old
import path, old tests and old commands keep working."""

import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import rehearse_content_controlled_facet_steering_chain_locally_with_tiny_model_and_mocked_judge as _implementation

sys.modules[__name__] = _implementation

if __name__ == "__main__":
    _implementation.main()
