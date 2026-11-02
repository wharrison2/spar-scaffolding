"""Alias of rehearse_content_controlled_facet_extraction_chain_locally_with_tiny_model_and_mocked_judge.py (its old name; the
default facet is moralizing delivery, so the old command is unchanged)."""
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import rehearse_content_controlled_facet_extraction_chain_locally_with_tiny_model_and_mocked_judge as _module  # noqa: E402

if __name__ == "__main__":
    sys.exit(_module.main(sys.argv[1:]))
sys.modules[__name__] = _module
