"""Alias of compute_content_controlled_facet_vectors_from_activations.py (its old, moralizing-delivery-only name). The old
module path, its functions and its command line keep working and are the very same module object."""
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import compute_content_controlled_facet_vectors_from_activations as _module  # noqa: E402

if __name__ == "__main__":
    sys.exit(_module.main())
sys.modules[__name__] = _module
