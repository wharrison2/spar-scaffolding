"""Alias of build_content_controlled_facet_scoring_run_folder.py (the implementation moved to a facet-neutral name; this module IS that module, so old imports, tests and
commands keep working; the default facet is moralizing delivery)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # when run as a script
from persona_vector_projection import build_content_controlled_facet_scoring_run_folder as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
