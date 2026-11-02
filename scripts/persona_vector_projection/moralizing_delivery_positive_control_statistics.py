"""Alias of content_controlled_facet_positive_control_statistics.py (the implementation moved to a facet-neutral name; this module IS
that module, so old imports, tests and commands keep working)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # when run as a script
from persona_vector_projection import content_controlled_facet_positive_control_statistics as _implementation

sys.modules[__name__] = _implementation
