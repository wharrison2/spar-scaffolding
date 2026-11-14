"""Alias of make_content_controlled_facet_positive_control_rewrites.py (the implementation moved to a facet-neutral name; this module IS
that module, so old imports, tests and commands keep working; the default facet is moralizing delivery)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # when run as a script
from persona_vector_projection import make_content_controlled_facet_positive_control_rewrites as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
