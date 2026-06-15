"""Put the repository root on ``sys.path`` so ``import visual_regression`` works.

The harness lives at the repo root (it is an experiment, not part of the installed ``openretailscience``
package), so it isn't importable via the editable install the way the package is.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
