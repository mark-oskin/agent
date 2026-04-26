from __future__ import annotations

import sys
from pathlib import Path

# Make `import agent` work when running pytest from the project root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
