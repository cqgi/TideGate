from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("TIDEGATE_ADMIN_TOKEN", "dev-admin")
os.environ.setdefault("MOCK_A_KEY", "mock-key")
