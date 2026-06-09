"""LegalRAG backend package."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "false")
