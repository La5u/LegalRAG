"""LegalRAG backend package."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
