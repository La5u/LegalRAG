ensure_ollama() {
  local python="$1"
  local log_path="$2"

  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is not installed or not on PATH."
    return 1
  fi

  if ! "$python" - <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
urlopen("http://127.0.0.1:11434/api/tags", timeout=1).read()
PY
  then
    nohup ollama serve >"$log_path" 2>&1 &
    echo "Started Ollama. Logs: $log_path"
  fi
}


wait_ollama_model() {
  local python="$1"
  echo "Waiting for Ollama model..."
  "$python" - <<'PY'
import json
from time import sleep
from urllib.request import urlopen

from backend.config import GENERATION_MODEL, OLLAMA_URL

for attempt in range(60):
    try:
        with urlopen(f"{OLLAMA_URL}/api/tags", timeout=1) as response:
            data = json.loads(response.read())
        model_names = {model["name"] for model in data.get("models", [])}
        if GENERATION_MODEL in model_names:
            break
    except Exception:
        pass

    if attempt == 59:
        raise SystemExit(
            f"Ollama model {GENERATION_MODEL!r} is not ready. "
            f"Run: ollama pull {GENERATION_MODEL}"
        )
    sleep(1)
PY
}
