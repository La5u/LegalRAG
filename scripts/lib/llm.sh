ensure_llm() {
  local python="$1"
  local ollama_log="$2"

  local provider="${LEGALRAG_LLM_PROVIDER:-ollama}"
  if [[ "$provider" == "vllm" ]]; then
    if [[ "${LEGALRAG_VLLM_REMOTE:-0}" == "1" ]]; then
      "$PROJECT_ROOT/scripts/remote-vllm" up
    fi
    wait_vllm "$python"
    return
  fi
  if [[ "$provider" == "openai_compatible" ]]; then
    return
  fi

  ensure_ollama "$python" "$ollama_log"
  wait_ollama_model "$python"
}


wait_vllm() {
  local python="$1"
  echo "Waiting for vLLM..."
  "$python" - <<'PY'
from time import sleep
from urllib.request import urlopen

from backend.config import VLLM_URL

for attempt in range(60):
    try:
        urlopen(f"{VLLM_URL}/v1/models", timeout=1).read()
        break
    except Exception:
        if attempt == 59:
            raise SystemExit(
                f"vLLM is not ready at {VLLM_URL}. "
                "Start it first, then run this command again."
            )
        sleep(1)
PY
}
