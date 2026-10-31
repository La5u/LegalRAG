container_engine() {
  case "${LEGALRAG_CONTAINER_ENGINE:-docker}" in
    docker | podman)
      echo "${LEGALRAG_CONTAINER_ENGINE:-docker}"
      ;;
    *)
      echo "LEGALRAG_CONTAINER_ENGINE must be docker or podman." >&2
      return 2
      ;;
  esac
}


ensure_neo4j_podman() {
  local container_name="legalrag-neo4j"
  local neo4j_password="${NEO4J_PASSWORD:-${LEGALRAG_NEO4J_PASSWORD:-legalrag-password}}"
  local state

  if ! command -v podman >/dev/null 2>&1; then
    echo "Podman is not installed or is not on PATH." >&2
    return 1
  fi

  state="$(podman inspect --format '{{.State.Status}}' "$container_name" 2>/dev/null || true)"
  if [[ "$state" == "running" ]]; then
    return
  fi
  if [[ -n "$state" ]]; then
    podman start "$container_name" >/dev/null
    return
  fi

  podman volume exists legalrag_neo4j_data || podman volume create legalrag_neo4j_data >/dev/null
  podman volume exists legalrag_neo4j_logs || podman volume create legalrag_neo4j_logs >/dev/null

  podman run -d \
    --name "$container_name" \
    -p 127.0.0.1:7474:7474 \
    -p 127.0.0.1:7687:7687 \
    -e "NEO4J_AUTH=neo4j/$neo4j_password" \
    -v legalrag_neo4j_data:/data \
    -v legalrag_neo4j_logs:/logs \
    neo4j:5.26-community >/dev/null
}


ensure_neo4j() {
  local engine
  engine="$(container_engine)"

  if [[ "$engine" == "podman" ]]; then
    ensure_neo4j_podman
    return
  fi

  if ! docker info >/dev/null 2>&1; then
    if command -v systemctl >/dev/null 2>&1; then
      sudo systemctl start docker
    fi
  fi

  if docker info >/dev/null 2>&1; then
    docker compose up -d neo4j
  else
    sudo docker compose up -d neo4j
  fi
}


wait_neo4j() {
  local python="$1"
  echo "Waiting for Neo4j Bolt at 127.0.0.1:7687..."
  "$python" - <<'PY'
from time import sleep

from backend.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from neo4j import GraphDatabase

for attempt in range(60):
    try:
        with GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        ) as driver:
            driver.verify_connectivity()
        break
    except Exception:
        if attempt == 59:
            raise SystemExit("Neo4j did not become ready within 60 seconds.")
        sleep(1)
PY
}
