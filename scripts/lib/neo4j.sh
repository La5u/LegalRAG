ensure_neo4j() {
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
