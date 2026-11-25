#!/usr/bin/env bash
# start_neo4j.sh — Start / stop / status the local Neo4j instance
#
# Neo4j runs as a standalone process (no Docker, no sudo required).
#
# Usage:
#   bash scripts/start_neo4j.sh start    # start Neo4j (default)
#   bash scripts/start_neo4j.sh stop     # stop Neo4j
#   bash scripts/start_neo4j.sh status   # show running status
#   bash scripts/start_neo4j.sh restart  # stop + start
#   bash scripts/start_neo4j.sh load     # start + ingest ontology v1.0.0
#
# Browser UI (once started):
#   http://localhost:7474  — user: neo4j  password: see NEO4J_PASSWORD in .env

set -euo pipefail

NEO4J_HOME="${NEO4J_HOME:-$HOME/tools/neo4j-community-5.20.0}"
JAVA_HOME="${JAVA_HOME:-$HOME/tools/jdk-17.0.19+10}"
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

_check_neo4j_bin() {
    if [[ ! -f "$NEO4J_HOME/bin/neo4j" ]]; then
        echo "[neo4j] ERROR: Neo4j not found at $NEO4J_HOME"
        echo "        Run the setup from docs to install Neo4j first."
        exit 1
    fi
}

_wait_for_bolt() {
    echo "[neo4j] Waiting for bolt port 7687..."
    local tries=0
    until nc -z localhost 7687 2>/dev/null; do
        tries=$((tries+1))
        if [[ $tries -ge 30 ]]; then
            echo "[neo4j] ERROR: Timed out waiting for Neo4j to start"
            exit 1
        fi
        sleep 2
    done
    echo "[neo4j] Ready at bolt://localhost:7687"
    echo "[neo4j] Browser: http://localhost:7474  (user neo4j, password in .env NEO4J_PASSWORD)"
}

CMD="${1:-start}"

case "$CMD" in
  start)
    _check_neo4j_bin
    "$NEO4J_HOME/bin/neo4j" start
    _wait_for_bolt
    ;;

  stop)
    _check_neo4j_bin
    "$NEO4J_HOME/bin/neo4j" stop
    echo "[neo4j] Stopped"
    ;;

  status)
    _check_neo4j_bin
    "$NEO4J_HOME/bin/neo4j" status
    if nc -z localhost 7687 2>/dev/null; then
        echo "[neo4j] Bolt port 7687: OPEN"
        echo "[neo4j] Browser:        http://localhost:7474"
    else
        echo "[neo4j] Bolt port 7687: CLOSED"
    fi
    ;;

  restart)
    _check_neo4j_bin
    "$NEO4J_HOME/bin/neo4j" stop 2>/dev/null || true
    "$NEO4J_HOME/bin/neo4j" start
    _wait_for_bolt
    ;;

  load)
    _check_neo4j_bin
    if ! nc -z localhost 7687 2>/dev/null; then
        "$NEO4J_HOME/bin/neo4j" start
        _wait_for_bolt
    fi
    echo "[neo4j] Loading ontology v1.0.0..."
    cd "$PROJECT_ROOT"
    if [[ -f ".env" ]]; then set -a; source .env; set +a; fi
    python scripts/ingest_to_neo4j.py --ontology-version 1.0.0
    ;;

  *)
    echo "Usage: $0 {start|stop|status|restart|load}"
    exit 1
    ;;
esac
