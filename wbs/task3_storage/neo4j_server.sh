#!/usr/bin/env bash
# WBS Task 3.1.1 - start/stop the Neo4j server the Cypher measurements run against.
#
# The WBS line item names "RDF/graph database". Until now this host had no JVM, so
# the delivered store was Oxigraph (embedded, SPARQL 1.1) and Neo4j existed in the
# repository as client code only. This script installs nothing: it points a
# home-directory JDK at a home-directory Neo4j tarball, on non-default ports so it
# cannot collide with anything else on a shared host.
#
#   bash wbs/task3_storage/neo4j_server.sh start     # configure (first run) and start
#   bash wbs/task3_storage/neo4j_server.sh status
#   bash wbs/task3_storage/neo4j_server.sh stop
#
# Ports: Bolt 7687, HTTP 7474 - the Neo4j Browser's own defaults. Override with
# NEO4J_BOLT_PORT / NEO4J_HTTP_PORT if something already holds them; if you do,
# the browser will keep offering 7687 from its cache and must be edited to match.
# Data:  artifacts/graph_db/neo4j, beside the Oxigraph store, so neither is hidden.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NEO4J_HOME="${NEO4J_HOME:-$HOME/.local/opt/neo4j-community-5.26.0}"
export JAVA_HOME="${JAVA_HOME:-$HOME/.local/opt/jdk21}"
export PATH="$JAVA_HOME/bin:$PATH"
export NEO4J_HOME

# The Neo4j Browser defaults to bolt://<host>:7687. Moving off that default was a
# mistake paid for twice: the browser kept offering 7687 from its own cache while
# the server listened elsewhere, which presents as a 30-second connect timeout
# right after the password is submitted. Standard ports, so the defaults line up.
BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"
PASSWORD="${NEO4J_PASSWORD:-jeju-ontology-2026}"
DATA_DIR="$ROOT/artifacts/graph_db/neo4j"
CONF="$NEO4J_HOME/conf/neo4j.conf"

die() { echo "error: $*" >&2; exit 1; }

[ -d "$NEO4J_HOME" ] || die "no Neo4j at $NEO4J_HOME"
[ -x "$JAVA_HOME/bin/java" ] || die "no JDK at $JAVA_HOME"

KEYS=(server.default_listen_address server.bolt.listen_address
       server.bolt.advertised_address server.http.listen_address
       server.http.advertised_address
       server.https.enabled server.directories.data
       server.directories.logs server.memory.heap.initial_size
       server.memory.heap.max_size server.memory.pagecache.size
       dbms.security.auth_enabled)

configure() {
    mkdir -p "$DATA_DIR" "$DATA_DIR/logs"
    [ -f "$CONF.orig" ] || cp "$CONF" "$CONF.orig"
    # Idempotent: strip any block we wrote before, then append the current one.
    if grep -q '^# --- jeju-ontology ---' "$CONF" 2>/dev/null; then
        sed -i '/^# --- jeju-ontology ---$/,/^# --- end jeju-ontology ---$/d' "$CONF"
    fi
    # Neo4j rejects a key declared twice, and the shipped conf already sets some of
    # these. Comment out any live occurrence before appending ours.
    for k in "${KEYS[@]}"; do
        sed -i "s|^[[:space:]]*${k}[[:space:]]*=|#(jeju) &|" "$CONF"
    done
    cat >> "$CONF" <<EOF
# --- jeju-ontology ---
server.default_listen_address=127.0.0.1
server.bolt.listen_address=127.0.0.1:${BOLT_PORT}
# advertised_address does NOT follow listen_address: without these two the HTTP
# discovery endpoint hands the Neo4j Browser the default bolt://...:7687, and the
# browser then fails to connect the moment credentials are submitted.
server.bolt.advertised_address=127.0.0.1:${BOLT_PORT}
server.http.listen_address=127.0.0.1:${HTTP_PORT}
server.http.advertised_address=127.0.0.1:${HTTP_PORT}
server.https.enabled=false
server.directories.data=${DATA_DIR}
server.directories.logs=${DATA_DIR}/logs
server.memory.heap.initial_size=1g
server.memory.heap.max_size=2g
server.memory.pagecache.size=1g
dbms.security.auth_enabled=true
# --- end jeju-ontology ---
EOF
    echo "configured: bolt 127.0.0.1:${BOLT_PORT}, http 127.0.0.1:${HTTP_PORT}"
    echo "data at ${DATA_DIR}"
}

set_password() {
    # Only possible before the database is first created.
    if [ ! -d "$DATA_DIR/dbms" ]; then
        "$NEO4J_HOME/bin/neo4j-admin" dbms set-initial-password "$PASSWORD" \
            >/dev/null 2>&1 && echo "initial password set" || \
            echo "initial password already set, leaving it"
    fi
}

wait_ready() {
    # Neo4j reports started before Bolt accepts connections; poll the port.
    for _ in $(seq 1 90); do
        if "$JAVA_HOME/bin/java" -version >/dev/null 2>&1 && \
           (exec 3<>/dev/tcp/127.0.0.1/"$BOLT_PORT") 2>/dev/null; then
            exec 3<&- 2>/dev/null || true
            echo "bolt ready on 127.0.0.1:${BOLT_PORT}"
            return 0
        fi
        sleep 1
    done
    echo "bolt did not come up; last 40 log lines:" >&2
    tail -40 "$DATA_DIR/logs/neo4j.log" 2>/dev/null >&2 || true
    return 1
}

case "${1:-start}" in
    start)
        configure
        set_password
        "$NEO4J_HOME/bin/neo4j" start
        wait_ready
        ;;
    stop)
        "$NEO4J_HOME/bin/neo4j" stop || true
        ;;
    status)
        "$NEO4J_HOME/bin/neo4j" status || true
        ;;
    *)
        die "usage: $0 {start|stop|status}"
        ;;
esac
