#!/usr/bin/env bash
# Sets up the operational side of the build in Lakebase. Idempotent.
#
#   bash lakebase/criar_lakebase.sh [perfil]
#
# 1. Lakebase project (Postgres 17, autoscaling, scale-to-zero).
# 2. Schema `investigacao`: the analyst's decisions, insert-only.
# 3. Synced table: the prioritized queue from UC → Postgres, so the app can read it without
#    waking up the warehouse. SNAPSHOT, because the queue is rewritten in full on every
#    scoring run; the daily job triggers the refresh after scoring.
# 4. CDF config: the `investigacao` schema → Delta in UC, continuous. It's the way
#    back: the analyst's decision becomes a label in the lakehouse with no hand-built pipeline.
#
# Requires the Databricks CLI with the `postgres` command group, psql and your own role in Lakebase.
# The token goes from the CLI straight to psql via PGPASSWORD; nothing touches the disk.

set -euo pipefail

PERFIL="${1:-field-toolbox}"
CATALOG="field_toolbox_catalog"
PROJETO="aurelia-speech-ops"
BRANCH="projects/${PROJETO}/branches/production"
ENDPOINT="${BRANCH}/endpoints/primary"
DB="databricks_postgres"
# The same database has two names: `databricks_postgres` inside Postgres and
# `databricks-postgres` as an API resource (CDF needs the second one).
DB_RECURSO="databricks-postgres"
AQUI="$(cd "$(dirname "$0")" && pwd)"

databricks postgres get-project "projects/${PROJETO}" -p "$PERFIL" >/dev/null 2>&1 || \
  databricks postgres create-project "$PROJETO" -p "$PERFIL" \
    --json '{"spec":{"display_name":"Aurélia Speech — investigação de chamados","pg_version":"17"}}'

HOST=$(databricks postgres get-endpoint "$ENDPOINT" -p "$PERFIL" -o json \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"]["hosts"]["host"])')
USUARIO=$(databricks current-user me -p "$PERFIL" -o json | python3 -c 'import sys,json;print(json.load(sys.stdin)["userName"])')
export PGPASSWORD=$(databricks postgres generate-database-credential "$ENDPOINT" -p "$PERFIL" -o json \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')

psql "host=${HOST} dbname=${DB} user=${USUARIO} sslmode=require" -v ON_ERROR_STOP=1 -f "${AQUI}/schema.sql"

# UC: schema for the synced table's online views and target schema for CDF.
databricks schemas create aurelia_speech_online "$CATALOG" -p "$PERFIL" >/dev/null 2>&1 || true
databricks schemas create aurelia_speech_investigacao "$CATALOG" -p "$PERFIL" >/dev/null 2>&1 || true

SYNCED="${CATALOG}.aurelia_speech_online.fila_priorizada"
databricks api get "/api/2.0/postgres/synced_tables/${SYNCED}" -p "$PERFIL" >/dev/null 2>&1 || \
databricks api post "/api/2.0/postgres/synced_tables?synced_table_id=${SYNCED}" -p "$PERFIL" --json "$(cat <<JSON
{
  "name": "${SYNCED}",
  "spec": {
    "postgres_database": "${DB}",
    "branch": "${BRANCH}",
    "source_table_full_name": "${CATALOG}.aurelia_speech.fila_priorizada",
    "primary_key_columns": ["chamado_id"],
    "scheduling_policy": "SNAPSHOT",
    "create_database_objects_if_missing": true,
    "new_pipeline_spec": {"storage_catalog": "${CATALOG}", "storage_schema": "aurelia_speech_online"}
  }
}
JSON
)" >/dev/null

databricks postgres list-cdf-configs "${BRANCH}/databases/${DB_RECURSO}" -p "$PERFIL" -o json 2>/dev/null \
  | grep -q '"postgres_schema": *"investigacao"' || \
  databricks postgres create-cdf-config "${BRANCH}/databases/${DB_RECURSO}" "$CATALOG" aurelia_speech_investigacao investigacao \
    --cdf-config-id investigacao -p "$PERFIL"

echo "Lakebase ready: ${HOST}"
echo "  queue (read):       aurelia_speech_online.fila_priorizada in Postgres"
echo "  decisions (write):  investigacao.decisoes → ${CATALOG}.aurelia_speech_investigacao"
