#!/usr/bin/env bash
# Grants the app's service principal what the bundle cannot: read access in Unity
# Catalog and its role and grants in Lakebase. The bundle already grants the LLM
# endpoint, the warehouse, the daily job and writing to the audio volume.
# Idempotent; run it once after the first deploy of each app.
#
#   bash backend/conceder_permissoes.sh [app] [profile]
#
# The role has to come from the API (`databricks postgres create-role`): a role made
# with CREATE ROLE in psql has the same name but refuses the app's OAuth token with
# a wrong-password message.

set -euo pipefail

APP="${1:-aurelia-speech-analytics-en}"
PERFIL="${2:-field-toolbox}"
CATALOG="field_toolbox_catalog"
SCHEMA="aurelia_speech"
BRANCH="projects/aurelia-speech-ops/branches/production"
ENDPOINT="${BRANCH}/endpoints/primary"
DB="databricks_postgres"

SP=$(databricks apps get "$APP" -p "$PERFIL" -o json \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["service_principal_client_id"])')
echo "App ${APP}: service principal ${SP}"

# Unity Catalog: the dashboard and the transcript in the queue read the build schema.
databricks grants update catalog "$CATALOG" -p "$PERFIL" \
  --json "{\"changes\":[{\"principal\":\"${SP}\",\"add\":[\"USE_CATALOG\"]}]}" >/dev/null
databricks grants update schema "${CATALOG}.${SCHEMA}" -p "$PERFIL" \
  --json "{\"changes\":[{\"principal\":\"${SP}\",\"add\":[\"USE_SCHEMA\",\"SELECT\",\"READ_VOLUME\"]}]}" >/dev/null

# Lakebase: a service-principal role with default privileges only. The role id must
# start with a letter and an application id may start with a digit, hence the prefix;
# the Postgres role name is still the application id.
EXISTE=$(databricks postgres list-roles "$BRANCH" -p "$PERFIL" -o json \
  | SP="$SP" python3 -c 'import sys,json,os;print(any(r.get("status",{}).get("postgres_role")==os.environ["SP"] for r in json.load(sys.stdin)))')
if [ "$EXISTE" != "True" ]; then
  databricks postgres create-role "$BRANCH" --role-id "sp-${SP}" -p "$PERFIL" \
    --json "{\"spec\":{\"identity_type\":\"SERVICE_PRINCIPAL\",\"postgres_role\":\"${SP}\",\"auth_method\":\"LAKEBASE_OAUTH_V1\"}}" >/dev/null
fi

HOST=$(databricks postgres get-endpoint "$ENDPOINT" -p "$PERFIL" -o json \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"]["hosts"]["host"])')
export PGPASSWORD=$(databricks postgres generate-database-credential "$ENDPOINT" -p "$PERFIL" -o json \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
USUARIO=$(databricks current-user me -p "$PERFIL" -o json | python3 -c 'import sys,json;print(json.load(sys.stdin)["userName"])')

# Reads the scored queue and the current decision; inserts decisions, never updates them.
psql "host=${HOST} dbname=${DB} user=${USUARIO} sslmode=require" -v ON_ERROR_STOP=1 -q <<SQL
GRANT USAGE ON SCHEMA aurelia_speech_online, investigacao TO "${SP}";
GRANT SELECT ON aurelia_speech_online.fila_priorizada TO "${SP}";
GRANT SELECT ON investigacao.decisao_vigente TO "${SP}";
GRANT SELECT, INSERT ON investigacao.decisoes TO "${SP}";
SQL

echo "Permissions granted to ${SP}"
