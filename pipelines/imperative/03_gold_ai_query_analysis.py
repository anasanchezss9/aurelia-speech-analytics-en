# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — Transcript analysis with Claude (missing-product claims)
# MAGIC
# MAGIC For each Silver transcription that **hasn't been analyzed yet**, calls
# MAGIC `ai_query` with Claude Sonnet 4.6 asking for structured JSON focused on
# MAGIC Aurélia's missing-product claims use case: identify delays, missing items, fraud
# MAGIC suspicions and missing products.
# MAGIC
# MAGIC **Idempotency**: `LEFT ANTI JOIN` + `INSERT INTO`. Reprocessing a
# MAGIC call requires deleting its row from Gold.

# COMMAND ----------

# MAGIC %pip install -q python-dotenv==1.0.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config
import os
from pathlib import Path
from dotenv import load_dotenv

for candidate in [
    Path.cwd() / ".env",
    Path.cwd().parent / ".env",
    Path("/Workspace/Users/ana.sanchez@databricks.com/aurelia_speech_pipelines/.env"),
]:
    if candidate.exists():
        load_dotenv(candidate)
        break

CATALOG = os.getenv("CATALOG", "field_toolbox_catalog")
SCHEMA = os.getenv("SCHEMA", "aurelia_speech")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-6")

# The widgets let the same gold run on the Whisper silver or the
# ai_transcribe one (02b) without duplicating the notebook. When empty, .env applies.
dbutils.widgets.text("table_silver", "")
dbutils.widgets.text("table_gold", "")
TABLE_SILVER = dbutils.widgets.get("table_silver") or os.getenv("TABLE_SILVER", "silver_transcripts")
TABLE_GOLD = dbutils.widgets.get("table_gold") or os.getenv("TABLE_GOLD", "gold_call_insights")

SILVER = f"{CATALOG}.{SCHEMA}.{TABLE_SILVER}"
GOLD = f"{CATALOG}.{SCHEMA}.{TABLE_GOLD}"

print(f"Silver:       {SILVER}")
print(f"Gold:         {GOLD}")
print(f"LLM endpoint: {LLM_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Create Gold (if it doesn't exist)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD} (
  call_id              STRING NOT NULL,
  analyzed_at          TIMESTAMP,
  llm_endpoint         STRING,

  -- General insights
  sentiment            STRING,
  emotion              STRING,
  intent               STRING,
  topics               ARRAY<STRING>,
  summary              STRING,
  next_action          STRING,
  compliance_ok        BOOLEAN,
  compliance_reason    STRING,
  churn_risk           DOUBLE,

  -- Missing-product claims specific
  tipo_problema        STRING,            -- atraso | falta_parcial | violacao_caixa | fraude_suspeita | outro
  suspeita_fraude      BOOLEAN,
  fraude_confianca     DOUBLE,            -- 0..1 self-reported by the model, NOT calibrated
  fraude_indicadores   ARRAY<STRING>,
  produtos_faltantes   ARRAY<STRING>,
  reincidencia         BOOLEAN,           -- does the consultant mention a recurring problem?
  consultora_codigo    STRING,
  pedido_numero        STRING,
  sinais_relato        ARRAY<STRING>,     -- behavior in the account, regardless of who diverted the goods
  relato_consistente   BOOLEAN,

  raw_response         STRING,
  CONSTRAINT {TABLE_GOLD}_pk PRIMARY KEY (call_id)
) USING DELTA
""")
print("OK — Gold ready.")

# COMMAND ----------

# DBTITLE 1,How many transcriptions haven't been analyzed yet?
to_analyze_count = spark.sql(f"""
SELECT count(*) AS n FROM {SILVER} s
LEFT ANTI JOIN {GOLD} g ON s.call_id = g.call_id
""").first()["n"]
print(f"New transcriptions to analyze: {to_analyze_count}")

if to_analyze_count == 0:
    dbutils.notebook.exit("no_new_transcripts")

# COMMAND ----------

# DBTITLE 1,Run ai_query (batch inference) on the new rows only
# The query below is the heart of the pipeline:
# - Builds the prompt with the transcription
# - Asks for structured JSON via responseFormat (json_schema)
# - Inserts straight into Gold only what hasn't been analyzed yet
#
# `ai_query` parallelizes calls automatically in batch — no
# pandas_udf or thread management needed.
#
# The prompt stays in Portuguese because the calls are in Portuguese and the
# measured results depend on its exact wording.

_sql = """
INSERT INTO __GOLD__
WITH novos AS (
  SELECT s.call_id, s.full_text
  FROM __SILVER__ s
  LEFT ANTI JOIN __GOLD__ g ON s.call_id = g.call_id
  WHERE s.full_text IS NOT NULL AND length(s.full_text) > 20
),
analise AS (
  SELECT
    call_id,
    full_text,
    ai_query(
      '__LLM__',
      CONCAT(
        'Você é analista de qualidade da Aurélia, especializado em chamadas ',
        'de consultoras (clientes B2B) reportando problemas em pedidos. ',
        'Estamos no projeto PPM Falta: detectar atrasos, faltas, violações ',
        'e suspeitas de fraude.\\n\\n',
        'Analise a transcrição e devolva JSON aderente ao schema. Seja ',
        'factual: só afirme o que estiver sustentado na transcrição. ',
        'Quando não houver evidência, use null ou string vazia.\\n\\n',
        'Em fraude_confianca devolva de 0 a 1 o quanto esta transcrição ',
        'sustenta a suspeita de desvio: 0 quando não há indício algum, 1 ',
        'quando a ligação declara o desvio de forma explícita. Julgue só o ',
        'que foi dito nesta ligação — você não tem o histórico da consultora. ',
        'Com suspeita_fraude false o valor deve expressar a dúvida residual, ',
        'em geral abaixo de 0.2, e não simplesmente 0.\\n\\n',
        'Definições de tipo_problema (escolha a que melhor descreve o caso):\\n',
        '- atraso: o pedido não chegou no prazo.\\n',
        '- falta_parcial: a caixa chegou íntegra, mas faltam itens, sem indício de desvio.\\n',
        '- violacao_caixa: a embalagem chegou danificada ou aberta, mas nenhum produto ',
        'sumiu e ninguém sugere desvio.\\n',
        '- fraude_suspeita: há indício de desvio intencional — caixa violada com itens ',
        'de valor sumidos, produtos trocados, enchimento no lugar de produto. Caixa ',
        'aberta COM produto sumido é fraude_suspeita, não violacao_caixa.\\n',
        '- outro: qualquer outro assunto (produto avariado ou vencido com caixa ',
        'intacta, cobrança, cadastro, dúvida, elogio).\\n\\n',
        'Em produtos_faltantes, devolva cada produto com o nome completo como foi ',
        'falado (ex.: "hidratante Diário de macadâmia", "Maré Oceano"), nunca só a ',
        'linha. Use esta lista de linhas da Aurélia apenas para corrigir a grafia ',
        'quando a transcrição errar: Âmbar, Maré, Lua, Memória, Riso, Raízes, ',
        'Diário, Aura (inclui Aura Artisan), Tempo, Brilho, Colo, Tez.\\n\\n',
        'Em sinais_relato, marque só o que foi dito sobre o RELATO da consultora, ',
        'independente de quem possa ter desviado: detalhes_vagos_ou_contraditorios ',
        '(não sabe dizer o que faltou, muda a versão), pressa_pelo_ressarcimento, ',
        'resiste_a_enviar_evidencia (evita foto, nota ou vídeo), ',
        'alega_nao_ter_conferido_na_entrega, historico_de_azar_recorrente (diz que ',
        'sempre acontece com ela), acusa_terceiros_sem_evidencia. Lista vazia quando ',
        'o relato é objetivo. Em relato_consistente, true quando a consultora sabe o ',
        'que aconteceu, mantém a mesma versão e oferece evidência.\\n\\n',
        'TRANSCRIÇÃO:\\n<<<\\n', full_text, '\\n>>>'
      ),
      responseFormat => '{
        "type": "json_schema",
        "json_schema": {
          "name": "ppm_falta_analysis",
          "schema": {
            "type": "object",
            "properties": {
              "sentiment":          {"type": "string", "enum": ["positivo","neutro","negativo"]},
              "emotion":            {"type": "string"},
              "intent":             {"type": "string"},
              "topics":             {"type": "array", "items": {"type": "string"}},
              "summary":            {"type": "string"},
              "next_action":        {"type": "string"},
              "compliance_ok":      {"type": "boolean"},
              "compliance_reason":  {"type": "string"},
              "churn_risk":         {"type": "number", "minimum": 0, "maximum": 1},
              "tipo_problema":      {"type": "string", "enum": ["atraso","falta_parcial","violacao_caixa","fraude_suspeita","outro"]},
              "suspeita_fraude":    {"type": "boolean"},
              "fraude_confianca":   {"type": "number", "minimum": 0, "maximum": 1},
              "fraude_indicadores": {"type": "array", "items": {"type": "string"}},
              "produtos_faltantes": {"type": "array", "items": {"type": "string"}},
              "reincidencia":       {"type": "boolean"},
              "consultora_codigo":  {"type": "string"},
              "pedido_numero":      {"type": "string"},
              "sinais_relato":      {"type": "array", "items": {"type": "string", "enum": ["detalhes_vagos_ou_contraditorios","pressa_pelo_ressarcimento","resiste_a_enviar_evidencia","alega_nao_ter_conferido_na_entrega","historico_de_azar_recorrente","acusa_terceiros_sem_evidencia"]}},
              "relato_consistente": {"type": "boolean"}
            },
            "required": ["sentiment","intent","tipo_problema","suspeita_fraude","fraude_confianca","summary","next_action","sinais_relato","relato_consistente"]
          },
          "strict": true
        }
      }',
      modelParameters => named_struct('temperature', 0.1, 'max_tokens', 2000)
    ) AS j
  FROM novos
)
SELECT
  call_id,
  current_timestamp() AS analyzed_at,
  '__LLM__'           AS llm_endpoint,

  j:sentiment::STRING                                AS sentiment,
  j:emotion::STRING                                  AS emotion,
  j:intent::STRING                                   AS intent,
  from_json(j:topics::STRING, 'ARRAY<STRING>')       AS topics,
  j:summary::STRING                                  AS summary,
  j:next_action::STRING                              AS next_action,
  j:compliance_ok::BOOLEAN                           AS compliance_ok,
  j:compliance_reason::STRING                        AS compliance_reason,
  j:churn_risk::DOUBLE                               AS churn_risk,

  j:tipo_problema::STRING                            AS tipo_problema,
  j:suspeita_fraude::BOOLEAN                         AS suspeita_fraude,
  j:fraude_confianca::DOUBLE                         AS fraude_confianca,
  from_json(j:fraude_indicadores::STRING, 'ARRAY<STRING>') AS fraude_indicadores,
  from_json(j:produtos_faltantes::STRING, 'ARRAY<STRING>') AS produtos_faltantes,
  j:reincidencia::BOOLEAN                            AS reincidencia,
  j:consultora_codigo::STRING                        AS consultora_codigo,
  j:pedido_numero::STRING                            AS pedido_numero,
  from_json(j:sinais_relato::STRING, 'ARRAY<STRING>') AS sinais_relato,
  j:relato_consistente::BOOLEAN                      AS relato_consistente,

  CAST(j AS STRING)                                  AS raw_response
FROM analise
"""

# Replace placeholders without an f-string (the JSON Schema has `{}`
# braces that would clash with an f-string or .format)
_sql = (_sql
        .replace("__GOLD__", GOLD)
        .replace("__SILVER__", SILVER)
        .replace("__LLM__", LLM_ENDPOINT))
spark.sql(_sql)

print("ai_query complete + insert into Gold.")

# COMMAND ----------

# DBTITLE 1,Results — business view
display(spark.sql(f"""
SELECT
  call_id,
  tipo_problema,
  suspeita_fraude,
  fraude_confianca,
  reincidencia,
  consultora_codigo,
  pedido_numero,
  produtos_faltantes,
  sentiment,
  churn_risk,
  summary,
  next_action
FROM {GOLD}
ORDER BY analyzed_at DESC
"""))

# COMMAND ----------

# DBTITLE 1,Focus on risky cases (fraud or repeat claims)
display(spark.sql(f"""
SELECT
  call_id,
  tipo_problema,
  fraude_confianca,
  fraude_indicadores,
  produtos_faltantes,
  next_action
FROM {GOLD}
WHERE suspeita_fraude = true OR reincidencia = true
ORDER BY analyzed_at DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why `ai_query` helps here
# MAGIC
# MAGIC | Topic | How `ai_query` helps |
# MAGIC |---|---|
# MAGIC | **Parallelism** | Spreads calls across the cluster automatically, no `pandas_udf` |
# MAGIC | **Structured output** | `responseFormat` with JSON Schema guarantees schema-compliant JSON |
# MAGIC | **Cost** | Runs in optimized batch mode; billed per token consumed |
# MAGIC | **Swappable model** | Change `LLM_ENDPOINT` in `.env` to try Llama, Haiku, GPT |
# MAGIC | **No credentials** | The cluster/job OAuth already authenticates to the endpoint |
# MAGIC
# MAGIC ### How to reprocess 1 call
# MAGIC
# MAGIC ```sql
# MAGIC DELETE FROM ${ana.gold} WHERE call_id = 'call_001_atraso';
# MAGIC -- Run this notebook again; it detects the "gap" and redoes only that one.
# MAGIC ```
