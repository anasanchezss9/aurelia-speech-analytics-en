# Databricks notebook source
# MAGIC %md
# MAGIC # Silver — Transcription with `ai_transcribe` (alternative to Whisper)
# MAGIC
# MAGIC Same job as `02_silver_transcription`, without a GPU endpoint: the
# MAGIC transcription is a serverless AI Function called directly from SQL.
# MAGIC
# MAGIC | | Whisper (`02_silver_transcription`) | `ai_transcribe` (this notebook) |
# MAGIC |---|---|---|
# MAGIC | Infra | GPU_MEDIUM endpoint on Model Serving | None — managed function |
# MAGIC | Preprocessing | Mono, 16 kHz resample, 5-min chunks | None — 8 kHz μ-law WAV goes in as is |
# MAGIC | Limit | 16 MB payload per request | 1 h of audio and 512 MB per call |
# MAGIC | Diarization | No | Yes, `speaker_id` per segment |
# MAGIC | Billing | GPU hours while the endpoint is up | `AI_FUNCTIONS` product, per use |
# MAGIC | Languages | PT-BR supported | Beta: officially English and Spanish only |
# MAGIC
# MAGIC Writes to its own silver table (`TABLE_SILVER_AIT`) with the same contract as the
# MAGIC Whisper silver — `call_id` and `full_text` — so the gold layer and the portal can read
# MAGIC either one by changing only the table name.
# MAGIC
# MAGIC **Prerequisite**: an admin must enable *AI Transcribe* in
# MAGIC Settings → Previews. Without it the call fails with
# MAGIC `AI Transcribe Preview is not enabled`.

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
VOLUME = os.getenv("VOLUME_LANDING", "audio_landing")
TABLE_BRONZE = os.getenv("TABLE_BRONZE", "bronze_audio_files_imperativo")
TABLE_SILVER_AIT = os.getenv("TABLE_SILVER_AIT", "silver_transcripts_ait")

BRONZE = f"{CATALOG}.{SCHEMA}.{TABLE_BRONZE}"
SILVER = f"{CATALOG}.{SCHEMA}.{TABLE_SILVER_AIT}"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

print(f"Bronze: {BRONZE}")
print(f"Silver: {SILVER}")
print(f"Volume: {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Create the empty Silver table (if it doesn't exist)
# `full_text` is the running text, same as in the Whisper silver, and is what the gold
# layer reads. `full_text_falantes` prefixes each passage with the speaker, for anyone who
# wants to bring diarization into the prompt. `segments` keeps the function's raw output.
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SILVER} (
  call_id             STRING NOT NULL,
  audio_path          STRING NOT NULL,
  transcribed_at      TIMESTAMP,
  full_text           STRING,
  full_text_falantes  STRING,
  duration_seconds    DOUBLE,
  speaker_count       INT,
  segments            ARRAY<STRUCT<start: DOUBLE, `end`: DOUBLE, speaker_id: STRING, text: STRING>>,
  error_message       STRING,
  transcriber         STRING,
  CONSTRAINT {TABLE_SILVER_AIT}_pk PRIMARY KEY (call_id)
) USING DELTA
""")
print("OK — Silver ready.")

# COMMAND ----------

# DBTITLE 1,How many audio files haven't been transcribed yet?
n = spark.sql(f"""
SELECT count(*) AS n FROM {BRONZE} b
LEFT ANTI JOIN {SILVER} s ON b.call_id = s.call_id
""").first()["n"]
print(f"New audio files to transcribe: {n}")

if n == 0:
    dbutils.notebook.exit("no_new_audio")

# COMMAND ----------

# DBTITLE 1,ai_transcribe + MERGE in a single statement
# A single statement: the function runs once per audio file and the result goes straight
# into the MERGE. No need for the staging table from the Whisper notebook, which exists
# because there the Python UDF would rerun on every action.
#
# The volume is read with recursiveFileLookup because the portal writes to /uploads and
# /recordings. The `dbfs:` prefix is stripped on both sides of the join: bronze stores the
# path with it and read_files sometimes doesn't.
spark.sql(f"""
MERGE INTO {SILVER} s
USING (
  WITH novos AS (
    SELECT b.call_id, b.audio_path
    FROM {BRONZE} b
    LEFT ANTI JOIN {SILVER} s ON b.call_id = s.call_id
  ),
  transcritos AS (
    SELECT n.call_id, n.audio_path, ai_transcribe(f.content) AS r
    FROM novos n
    JOIN read_files(
      '{VOLUME_PATH}/',
      format => 'binaryFile',
      recursiveFileLookup => true,
      pathGlobFilter => '*.{{mp3,wav,m4a}}'
    ) f
      ON regexp_replace(f.path, '^dbfs:', '') = regexp_replace(n.audio_path, '^dbfs:', '')
  ),
  segmentos AS (
    SELECT
      call_id,
      audio_path,
      r:error_message::STRING              AS error_message,
      r:response.duration_seconds::DOUBLE  AS duration_seconds,
      from_json(
        to_json(r:response.segments),
        'ARRAY<STRUCT<start: DOUBLE, `end`: DOUBLE, speaker_id: STRING, text: STRING>>'
      ) AS segments
    FROM transcritos
  )
  SELECT
    call_id,
    audio_path,
    current_timestamp() AS transcribed_at,
    array_join(transform(segments, x -> trim(x.text)), ' ') AS full_text,
    array_join(
      transform(segments, x -> concat('[Falante ', coalesce(x.speaker_id, '?'), '] ', trim(x.text))),
      '\\n'
    ) AS full_text_falantes,
    duration_seconds,
    size(array_distinct(filter(transform(segments, x -> x.speaker_id), x -> x IS NOT NULL))) AS speaker_count,
    segments,
    error_message,
    'ai_transcribe' AS transcriber
  FROM segmentos
) u
ON s.call_id = u.call_id
WHEN NOT MATCHED THEN INSERT *
""")
print("MERGE complete.")

# COMMAND ----------

# DBTITLE 1,Results
display(spark.sql(f"""
SELECT call_id, duration_seconds, speaker_count, error_message,
       length(full_text) AS chars, substr(full_text, 1, 300) AS preview
FROM {SILVER}
ORDER BY transcribed_at DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wiring it into gold and the portal
# MAGIC
# MAGIC `03_gold_ai_query_analysis` accepts the `table_silver` and
# MAGIC `table_gold` widgets. To build the gold table for this version without touching the Whisper one:
# MAGIC
# MAGIC ```python
# MAGIC dbutils.notebook.run("03_gold_ai_query_analysis", 0, {
# MAGIC     "table_silver": "silver_transcripts_ait",
# MAGIC     "table_gold":   "gold_call_insights_ait",
# MAGIC })
# MAGIC ```
# MAGIC
# MAGIC The portal reads `call_id` and `full_text` from silver. Pointing `PORTAL_TABLE_SILVER`
# MAGIC and `PORTAL_TABLE_GOLD` in `app.yaml` to the `_ait` tables switches the displayed
# MAGIC version without changing code.
