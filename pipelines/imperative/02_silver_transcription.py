# Databricks notebook source
# MAGIC %md
# MAGIC # Silver — Transcription via the Whisper endpoint
# MAGIC
# MAGIC Takes Bronze audio files that **have not been transcribed yet**, reads the bytes
# MAGIC from the volume, sends them to the Whisper endpoint, and saves the text in Silver.
# MAGIC
# MAGIC **Pattern**: `pandas_udf` spreads the calls across the cluster + `MERGE INTO`
# MAGIC so that each audio file is transcribed **exactly once**, even if the
# MAGIC pipeline runs several times a day.

# COMMAND ----------

# MAGIC %pip install -q python-dotenv==1.0.1 soundfile==0.12.1
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
TABLE_BRONZE = os.getenv("TABLE_BRONZE", "bronze_audio_files_imperativo")
TABLE_SILVER = os.getenv("TABLE_SILVER", "silver_transcripts")
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "aurelia-whisper-turbo")

BRONZE = f"{CATALOG}.{SCHEMA}.{TABLE_BRONZE}"
SILVER = f"{CATALOG}.{SCHEMA}.{TABLE_SILVER}"

print(f"Bronze:           {BRONZE}")
print(f"Silver:           {SILVER}")
print(f"Whisper endpoint: {WHISPER_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Create the empty Silver table (if it doesn't exist)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SILVER} (
  call_id          STRING NOT NULL,
  audio_path       STRING NOT NULL,
  transcribed_at   TIMESTAMP,
  full_text        STRING,
  whisper_endpoint STRING,
  CONSTRAINT {TABLE_SILVER}_pk PRIMARY KEY (call_id)
) USING DELTA
""")
print("OK — Silver ready.")

# COMMAND ----------

# DBTITLE 1,Find only the NEW audio files (anti-join with Silver)
to_process = spark.sql(f"""
SELECT b.call_id, b.audio_path
FROM {BRONZE} b
LEFT ANTI JOIN {SILVER} s
  ON b.call_id = s.call_id
""")

n = to_process.count()
print(f"New audio files to transcribe: {n}")
display(to_process)

# COMMAND ----------

# DBTITLE 1,Capture host and token on the DRIVER and pass them to the UDF
# On serverless, executors don't inherit Databricks credentials. The fix is to
# grab host + token on the driver and pass them to the UDF via broadcast/closure.
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
DATABRICKS_HOST = ctx.apiUrl().get()
DATABRICKS_TOKEN = ctx.apiToken().get()

# Quick check — fail early if there's no token
assert DATABRICKS_HOST and DATABRICKS_TOKEN, "Could not get host/token from the context."
ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{WHISPER_ENDPOINT}/invocations"
print(f"Endpoint URL: {ENDPOINT_URL}")

# COMMAND ----------

# DBTITLE 1,UDF that calls the Whisper endpoint via REST
import base64
import pandas as pd
from pyspark.sql.functions import pandas_udf

# Captured in the closure — Spark serializes them to the executors
_url = ENDPOINT_URL
_token = DATABRICKS_TOKEN

# Model Serving endpoints have a hard limit of 16 MB per request. Real .wav
# files can be 30+ MB. The strategy:
#   1) soundfile reads WAV natively (no ffmpeg, no librosa/numba).
#   2) We convert to mono by averaging the channels.
#   3) Resample to 16kHz with scipy.signal.resample_poly (no JIT).
#   4) We save as OGG Vorbis (~80x smaller than WAV) or FLAC (lossless ~3x).
#
# We avoid librosa because it loads numba, which tries to cache JIT output in
# /local_disk0/.ephemeral_nfs/, which is not writable on serverless.
@pandas_udf("string")
def transcribe_udf(audio_paths: pd.Series) -> pd.Series:
    import io
    import requests
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    TARGET_SR = 16000

    out = []
    headers = {"Authorization": f"Bearer {_token}", "Content-Type": "application/json"}

    def _send(audio_bytes: bytes) -> str:
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        r = requests.post(
            _url,
            headers=headers,
            json={"dataframe_records": [{"audio": audio_b64}]},
            timeout=600,
        )
        r.raise_for_status()
        preds = r.json().get("predictions", [])
        return (preds[0] if isinstance(preds, list) and preds else "").strip()

    for path in audio_paths:
        try:
            local = path.replace("dbfs:", "") if path.startswith("dbfs:") else path

            # 1) Read the WAV (soundfile covers WAV/FLAC natively)
            y, sr = sf.read(local, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)  # mono

            # 2) Resample to 16kHz mono (Whisper's native format)
            if sr != TARGET_SR:
                from math import gcd
                g = gcd(sr, TARGET_SR)
                y = resample_poly(y, TARGET_SR // g, sr // g).astype("float32")
                sr = TARGET_SR

            # 3) Save as WAV PCM-16 and try to send it.
            # Long audio can exceed 16MB; in that case we split it into
            # 8-minute chunks (= ~7.7 MB each) and concatenate the texts.
            #
            # Why not FLAC/OGG? On DBR Serverless libvorbis segfaults
            # and libFLAC occasionally does too. Plain WAV is the most
            # stable path + chunking solves the payload limit.
            buf = io.BytesIO()
            sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
            audio_bytes = buf.getvalue()

            # Hard payload limit of the endpoint: 16 MiB. Base64 takes 4/3 of
            # the original size, so the maximum raw WAV is ~12 MB. Safety
            # margin: 11 MB.
            #
            # Sizing: 16 kHz mono PCM-16 = 32_000 bytes/second.
            # 11 MB / 32_000 ≈ 360 seconds ⇒ 5-minute chunks.
            MAX_RAW = 11 * 1024 * 1024

            if len(audio_bytes) <= MAX_RAW:
                text = _send(audio_bytes)
            else:
                chunk_secs = 300  # 5 min
                samples_per_chunk = chunk_secs * sr
                texts = []
                for i in range(0, len(y), samples_per_chunk):
                    chunk = y[i : i + samples_per_chunk]
                    cbuf = io.BytesIO()
                    sf.write(cbuf, chunk, sr, format="WAV", subtype="PCM_16")
                    texts.append(_send(cbuf.getvalue()))
                text = " ".join(t for t in texts if t)

            out.append(text)
        except Exception as exc:
            out.append(f"[ERRO_TRANSCRICAO: {type(exc).__name__}: {exc}]")
    return pd.Series(out)

# COMMAND ----------

# DBTITLE 1,Apply the UDF and prepare the DF for the MERGE
if n == 0:
    print("Nothing to do — every audio file is already transcribed.")
    dbutils.notebook.exit("no_new_audio")

transcribed = (
    to_process
    .withColumn("full_text", transcribe_udf("audio_path"))
    .selectExpr(
        "call_id",
        "audio_path",
        "current_timestamp() AS transcribed_at",
        "full_text",
        f"'{WHISPER_ENDPOINT}' AS whisper_endpoint",
    )
)

# Materialize into a staging table before the MERGE.
#
# The obvious path would be `.cache()`, but serverless rejects it with
# NOT_SUPPORTED_WITH_SERVERLESS: PERSIST TABLE — and serverless is exactly what
# the app's "Processar" button uses, because it calls `jobs.submit` with no cluster spec.
# Without materializing, every action (count, display, merge) would rerun the UDF and
# pay for a new Whisper call per audio file.
STAGING = f"{SILVER}_staging"
transcribed.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(STAGING)
staged = spark.table(STAGING)
print(f"Transcribed {staged.count()} audio files.")
display(staged.select("call_id", "full_text"))

# COMMAND ----------

# DBTITLE 1,MERGE INTO Silver — inserts only what is new
from delta.tables import DeltaTable

silver_table = DeltaTable.forName(spark, SILVER)
(
    silver_table.alias("s")
    .merge(
        staged.alias("u"),
        "s.call_id = u.call_id",
    )
    .whenNotMatchedInsertAll()
    # No UPDATE: if an audio file was already transcribed, we keep the original version.
    # To reprocess, delete the row from Silver and run again.
    .execute()
)
print("MERGE complete.")

spark.sql(f"DROP TABLE IF EXISTS {STAGING}")

# COMMAND ----------

# DBTITLE 1,Results
display(spark.table(SILVER).orderBy("transcribed_at"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why MERGE and not append?
# MAGIC
# MAGIC The `LEFT ANTI JOIN` in step 3 already ensures only new audio files reach
# MAGIC the DataFrame. But in environments with retries (the job fails midway and reruns),
# MAGIC two processes can race. `MERGE INTO ... WHEN NOT MATCHED`
# MAGIC guarantees **transactional idempotency** — Delta handles it atomically.
