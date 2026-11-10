# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — Audio ingestion via Auto Loader
# MAGIC
# MAGIC Reads new files from the `audio_landing` volume and materializes them into a
# MAGIC Delta table with **one row per audio file**.
# MAGIC
# MAGIC **Pattern**: Auto Loader in `binaryFile` mode + `trigger(availableNow=True)`
# MAGIC — each run processes only new files, in batch.
# MAGIC
# MAGIC > Why `binaryFile` without storing the bytes? Because `binaryFile` reads
# MAGIC > the **raw** file and exposes `path`, `modificationTime` and `length`.
# MAGIC > We don't store the bytes in Bronze (audio volumes can grow
# MAGIC > a lot); we keep only the pointer. The bytes are read in Silver.

# COMMAND ----------

# MAGIC %pip install -q python-dotenv==1.0.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Settings from .env
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

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
CHECKPOINT = f"{VOLUME_PATH}/_checkpoints/bronze"
TARGET = f"{CATALOG}.{SCHEMA}.{TABLE_BRONZE}"

print(f"Source:     {VOLUME_PATH}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Target:     {TARGET}")

# COMMAND ----------

# DBTITLE 1,Auto Loader — reads new audio files
from pyspark.sql.functions import (
    col,
    current_timestamp,
    regexp_extract,
    input_file_name,
)

from pyspark.sql.functions import lower, when

stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.includeExistingFiles", "true")
    # Accepts the 3 most common formats (Havaia/Genesys export WAV; gTTS/MP3
    # from the demo; M4A/AAC from some mobile recorders).
    .option("pathGlobFilter", "*.{mp3,wav,m4a}")
    .load(VOLUME_PATH)
    .select(
        col("path").alias("audio_path"),
        col("length").alias("size_bytes"),
        col("modificationTime").alias("source_modified_at"),
        # Extracts the file name (without extension) as a stable call_id.
        regexp_extract(col("path"), r"([^/]+)\.(mp3|wav|m4a)$", 1).alias("call_id"),
        # Extracts the extension as audio_format (mp3, wav, m4a).
        lower(regexp_extract(col("path"), r"\.([a-zA-Z0-9]+)$", 1)).alias("audio_format"),
        # `source` tells where the audio came from:
        #   upload    -> file sent through the portal (/uploads/ subfolder)
        #   recording -> live recording from the portal (/recordings/ subfolder)
        #   landing   -> dropped straight into the volume (Havaia, Lakeflow Connect, etc)
        when(col("path").contains("/uploads/"), "upload")
        .when(col("path").contains("/recordings/"), "recording")
        .otherwise("landing")
        .alias("source"),
        current_timestamp().alias("ingested_at"),
    )
)

# COMMAND ----------

# DBTITLE 1,Write in batch (trigger=availableNow)
(
    stream.writeStream.format("delta")
    .option("checkpointLocation", CHECKPOINT)
    .option("mergeSchema", "true")
    .outputMode("append")
    .trigger(availableNow=True)  # processes the backlog and stops
    .toTable(TARGET)
    .awaitTermination()
)

# COMMAND ----------

# DBTITLE 1,Check the result
display(spark.table(TARGET).orderBy("ingested_at"))

# COMMAND ----------

# DBTITLE 1,How many new files arrived in this run?
count = spark.table(TARGET).count()
print(f"Running total in bronze: {count} files.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes on idempotency
# MAGIC
# MAGIC - Auto Loader keeps a `_checkpoint` that tracks which files have already
# MAGIC   been read. On a 2nd run, **only the new ones** go into bronze.
# MAGIC - If you delete the checkpoint, Auto Loader will **reprocess everything**.
# MAGIC - The `call_id` (extracted from the file name) is the stable key used
# MAGIC   by Silver to avoid redoing transcriptions that are already done.
