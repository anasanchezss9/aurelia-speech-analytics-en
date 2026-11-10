-- ===================================================================
-- Silver — Transcription via Whisper, 100% SQL with ai_query
-- ===================================================================
-- Reads the bytes from the volume with read_files(format='binaryFile') using the
-- bronze audio_path, and calls the Whisper endpoint via ai_query.
--
-- IMPORTANT: ai_query can call ANY Model Serving
-- endpoint — including custom pyfunc models like Whisper.
-- The Whisper endpoint is an MLflow pyfunc that
-- expects a DataFrame with a base64 `audio` column:
--   {"dataframe_records": [{"audio": "<base64_audio>"}]}
-- and returns:
--   {"predictions": ["<transcribed text>"]}
--
-- For ai_query to send that format, we pass `request` as a
-- STRUCT — Spark turns each row into dataframe_records automatically.
--
-- The STREAMING TABLE guarantees each call_id is processed EXACTLY
-- ONCE — when a new row lands in bronze, SDP only runs
-- Whisper for that new audio file, with no manual MERGE.
-- ===================================================================

CREATE OR REFRESH STREAMING TABLE silver_transcripts
  COMMENT 'Call transcriptions, produced by the Whisper Turbo endpoint.'
  TBLPROPERTIES ('quality' = 'silver')
AS
WITH novos_audios_com_bytes AS (
  -- Takes only the new audio files (STREAM = only unseen rows) and reads the
  -- bytes from the volume with a 2nd read_files call. That yields one row
  -- per file with the `content` column.
  --
  -- CAUTION: a STREAMING TABLE exposes `_metadata.file_path` as a plain
  -- POSIX path (`/Volumes/...`), but BATCH read_files returns it with a
  -- `dbfs:` prefix. We normalize in the JOIN by stripping the prefix.
  SELECT
    b.call_id,
    b.audio_path,
    base64(f.content) AS audio_b64
  FROM STREAM bronze_audio_files b
  JOIN read_files('/Volumes/${catalog}/${schema}/${volume_landing}/', format => 'binaryFile') f
    ON regexp_replace(f.path, '^dbfs:', '') = b.audio_path
)
SELECT
  call_id,
  audio_path,
  current_timestamp() AS transcribed_at,
  '${whisper_endpoint}' AS whisper_endpoint,

  -- ai_query sends STRUCT(audio=audio_b64), which becomes dataframe_records
  -- with the `audio` column the Whisper pyfunc expects.
  ai_query(
    '${whisper_endpoint}',
    named_struct('audio', audio_b64),
    returnType => 'STRING'
  ) AS full_text
FROM novos_audios_com_bytes;


-- Final flattened view — makes consumption easier in Gold and BI
CREATE OR REFRESH MATERIALIZED VIEW silver_transcripts_clean
  COMMENT 'Normalized transcriptions for downstream consumption.'
  TBLPROPERTIES ('quality' = 'silver')
AS
SELECT
  call_id,
  audio_path,
  transcribed_at,
  whisper_endpoint,
  full_text,
  CASE
    WHEN full_text IS NULL OR length(full_text) < 10
      THEN 'TRANSCRIPTION_EMPTY_OR_FAILED'
    ELSE NULL
  END AS transcription_error
FROM silver_transcripts;
