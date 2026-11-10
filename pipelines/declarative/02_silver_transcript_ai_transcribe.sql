-- ===================================================================
-- Silver — Transcription with ai_transcribe, 100% SQL and no endpoint
-- ===================================================================
-- Alternative to 02_silver_transcript.sql. Publishes the same tables
-- (silver_transcripts and silver_transcripts_clean) with the same columns
-- gold reads, so switching versions means swapping this file for the other
-- in the `libraries` of pipeline.yml. Both in the same pipeline
-- collide on the table name — it's one or the other.
--
-- What changes compared to Whisper:
--   - There is no GPU endpoint to start, warm up or forget running.
--   - Telephony WAV (8 kHz stereo μ-law) goes in as is: no mono,
--     no resampling, no chunking. The limit is 1 h and 512 MB per call.
--   - Each segment comes with a speaker_id (automatic diarization).
--   - Beta, and officially English and Spanish only. It works in PT-BR, but
--     with no guarantee — compare it with Whisper on your audio before switching.
--
-- Requires the "AI Transcribe" preview enabled in the workspace and serverless
-- environment 3+ (the serverless pipeline already meets that).
-- ===================================================================

CREATE OR REFRESH STREAMING TABLE silver_transcripts
  COMMENT 'Call transcriptions, produced by the ai_transcribe AI Function.'
  TBLPROPERTIES ('quality' = 'silver')
AS
WITH novos_audios AS (
  -- recursiveFileLookup because the portal writes to /uploads and /recordings.
  SELECT
    b.call_id,
    b.audio_path,
    ai_transcribe(f.content) AS r
  FROM STREAM bronze_audio_files b
  JOIN read_files(
    '/Volumes/${catalog}/${schema}/${volume_landing}/',
    format => 'binaryFile',
    recursiveFileLookup => true,
    pathGlobFilter => '*.{mp3,wav,m4a}'
  ) f
    ON regexp_replace(f.path, '^dbfs:', '') = regexp_replace(b.audio_path, '^dbfs:', '')
),
segmentos AS (
  SELECT
    call_id,
    audio_path,
    r:error_message::STRING             AS error_message,
    r:response.duration_seconds::DOUBLE AS duration_seconds,
    from_json(
      to_json(r:response.segments),
      'ARRAY<STRUCT<start: DOUBLE, `end`: DOUBLE, speaker_id: STRING, text: STRING>>'
    ) AS segments
  FROM novos_audios
)
SELECT
  call_id,
  audio_path,
  current_timestamp() AS transcribed_at,
  'ai_transcribe'     AS whisper_endpoint,
  array_join(transform(segments, x -> trim(x.text)), ' ') AS full_text,
  array_join(
    transform(segments, x -> concat('[Falante ', coalesce(x.speaker_id, '?'), '] ', trim(x.text))),
    '\n'
  ) AS full_text_falantes,
  duration_seconds,
  size(array_distinct(filter(transform(segments, x -> x.speaker_id), x -> x IS NOT NULL))) AS speaker_count,
  segments,
  error_message
FROM segmentos;


-- Same view as in 02_silver_transcript.sql, with the function's error when there is one.
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
  full_text_falantes,
  speaker_count,
  CASE
    WHEN error_message IS NOT NULL THEN concat('AI_TRANSCRIBE_ERROR: ', error_message)
    WHEN full_text IS NULL OR length(full_text) < 10 THEN 'TRANSCRIPTION_EMPTY_OR_FAILED'
    ELSE NULL
  END AS transcription_error
FROM silver_transcripts;
