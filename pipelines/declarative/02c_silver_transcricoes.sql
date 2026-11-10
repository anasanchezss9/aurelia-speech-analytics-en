-- ===================================================================
-- Silver — every transcription in a single table
-- ===================================================================
-- Gold shouldn't know where the transcription came from. Two flows write
-- to the same STREAMING TABLE: the audio one, transcribed by ai_transcribe (or
-- by Whisper, depending on the 02_* in `libraries`), and the text one the
-- telephony platform already exports. A new source is one more CREATE FLOW, without
-- touching gold.
-- ===================================================================

CREATE OR REFRESH STREAMING TABLE silver_transcricoes (
  call_id       STRING    NOT NULL,
  full_text     STRING,
  origem        STRING    COMMENT 'audio | texto',
  transcrito_em TIMESTAMP,
  CONSTRAINT texto_suficiente EXPECT (length(full_text) > 20) ON VIOLATION DROP ROW
)
COMMENT 'Transcriptions of every call, from audio or already as text.'
TBLPROPERTIES ('quality' = 'silver');

CREATE FLOW transcricoes_do_audio
AS INSERT INTO silver_transcricoes BY NAME
SELECT call_id, full_text, 'audio' AS origem, transcribed_at AS transcrito_em
FROM STREAM(silver_transcripts)
WHERE full_text IS NOT NULL;

CREATE FLOW transcricoes_em_texto
AS INSERT INTO silver_transcricoes BY NAME
SELECT call_id, full_text, 'texto' AS origem, gerado_em AS transcrito_em
FROM STREAM(bronze_transcricoes_texto);
