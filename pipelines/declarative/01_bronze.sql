-- ===================================================================
-- Bronze — everything that arrives, as it arrives, as STREAMING TABLEs
-- ===================================================================
-- Three sources, one pattern: STREAM read_files, the declarative
-- equivalent of Auto Loader. SDP keeps the checkpoint, so each file is read
-- once; in triggered mode the table processes the backlog and stops.
--
--   audio           → Volume ${volume_landing} (pointer + metadata, no bytes)
--   operational     → Volume ${volume_operacional}, one folder per table
--                     from the order system and the customer-service system
--   text            → transcriptions the telephony platform already exports
--
-- A file overwritten with the same name is not read again. When you regenerate the
-- synthetic data, run the pipeline with a full refresh.
--
-- `data_hora` arrives as a timestamp without time zone (TIMESTAMP_NTZ), which a Delta
-- table only accepts with the timestampNtz feature enabled; the CAST avoids that.
-- ===================================================================

CREATE OR REFRESH STREAMING TABLE bronze_audio_files
  COMMENT 'Raw audio of consultant calls (pointer + metadata; no bytes).'
  TBLPROPERTIES ('quality' = 'bronze', 'pipelines.reset.allowed' = 'false')
AS
SELECT
  regexp_extract(_metadata.file_path, '([^/]+)\\.(mp3|wav|m4a)$', 1) AS call_id,
  _metadata.file_path                                                AS audio_path,
  _metadata.file_size                                                AS size_bytes,
  _metadata.file_modification_time                                   AS source_modified_at,
  current_timestamp()                                                AS ingested_at
FROM STREAM read_files(
  '/Volumes/${catalog}/${schema}/${volume_landing}/',
  format              => 'binaryFile',
  recursiveFileLookup => true,
  pathGlobFilter      => '*.{mp3,wav,m4a}'
);

-- ----- Order system ---------------------------------------------------------

CREATE OR REFRESH STREAMING TABLE bronze_consultoras
  COMMENT 'Consultant registry.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/consultoras', format => 'parquet');

CREATE OR REFRESH STREAMING TABLE bronze_produtos
  COMMENT 'Product catalog, with price and weight.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/produtos', format => 'parquet');

CREATE OR REFRESH STREAMING TABLE bronze_centros_distribuicao
  COMMENT 'Distribution centers.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/centros_distribuicao', format => 'parquet');

CREATE OR REFRESH STREAMING TABLE bronze_pedidos
  COMMENT 'Orders per cycle, with DC and carrier.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/pedidos', format => 'parquet');

CREATE OR REFRESH STREAMING TABLE bronze_pedido_itens
  COMMENT 'Items of each order.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/pedido_itens', format => 'parquet');

CREATE OR REFRESH STREAMING TABLE bronze_entregas
  COMMENT 'Deliveries: dates, status, weight at dispatch and at the hub, proof of delivery.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/entregas', format => 'parquet');

-- ----- Customer-service system ----------------------------------------------

CREATE OR REFRESH STREAMING TABLE bronze_chamados
  COMMENT 'Missing-product claims opened by consultants.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/chamados', format => 'parquet');

CREATE OR REFRESH STREAMING TABLE bronze_chamado_itens
  COMMENT 'Items the consultant says were missing.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/chamado_itens', format => 'parquet');

CREATE OR REFRESH STREAMING TABLE bronze_investigacoes
  COMMENT 'Investigation outcomes — the only fraud label there is.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/investigacoes', format => 'parquet');

CREATE OR REFRESH STREAMING TABLE bronze_ligacoes
  COMMENT 'Contact-center calls, with the claim linked in the CRM.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT * EXCEPT (data_hora), CAST(data_hora AS TIMESTAMP) AS data_hora,
          _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/ligacoes', format => 'parquet');

CREATE OR REFRESH STREAMING TABLE bronze_transcricoes_texto
  COMMENT 'Transcriptions the telephony platform already exports as text.' TBLPROPERTIES ('quality' = 'bronze')
AS SELECT * EXCEPT (data_hora), CAST(data_hora AS TIMESTAMP) AS data_hora,
          _metadata.file_path AS _arquivo, current_timestamp() AS _ingerido_em
FROM STREAM read_files('/Volumes/${catalog}/${schema}/${volume_operacional}/transcricoes_texto', format => 'parquet');
