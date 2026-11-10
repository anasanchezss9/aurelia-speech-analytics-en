-- The analyst's decisions on the investigation queue.
--
-- Insert-only: a new decision doesn't erase the previous one, and the current one is the most
-- recent per claim. It's the audit trail (who decided what, with which
-- score on screen) and it's the label that flows back to the lakehouse through Lakebase CDF
-- to retrain the model — especially for claims with a call, which are the
-- ones missing from training.
--
-- Simple types on purpose: CDF replicates this schema as Delta in Unity
-- Catalog, and a Postgres array becomes a problem on the other side.

CREATE SCHEMA IF NOT EXISTS investigacao;

CREATE TABLE IF NOT EXISTS investigacao.decisoes (
  decisao_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  chamado_id        TEXT        NOT NULL,
  decisao           TEXT        NOT NULL
                    CHECK (decisao IN ('fraude_confirmada', 'procedente', 'improcedente', 'pedir_evidencia')),
  analista          TEXT        NOT NULL,
  comentario        TEXT,
  score_no_momento  DOUBLE PRECISION,
  modelo_versao     INTEGER,
  tinha_ligacao     BOOLEAN,
  origem            TEXT        NOT NULL DEFAULT 'app' CHECK (origem IN ('app', 'simulacao')),
  decidido_em       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Lakebase CDF skips the table without this ("Does not have REPLICA IDENTITY
-- FULL"): logical replication needs the whole row on every change.
ALTER TABLE investigacao.decisoes REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS decisoes_chamado_recente
  ON investigacao.decisoes (chamado_id, decidido_em DESC);

CREATE OR REPLACE VIEW investigacao.decisao_vigente AS
SELECT DISTINCT ON (chamado_id) *
FROM investigacao.decisoes
ORDER BY chamado_id, decidido_em DESC, decisao_id DESC;
