-- ===================================================================
-- Silver — operational data, typed, validated and with per-row calculations
-- ===================================================================
-- Each silver table answers one question once, so gold doesn't
-- repeat the calculation: what the order is worth, how much weight the box lost and
-- where, how much the claim asks for and how much of it is perfume, what the outcome was.
--
-- The expectations drop rows that make no sense (consultant
-- code in the wrong format, zero weight, delivery before dispatch) and
-- SDP records how many each one dropped in the pipeline event log.
-- ===================================================================

CREATE OR REFRESH MATERIALIZED VIEW silver_consultoras (
  CONSTRAINT codigo_valido EXPECT (consultora_codigo RLIKE '^[0-9]{5}$') ON VIOLATION DROP ROW
)
COMMENT 'Consultants, one row per code.'
TBLPROPERTIES ('quality' = 'silver')
AS
SELECT consultora_codigo, uf, regiao, data_cadastro, nivel, tem_app
FROM bronze_consultoras
QUALIFY row_number() OVER (PARTITION BY consultora_codigo ORDER BY _ingerido_em DESC) = 1;

CREATE OR REFRESH MATERIALIZED VIEW silver_produtos
COMMENT 'Product catalog.'
TBLPROPERTIES ('quality' = 'silver')
AS
SELECT sku, nome, linha, categoria, preco, peso_kg
FROM bronze_produtos
QUALIFY row_number() OVER (PARTITION BY sku ORDER BY _ingerido_em DESC) = 1;

CREATE OR REFRESH MATERIALIZED VIEW silver_pedidos (
  CONSTRAINT pedido_com_itens EXPECT (qtd_itens > 0) ON VIOLATION DROP ROW
)
COMMENT 'Orders with value, item count and perfume share.'
TBLPROPERTIES ('quality' = 'silver')
AS
WITH itens AS (
  SELECT
    i.pedido_numero,
    sum(i.quantidade)                                                      AS qtd_itens,
    sum(i.quantidade * i.preco_unitario)                                   AS valor_pedido,
    sum(CASE WHEN p.categoria = 'perfumaria' THEN i.quantidade ELSE 0 END) AS qtd_perfumaria
  FROM bronze_pedido_itens i
  JOIN silver_produtos p USING (sku)
  GROUP BY i.pedido_numero
)
SELECT
  p.pedido_numero, p.consultora_codigo, p.ciclo, p.data_pedido, p.uf, p.cd,
  p.transportadora, p.canal AS canal_pedido,
  i.qtd_itens, round(i.valor_pedido, 2) AS valor_pedido, i.qtd_perfumaria
FROM bronze_pedidos p
LEFT JOIN itens i USING (pedido_numero);

-- The box is weighed twice: at DC dispatch and at the last-mile hub.
-- The theoretical weight is the sum of the items. The two divergences separate the
-- causes of missing items: the DC dispatching light points to a picking error; the hub weighing
-- less than dispatch points to loss in transit; everything matching with a
-- signed proof of delivery is what is left for fraud to explain.
CREATE OR REFRESH MATERIALIZED VIEW silver_entregas (
  CONSTRAINT peso_teorico_positivo EXPECT (peso_teorico_kg > 0) ON VIOLATION DROP ROW,
  CONSTRAINT entrega_depois_do_envio EXPECT (data_entrega IS NULL OR data_entrega >= data_envio) ON VIOLATION DROP ROW
)
COMMENT 'Deliveries with the weight divergence at the DC and at the hub.'
TBLPROPERTIES ('quality' = 'silver')
AS
SELECT
  pedido_numero, data_envio, data_prevista, data_entrega, status_entrega,
  comprovante_assinado,
  peso_teorico_kg, peso_expedido_kg, peso_hub_kg,
  datediff(coalesce(data_entrega, data_prevista), data_prevista)                AS dias_atraso,
  round((peso_teorico_kg - peso_expedido_kg) / peso_teorico_kg, 4)              AS div_peso_cd,
  round((peso_expedido_kg - peso_hub_kg) / peso_expedido_kg, 4)                 AS div_peso_hub,
  peso_hub_kg IS NOT NULL                                                       AS pesado_no_hub
FROM bronze_entregas;

CREATE OR REFRESH MATERIALIZED VIEW silver_chamados (
  CONSTRAINT valor_nao_negativo EXPECT (valor_reclamado >= 0) ON VIOLATION DROP ROW,
  CONSTRAINT pedido_conhecido EXPECT (pedido_existe) ON VIOLATION DROP ROW
)
COMMENT 'Claims with what was claimed and the investigation outcome, when there was one.'
TBLPROPERTIES ('quality' = 'silver')
AS
WITH itens AS (
  SELECT
    ci.chamado_id,
    sum(ci.quantidade)                                                                  AS qtd_itens_reclamados,
    sum(CASE WHEN p.categoria = 'perfumaria' THEN ci.quantidade ELSE 0 END)             AS qtd_perfumaria_reclamada,
    sum(CASE WHEN p.categoria = 'perfumaria' THEN ci.quantidade * p.preco ELSE 0 END)   AS valor_perfumaria_reclamada,
    collect_list(p.nome)                                                                AS produtos_reclamados
  FROM bronze_chamado_itens ci
  JOIN silver_produtos p USING (sku)
  GROUP BY ci.chamado_id
)
SELECT
  c.chamado_id, c.pedido_numero, c.consultora_codigo, c.data_abertura,
  c.canal AS canal_chamado, c.motivo_declarado, c.valor_reclamado,
  c.status, c.valor_ressarcido,
  coalesce(i.qtd_itens_reclamados, 0)        AS qtd_itens_reclamados,
  coalesce(i.qtd_perfumaria_reclamada, 0)    AS qtd_perfumaria_reclamada,
  coalesce(i.valor_perfumaria_reclamada, 0)  AS valor_perfumaria_reclamada,
  i.produtos_reclamados,
  inv.resultado, inv.data_conclusao, inv.analista_id,
  p.pedido_numero IS NOT NULL                AS pedido_existe
FROM bronze_chamados c
LEFT JOIN itens i USING (chamado_id)
LEFT JOIN bronze_investigacoes inv USING (chamado_id)
LEFT JOIN silver_pedidos p ON p.pedido_numero = c.pedido_numero;

CREATE OR REFRESH MATERIALIZED VIEW silver_ligacoes
COMMENT 'Contact-center calls with the linked claim and the transcription source.'
TBLPROPERTIES ('quality' = 'silver')
AS
SELECT call_id, chamado_id, data_hora, duracao_s, origem, arquivo
FROM bronze_ligacoes;

-- Analyst decisions, written by the app to Lakebase and replicated to UC via
-- CDF (`${decisoes_lakebase}`, a change history with _pg_change_type).
-- First the latest version of each decision (deleted ones drop out), then the most
-- recent decision for each claim. Those with origin 'simulacao' are tests and don't
-- become labels.
CREATE OR REFRESH MATERIALIZED VIEW silver_decisoes_analista
COMMENT 'Current analyst decision per claim, from Lakebase.'
TBLPROPERTIES ('quality' = 'silver')
AS
WITH versao_final AS (
  SELECT *
  FROM ${decisoes_lakebase}
  QUALIFY row_number() OVER (PARTITION BY decisao_id ORDER BY _pg_lsn DESC, _sort_by DESC) = 1
)
SELECT chamado_id, decisao, analista, comentario, score_no_momento, modelo_versao, decidido_em
FROM versao_final
WHERE _pg_change_type <> 'delete' AND origem = 'app'
QUALIFY row_number() OVER (PARTITION BY chamado_id ORDER BY decidido_em DESC, decisao_id DESC) = 1;
