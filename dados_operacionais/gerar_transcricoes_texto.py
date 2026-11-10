# Databricks notebook source
# MAGIC %md
# MAGIC # Text transcripts of the synthetic calls
# MAGIC
# MAGIC Of the 1,230 calls in the operational world, 30 have audio (`sintetico/`). The
# MAGIC other 1,200 arrive as text, as if the telephony platform already exported
# MAGIC the transcript — that is what gives the gold layer volume to analyze and
# MAGIC gives the model the call signal on more than 30 claims.
# MAGIC
# MAGIC Each transcript is written by an LLM from the facts of the claim (motive,
# MAGIC items, delivery, carrier) and the real cause, which only the generator knows.
# MAGIC Fraud does not confess: in some calls the consultant lets subtle signs slip,
# MAGIC in the others she is convincing. That way the gold layer adds a partial
# MAGIC signal, not a leaked label.
# MAGIC
# MAGIC The model that writes (Haiku) is different from the one that analyzes in the
# MAGIC gold layer (Sonnet), so the same model does not judge its own text.

# COMMAND ----------

dbutils.widgets.text("catalog", "field_toolbox_catalog")
dbutils.widgets.text("schema", "aurelia_speech")
dbutils.widgets.text("modelo", "databricks-claude-haiku-4-5")
dbutils.widgets.text("limite", "0")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODELO = dbutils.widgets.get("modelo")
LIMITE = int(dbutils.widgets.get("limite") or 0)
OPER = f"/Volumes/{CATALOG}/{SCHEMA}/operacional"
AVAL = f"/Volumes/{CATALOG}/{SCHEMA}/avaliacao"
DESTINO = f"{OPER}/transcricoes_texto"

for nome, pasta in [("lig", "ligacoes"), ("ch", "chamados"), ("ped", "pedidos"), ("ent", "entregas"),
                    ("cons", "consultoras"), ("ci", "chamado_itens"), ("prod", "produtos")]:
    spark.read.parquet(f"{OPER}/{pasta}").createOrReplaceTempView(nome)
spark.read.parquet(f"{AVAL}/gabarito_chamados").createOrReplaceTempView("gab")

# COMMAND ----------

# DBTITLE 1,Facts of each call
contexto = spark.sql(f"""
WITH itens AS (
  SELECT ci.chamado_id,
         array_join(collect_list(concat(CAST(ci.quantidade AS STRING), ' ', p.nome)), ', ') AS itens
  FROM ci JOIN prod p USING (sku)
  GROUP BY ci.chamado_id
)
SELECT
  l.call_id, l.chamado_id, l.data_hora, l.duracao_s,
  ch.consultora_codigo, ch.pedido_numero, ch.motivo_declarado, ch.valor_reclamado,
  coalesce(i.itens, '') AS itens,
  ped.transportadora, ped.uf, ent.status_entrega, ent.data_prevista, ent.data_entrega,
  cons.nivel, cons.data_cadastro,
  g.mecanismo,
  pmod(hash(l.call_id), 10) < 5 AS fraude_convincente
FROM lig l
JOIN ch   ON ch.chamado_id = l.chamado_id
JOIN ped  ON ped.pedido_numero = ch.pedido_numero
JOIN ent  ON ent.pedido_numero = ch.pedido_numero
JOIN cons ON cons.consultora_codigo = ch.consultora_codigo
JOIN gab g ON g.chamado_id = l.chamado_id
LEFT JOIN itens i ON i.chamado_id = l.chamado_id
WHERE l.origem = 'roteiro_texto'
ORDER BY l.call_id
{f"LIMIT {LIMITE}" if LIMITE else ""}
""")
contexto.createOrReplaceTempView("contexto")
display(contexto.groupBy("mecanismo").count())

# COMMAND ----------

# DBTITLE 1,The LLM writes each call
# Stays in Portuguese: the calls are in Portuguese and the measured results depend on its exact wording.
COMPORTAMENTO = """
CASE mecanismo
  WHEN 'erro_separacao' THEN 'A caixa chegou lacrada e íntegra, mas faltam os itens. A consultora é coerente, sabe exatamente o que faltou, oferece foto e nota fiscal.'
  WHEN 'furto_transporte' THEN 'A caixa chegou com a fita rasgada ou refeita e faltam os itens. A consultora é coerente, descreve a embalagem com detalhe e aceita mandar foto.'
  WHEN 'extravio' THEN 'O pedido não chegou. O rastreio parou há dias. A consultora quer saber o que aconteceu e quando recebe.'
  WHEN 'atraso' THEN 'O pedido está atrasado. A consultora quer a nova previsão de entrega.'
  WHEN 'outro' THEN 'O assunto é um produto avariado com a caixa intacta, uma dúvida de cobrança ou de cadastro.'
  WHEN 'fraude' THEN IF(fraude_convincente,
    'A consultora está mentindo para ser ressarcida, mas é convincente: história coerente, tom calmo, nenhum sinal óbvio. Ela nunca admite nada.',
    'A consultora está mentindo para ser ressarcida e nunca admite. Deixe sinais sutis, sem exagero: detalhes vagos ou que mudam durante a conversa, pressa pelo crédito, resistência a mandar foto, insistência em que o entregador não deixou conferir, menção a outras vezes em que "deu azar".')
  WHEN 'fraude_oportunista' THEN 'A consultora exagera a reclamação para ganhar um item a mais. Ela é quase convincente; no máximo hesita quando o agente pede detalhe.'
END
"""

# The prompt stays in Portuguese: the calls are in Portuguese and the measured results depend on its exact wording.
geradas = spark.sql(f"""
SELECT
  call_id, chamado_id, data_hora, duracao_s,
  ai_query(
    '{MODELO}',
    concat(
      'Escreva a transcrição de uma ligação para a Central de Relacionamento da Aurélia ',
      'Cosméticos, empresa de venda direta. Quem liga é uma consultora (ou consultor) ',
      'que revende os produtos; quem atende é um agente.\\n\\n',
      'Fatos do caso:\\n',
      '- código da consultora: ', consultora_codigo, ' (ela dita o código)\\n',
      '- número do pedido: ', pedido_numero, ' (ela dita o número)\\n',
      '- motivo do chamado: ', motivo_declarado, '\\n',
      '- itens reclamados: ', if(itens = '', 'nenhum item específico', itens), '\\n',
      '- transportadora: ', transportadora, ' · estado: ', uf, '\\n',
      '- situação da entrega no sistema: ', status_entrega,
      ', prevista para ', CAST(data_prevista AS STRING),
      coalesce(concat(', entregue em ', CAST(data_entrega AS STRING)), ''), '\\n',
      '- nível da consultora: ', nivel, ', cadastrada desde ', CAST(data_cadastro AS STRING), '\\n\\n',
      'Como a ligação acontece: ', {COMPORTAMENTO}, '\\n\\n',
      'Regras: português do Brasil coloquial, de 2 a 4 minutos de conversa (de 25 a 45 ',
      'falas). Invente os nomes. Escreva como sai de um sistema de transcrição ',
      'automática: texto corrido, sem rótulo de quem fala, sem marcação de cena, ',
      'números ditos por extenso ou em dígitos como a pessoa falaria. O agente segue ',
      'o procedimento: confirma código e pedido, registra o protocolo e diz o próximo passo.'
    ),
    responseFormat => '{{"type":"json_schema","json_schema":{{"name":"transcricao","schema":{{"type":"object","properties":{{"transcricao":{{"type":"string"}}}},"required":["transcricao"]}},"strict":true}}}}',
    modelParameters => named_struct('temperature', 0.9, 'max_tokens', 3000),
    failOnError => false
  ) AS r
FROM contexto
""")

# Serverless does not accept .cache(): materialize once into a staging table to
# count and filter without paying for the LLM again on every action.
STAGING = f"{CATALOG}.{SCHEMA}._staging_transcricoes_texto"
(geradas
    .selectExpr("call_id", "chamado_id", "data_hora", "duracao_s",
                "r.result:transcricao::STRING AS full_text",
                "r.errorMessage AS erro",
                f"'{MODELO}' AS modelo_gerador",
                "current_timestamp() AS gerado_em")
    .write.mode("overwrite").saveAsTable(STAGING))

resultado = spark.table(STAGING)
ok = resultado.where("full_text IS NOT NULL AND length(full_text) > 200")
print(f"generated: {ok.count()} of {resultado.count()}")
display(resultado.where("full_text IS NULL OR length(full_text) <= 200").select("call_id", "erro"))

# COMMAND ----------

# DBTITLE 1,Write to the Volume, where the pipeline reads
(ok.drop("erro").coalesce(1)
   .write.mode("overwrite").parquet(DESTINO))
spark.sql(f"DROP TABLE {STAGING}")
print(DESTINO)
display(spark.read.parquet(DESTINO).selectExpr("call_id", "left(full_text, 300) AS inicio").limit(5))
