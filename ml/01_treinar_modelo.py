# Databricks notebook source
# MAGIC %md
# MAGIC # Fraud model for missing-product claims
# MAGIC
# MAGIC Predicts the probability that a claim is confirmed fraud, from
# MAGIC `gold_features_chamado`. Trains only on investigated claims — the only
# MAGIC label that exists — and tests on the most recent ones, to measure what the model would do
# MAGIC with claims it hasn't seen yet.
# MAGIC
# MAGIC Three candidates, on the same test set:
# MAGIC
# MAGIC | Candidate | What it uses |
# MAGIC |---|---|
# MAGIC | `llm_sozinho` | the `fraude_confianca` the LLM gave the call, no model |
# MAGIC | `operacional` | claim, delivery, weight, history and route, without the call |
# MAGIC | `completo` | the operational features plus the call signal |
# MAGIC
# MAGIC The full model only becomes champion if its PR-AUC gain over the operational one is real:
# MAGIC the 95% interval of a paired bootstrap entirely above zero. Depending on the
# MAGIC call has a cost (the gold layer has to be up to date), and noise doesn't pay that cost.
# MAGIC The champion is registered in Unity Catalog with the alias `campeao`. The output is a
# MAGIC calibrated probability: the analyst reads "0.7" as seven in ten, and the number the
# MAGIC LLM self-reports doesn't support that reading.

# COMMAND ----------

# MAGIC %pip install -q "mlflow[databricks]>=3.1" "scikit-learn==1.6.1" "matplotlib>=3.8"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "field_toolbox_catalog")
dbutils.widgets.text("schema", "aurelia_speech")
dbutils.widgets.text("experimento", "/Users/ana.sanchez@databricks.com/aurelia_fraude_chamado")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
NOME_MODELO = f"{CATALOG}.{SCHEMA}.modelo_fraude_chamado"
K = 50  # claims the investigation team can open per cycle
REAMOSTRAS = 1000  # paired bootstrap of the full model's PR-AUC gain over the operational one

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(dbutils.widgets.get("experimento"))

# COMMAND ----------

# DBTITLE 1,Features and label
CATEGORICAS = ["motivo_declarado", "canal_chamado", "cd", "transportadora", "uf", "regiao",
               "status_entrega", "nivel"]
NUMERICAS = ["valor_reclamado", "qtd_itens_reclamados", "qtd_perfumaria_reclamada",
             "share_valor_perfumaria", "share_pedido_reclamado", "comprovante_assinado", "dias_atraso",
             "div_peso_cd", "div_peso_hub", "pesado_no_hub", "dias_entrega_abertura",
             "rota_chamados_100_pedidos", "tem_app", "meses_casa", "chamados_180d", "pedidos_180d",
             "chamados_por_pedido_180d", "fraudes_confirmadas_antes"]
CATEGORICAS_LIGACAO = ["llm_tipo_problema", "llm_sentiment"]
NUMERICAS_LIGACAO = ["tem_ligacao", "llm_suspeita_fraude", "llm_fraude_confianca",
                     "llm_n_sinais_relato", "llm_relato_consistente"]

df = spark.table(f"{CATALOG}.{SCHEMA}.gold_features_chamado").toPandas()
for c in NUMERICAS + NUMERICAS_LIGACAO:
    df[c] = pd.to_numeric(df[c].astype("float64"), errors="coerce")
for c in CATEGORICAS + CATEGORICAS_LIGACAO:
    df[c] = df[c].astype("string").fillna("ausente")

# Split by date, not by position: with an unstable sort, same-day claims at the
# boundary switched sides depending on the order in which Spark returns the rows,
# and the PR-AUC swung ~0.02 between runs on the same data.
rotulado = (df[df["rotulo_fraude"].notna()]
            .sort_values(["data_abertura", "chamado_id"], kind="mergesort").reset_index(drop=True))
data_corte = rotulado["data_abertura"].iloc[int(len(rotulado) * 0.75)]
treino = rotulado[rotulado["data_abertura"] < data_corte]
teste = rotulado[rotulado["data_abertura"] >= data_corte]
y_treino, y_teste = treino["rotulo_fraude"].astype(int), teste["rotulo_fraude"].astype(int)
print(f"labeled: {len(rotulado)} · train {len(treino)} through {treino['data_abertura'].max()} "
      f"· test {len(teste)} since {data_corte} · test prevalence {y_teste.mean():.1%}")

# COMMAND ----------

# DBTITLE 1,Queue metrics
def metricas(y, p, prefixo=""):
    ordem = np.argsort(-p)
    topo = ordem[:K]
    return {
        f"{prefixo}pr_auc": average_precision_score(y, p),
        f"{prefixo}roc_auc": roc_auc_score(y, p),
        f"{prefixo}precisao_top{K}": float(np.asarray(y)[topo].mean()),
        f"{prefixo}recall_top{K}": float(np.asarray(y)[topo].sum() / max(1, np.asarray(y).sum())),
        f"{prefixo}brier": brier_score_loss(y, np.clip(p, 0, 1)),
    }


def montar(categoricas, numericas):
    previsor = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categoricas),
         ("num", "passthrough", numericas)])
    gbm = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0,
        categorical_features=list(range(len(categoricas))), early_stopping=True,
        validation_fraction=0.15, random_state=42)
    base = Pipeline([("prep", previsor), ("gbm", gbm)])
    # Isotonic with cross-validation on the training set: the ranking already comes from the GBM,
    # calibration only fixes the scale so the score becomes a probability.
    return CalibratedClassifierCV(base, method="isotonic", cv=5)


from modelo_fraude import ModeloFraude  # module next to the notebook; ships along via code_paths

# COMMAND ----------

# DBTITLE 1,Candidate 1 — the LLM alone
com_ligacao = teste["tem_ligacao"] == 1
with mlflow.start_run(run_name="llm_sozinho"):
    p_llm = teste["llm_fraude_confianca"].fillna(0).to_numpy()
    m = metricas(y_teste, p_llm)
    m.update(metricas(y_teste[com_ligacao], p_llm[com_ligacao.to_numpy()], "com_ligacao_"))
    mlflow.log_metrics(m)
    resultados = {"llm_sozinho": m}
print(pd.Series(resultados["llm_sozinho"]).round(3))

# COMMAND ----------

# DBTITLE 1,Candidates 2 and 3 — operational and full
modelos = {}
for nome, cats, nums in [("operacional", CATEGORICAS, NUMERICAS),
                         ("completo", CATEGORICAS + CATEGORICAS_LIGACAO, NUMERICAS + NUMERICAS_LIGACAO)]:
    with mlflow.start_run(run_name=nome) as run:
        modelo = montar(cats, nums).fit(treino[cats + nums], y_treino)
        p = modelo.predict_proba(teste[cats + nums])[:, 1]
        m = metricas(y_teste, p)
        m.update(metricas(y_teste[com_ligacao], p[com_ligacao.to_numpy()], "com_ligacao_"))
        mlflow.log_params({"features": len(cats + nums), "treino": len(treino), "teste": len(teste),
                           "corte_teste": str(data_corte), "calibracao": "isotonica_cv5", "k_fila": K})
        mlflow.log_metrics(m)
        resultados[nome] = m
        modelos[nome] = (modelo, cats, nums, p, run.info.run_id)

comparacao = pd.DataFrame(resultados).T
display(comparacao.round(3).reset_index().rename(columns={"index": "candidato"}))

# COMMAND ----------

# DBTITLE 1,Champion curves and importance
# The full model depends on the call gold layer being up to date; it's only worth it if the gain is
# real. Paired bootstrap: the same test resamples for both models,
# and the full model only wins if the 95% interval of the gain stays above zero.
sorteio = np.random.default_rng(0)
yv = y_teste.to_numpy()
p_op, p_co = modelos["operacional"][3], modelos["completo"][3]
ganhos = []
for _ in range(REAMOSTRAS):
    i = sorteio.integers(0, len(yv), len(yv))
    if yv[i].sum() == 0:
        continue
    ganhos.append(average_precision_score(yv[i], p_co[i]) - average_precision_score(yv[i], p_op[i]))
ganho = resultados["completo"]["pr_auc"] - resultados["operacional"]["pr_auc"]
ic_inf, ic_sup = np.percentile(ganhos, [2.5, 97.5])
campeao = "completo" if ic_inf > 0 else "operacional"
with mlflow.start_run(run_id=modelos["completo"][4]):
    mlflow.log_metrics({"ganho_pr_auc": ganho, "ganho_ic95_inf": ic_inf, "ganho_ic95_sup": ic_sup})
print(f"PR-AUC gain from the call: {ganho:+.3f} (95% CI {ic_inf:+.3f} to {ic_sup:+.3f}) → champion: {campeao}")
modelo, cats, nums, p_camp, run_id = modelos[campeao]

fig, eixos = plt.subplots(1, 3, figsize=(18, 5))
for nome, p in [("llm_sozinho", p_llm), ("operacional", modelos["operacional"][3]), ("completo", modelos["completo"][3])]:
    prec, rec, _ = precision_recall_curve(y_teste, p)
    eixos[0].plot(rec, prec, label=f"{nome} (PR-AUC {resultados[nome]['pr_auc']:.2f})")
eixos[0].axhline(y_teste.mean(), ls="--", c="gray", label="random")
eixos[0].set(xlabel="recall", ylabel="precision", title="Precision × recall on the test set"); eixos[0].legend()

frac, media = calibration_curve(y_teste, p_camp, n_bins=8, strategy="quantile")
eixos[1].plot(media, frac, "o-", label=campeao)
frac_llm, media_llm = calibration_curve(y_teste, np.clip(p_llm, 0, 1), n_bins=8, strategy="quantile")
eixos[1].plot(media_llm, frac_llm, "s--", label="llm_sozinho")
eixos[1].plot([0, 1], [0, 1], c="gray", ls=":")
eixos[1].set(xlabel="predicted probability", ylabel="observed fraud", title="Calibration"); eixos[1].legend()

imp = permutation_importance(modelo, teste[cats + nums], y_teste, scoring="average_precision",
                             n_repeats=5, random_state=0)
importancia = pd.Series(imp.importances_mean, index=cats + nums).sort_values().tail(12)
importancia.plot.barh(ax=eixos[2], title="Permutation importance (PR-AUC drop)")
plt.tight_layout()

with mlflow.start_run(run_id=run_id):
    mlflow.log_figure(fig, "avaliacao.png")
    mlflow.log_text(importancia.sort_values(ascending=False).to_string(), "importancia.txt")
display(fig)

# COMMAND ----------

# DBTITLE 1,Registers the champion in Unity Catalog
from mlflow.models import infer_signature

exemplo = teste[cats + nums].head(5)
wrapper = ModeloFraude(modelo, cats, nums)
assinatura = infer_signature(exemplo, wrapper.predict(None, exemplo))
with mlflow.start_run(run_id=run_id):
    info = mlflow.pyfunc.log_model(
        name="modelo",
        python_model=wrapper,
        signature=assinatura,
        input_example=exemplo,
        registered_model_name=NOME_MODELO,
        pip_requirements=["mlflow>=3.1", "scikit-learn==1.6.1", "pandas", "numpy"],
        code_paths=["modelo_fraude.py"],
    )
versao = info.registered_model_version
cliente = mlflow.MlflowClient()
cliente.set_registered_model_alias(NOME_MODELO, "campeao", versao)
cliente.update_model_version(NOME_MODELO, versao, description=(
    f"Candidate '{campeao}'. Temporal test since {data_corte}: PR-AUC {resultados[campeao]['pr_auc']:.3f}, "
    f"gain of the full model over the operational one {ganho:+.3f} (95% CI {ic_inf:+.3f} to {ic_sup:+.3f}), "
    f"precision in the top {K} {resultados[campeao][f'precisao_top{K}']:.0%}. "
    "Output: calibrated (isotonic) probability of confirmed fraud."))
print(f"{NOME_MODELO} v{versao} @campeao ({campeao})")
