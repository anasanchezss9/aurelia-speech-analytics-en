# Fraud model for missing-product claims

Predicts the probability that a missing-product claim is confirmed fraud, so the
investigation team opens the cases most worth opening first. Reads
`gold_features_chamado` (Lakeflow pipeline) and returns a calibrated probability.

| File | What it does |
|---|---|
| `01_treinar_modelo.py` | trains three candidates, compares them on the same test set, registers the champion in UC |
| `modelo_fraude.py` | the pyfunc wrapper that returns the probability; ships with the model via `code_paths` |
| `02_pontuar_chamados.py` | scores every claim and writes the prioritized queue |

```
gold_features_chamado ──► 01_treinar_modelo ──► field_toolbox_catalog.aurelia_speech.modelo_fraude_chamado @campeao
                                                      │
                     ┌────────────────────────────────┼─────────────────────────────┐
                     ▼                                ▼                             ▼
         02_pontuar_chamados (batch)      endpoint aurelia-fraude-chamado     (retraining with the
         score_fraude_chamado             (on-demand score in the app)         analyst's decisions)
         fila_priorizada
```

The `aurelia-speech-diario` job runs the pipeline, the scoring and the queue refresh in
Lakebase (scheduled for 6 a.m., paused). Training runs separately: retraining every day with no new labels just swaps
one model for an identical one.

## How the model is trained

- **Only on what has a label.** A label exists where there was an investigation (~20% of
  closed claims). The policy investigates expensive and repeat cases more, so the
  model learns the fraud the investigation usually finds — a bias stated in the
  generator's README.
- **Labels from two sources.** The formal investigation and, where there was none, the
  analyst's decision in the app, which comes back from Lakebase through CDF (`origem_rotulo`).
- **Temporal test, split by date.** Claims investigated from
  June 9, 2026 onward are held out. A random split would mix future and past and give an
  optimistic metric.
- **Gradient boosting with isotonic calibration.** The GBM ranks; calibration
  fixes the scale so that "0.7" means seven in ten. The `fraude_confianca` the LLM
  reports does not have that property.
- **Three candidates on the same test set**, and the full model only becomes champion if its
  PR-AUC gain over the operational one is real: the 95% interval of a paired bootstrap
  (1,000 resamples of the test set) entirely above zero. Depending on the call has a cost,
  and noise doesn't pay for it.

## Result (temporal test: 512 claims, 14% fraud)

| Candidate | PR-AUC | Precision in the top 50 | Brier |
|---|---|---|---|
| LLM alone (`fraude_confianca`) | 0.18 | 30% | 0.157 |
| Operational | **0.83** | **88%** | 0.044 |
| Full (operational + call) | 0.83 | 84% | 0.044 |

Gain of the full model over the operational one: −0.001 PR-AUC, 95% CI from −0.022 to
+0.021. Champion: **operational**, version 5, served at `aurelia-fraude-chamado`.

On the open queue (564 claims, 10% fraud according to the generator's answer key), in score
order:

| Top | Actual fraud | Frauds found out of 58 | Fraud value up front |
|---|---|---|---|
| 20 | 100% | 20 | R$ 11.4k |
| 50 | 96% | 48 | R$ 21.7k |
| 100 | 58% | 58 | all R$ 23.2k |

## Why the call was left out of the score

The first version of the gold layer extracted `suspeita_fraude`, which answers "was there diversion?".
It flags 95% of the in-transit theft calls, which the box weight at the hub already
explains, and only 23% of consultant fraud. As a feature, it added nothing.

The second version extracts `sinais_relato`: the consultant's behavior in her account,
regardless of who diverted the goods (vague or contradictory account, rushing for the credit,
reluctance to send a photo, "I couldn't check it on delivery"). That signal separates:

| Labeled claims with a call | Claims | Fraud |
|---|---|---|
| No signal in the account | 217 | 10% |
| One signal | 16 | 38% |
| Two or more | 19 | **89%** |

Even so, the full model didn't win, and the reason is coverage: transcription started
in the last cycles, and training has **90 labeled claims with a call, 16 of them
fraud**. That is too few for the GBM to learn the signal's weight. The decision was to keep the
operational score and show the signals in the consultant's account in the queue, as evidence for the
analyst. Each of the analyst's decisions on a claim with a call is a missing label;
with a few hundred, the full model should overtake the operational one.

## Pitfalls

- **Splitting by position made the metric swing.** Sorting by date with an unstable
  sort and cutting at 75% sent same-day claims at the boundary to
  training or to test depending on the order in which Spark returned the rows. The
  PR-AUC varied ~0.02 between runs on the same data — more than the difference
  between the candidates. One run even elected the full model with 0.840 against
  0.812. The split is now by date, with ties broken by id, and two consecutive runs
  give the same number.
- **A class defined in the notebook doesn't cross Python versions.** cloudpickle
  stores the wrapper code by value; saved on serverless with 3.10, the model
  wouldn't load in the job with 3.12 (`code() argument 13 must be str, not int`). The
  wrapper lives in `modelo_fraude.py` and ships via `code_paths`, stored by reference.

- **Decimals and booleans with nulls arrive from Spark as `object`.** The wrapper converts
  with `pd.to_numeric`, but a boolean with `None` needs `astype("float64")` first.
  Scoring does this with the column list the model itself stores.
- **`display` of a DataFrame with a mixed column fails on the Arrow conversion** — numbers
  and text in the same column. It's the last cell of scoring; the tables were already
  written, but the job exited with an error.
- **The endpoint is scale-to-zero.** The first call after it has been idle takes
  a few extra seconds. Check the billing, not just the configuration.
