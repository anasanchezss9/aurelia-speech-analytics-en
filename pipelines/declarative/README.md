# Declarative pipeline (Lakeflow) — 100% SQL

A single journey, from file to model feature. The call becomes a transcription, the
transcription becomes structured analysis, and the analysis is joined with operational data —
order, delivery, consultant, investigation — in the gold tables that the fraud
model, the analyst queue and Genie read.

```
declarative/
├── pipeline.yml                            pipeline config (bundle snippet)
├── 01_bronze.sql                           audio, 10 operational tables and text transcriptions
├── 02_silver_transcript_ai_transcribe.sql  audio → text with ai_transcribe (default)
├── 02_silver_transcript.sql                audio → text with Whisper on Model Serving (alternative)
├── 02c_silver_transcricoes.sql             both transcription sources in one table, via two flows
├── 03_gold_insights.sql                    ai_query with JSON Schema → business fields per call
├── 04_silver_operacional.sql               typed operational data, with quality expectations
└── 05_gold_operacional.sql                 features per claim, queue, validation, per-cycle aggregates
```

```
Volume audio_landing ──► bronze_audio_files ──► silver_transcripts (ai_transcribe) ─┐
Volume operacional/transcricoes_texto ──► bronze_transcricoes_texto ───────────────┴► silver_transcricoes
                                                                                         │
                                                                                         ▼
                                                                                gold_call_insights (ai_query)
Volume operacional/* ──► bronze_* ──► silver_pedidos, silver_entregas,                   │
                                      silver_chamados, silver_consultoras ───────────────┤
                                                                                         ▼
                                       gold_features_chamado · gold_fila_investigacao · gold_ligacoes_validadas
                                       gold_chamados_ciclo · gold_pedidos_ciclo
```

## The decisions worth explaining

**Two transcription sources, one gold.** Some calls arrive as audio and
need to be transcribed; for others the telephony platform already exports text. The two flows in
`02c` write to the same `silver_transcricoes`, and gold reads only that table. A new source is one
more `CREATE FLOW`, without touching gold.

**Each call goes through the LLM once.** Gold is a `STREAMING TABLE` over
`STREAM silver_transcricoes`: `ai_query` runs only on what is new. Reprocessing everything
on every run would multiply the token cost by the number of runs.

**The box weight separates the causes.** `silver_entregas` computes the weight
divergence at the DC (dispatched lighter than the theoretical weight → picking error) and at the hub (weighed
less than at dispatch → loss in transit). The call says *that* something went missing; the
weight says *where*.

**Features only use what was known when the claim was opened.** In `gold_features_chamado`, the
consultant's history counts backward from the claim date, and a previous fraud
only counts if its investigation had already concluded. Without that the model learns
from the future and the test metric lies.

**The number dictated over the phone is checked, not trusted.** A repeated digit disappears in the
transcription ("793337" comes out as "7937"), and no prompt can recover it. `gold_ligacoes_validadas`
checks the extracted consultant and order against the registry and, when they don't match,
picks the consultant's order that is at most 2 digits off.

## Results on fevm-field-toolbox (05/11/2026)

1,229 calls (30 as audio, 1,199 as text), 9,884 claims, 210,909 orders.
Full run in ~10 min, serverless.

| Measure | Result |
|---|---|
| Ground truth for the 30 audio calls | word error rate 5.2%; `tipo_problema`, `suspeita_fraude` and products 100% |
| Correct order, from what the LLM extracted alone | 80% on audio · 86.6% on text |
| Correct order, after validation against the registry | **100%** on audio · **97.7%** on text |
| Quality expectations | 8, no rows dropped (clean synthetic data) |

What the LLM sees in the call, by the claim's real cause (1,199 text calls):

| Real cause | `suspeita_fraude` flagged |
|---|---|
| Theft in transit | 100% |
| Consultant fraud | 23% |
| Opportunistic fraud | 18% |
| Picking error, delay, lost shipment, other | 0% |

This is why the model exists: from the call alone, "the box was open and the perfume was gone" sounds
the same whether the carrier took it or the consultant did. The LLM gets right
that goods were diverted and has no way of knowing who diverted them. The call signal goes in as a
feature, next to the weight, the history and the route.

## How to deploy

`pipeline.yml` is a bundle snippet (`include: pipelines/declarative/pipeline.yml`
in `databricks.yml`). The parameters under `configuration` become `${var}` in the SQL files:
`catalog`, `schema`, `volume_landing`, `volume_operacional`, `whisper_endpoint`,
`llm_endpoint`.

Requires the **AI Transcribe** preview enabled in the workspace. A file overwritten with the
same name is not read again by `read_files`: when you regenerate the synthetic data, run with
**full refresh**.

## When the imperative pipeline (`../imperative/`) still makes sense

When the logic needs a Python library SQL doesn't have (audio
preprocessing, OCR, a local model), or fine-grained control over parallelism. It writes to
its own tables (`bronze_audio_files_imperativo`, `*_ait`), because a declarative
pipeline table doesn't accept writes from outside the pipeline.
