# Lakebase — the analyst's queue and the analyst's decisions

The lakehouse decides the order of the queue; the analyst decides each case. Lakebase is where
the two meet: the app reads the queue without waking up a warehouse and writes the
decision to a transactional table, and the decision flows back to the lakehouse as a label.

```
UC fila_priorizada ──synced table (SNAPSHOT)──► Postgres aurelia_speech_online.fila_priorizada ──► app reads
                                                                                                    │
UC aurelia_speech_investigacao.lb_decisoes_history ◄──CDF (continuous)── Postgres investigacao.decisoes ◄─ app writes
          │
          ▼
silver_decisoes_analista ──► gold_features_chamado.rotulo_fraude (origem_rotulo = 'analista') ──► retraining
```

| Piece | What it is |
|---|---|
| Project `aurelia-speech-ops` | Postgres 17, autoscaling, scale-to-zero, branch `production` |
| `investigacao.decisoes` | one row per decision, insert-only; the current one is the most recent per claim |
| `investigacao.decisao_vigente` | view with each claim's current decision |
| `aurelia_speech_online.fila_priorizada` | synced table of the scored queue; the daily job triggers the refresh |
| CDF `investigacao` | replicates the `investigacao` schema as Delta in `field_toolbox_catalog.aurelia_speech_investigacao` |

```bash
bash lakebase/criar_lakebase.sh field-toolbox   # idempotent
```

## Why each choice

**Insert-only decisions.** Changing your mind is a new row, not an `UPDATE`. The trail
of who decided what, and with which score on screen, is the audit a fraud case
calls for — and it's what lets you measure later whether the analyst agreed with the model.

**SNAPSHOT for the queue.** Scoring rewrites the whole queue on every run; an incremental
sync would have nothing to reuse. The job runs pipeline → scoring →
synced table refresh, in that order.

**CDF back, not a hand-made copy.** The analyst's decision is the label the model is missing
to learn the call signal: training has only 90 labeled claims
with a call. Lakebase CDF carries every Postgres change to Delta within
seconds, and the pipeline turns it into a label. `origem = 'simulacao'` never becomes a
label, so tests don't contaminate training.

## Result

- 564 queue claims in Postgres; reading the queue in ~160 ms measured from outside the
  workspace.
- A decision written to Postgres showed up in `lb_decisoes_history` within seconds and,
  after the pipeline, became `rotulo_fraude = 1` with `origem_rotulo = 'analista'` on a
  claim that was still open. Inserts and deletes arrive with `_pg_change_type`.

## Pitfalls

- **The database has two names.** `databricks_postgres` inside Postgres and
  `databricks-postgres` as an API resource. CDF needs the second one and answers
  "CdfConfig not found" with the first.
- **Without `REPLICA IDENTITY FULL`, CDF skips the table** (`CDF_STATE_SKIPPED`), and
  changing it afterward isn't enough: the configuration evaluates the tables when it's created.
  The CDF config had to be deleted and recreated. `schema.sql` already creates the table this way.
- **App role via the API, not via `CREATE ROLE`.** The app's service principal
  needs a role created with `databricks postgres create-role`; psql's `CREATE ROLE`
  creates a role with the same name that rejects the OAuth token with a wrong-password
  message.
