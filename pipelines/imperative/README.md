# Imperative pipeline (PySpark)

A 3-stage pipeline, each stage in its own notebook, that can be chained in a
multi-task Job.

| Stage | Notebook | Output table | Dominant cost |
|---|---|---|---|
| Bronze | `01_bronze_audio_ingestion.py` | `bronze_audio_files_imperativo` | Volume I/O |
| Silver | `02_silver_transcription.py` | `silver_transcripts` | Whisper endpoint GPU |
| Gold | `03_gold_ai_query_analysis.py` | `gold_call_insights` | Claude tokens |

## How it runs in batch (even though it uses Spark Streaming)

The Bronze stage uses **Auto Loader** with `trigger(availableNow=True)` —
which means: "read everything not read yet, write it, and stop".
That is exactly the behavior of an idempotent batch job, but with the
streaming infrastructure guaranteeing that **only new files**
are processed.

The Silver stage uses `LEFT ANTI JOIN` + `MERGE INTO` — classic incremental
batch semantics.

The Gold stage also uses `LEFT ANTI JOIN` + `INSERT INTO` before
`ai_query`, so the LLM **is only called for new transcriptions**.

## How to schedule it as a Job

```yaml
# excerpt from databricks.yml
resources:
  jobs:
    aurelia_speech_pipeline:
      name: aurelia-speech-imperative
      schedule:
        quartz_cron_expression: "0 0 */4 * * ?"  # every 4 hours
        timezone_id: America/Sao_Paulo
      tasks:
        - task_key: bronze
          notebook_task:
            notebook_path: ./01_bronze_audio_ingestion.py
        - task_key: silver
          depends_on: [{ task_key: bronze }]
          notebook_task:
            notebook_path: ./02_silver_transcription.py
        - task_key: gold
          depends_on: [{ task_key: silver }]
          notebook_task:
            notebook_path: ./03_gold_ai_query_analysis.py
```

## Strengths of the imperative approach

- **Flexibility**: you control every step and can inject custom
  logic (e.g. skip audio > X minutes, retry with a specific policy).
- **Familiar**: a team that already knows PySpark can dive right in.
- **Rich Python integrations**: you can use libraries that don't exist
  in SQL (e.g. splitting audio with `pydub` before transcription).

## Weaknesses vs SDP

- **More code**: ~300 lines vs ~120 for SDP.
- **You manage idempotency**: checkpoints, MERGE, ANTI JOIN are on you.
- **Manual lineage**: you need to set up tags / tables for UC to see it.
- **No pipeline dashboard**: you get the Spark UI, but no per-stage
  views, expectations, or consolidated data quality.
