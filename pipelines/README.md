# Speech Analytics pipelines — Aurélia missing-product claims

Two complete pipelines, side by side, that ingest call audio from Aurélia
consultants, transcribe it with Whisper and extract business insights
with Claude — focused on the **missing-product claims** use case (delays, missing items,
possible delivery fraud).

```
pipelines/
├── .env.example            ← copy to .env and adjust catalog/schema
├── 00_setup_env.py         ← creates the volume + generates 3 synthetic audios via gTTS
│
├── imperative/             ← Pipeline 1: imperative Spark (PySpark + Auto Loader)
│   ├── 01_bronze_audio_ingestion.py
│   ├── 02_silver_transcription.py      ← Whisper on Model Serving
│   ├── 02b_silver_ai_transcribe.py     ← ai_transcribe, no endpoint
│   └── 03_gold_ai_query_analysis.py    ← widgets table_silver/table_gold
│
├── declarative/            ← Pipeline 2: Spark Declarative Pipelines (SDP), 100% SQL
│   ├── pipeline.yml        ← serverless pipeline config
│   ├── 01_bronze.sql
│   ├── 02_silver_transcript.sql                 ← Whisper
│   ├── 02_silver_transcript_ai_transcribe.sql   ← ai_transcribe (swap it in pipeline.yml)
│   └── 03_gold_insights.sql
│
└── comparacao/
    └── 04_comparacao_whisper_vs_ai_transcribe.py ← latency, quality and cost
```

## Two versions of the transcription

The silver layer exists with **Whisper** (its own GPU endpoint) and with **`ai_transcribe`**
(serverless AI Function, Beta). Both publish `call_id` and `full_text`, which is
everything the gold layer and the portal read.

In the imperative pipeline they coexist: `02b` writes to `TABLE_SILVER_AIT` and `03`, with
the `table_silver` and `table_gold` widgets, builds the matching gold table. In the
declarative pipeline it is one or the other, because both publish `silver_transcripts`.

`ai_transcribe` requires the *AI Transcribe* preview to be enabled by an admin. The
comparison results are in
[`../../docs/comparacao-whisper-vs-ai-transcribe.md`](../../docs/comparacao-whisper-vs-ai-transcribe.md).

## Both pipelines do the same thing

| Stage | What happens |
|---|---|
| **Bronze** | Auto Loader / `read_files` reads new audio files from the `/Volumes/.../audio_landing/` volume |
| **Silver** | Each audio file is transcribed through the Whisper endpoint. **Only new audio files** are processed |
| **Gold** | `ai_query` calls Claude and returns structured JSON with sentiment, intent, fraud suspicion, missing products, next action… |

Both run in **batch mode** (`trigger=availableNow` in the imperative pipeline, *triggered*
mode in SDP) — in line with the decision from the meeting with Aurélia.

## How do they differ — when to use each?

| | **Imperative (PySpark)** | **Declarative (SDP / Lakeflow)** |
|---|---|---|
| Language | Python + PySpark | Pure SQL (100%) |
| Lines of code | ~300 | ~120 |
| Orchestration | Chained multi-task Job | Single declarative pipeline |
| Maintained by | Data engineers | Engineers + SQL analysts |
| Idempotency | Manual (checkpoints + MERGE) | Automatic (SDP manages it) |
| CDC / new-only | `withColumn` + `MERGE INTO` | `STREAMING TABLE` does it on its own |
| Lineage | Manual in UC | Automatic and visual in the UI |
| Schedule and retry | Configure in the Job | Built into the pipeline |
| Observability | Spark UI logs | Pipeline UI with expectations, metrics, events |
| Recommended for | Complex logic, external integrations | Standard ELT, governance, multiple teams |

> For Aurélia, **SDP is the natural choice** when the team is already
> comfortable with SQL. It runs end to end with 1/3 of the code, automatic
> governance and ready-made pipeline dashboards. See the detailed advantages
> in `declarative/README.md`.

## How to run — first time

### 1. Configure the environment

```bash
cd pipelines
cp .env.example .env
# edit .env if you want to change catalog/schema
```

### 2. Run the setup (once)

Run the `00_setup_env.py` notebook on a cluster or serverless compute. It:
- Creates the `/Volumes/{catalog}/{schema}/audio_landing/` volume
- Generates 3 synthetic PT-BR audio files via gTTS:
  1. Delay complaint (regular consultant)
  2. Fraud suspicion (an expensive product went missing, suspected theft)
  3. Partial shortage of items (picking problem)
- Uploads the `.mp3` files to the volume

### 3. Whisper endpoint

The `aurelia-whisper-turbo` endpoint is already running in the workspace. If it
isn't, first run the `notebooks/deploy_whisper_job.py` notebook from the
project root.

### 4. Pick a pipeline

- **Imperative**: run the 3 notebooks in sequence or create a Job in
  `imperative/` with 3 chained tasks.
- **Declarative (SDP)**: import the SQL files in `declarative/` and create a
  serverless pipeline pointing to them.

## `.env` files

The Python scripts read `.env` automatically via `python-dotenv` (already
available in every Databricks runtime). The SDP SQL files use
[pipeline parameters](declarative/pipeline.yml).
