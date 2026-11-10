# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — Speech Analytics for missing-product claims (Aurélia)
# MAGIC
# MAGIC This notebook prepares the environment for the 2 pipelines (imperative + SDP).
# MAGIC Run it once.
# MAGIC
# MAGIC **What it does:**
# MAGIC 1. Reads configs from `.env`
# MAGIC 2. Creates the `audio_landing` volume in Unity Catalog
# MAGIC 3. Generates 3 synthetic PT-BR audio files via Google Text-to-Speech (gTTS):
# MAGIC    - **call_001_atraso.mp3** — consultant complains about a delay, routine
# MAGIC    - **call_002_suspeita_fraude.mp3** — box arrived open with a product missing
# MAGIC    - **call_003_falta_parcial.mp3** — order arrived with missing items
# MAGIC 4. Uploads the `.mp3` files to the volume
# MAGIC
# MAGIC > **Why gTTS?** It needs no API key, produces realistic PT-BR MP3, and
# MAGIC > MP3 can be decoded by Whisper (libsndfile + audioread).

# COMMAND ----------

# MAGIC %pip install -q gTTS==2.5.4 python-dotenv==1.0.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load settings from .env (with fallback)
import os
from pathlib import Path
from dotenv import load_dotenv

# Look for .env in the notebook's directory
here = Path.cwd()
candidates = [
    here / ".env",
    here.parent / ".env",
    Path("/Workspace/Users/ana.sanchez@databricks.com/aurelia_speech_pipelines/.env"),
]
for candidate in candidates:
    if candidate.exists():
        load_dotenv(candidate)
        print(f"Loaded config from: {candidate}")
        break
else:
    print("No .env found; using defaults.")

CATALOG = os.getenv("CATALOG", "field_toolbox_catalog")
SCHEMA = os.getenv("SCHEMA", "aurelia_speech")
VOLUME = os.getenv("VOLUME_LANDING", "audio_landing")

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
print(f"Catalog: {CATALOG}")
print(f"Schema:  {SCHEMA}")
print(f"Volume:  {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Create schema and volume if they don't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"OK — volume available at {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Realistic scripts (3 scenarios)
# Each call is between a CONSULTANT (Aurélia customer) and a customer-service AGENT.
# The scripts were written to sound like real calls, with natural
# speech, pauses and everyday terms.
# The scripts stay in Portuguese because the calls are in Portuguese and the
# measured results depend on their exact wording.

CALLS = [
    {
        "filename": "call_001_atraso.mp3",
        "scenario": "Reclamação de atraso simples",
        "script": (
            "Olá, tudo bem? Aqui é a Maria Aparecida, sou consultora Aurélia "
            "lá de Belo Horizonte, código de consultora oito quatro sete dois "
            "três. Estou ligando porque meu pedido número nove zero dois um "
            "está atrasado já tem quase dez dias e minhas clientes estão "
            "me cobrando. "
            "Tudo bem, dona Maria. Lamento o ocorrido. Deixa eu verificar "
            "aqui no sistema. Aproveitando, a senhora não gostaria de "
            "conhecer a nossa promoção da linha Raízes? "
            "Não, eu realmente preciso resolver esse atraso. Eu não posso "
            "ficar com as clientes esperando, isso prejudica meu trabalho. "
            "Por favor, me diga só quando vai chegar. "
            "Entendo. Vou abrir um protocolo de urgência aqui pra senhora. "
            "Em até quarenta e oito horas a transportadora retorna com um "
            "prazo concreto. Posso confirmar o seu telefone para contato? "
            "Pode confirmar sim, é o mesmo cadastrado. Obrigada."
        ),
    },
    {
        "filename": "call_002_suspeita_fraude.mp3",
        "scenario": "Caixa chegou aberta com produtos faltando — possível roubo no transporte",
        "script": (
            "Alô, boa tarde. Aqui é o José Carlos, consultor Aurélia, "
            "código sete dois nove cinco um. Estou bem chateado, viu. "
            "Recebi minha caixa hoje pela manhã e ela veio com a fita "
            "violada, parecia que tinha sido aberta. Quando eu fui "
            "conferir, faltavam dois perfumes Âmbar Exclusivo "
            "masculino e três cremes Diário. Justo os produtos mais "
            "caros do pedido, isso aqui foi roubo na certeza. "
            "Lamento muito o ocorrido, senhor José. O senhor consegue "
            "tirar fotos da caixa e da nota fiscal? "
            "Já tirei, mandei pelo aplicativo. Mas olha, isso não é a "
            "primeira vez que acontece comigo, no mês passado também "
            "veio faltando uma coisinha. Eu acho que tem alguém na "
            "transportadora mexendo nas caixas. Eu vou ter que devolver "
            "o dinheiro pras minhas clientes do meu bolso, isso é um "
            "prejuízo enorme. "
            "Vou registrar como ocorrência de violação de embalagem e "
            "encaminhar pra área de fraude e segurança. O senhor recebe "
            "um retorno em até cinco dias úteis, com o ressarcimento ou "
            "o reenvio dos produtos. "
            "Cinco dias úteis é muito tempo, viu. Mas tudo bem, vou "
            "esperar. Mas por favor, vê com a transportadora, isso "
            "está demais."
        ),
    },
    {
        "filename": "call_003_falta_parcial.mp3",
        "scenario": "Pedido com falta de itens — possível erro de separação",
        "script": (
            "Oi, boa tarde. Sou a Cláudia, consultora aqui de Recife, "
            "meu código é um cinco quatro sete dois oito. Tô ligando "
            "porque meu pedido chegou hoje, mas faltou bastante coisa. "
            "Veio um batom Aura a menos, faltou um sabonete líquido "
            "Erva Doce, e veio só um esmalte quando eu pedi quatro. "
            "Entendi, dona Cláudia. A caixa estava lacrada quando "
            "chegou? "
            "Estava sim, a fita tava normal, parece que eles esqueceram "
            "de colocar mesmo lá no centro de distribuição. Eu confiro "
            "tudinho com a nota e tá faltando exatamente esses itens. "
            "Provavelmente foi erro de separação na expedição. Vou "
            "abrir aqui um chamado de complemento de pedido. Os produtos "
            "vão ser enviados sem custo no próximo ciclo. "
            "Tudo bem. Mas pode olhar isso direitinho lá, porque a "
            "última vez também veio faltando. Tá ficando frequente "
            "esses errinhos."
        ),
    },
]

print(f"{len(CALLS)} synthetic audio files will be generated.")

# COMMAND ----------

# DBTITLE 1,Generate the MP3s and upload them to the volume
# On serverless compute (Shared UC mode) we CANNOT use `dbutils.fs.cp`
# to copy from `file:/tmp/...`. The fix is to write the MP3s straight to the
# Volume through Python's filesystem — /Volumes/... paths are treated as the
# local filesystem by the runtime and work on serverless.

import os
import time
from gtts import gTTS

for call in CALLS:
    target = f"{VOLUME_PATH}/{call['filename']}"
    # Idempotent: skip if it already exists
    if os.path.exists(target):
        size = os.path.getsize(target)
        print(f"[skip] {call['filename']} already exists ({size:,} bytes)")
        continue

    print(f"[generating] {call['filename']} — {call['scenario']}")
    tts = gTTS(text=call["script"], lang="pt", slow=False)
    # gTTS accepts any filename, including paths in /Volumes
    tts.save(target)
    size = os.path.getsize(target)
    print(f"   -> {target} ({size:,} bytes)")
    time.sleep(0.3)  # be polite to the free gTTS API

print("Done.")

# COMMAND ----------

# DBTITLE 1,Also save the scripts as a reference (to validate the LLM later)
import json

ref_path = f"{VOLUME_PATH}/_reference_scripts.json"
with open(ref_path, "w", encoding="utf-8") as f:
    json.dump(CALLS, f, ensure_ascii=False, indent=2)
print(f"Reference scripts saved to: {ref_path}")

# COMMAND ----------

# DBTITLE 1,List the files ready for ingestion
files = sorted(os.listdir(VOLUME_PATH))
print(f"\n{len(files)} files in {VOLUME_PATH}:\n")
for name in files:
    full = f"{VOLUME_PATH}/{name}"
    size = os.path.getsize(full)
    print(f"  {name:<40} {size:>10,} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC Next steps:
# MAGIC
# MAGIC | Pipeline | Where |
# MAGIC |---|---|
# MAGIC | Imperative (PySpark) | `pipelines/imperative/01_..02_..03_*.py` |
# MAGIC | Declarative (SDP SQL) | `pipelines/declarative/*.sql` |
# MAGIC
# MAGIC Add new audio files later with:
# MAGIC ```python
# MAGIC dbutils.fs.cp("file:/path/local.mp3", "/Volumes/.../audio_landing/")
# MAGIC ```
# MAGIC and the pipelines automatically process only the new files.
