# Databricks notebook source
# MAGIC %md
# MAGIC # Whisper Large V3 Turbo on Model Serving (GPU)
# MAGIC
# MAGIC **The contract is simple on purpose**: the `audio` column comes in with the audio
# MAGIC in base64, a string with the transcription comes out. The app and the
# MAGIC transcription pipeline speak the same contract.
# MAGIC
# MAGIC The weights go **inside** the MLflow model instead of being downloaded from Hugging
# MAGIC Face when the container starts. The serving container doesn't have the same internet
# MAGIC egress as the job, and a download in `load_context` would turn every
# MAGIC cold start into an external point of failure.

# COMMAND ----------

# MAGIC %pip install -q -U mlflow==3.6.0 huggingface_hub==0.26.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Parameters
CATALOG = "field_toolbox_catalog"
SCHEMA = "aurelia_speech"
MODEL_NAME = "whisper_large_v3_turbo"
ENDPOINT_NAME = "aurelia-whisper-turbo"

HF_MODEL_ID = "openai/whisper-large-v3-turbo"

# GPU_MEDIUM = A10G, the recommended one for Turbo. scale-to-zero because it's a demo.
WORKLOAD_TYPE = "GPU_MEDIUM"
WORKLOAD_SIZE = "Small"

UC_MODEL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
print(f"model:    {UC_MODEL}")
print(f"endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Downloads the weights from Hugging Face
import os
import tempfile

from huggingface_hub import snapshot_download

destino = tempfile.mkdtemp(prefix="whisper-")
caminho = snapshot_download(
    repo_id=HF_MODEL_ID,
    local_dir=destino,
    # Only what transformers needs: safetensors weights, config and tokenizer.
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
)

total = 0
for raiz, _, arquivos in os.walk(caminho):
    for nome in arquivos:
        total += os.path.getsize(os.path.join(raiz, nome))
print(f"downloaded to {caminho} — {total / 1e9:.2f} GB")
print(sorted(os.listdir(caminho)))

# COMMAND ----------

# DBTITLE 1,pyfunc wrapper — same contract as the old endpoint
import mlflow


class WhisperTurbo(mlflow.pyfunc.PythonModel):
    """Transcribes base64 audio. `audio` goes in, the text comes out.

    Also accepts `audio_b64` and `language` to tolerate the other dialect the
    app client knows how to speak, but the declared signature is the simple one, which is the
    one the app and the pipeline use.
    """

    def load_context(self, context):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        pesos = context.artifacts["model"]
        usa_gpu = torch.cuda.is_available()
        device = "cuda:0" if usa_gpu else "cpu"
        dtype = torch.float16 if usa_gpu else torch.float32

        modelo = AutoModelForSpeechSeq2Seq.from_pretrained(
            pesos, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        modelo.to(device)
        processor = AutoProcessor.from_pretrained(pesos)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=modelo,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=False,
        )

    def predict(self, context, model_input):
        import base64
        import io

        import librosa
        import pandas as pd

        linhas = (
            model_input.to_dict(orient="records")
            if isinstance(model_input, pd.DataFrame)
            else list(model_input)
        )

        saidas = []
        for linha in linhas:
            b64 = linha.get("audio") or linha.get("audio_b64")
            idioma = (linha.get("language") or "pt").lower()
            if not b64:
                saidas.append("")
                continue
            audio, _ = librosa.load(io.BytesIO(base64.b64decode(b64)), sr=16000, mono=True)
            out = self.pipe(audio, generate_kwargs={"language": idioma, "task": "transcribe"})
            saidas.append(out["text"].strip())
        return saidas


print("wrapper defined")

# COMMAND ----------

# DBTITLE 1,Registers in Unity Catalog
import base64
import io
import wave

import pandas as pd
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

# A 0.5s WAV of silence just to get the signature right: `audio` goes in
# as a string, a string comes out.
buf = io.BytesIO()
with wave.open(buf, "wb") as w_:
    w_.setnchannels(1)
    w_.setsampwidth(2)
    w_.setframerate(16000)
    w_.writeframes(b"\x00\x00" * 8000)
exemplo_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

entrada = pd.DataFrame([{"audio": exemplo_b64}])
assinatura = infer_signature(entrada, ["transcricao de exemplo"])

with mlflow.start_run(run_name="whisper-turbo-field-toolbox") as run:
    mlflow.pyfunc.log_model(
        name="model",
        python_model=WhisperTurbo(),
        artifacts={"model": caminho},
        signature=assinatura,
        registered_model_name=UC_MODEL,
        pip_requirements=[
            "mlflow==3.6.0",
            "transformers==4.46.3",
            "torch==2.4.1",
            "accelerate==1.1.1",
            "librosa==0.10.2",
            "soundfile==0.12.1",
        ],
    )
    run_id = run.info.run_id

print(f"run: {run_id}")

# COMMAND ----------

# DBTITLE 1,Registered version
from mlflow.tracking import MlflowClient

cliente = MlflowClient(registry_uri="databricks-uc")
versao = max(int(v.version) for v in cliente.search_model_versions(f"name='{UC_MODEL}'"))
print(f"version: {versao}")

# COMMAND ----------

# DBTITLE 1,Creates the GPU endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

entidade = ServedEntityInput(
    entity_name=UC_MODEL,
    entity_version=str(versao),
    name="whisper-turbo",
    workload_type=WORKLOAD_TYPE,
    workload_size=WORKLOAD_SIZE,
    scale_to_zero_enabled=True,
)

existe = [e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME]
if existe:
    print("updating existing endpoint")
    w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[entidade])
else:
    print("creating endpoint")
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[entidade]),
    )

print("deployment triggered — takes 15 to 30 min to become READY")

# COMMAND ----------

import json

dbutils.notebook.exit(json.dumps({"modelo": UC_MODEL, "versao": versao, "endpoint": ENDPOINT_NAME}))
