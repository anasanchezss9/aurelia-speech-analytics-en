"""
Deploy Whisper Large V3 Turbo on Model Serving — Job version (non-notebook).

Run this script on a cluster with DBR 15.4 LTS ML GPU. It:
  1. installs dependencies via %pip (in a notebook it would be; here we use a pip subprocess)
  2. registers the pyfunc model in Unity Catalog
  3. creates/updates the GPU serving endpoint
"""

import base64
import io
import os
import subprocess
import sys

# ----- 1. Install dependencies -----
PKGS = [
    "mlflow==3.6.0",
    "transformers==4.46.3",
    "accelerate==1.1.1",
    "librosa==0.10.2",
    "soundfile==0.12.1",
]
print(f"Installing: {PKGS}")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *PKGS])

# ----- 2. Parameters -----
CATALOG = os.environ.get("CATALOG", "field_toolbox_catalog")
SCHEMA = os.environ.get("SCHEMA", "aurelia_speech")
MODEL_NAME = os.environ.get("MODEL_NAME", "whisper_large_v3_turbo")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "aurelia-whisper-turbo")
WORKLOAD_TYPE = os.environ.get("WORKLOAD_TYPE", "GPU_MEDIUM")
WORKLOAD_SIZE = os.environ.get("WORKLOAD_SIZE", "Small")

UC_MODEL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
print(f"UC model:  {UC_MODEL}")
print(f"Endpoint:  {ENDPOINT_NAME}")

# ----- 3. Ensures catalog/schema -----
from pyspark.sql import SparkSession  # noqa: E402

spark = SparkSession.builder.getOrCreate()
# The catalog must already exist (in workspaces with Default Storage we can't create it
# ad hoc). The schema is created if it doesn't exist.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema OK: {CATALOG}.{SCHEMA}")


# ----- 4. Defines the pyfunc -----
import mlflow  # noqa: E402
import mlflow.pyfunc  # noqa: E402


class WhisperTurbo(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3-turbo"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=False,
        )

    def _decode_audio(self, audio_b64: str):
        import io as _io
        import librosa
        raw = base64.b64decode(audio_b64)
        y, sr = librosa.load(_io.BytesIO(raw), sr=16000, mono=True)
        return y, sr

    def predict(self, context, model_input):
        import pandas as pd
        rows = (
            model_input.to_dict(orient="records")
            if hasattr(model_input, "to_dict")
            else list(model_input)
        )
        results = []
        for row in rows:
            audio_b64 = row.get("audio_b64") or row.get("audio")
            language = (row.get("language") or "pt").lower()
            audio, sr = self._decode_audio(audio_b64)
            duration = float(len(audio)) / float(sr)
            out = self.pipe(audio, generate_kwargs={"language": language, "task": "transcribe"})
            results.append({
                "text": out["text"].strip(),
                "language": language,
                "duration": round(duration, 2),
            })
        return pd.DataFrame(results)


# ----- 5. Generates input/output examples (synthetic silence) -----
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import pandas as pd  # noqa: E402

sr = 16000
test_audio = (np.random.randn(sr * 2) * 0.01).astype(np.float32)
buf = io.BytesIO()
sf.write(buf, test_audio, sr, format="WAV")
audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

input_example = pd.DataFrame([{"audio_b64": audio_b64, "language": "pt"}])
output_example = pd.DataFrame([{"text": "exemplo", "language": "pt", "duration": 2.0}])

from mlflow.models import infer_signature  # noqa: E402

signature = infer_signature(input_example, output_example)

# ----- 6. Log to UC -----
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="whisper-turbo-deploy") as run:
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WhisperTurbo(),
        signature=signature,
        input_example=input_example,
        registered_model_name=UC_MODEL,
        pip_requirements=PKGS,
    )
    run_id = run.info.run_id

print(f"Run id: {run_id}")
print(f"Model registered at: {UC_MODEL}")

# ----- 7. Gets the latest version -----
from mlflow.tracking import MlflowClient  # noqa: E402

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{UC_MODEL}'")
latest_version = max(int(v.version) for v in versions)
print(f"Latest version: {latest_version}")

# ----- 8. Creates/updates the endpoint -----
from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.service.serving import (  # noqa: E402
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

w = WorkspaceClient()
served_entity = ServedEntityInput(
    entity_name=UC_MODEL,
    entity_version=str(latest_version),
    name="whisper-turbo",
    workload_type=ServingModelWorkloadType(WORKLOAD_TYPE),
    workload_size=WORKLOAD_SIZE,
    scale_to_zero_enabled=True,
)
existing = [e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME]
if existing:
    print(f"Updating endpoint {ENDPOINT_NAME}...")
    w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[served_entity])
else:
    print(f"Creating endpoint {ENDPOINT_NAME}...")
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),
    )

print(f"OK — deployment started for {ENDPOINT_NAME}.")
print("Track it with:")
print(f"  databricks --profile <profile> serving-endpoints get {ENDPOINT_NAME}")
