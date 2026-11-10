"""FastAPI app: Speech Analytics demo (Aurélia).

Endpoints:
  GET  /health                  - liveness
  GET  /api/recursos            - what is enabled in this deploy (live recording)
  GET  /api/presets             - preconfigured analysis options
  POST /api/analyze             - runs the LLM over a transcript
  WS   /ws/transcribe           - streaming: receives audio chunks,
                                  returns a partial transcript
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llm_client import PRESET_FIELDS, LLMClient
from fila_routes import router as fila_router
from portal_routes import router as portal_router
from whisper_client import WhisperClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speech-analytics")

app = FastAPI(title="Aurélia Speech Analytics Demo", version="0.1.0")

# CORS — in dev we run React on :5173 and the backend on :8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper = WhisperClient()
llm = LLMClient()

# Portal routes (upload, dashboard, calls, audio streaming) and queue routes (Lakebase)
app.include_router(portal_router)
app.include_router(fila_router)


# ---------- Models ----------


class PresetOut(BaseModel):
    key: str
    label: str
    description: str


class AnalyzeIn(BaseModel):
    transcript: str = Field(..., min_length=1)
    presets: list[str] = Field(default_factory=list)
    custom_questions: list[str] = Field(default_factory=list)


class AnalyzeOut(BaseModel):
    result: dict


# Friendly labels for the UI
PRESET_LABELS: dict[str, str] = {
    "sentiment": "Sentiment (positive/neutral/negative)",
    "emotion": "Predominant emotion",
    "intent": "Call intent",
    "topics": "Topics / subjects",
    "churn_risk": "Churn risk (0 to 1)",
    "compliance_ok": "Compliance — mandatory script",
    "summary": "Executive summary",
    "next_action": "Recommended next action",
}


# ---------- REST ----------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/recursos")
def recursos() -> dict[str, bool]:
    # Live recording depends on a GPU Whisper endpoint; without it, the Configure
    # and Record tabs are hidden instead of failing on the first utterance.
    return {"gravacao_ao_vivo": bool(os.getenv("WHISPER_ENDPOINT"))}


@app.get("/api/presets", response_model=list[PresetOut])
def list_presets() -> list[PresetOut]:
    return [
        PresetOut(
            key=k,
            label=PRESET_LABELS.get(k, k),
            description=PRESET_FIELDS[k].get("description", ""),
        )
        for k in PRESET_FIELDS
    ]


@app.post("/api/warmup")
async def warmup() -> dict[str, str]:
    """
    Warms up the Whisper endpoint by sending a tiny WAV of silence.
    Call it when entering the recording screen to avoid a ~10s cold start
    on the first real request.
    """
    import io
    import numpy as np
    import soundfile as sf

    sr = 16000
    silence = np.zeros(int(sr * 0.5), dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, silence, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    import base64
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    def _call():
        return whisper.w.serving_endpoints.query(
            name=whisper.endpoint_name,
            dataframe_records=[{"audio_b64": audio_b64, "language": "pt"}],
        )

    try:
        await asyncio.to_thread(_call)
        return {"status": "warm"}
    except Exception as exc:
        logger.exception("Warmup failed")
        return {"status": "error", "message": str(exc)}


@app.post("/api/analyze", response_model=AnalyzeOut)
async def analyze(payload: AnalyzeIn) -> AnalyzeOut:
    logger.info(
        "analyze: presets=%s custom=%d transcript_len=%d",
        payload.presets,
        len(payload.custom_questions),
        len(payload.transcript),
    )
    result = await asyncio.to_thread(
        llm.analyze,
        payload.transcript,
        payload.presets,
        payload.custom_questions,
    )
    return AnalyzeOut(result=result)


# ---------- WebSocket: audio streaming ----------
#
# Simple protocol (JSON text or binary):
#   - client sends binary: audio chunk (WebM/Opus from MediaRecorder)
#   - client sends text:   {"type":"flush"} -> force transcription now
#                          {"type":"end"}   -> close
#   - server sends text:   {"type":"partial", "text": "..."}
#                          {"type":"final",   "text": "..."}
#                          {"type":"error",   "message": "..."}
#
# Strategy: we accumulate the chunks in a buffer; on each FLUSH or every N seconds,
# we concatenate them and send them to Whisper. WebM/Opus is cumulative (each chunk
# must be decoded together with the previous ones), so we send the whole buffer
# on every flush — the backend is stateless per session.


SESSION_FLUSH_SECONDS = float(os.getenv("WHISPER_FLUSH_SECONDS", "3.5"))


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("WS connected")

    audio_buffer = io.BytesIO()
    last_flush = asyncio.get_event_loop().time()
    closed = False
    chunk_count = 0
    total_bytes = 0
    flush_count = 0

    async def do_transcribe(final: bool = False) -> None:
        nonlocal last_flush, flush_count
        data = audio_buffer.getvalue()
        flush_count += 1
        logger.info(
            "flush#%d final=%s buffer_bytes=%d", flush_count, final, len(data)
        )
        if len(data) < 2048:
            logger.info("flush#%d: buffer too small, skip", flush_count)
            return
        try:
            t0 = asyncio.get_event_loop().time()
            result = await asyncio.to_thread(whisper.transcribe, data, "pt")
            dt = asyncio.get_event_loop().time() - t0
            logger.info(
                "flush#%d: whisper ok in %.2fs, text_len=%d",
                flush_count,
                dt,
                len(result.text),
            )
        except Exception as exc:
            logger.exception("Transcription failed (flush#%d)", flush_count)
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
            return

        msg_type: Literal["partial", "final"] = "final" if final else "partial"
        await ws.send_json(
            {
                "type": msg_type,
                "text": result.text,
                "duration": result.duration,
            }
        )
        last_flush = asyncio.get_event_loop().time()

    try:
        while not closed:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                logger.info("WS received disconnect")
                break

            if "bytes" in msg and msg["bytes"]:
                chunk = msg["bytes"]
                audio_buffer.write(chunk)
                chunk_count += 1
                total_bytes += len(chunk)
                if chunk_count <= 3 or chunk_count % 5 == 0:
                    logger.info(
                        "chunk#%d size=%d total=%d",
                        chunk_count,
                        len(chunk),
                        total_bytes,
                    )

                now = asyncio.get_event_loop().time()
                if now - last_flush >= SESSION_FLUSH_SECONDS:
                    await do_transcribe(final=False)
            elif "text" in msg and msg["text"]:
                import json

                try:
                    cmd = json.loads(msg["text"])
                except json.JSONDecodeError:
                    cmd = {}
                ctype = cmd.get("type")
                logger.info(
                    "cmd=%s chunks=%d bytes=%d", ctype, chunk_count, total_bytes
                )
                if ctype == "flush":
                    await do_transcribe(final=False)
                elif ctype == "end":
                    await do_transcribe(final=True)
                    closed = True
                    break
    except WebSocketDisconnect:
        logger.info(
            "WS disconnected by client (chunks=%d bytes=%d)",
            chunk_count,
            total_bytes,
        )
    except Exception:
        logger.exception("WS error")
        try:
            await ws.send_json({"type": "error", "message": "internal"})
        except Exception:
            pass
    finally:
        logger.info(
            "WS closing: chunks=%d bytes=%d flushes=%d",
            chunk_count,
            total_bytes,
            flush_count,
        )
        try:
            await ws.close()
        except Exception:
            pass


# ---------- Serve frontend ----------
# Simple strategy: standalone HTML/CSS/JS in ./static, no build step.
# The portal is plain HTML and JS in static/, served by FastAPI itself.

_HERE = Path(__file__).parent
STATIC_DIR = _HERE / "static"

if STATIC_DIR.exists():
    logger.info("Serving static frontend from %s", STATIC_DIR)
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )

    @app.get("/")
    def root():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        return {"error": "index.html not found"}
else:
    logger.warning("static/ folder not found — running API only.")
