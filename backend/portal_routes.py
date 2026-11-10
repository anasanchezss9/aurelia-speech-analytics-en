"""Routes for the Aurélia audio analysis portal.

Endpoints (prefix /api/portal):
  POST /api/portal/upload         - upload audio (.wav/.mp3/.m4a) to the volume
  POST /api/portal/process        - triggers the daily job (pipeline, scoring, Lakebase queue)
  GET  /api/portal/processing     - status of the last pipeline run
  GET  /api/portal/calls          - lists calls with insights (filters via query)
  GET  /api/portal/calls/{id}     - detail of one call
  GET  /api/portal/audio/{id}     - streams the .wav/.mp3 from the volume
  GET  /api/portal/dashboard      - aggregated KPIs for the main dashboard

Assumptions:
- The app runs as a Databricks App, with automatic OAuth via WorkspaceClient.
- Tables and volume live in ${CATALOG}.${SCHEMA}.* (configured via env).
- Processing is the `aurelia-speech-diario` job (PORTAL_JOB_ID), triggered through
  the Jobs API: the Lakeflow pipeline picks up new audio from the volume, including
  uploads, and the queue is scored and synced to Lakebase in the same run.
- The volume is accessed through the Files API, not the filesystem: the app
  container does not mount /Volumes, so open()/os.path.exists() there always fail.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from traducao import traduzir

logger = logging.getLogger("portal")

router = APIRouter(prefix="/api/portal", tags=["portal"])

# ---------- Config ----------

CATALOG = os.getenv("PORTAL_CATALOG", "field_toolbox_catalog")
SCHEMA = os.getenv("PORTAL_SCHEMA", "aurelia_speech")
VOLUME = os.getenv("PORTAL_VOLUME", "audio_landing")
TABLE_BRONZE = os.getenv("PORTAL_TABLE_BRONZE", "bronze_audio_files")
TABLE_SILVER = os.getenv("PORTAL_TABLE_SILVER", "silver_transcripts")
TABLE_GOLD = os.getenv("PORTAL_TABLE_GOLD", "gold_call_insights")
TABLE_TEXTO = os.getenv("PORTAL_TABLE_TEXTO", "bronze_transcricoes_texto")
WAREHOUSE_ID = os.getenv("PORTAL_WAREHOUSE_ID", "")  # SQL Warehouse the app reads from
JOB_ID = int(os.getenv("PORTAL_JOB_ID", "0"))

# The declarative pipeline's bronze table only stores the file pointer. Format and
# source come from the path: uploads are written to /uploads, recordings to /recordings.
BRONZE = (
    "(SELECT call_id, audio_path, ingested_at, "
    r"lower(regexp_extract(audio_path, '\.([a-zA-Z0-9]+)$', 1)) AS audio_format, "
    "CASE WHEN audio_path LIKE '%/uploads/%' THEN 'upload' "
    "WHEN audio_path LIKE '%/recordings/%' THEN 'recording' ELSE 'landing' END AS source "
    f"FROM {CATALOG}.{SCHEMA}.{TABLE_BRONZE})"
)

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
UPLOADS_SUBDIR = "uploads"
RECORDINGS_SUBDIR = "recordings"

ALLOWED_EXT = {".wav", ".mp3", ".m4a"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB / audio file

_w: WorkspaceClient | None = None


def workspace() -> WorkspaceClient:
    """Lazy singleton — the App's auth is injected at runtime."""
    global _w
    if _w is None:
        _w = WorkspaceClient()
    return _w


# ---------- State (in memory, enough for a single-instance demo) ----------

_processing_state: dict[str, Any] = {
    "status": "idle",  # idle | running | success | failed
    "run_id": None,
    "started_at": None,
    "finished_at": None,
    "message": "",
    "audios_pending_at_start": 0,
}
_processing_lock = threading.Lock()


# ---------- Helpers ----------


def _safe_filename(name: str) -> str:
    """Removes path traversal and dangerous characters."""
    base = Path(name).name  # strips directories
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or "audio"


def _volume_tamanho(caminho: str) -> int:
    """Size in bytes of a volume file, via the Files API."""
    md = workspace().files.get_metadata(caminho)
    return int(md.content_length or 0)


def _volume_leitura(caminho: str, inicio: int | None = None, fim: int | None = None):
    """Opens a volume file for reading, optionally only a byte range.

    The Databricks App container does not have the volume mounted as a filesystem —
    an `open("/Volumes/...")` here raises FileNotFoundError, even with the right
    Unity Catalog permissions. All audio access goes through the Files API.

    We use `api_client.do` instead of `files.download` because it is the only way
    to pass the Range header through: without it the browser's `<audio>` cannot
    seek and hangs on large files.
    """
    cabecalhos = {}
    if inicio is not None:
        cabecalhos["Range"] = f"bytes={inicio}-{fim}"
    resposta = workspace().api_client.do(
        "GET",
        f"/api/2.0/fs/files{caminho}",
        headers=cabecalhos,
        raw=True,
    )
    return resposta["contents"]


def _sql(stmt: str, parameters: Optional[list[dict]] = None) -> dict:
    """Runs SQL via the Statement Execution API."""
    if not WAREHOUSE_ID:
        raise HTTPException(
            status_code=500,
            detail="PORTAL_WAREHOUSE_ID is not configured in app.yaml",
        )
    w = workspace()
    payload: dict[str, Any] = {
        "warehouse_id": WAREHOUSE_ID,
        "statement": stmt,
        "wait_timeout": "30s",
        "format": "JSON_ARRAY",
        "disposition": "INLINE",
    }
    if parameters:
        payload["parameters"] = parameters

    res = w.api_client.do(
        "POST",
        "/api/2.0/sql/statements",
        body=payload,
    )
    state = res.get("status", {}).get("state")
    if state != "SUCCEEDED":
        err = res.get("status", {}).get("error", {}).get("message", "")
        logger.error("SQL failed (%s): %s\nstmt=%s", state, err, stmt[:300])
        raise HTTPException(status_code=502, detail=f"SQL {state}: {err}")
    return res


def _rows_to_dicts(res: dict) -> list[dict]:
    """Converts a Statement API result into list[dict]."""
    cols = [c["name"] for c in res.get("manifest", {}).get("schema", {}).get("columns", [])]
    data = res.get("result", {}).get("data_array") or []
    out = []
    for row in data:
        d = dict(zip(cols, row))
        out.append(d)
    return out


# ---------- Models ----------


class UploadOut(BaseModel):
    filename: str
    size_bytes: int
    saved_to: str
    call_id: str


class ProcessOut(BaseModel):
    status: str
    run_id: Optional[int] = None
    message: str = ""


class ProcessingStatusOut(BaseModel):
    status: str
    run_id: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: str = ""
    audios_pending_at_start: int = 0


class CallSummary(BaseModel):
    call_id: str
    audio_path: str
    audio_format: str
    source: str
    ingested_at: str
    analyzed_at: Optional[str] = None
    transcript_preview: Optional[str] = None
    sentiment: Optional[str] = None
    tipo_problema: Optional[str] = None
    suspeita_fraude: Optional[bool] = None
    reincidencia: Optional[bool] = None
    consultora_codigo: Optional[str] = None
    pedido_numero: Optional[str] = None
    churn_risk: Optional[float] = None


class CallDetail(CallSummary):
    full_text: Optional[str] = None
    summary: Optional[str] = None
    next_action: Optional[str] = None
    emotion: Optional[str] = None
    intent: Optional[str] = None
    topics: list[str] = []
    # Self-reported by the model, not calibrated — see the label on the detail screen.
    fraude_confianca: Optional[float] = None
    fraude_indicadores: list[str] = []
    produtos_faltantes: list[str] = []
    compliance_ok: Optional[bool] = None
    compliance_reason: Optional[str] = None


class DashboardKPIs(BaseModel):
    total_calls: int
    analyzed_calls: int
    pending_calls: int
    suspeita_fraude_count: int
    reincidencia_count: int
    avg_churn_risk: float


class DashboardOut(BaseModel):
    kpis: DashboardKPIs
    by_tipo_problema: list[dict[str, Any]]
    by_sentiment: list[dict[str, Any]]
    top_consultoras: list[dict[str, Any]]
    top_produtos_faltantes: list[dict[str, Any]]
    last_calls: list[CallSummary]


# ---------- Endpoints ----------


@router.post("/upload", response_model=UploadOut)
async def upload_audio(file: UploadFile) -> UploadOut:
    """Receives an audio file from the portal and saves it to /Volumes/.../uploads/<filename>."""
    name = _safe_filename(file.filename or "audio")
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Format {ext!r} not supported. Accepted: {sorted(ALLOWED_EXT)}",
        )

    target_path = f"{VOLUME_PATH}/{UPLOADS_SUBDIR}/{name}"

    # The file goes through a local temp file before being uploaded via the Files
    # API: the volume is not a filesystem inside the app, and this way the size limit
    # is enforced while reading, without loading the whole audio into memory.
    total = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=ext) as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File > {MAX_UPLOAD_BYTES // (1024*1024)}MB",
                    )
                buffer.write(chunk)
            buffer.flush()
            buffer.seek(0)
            workspace().files.upload(target_path, buffer, overwrite=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload of %s failed", name)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    call_id = Path(name).stem
    logger.info("Upload OK: %s (%d bytes) -> %s", name, total, target_path)

    return UploadOut(
        filename=name,
        size_bytes=total,
        saved_to=target_path,
        call_id=call_id,
    )


def _count_pending_audios() -> int:
    """Counts how many files in the volume are not yet in bronze."""
    try:
        res = _sql(
            f"""
            WITH files AS (
              SELECT path FROM read_files('{VOLUME_PATH}/', format=>'binaryFile')
              WHERE lower(path) RLIKE '\\\\.(mp3|wav|m4a)$'
            ),
            ingested AS (
              SELECT regexp_replace(audio_path, '^dbfs:', '') AS path
              FROM {CATALOG}.{SCHEMA}.{TABLE_BRONZE}
            )
            SELECT count(*) AS n
            FROM files f
            LEFT ANTI JOIN ingested i
              ON regexp_replace(f.path, '^dbfs:', '') = i.path
            """
        )
        rows = _rows_to_dicts(res)
        return int(rows[0]["n"]) if rows else 0
    except Exception as e:
        logger.warning("Could not count pending files: %s", e)
        return 0


def _run_pipeline_async(run_id_holder: dict[str, Any]) -> None:
    """Triggers the daily job and follows it until it finishes, in the background."""
    w = workspace()
    try:
        if not JOB_ID:
            raise RuntimeError("PORTAL_JOB_ID is not configured in app.yaml")
        run = w.jobs.run_now(job_id=JOB_ID)
        run_id_holder["run_id"] = run.run_id
        with _processing_lock:
            _processing_state["run_id"] = run.run_id
            _processing_state["message"] = "Pipeline, scoring and queue in progress…"

        run.result()  # blocks until a terminal state

        final = w.jobs.get_run(run.run_id)
        result_state = (final.state.result_state.value if final.state and final.state.result_state else "UNKNOWN")
        with _processing_lock:
            _processing_state["status"] = "success" if result_state == "SUCCESS" else "failed"
            _processing_state["finished_at"] = datetime.utcnow().isoformat()
            _processing_state["message"] = f"Job {result_state}"
        logger.info("Job finished: %s", result_state)
    except Exception as e:
        logger.exception("Job run failed")
        with _processing_lock:
            _processing_state["status"] = "failed"
            _processing_state["finished_at"] = datetime.utcnow().isoformat()
            _processing_state["message"] = str(e)


@router.post("/process", response_model=ProcessOut)
def trigger_processing() -> ProcessOut:
    """Triggers the pipeline (asynchronous). Only 1 run at a time."""
    with _processing_lock:
        if _processing_state["status"] == "running":
            return ProcessOut(
                status="already_running",
                run_id=_processing_state.get("run_id"),
                message="A processing run is already in progress.",
            )
        _processing_state.update(
            status="running",
            run_id=None,
            started_at=datetime.utcnow().isoformat(),
            finished_at=None,
            message="Triggering pipeline…",
            audios_pending_at_start=_count_pending_audios(),
        )

    holder: dict[str, Any] = {}
    t = threading.Thread(target=_run_pipeline_async, args=(holder,), daemon=True)
    t.start()
    return ProcessOut(status="started", message="Pipeline triggered in the background.")


@router.get("/processing", response_model=ProcessingStatusOut)
def processing_status() -> ProcessingStatusOut:
    with _processing_lock:
        return ProcessingStatusOut(**_processing_state)


@router.get("/calls", response_model=list[CallSummary])
def list_calls(
    limit: int = Query(100, ge=1, le=500),
    tipo_problema: Optional[str] = None,
    suspeita_fraude: Optional[bool] = None,
    source: Optional[str] = None,
    only_analyzed: bool = False,
) -> list[CallSummary]:
    """Lists calls with their insights. Bronze LEFT JOIN gold so audio files
    still being processed show up too."""
    where = []
    if tipo_problema:
        where.append(f"g.tipo_problema = '{tipo_problema}'")
    if suspeita_fraude is not None:
        where.append(f"g.suspeita_fraude = {str(suspeita_fraude).lower()}")
    if source:
        where.append(f"b.source = '{source}'")
    if only_analyzed:
        where.append("g.call_id IS NOT NULL")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    res = _sql(
        f"""
        SELECT
          b.call_id,
          b.audio_path,
          b.audio_format,
          b.source,
          CAST(b.ingested_at AS STRING) AS ingested_at,
          CAST(g.analyzed_at AS STRING) AS analyzed_at,
          substr(s.full_text, 1, 200) AS transcript_preview,
          g.sentiment,
          g.tipo_problema,
          g.suspeita_fraude,
          g.reincidencia,
          g.consultora_codigo,
          g.pedido_numero,
          g.churn_risk
        FROM {BRONZE} b
        LEFT JOIN {CATALOG}.{SCHEMA}.{TABLE_SILVER} s ON s.call_id = b.call_id
        LEFT JOIN {CATALOG}.{SCHEMA}.{TABLE_GOLD}   g ON g.call_id = b.call_id
        {where_sql}
        ORDER BY COALESCE(g.analyzed_at, b.ingested_at) DESC
        LIMIT {limit}
        """
    )
    return [CallSummary(**r) for r in _rows_to_dicts(res)]


def _parse_array(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        # May come as a JSON-array string OR CSV
        try:
            j = json.loads(v)
            if isinstance(j, list):
                return [str(x) for x in j]
        except Exception:
            pass
        if v.startswith("[") and v.endswith("]"):
            return [v]
        return [v] if v else []
    return [str(v)]


@router.get("/calls/{call_id}", response_model=CallDetail)
def get_call(call_id: str) -> CallDetail:
    """Detail of one call — used by the drill-down."""
    res = _sql(
        f"""
        SELECT
          b.call_id,
          b.audio_path,
          b.audio_format,
          b.source,
          CAST(b.ingested_at AS STRING) AS ingested_at,
          CAST(g.analyzed_at AS STRING) AS analyzed_at,
          s.full_text,
          substr(s.full_text, 1, 200) AS transcript_preview,
          g.sentiment, g.emotion, g.intent,
          g.summary, g.next_action,
          g.tipo_problema, g.suspeita_fraude, g.fraude_confianca, g.reincidencia,
          g.consultora_codigo, g.pedido_numero, g.churn_risk,
          g.compliance_ok, g.compliance_reason,
          to_json(g.topics)             AS topics_json,
          to_json(g.fraude_indicadores) AS fraude_indicadores_json,
          to_json(g.produtos_faltantes) AS produtos_faltantes_json
        FROM {BRONZE} b
        LEFT JOIN {CATALOG}.{SCHEMA}.{TABLE_SILVER} s ON s.call_id = b.call_id
        LEFT JOIN {CATALOG}.{SCHEMA}.{TABLE_GOLD}   g ON g.call_id = b.call_id
        WHERE b.call_id = '{call_id}'
        LIMIT 1
        """
    )
    rows = _rows_to_dicts(res)
    if not rows:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")
    r = rows[0]
    topics = _parse_array(r.get("topics_json"))
    indicadores = _parse_array(r.get("fraude_indicadores_json"))
    livres = [r.get("summary"), r.get("next_action"), r.get("compliance_reason"), r.get("emotion")]
    traduzidos = traduzir(livres + topics + indicadores)
    r["summary"], r["next_action"], r["compliance_reason"], r["emotion"] = traduzidos[:4]
    topics, indicadores = traduzidos[4:4 + len(topics)], traduzidos[4 + len(topics):]
    return CallDetail(
        call_id=r["call_id"],
        audio_path=r["audio_path"],
        audio_format=r.get("audio_format") or "",
        source=r.get("source") or "landing",
        ingested_at=r.get("ingested_at") or "",
        analyzed_at=r.get("analyzed_at"),
        transcript_preview=r.get("transcript_preview"),
        full_text=r.get("full_text"),
        sentiment=r.get("sentiment"),
        emotion=r.get("emotion"),
        intent=r.get("intent"),
        summary=r.get("summary"),
        next_action=r.get("next_action"),
        tipo_problema=r.get("tipo_problema"),
        suspeita_fraude=r.get("suspeita_fraude"),
        fraude_confianca=r.get("fraude_confianca"),
        reincidencia=r.get("reincidencia"),
        consultora_codigo=r.get("consultora_codigo"),
        pedido_numero=r.get("pedido_numero"),
        churn_risk=r.get("churn_risk"),
        compliance_ok=r.get("compliance_ok"),
        compliance_reason=r.get("compliance_reason"),
        topics=topics,
        fraude_indicadores=indicadores,
        produtos_faltantes=_parse_array(r.get("produtos_faltantes_json")),
    )


@router.get("/audio/{call_id}")
def stream_audio(call_id: str, request: Request):
    """Streams the audio file directly from the volume.

    Implements HTTP Range Requests (RFC 7233) — without it the browser's
    `<audio>` cannot seek or play large files (>20 MB) because it won't
    start playback until the whole file is in memory.
    """
    # Find the real path via bronze (prevents path traversal)
    res = _sql(
        f"""
        SELECT regexp_replace(audio_path, '^dbfs:', '') AS path,
               audio_format
        FROM {BRONZE} b
        WHERE call_id = '{call_id}'
        LIMIT 1
        """
    )
    rows = _rows_to_dicts(res)
    if not rows:
        raise HTTPException(status_code=404, detail="Audio not found")
    path = rows[0]["path"]
    fmt = rows[0].get("audio_format") or "wav"
    media_type = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
    }.get(fmt, "application/octet-stream")

    if not path:
        raise HTTPException(status_code=404, detail="Audio has no path in bronze")
    try:
        file_size = _volume_tamanho(path)
    except Exception as e:
        logger.exception("Audio %s not accessible in the volume", path)
        raise HTTPException(status_code=404, detail=f"File {path} is not in the volume: {e}")

    range_header = request.headers.get("range") or request.headers.get("Range")

    # Case 1: client did NOT request a range -> we send the whole file, but
    # still signal that we accept ranges (some browsers check the header
    # before playing and only start playback if they see Accept-Ranges).
    if not range_header:
        def iter_full():
            fonte = _volume_leitura(path)
            try:
                while True:
                    chunk = fonte.read(1024 * 64)
                    if not chunk:
                        break
                    yield chunk
            finally:
                fonte.close()

        return StreamingResponse(
            iter_full(),
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "no-cache",
            },
        )

    # Case 2: client requested range "bytes=start-end"
    try:
        units, _, rng = range_header.partition("=")
        if units.strip().lower() != "bytes":
            raise ValueError("bytes only")
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        if start < 0 or start >= file_size:
            raise ValueError("start outside the file")
        end = min(end, file_size - 1)
    except Exception:
        # Malformed Range header -> 416 Range Not Satisfiable
        raise HTTPException(
            status_code=416,
            detail="Range Not Satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    length = end - start + 1

    def iter_range():
        fonte = _volume_leitura(path, start, end)
        try:
            remaining = length
            while remaining > 0:
                chunk = fonte.read(min(1024 * 64, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
        finally:
            fonte.close()

    return StreamingResponse(
        iter_range(),
        status_code=206,  # Partial Content
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
            "Cache-Control": "no-cache",
        },
    )


@router.get("/dashboard", response_model=DashboardOut)
def dashboard() -> DashboardOut:
    """KPIs + aggregations for the main dashboard."""
    # Main KPIs
    res_kpi = _sql(
        f"""
        SELECT
          (SELECT count(*) FROM {CATALOG}.{SCHEMA}.{TABLE_BRONZE})
            + (SELECT count(*) FROM {CATALOG}.{SCHEMA}.{TABLE_TEXTO})                    AS total_calls,
          (SELECT count(*) FROM {CATALOG}.{SCHEMA}.{TABLE_GOLD})                         AS analyzed_calls,
          (SELECT count(*) FROM {CATALOG}.{SCHEMA}.{TABLE_GOLD} WHERE suspeita_fraude)   AS suspeita_fraude_count,
          (SELECT count(*) FROM {CATALOG}.{SCHEMA}.{TABLE_GOLD} WHERE reincidencia)      AS reincidencia_count,
          (SELECT COALESCE(avg(churn_risk),0) FROM {CATALOG}.{SCHEMA}.{TABLE_GOLD})      AS avg_churn_risk
        """
    )
    k = _rows_to_dicts(res_kpi)[0]
    kpis = DashboardKPIs(
        total_calls=int(k["total_calls"]),
        analyzed_calls=int(k["analyzed_calls"]),
        pending_calls=max(int(k["total_calls"]) - int(k["analyzed_calls"]), 0),
        suspeita_fraude_count=int(k["suspeita_fraude_count"]),
        reincidencia_count=int(k["reincidencia_count"]),
        avg_churn_risk=float(k["avg_churn_risk"] or 0.0),
    )

    res_tipo = _sql(
        f"""
        SELECT COALESCE(tipo_problema, 'sem_classificacao') AS tipo, count(*) AS n
        FROM {CATALOG}.{SCHEMA}.{TABLE_GOLD}
        GROUP BY 1 ORDER BY n DESC
        """
    )
    by_tipo = [{"tipo": r["tipo"], "n": int(r["n"])} for r in _rows_to_dicts(res_tipo)]

    res_sent = _sql(
        f"""
        SELECT COALESCE(sentiment, 'sem_classificacao') AS sentiment, count(*) AS n
        FROM {CATALOG}.{SCHEMA}.{TABLE_GOLD}
        GROUP BY 1 ORDER BY n DESC
        """
    )
    by_sent = [{"sentiment": r["sentiment"], "n": int(r["n"])} for r in _rows_to_dicts(res_sent)]

    res_cons = _sql(
        f"""
        SELECT consultora_codigo,
               count(*) AS chamadas,
               sum(CASE WHEN suspeita_fraude THEN 1 ELSE 0 END) AS fraudes,
               sum(CASE WHEN reincidencia    THEN 1 ELSE 0 END) AS reincidencias
        FROM {CATALOG}.{SCHEMA}.{TABLE_GOLD}
        WHERE consultora_codigo IS NOT NULL AND consultora_codigo != ''
        GROUP BY consultora_codigo
        ORDER BY chamadas DESC
        LIMIT 10
        """
    )
    top_cons = _rows_to_dicts(res_cons)

    # Top missing products — explodes the array
    res_prod = _sql(
        f"""
        SELECT produto, count(*) AS n
        FROM (
          SELECT explode(produtos_faltantes) AS produto
          FROM {CATALOG}.{SCHEMA}.{TABLE_GOLD}
          WHERE produtos_faltantes IS NOT NULL AND size(produtos_faltantes) > 0
        )
        GROUP BY produto
        ORDER BY n DESC
        LIMIT 10
        """
    )
    top_prod = _rows_to_dicts(res_prod)

    res_last = _sql(
        f"""
        SELECT
          b.call_id, b.audio_path, b.audio_format, b.source,
          CAST(b.ingested_at AS STRING)  AS ingested_at,
          CAST(g.analyzed_at AS STRING)  AS analyzed_at,
          substr(s.full_text, 1, 200)    AS transcript_preview,
          g.sentiment, g.tipo_problema, g.suspeita_fraude, g.reincidencia,
          g.consultora_codigo, g.pedido_numero, g.churn_risk
        FROM {BRONZE} b
        LEFT JOIN {CATALOG}.{SCHEMA}.{TABLE_SILVER} s ON s.call_id = b.call_id
        LEFT JOIN {CATALOG}.{SCHEMA}.{TABLE_GOLD}   g ON g.call_id = b.call_id
        ORDER BY COALESCE(g.analyzed_at, b.ingested_at) DESC
        LIMIT 8
        """
    )
    last_calls = [CallSummary(**r) for r in _rows_to_dicts(res_last)]

    return DashboardOut(
        kpis=kpis,
        by_tipo_problema=by_tipo,
        by_sentiment=by_sent,
        top_consultoras=top_cons,
        top_produtos_faltantes=top_prod,
        last_calls=last_calls,
    )
