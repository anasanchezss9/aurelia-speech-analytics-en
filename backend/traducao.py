"""English rendering of the Portuguese text the LLM wrote about each call.

The calls are in Portuguese, and so is what the gold layer extracted from them in
free text (call summary, next action, fraud indicators). The lakehouse stays the
single source: this module translates at read time with the Foundation Model
endpoint the app already declares, and keeps each translation in memory, so a
case opened twice costs one call. If the endpoint fails, the original text comes
back — the analyst still sees the case, only in Portuguese.

APP_IDIOMA=pt turns translation off.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import requests
from databricks.sdk import WorkspaceClient

logger = logging.getLogger("traducao")

MODELO = os.getenv("LLM_MODEL", "databricks-claude-sonnet-4-6")
ATIVA = os.getenv("APP_IDIOMA", "en").lower() == "en"

_INSTRUCAO = (
    "You translate Brazilian Portuguese into US English for a fraud-investigation "
    "analyst. Keep product names, person names, codes and numbers exactly as they are. "
    "Reply with only a JSON array of the English translations, one per input string, "
    "in the same order, nothing else."
)

_cache: dict[str, str] = {}
_trava = threading.Lock()
_cliente: WorkspaceClient | None = None


def _workspace() -> WorkspaceClient:
    global _cliente
    if _cliente is None:
        _cliente = WorkspaceClient()
    return _cliente


def traduzir(textos: list[str | None]) -> list[str | None]:
    if not ATIVA:
        return textos
    pendentes = [t for t in dict.fromkeys(textos) if t and t not in _cache]
    if pendentes:
        try:
            w = _workspace()
            r = requests.post(
                f"{w.config.host.rstrip('/')}/serving-endpoints/{MODELO}/invocations",
                headers=w.config.authenticate(),
                timeout=30,
                # No response_format: with a strict json_schema this endpoint returns the
                # input untranslated, so the array is asked for in the instruction and
                # checked below.
                json={
                    "messages": [
                        {"role": "system", "content": _INSTRUCAO},
                        {"role": "user", "content": json.dumps(pendentes, ensure_ascii=False)},
                    ],
                    "temperature": 0,
                    "max_tokens": 4000,
                },
            )
            r.raise_for_status()
            conteudo = r.json()["choices"][0]["message"]["content"].strip()
            saida = json.loads(conteudo[conteudo.find("["):conteudo.rfind("]") + 1])
            if not all(isinstance(s, str) for s in saida):
                saida = []
            if len(saida) == len(pendentes):
                with _trava:
                    _cache.update(zip(pendentes, saida))
            else:
                logger.warning("Translation returned %d of %d strings; showing the originals",
                               len(saida), len(pendentes))
        except Exception:
            logger.exception("Translation failed; showing the original text")
    return [_cache.get(t, t) if t else t for t in textos]
