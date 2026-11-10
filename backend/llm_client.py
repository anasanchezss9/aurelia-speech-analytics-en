"""Client for LLM-based analysis (Claude Sonnet 4.6 + AI Functions)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


# Dynamic JSON schema — built from the options chosen by the user.
PRESET_FIELDS: dict[str, dict[str, Any]] = {
    "sentiment": {
        "type": "string",
        "enum": ["positivo", "neutro", "negativo"],
        "description": "Overall customer sentiment on the call",
    },
    "emotion": {
        "type": "string",
        "description": "Predominant emotion (e.g., frustration, joy, anger, calm)",
    },
    "intent": {
        "type": "string",
        "enum": [
            "reclamacao",
            "suporte_tecnico",
            "cancelamento",
            "duvida_produto",
            "elogio",
            "outro",
        ],
        "description": "Main intent of the call",
    },
    "topics": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Topics/subjects mentioned (up to 5)",
    },
    "churn_risk": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": "Churn probability (0=low, 1=high)",
    },
    "compliance_ok": {
        "type": "boolean",
        "description": "True if the agent followed the mandatory script (greeting, recording notice, LGPD)",
    },
    "summary": {
        "type": "string",
        "description": "Call summary in up to 3 sentences",
    },
    "next_action": {
        "type": "string",
        "description": "Recommended next action to close the loop",
    },
}


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        workspace: WorkspaceClient | None = None,
    ):
        self.model = model or os.getenv("LLM_MODEL", "databricks-claude-sonnet-4-6")
        self.w = workspace or WorkspaceClient()

    def analyze(
        self,
        transcript: str,
        presets: list[str],
        custom_questions: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Runs the analysis over the transcript.

        - presets: list of keys in PRESET_FIELDS (e.g. ['sentiment', 'intent'])
        - custom_questions: the user's free-form questions (each one becomes a field
          'q1', 'q2'... with a string answer).
        """
        custom_questions = custom_questions or []

        # Builds the JSON Schema dynamically from the selections
        properties: dict[str, Any] = {}
        required: list[str] = []
        for key in presets:
            if key in PRESET_FIELDS:
                properties[key] = PRESET_FIELDS[key]
                required.append(key)
        for i, q in enumerate(custom_questions, start=1):
            field = f"q{i}"
            properties[field] = {
                "type": "string",
                "description": f"Answer to the question: {q}",
            }
            required.append(field)

        if not properties:
            return {"error": "No analysis selected."}

        json_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

        # Prompt: makes explicit what must be returned
        # The prompt text stays in Portuguese because the calls are in Portuguese.
        questions_block = ""
        if custom_questions:
            questions_block = "\n\nPERGUNTAS CUSTOMIZADAS (responda cada uma):\n" + "\n".join(
                f"- q{i}: {q}" for i, q in enumerate(custom_questions, start=1)
            )

        system_msg = (
            "Você é um analista de qualidade da Aurélia especializado em speech analytics "
            "de chamadas entre consultoras/clientes e o atendimento. Analise a transcrição "
            "abaixo e devolva EXCLUSIVAMENTE um JSON aderente ao schema solicitado. "
            "Seja factual: só afirme o que está sustentado na transcrição. "
            "Quando não houver evidência, marque como 'não identificado'."
        )

        user_msg = f"TRANSCRIÇÃO:\n\"\"\"\n{transcript}\n\"\"\"{questions_block}"

        # The Databricks SDK (.serving_endpoints.query) does not expose
        # response_format/json_schema. We call the REST API directly with the app's
        # OAuth token (auto-refreshed by WorkspaceClient).
        host = self.w.config.host.rstrip("/")
        token = self.w.config.authenticate()  # dict with the Authorization header
        url = f"{host}/serving-endpoints/{self.model}/invocations"
        payload = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 1500,
            "temperature": 0.1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "speech_analysis",
                    "schema": json_schema,
                    "strict": True,
                },
            },
        }

        try:
            r = requests.post(url, json=payload, headers=token, timeout=60)
        except Exception:
            logger.exception("Failed to call LLM %s", self.model)
            raise

        if r.status_code >= 400:
            logger.error("LLM HTTP %s: %s", r.status_code, r.text[:500])
            r.raise_for_status()

        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM returned no choices: {data}")
        content = choices[0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, returning raw")
            return {"raw": content}
