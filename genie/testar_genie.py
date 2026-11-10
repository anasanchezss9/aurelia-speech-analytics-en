"""Asks the Genie space questions through the Conversation API and shows the SQL and the answer.

Usage:
    python testar_genie.py --space-id <id> "question 1" "question 2"
"""

import argparse
import json
import subprocess
import time


def api(perfil, metodo, caminho, corpo=None):
    cmd = ["databricks", "api", metodo, caminho, "-p", perfil, "-o", "json"]
    if corpo is not None:
        cmd += ["--json", json.dumps(corpo)]
    saida = subprocess.run(cmd, capture_output=True, text=True)
    if saida.returncode:
        raise RuntimeError(saida.stderr or saida.stdout)
    return json.loads(saida.stdout or "{}")


def perguntar(perfil, space, pergunta):
    r = api(perfil, "post", f"/api/2.0/genie/spaces/{space}/start-conversation", {"content": pergunta})
    conversa, mensagem = r["conversation"]["id"], r["message"]["id"]
    base = f"/api/2.0/genie/spaces/{space}/conversations/{conversa}/messages/{mensagem}"
    for _ in range(60):
        m = api(perfil, "get", base)
        if m["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(3)
    print(f"\n=== {pergunta}\nstatus: {m['status']}")
    for anexo in m.get("attachments", []):
        if "text" in anexo:
            print("text:", anexo["text"].get("content", "")[:600])
        if "query" in anexo:
            print("sql:", anexo["query"].get("query", "").strip()[:900])
            linhas = api(perfil, "get", f"{base}/attachments/{anexo['attachment_id']}/query-result")
            dados = linhas.get("statement_response", {}).get("result", {}).get("data_array", [])
            for linha in dados[:6]:
                print("  ", linha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="field-toolbox")
    ap.add_argument("--space-id", required=True)
    ap.add_argument("perguntas", nargs="+")
    a = ap.parse_args()
    for p in a.perguntas:
        perguntar(a.profile, a.space_id, p)


if __name__ == "__main__":
    main()
