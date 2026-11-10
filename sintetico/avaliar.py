"""Compares what the pipeline extracted from the synthetic calls against the ground truth.

Reads the silver (transcript) and the gold (business fields) through a SQL
warehouse and measures, call by call:

- transcript WER against the script, with the same normalization as the
  comparison notebook (no accents, no punctuation and no numbers, because the
  script says "oito quatro" (eight four) and the transcript writes "84");
- accuracy of tipo_problema, suspeita_fraude, reincidencia and sentiment;
- whether the consultant code and the order number came with the right digits;
- whether each missing product in the ground truth appears in the extracted list.

Usage:
    python avaliar.py --profile field-toolbox --warehouse 9fc569e6e984dcde \\
        --silver field_toolbox_catalog.aurelia_speech.sint_silver_transcripts_ait \\
        --gold field_toolbox_catalog.aurelia_speech.sint_gold_call_insights_ait
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import jiwer
from databricks.sdk import WorkspaceClient

AQUI = Path(__file__).parent

NUMEROS = set(
    "zero um uma dois duas tres quatro cinco seis sete oito nove dez onze doze treze quatorze catorze "
    "quinze dezesseis dezessete dezoito dezenove vinte trinta quarenta cinquenta sessenta setenta "
    "oitenta noventa cem cento duzentos trezentos quatrocentos quinhentos mil".split()
)
CAMPOS = ("tipo_problema", "suspeita_fraude", "reincidencia", "sentiment")


def norm(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", (texto or "").lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9 ]+", " ", texto)
    return " ".join(p for p in texto.split() if not p.isdigit() and p not in NUMEROS)


def digitos(texto) -> str:
    return re.sub(r"\D", "", str(texto or ""))


def consultar(cliente: WorkspaceClient, warehouse: str, sql: str) -> list[dict]:
    resposta = cliente.statement_execution.execute_statement(
        statement=sql, warehouse_id=warehouse, wait_timeout="50s"
    )
    while resposta.status.state.value in ("PENDING", "RUNNING"):
        resposta = cliente.statement_execution.get_statement(resposta.statement_id)
    if resposta.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"{resposta.status.state.value}: {resposta.status.error}")
    colunas = [c.name for c in resposta.manifest.schema.columns]
    return [dict(zip(colunas, linha)) for linha in (resposta.result.data_array or [])]


def produto_encontrado(esperado: str, extraidos: list[str]) -> bool:
    """The first two words that distinguish the product appear in the extracted list.

    The comparison ignores spaces because the transcriber splits compound brand
    names into two words; a wrong letter in the brand name counts as an error.
    """
    genericas = {"perfume", "perfumes", "kit", "creme", "hidratante", "dois", "duas", "tres"}
    chaves = [p for p in norm(esperado).split() if len(p) > 3 and p not in genericas]
    texto = "".join(norm(e).replace(" ", "") for e in extraidos)
    return bool(chaves) and all(k in texto for k in chaves[:2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--roteiros", default=str(AQUI / "roteiros.json"))
    args = parser.parse_args()

    roteiros = {r["call_id"]: r for r in json.loads(Path(args.roteiros).read_text(encoding="utf-8"))}
    cliente = WorkspaceClient(profile=args.profile)
    silver = {
        r["call_id"]: r["full_text"]
        for r in consultar(cliente, args.warehouse, f"SELECT call_id, full_text FROM {args.silver}")
    }
    gold = {
        r["call_id"]: r
        for r in consultar(
            cliente,
            args.warehouse,
            f"SELECT call_id, tipo_problema, CAST(suspeita_fraude AS STRING) AS suspeita_fraude, "
            f"CAST(reincidencia AS STRING) AS reincidencia, sentiment, consultora_codigo, pedido_numero, "
            f"to_json(produtos_faltantes) AS produtos_faltantes FROM {args.gold}",
        )
    }

    acertos = Counter()
    confusao = Counter()
    wers = []
    produtos_total = produtos_ok = 0
    avaliadas = 0
    print(f"{'call':<10} {'WER':>5}  {'expected type':<16} {'extracted type':<16} mismatches")
    for call_id, roteiro in roteiros.items():
        if call_id not in gold:
            continue
        avaliadas += 1
        esperado, obtido = roteiro["gabarito"], gold[call_id]
        obtido_tipado = {
            "tipo_problema": obtido["tipo_problema"],
            "suspeita_fraude": obtido["suspeita_fraude"] == "true",
            "reincidencia": obtido["reincidencia"] == "true",
            "sentiment": obtido["sentiment"],
        }
        erros = []
        for campo in CAMPOS:
            if obtido_tipado[campo] == esperado[campo]:
                acertos[campo] += 1
            else:
                erros.append(f"{campo}={obtido_tipado[campo]} (expected {esperado[campo]})")
        confusao[(esperado["tipo_problema"], obtido_tipado["tipo_problema"])] += 1
        for campo in ("consultora_codigo", "pedido_numero"):
            if digitos(obtido[campo]) == esperado[campo]:
                acertos[campo] += 1
            else:
                erros.append(f"{campo}={obtido[campo]!r} (expected {esperado[campo]})")
        extraidos = json.loads(obtido["produtos_faltantes"] or "[]")
        for produto in esperado["produtos_faltantes"]:
            produtos_total += 1
            if produto_encontrado(produto, extraidos):
                produtos_ok += 1
            else:
                erros.append(f"product '{produto}' not extracted ({extraidos})")

        wer = None
        if call_id in silver:
            wer = jiwer.wer(norm(roteiro["script"]), norm(silver[call_id]))
            wers.append(wer)
        wer_txt = f"{wer:5.1%}" if wer is not None else "  -  "
        print(f"{call_id:<10} {wer_txt}  {esperado['tipo_problema']:<16} {obtido_tipado['tipo_problema']:<16} {'; '.join(erros)}")

    if not avaliadas:
        raise SystemExit("None of the ground-truth calls is in the gold table.")
    print(f"\n{avaliadas} calls evaluated")
    if wers:
        print(f"Mean transcript WER: {sum(wers) / len(wers):.1%} (worst {max(wers):.1%})")
    for campo in CAMPOS + ("consultora_codigo", "pedido_numero"):
        print(f"  {campo:<18} {acertos[campo]:>3}/{avaliadas}  {acertos[campo] / avaliadas:.0%}")
    if produtos_total:
        print(f"  {'missing products':<18} {produtos_ok:>3}/{produtos_total}  {produtos_ok / produtos_total:.0%}")
    print("\nProblem type (expected -> extracted) when they differ:")
    for (esperado, obtido), n in sorted(confusao.items()):
        if esperado != obtido:
            print(f"  {esperado} -> {obtido}: {n}")


if __name__ == "__main__":
    main()
