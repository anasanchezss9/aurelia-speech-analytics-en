"""Generates the scripts for the synthetic missing-product claim calls, with the ground truth for each one.

Each scenario below fixes what the gold layer should extract (tipo_problema,
suspeita_fraude, reincidencia, sentiment) and describes what happens in the call.
The consultant code and the order number are drawn here, seeded with the call id,
and the LLM writes the dialogue and lists the products that were missing. The
code checks that the numbers and the products really appear in the speech before
accepting the script.

Usage:
    python gerar_roteiros.py --profile field-toolbox
    python gerar_roteiros.py --profile field-toolbox --somente sint_007,sint_012
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from databricks.sdk import WorkspaceClient

AQUI = Path(__file__).parent
SAIDA = AQUI / "roteiros.json"
MODELO = "databricks-claude-sonnet-5-5"

# (id, tipo_problema, suspeita_fraude, reincidencia, sentiment, what happens)
# The descriptions stay in Portuguese: the calls are in Portuguese and the measured results depend on their exact wording.
CENARIOS = [
    ("sint_001", "atraso", False, False, "negativo", "Pedido atrasado há oito dias; as clientes da consultora estão cobrando. O agente abre protocolo com a transportadora."),
    ("sint_002", "atraso", False, True, "negativo", "Terceiro ciclo seguido em que o pedido atrasa. A consultora ameaça parar de vender e pede para falar com supervisor."),
    ("sint_003", "atraso", False, False, "neutro", "Consultora quer só saber a nova previsão de entrega; o rastreio parou numa cidade do caminho. Tom calmo."),
    ("sint_004", "atraso", False, False, "positivo", "Pedido atrasado, mas o agente encontra que sai para entrega amanhã e oferece frete grátis no próximo ciclo. Consultora termina agradecida."),
    ("sint_005", "atraso", False, True, "negativo", "Consultor homem do interior; o pedido sempre atrasa porque a transportadora não entra no bairro. Já reclamou antes."),
    ("sint_006", "atraso", False, False, "neutro", "Pedido de Natal atrasado; consultora pergunta se dá para cancelar parte dos itens. O agente explica a política."),
    ("sint_007", "falta_parcial", False, False, "negativo", "Caixa lacrada chegou sem dois itens; a nota fiscal lista os dois. Provável erro de separação no centro de distribuição."),
    ("sint_008", "falta_parcial", False, True, "negativo", "Faltou um item pela segunda vez no mês; caixa lacrada. A consultora pede reenvio expresso."),
    ("sint_009", "falta_parcial", False, False, "neutro", "Veio um perfume a menos e um hidratante de outra fragrância no lugar do pedido. Caixa intacta."),
    ("sint_010", "falta_parcial", False, False, "positivo", "Faltou um kit de presente; o agente agenda o complemento sem custo para o mesmo dia útil. Consultora satisfeita."),
    ("sint_011", "falta_parcial", False, True, "negativo", "Consultora experiente com dez anos de Aurélia: falta o refil de um item pela terceira vez; ela confere tudo com a nota."),
    ("sint_012", "falta_parcial", False, False, "neutro", "Pedido grande com trinta itens; faltaram três itens pequenos. Caixa lacrada, sem sinal de violação."),
    ("sint_013", "falta_parcial", False, False, "negativo", "Faltou o item mais caro do pedido, mas a caixa estava lacrada e com a fita original. A consultora acha que foi erro na expedição."),
    ("sint_014", "violacao_caixa", False, False, "negativo", "Caixa chegou amassada e com a fita rasgada pela chuva, mas todos os itens estavam lá. Consultora quer registrar a ocorrência."),
    ("sint_015", "violacao_caixa", False, False, "neutro", "Entregador deixou a caixa com o vizinho; a embalagem veio com a fita refeita, mas a consultora conferiu e não falta nada."),
    ("sint_016", "violacao_caixa", False, True, "negativo", "Segunda vez que a caixa chega furada nas laterais; um frasco vazou e manchou os outros, mas nada sumiu."),
    ("sint_017", "violacao_caixa", False, False, "positivo", "Caixa veio com o lacre rompido; o agente orienta a fotografar e já registra a troca da embalagem. Consultora agradece."),
    ("sint_018", "violacao_caixa", False, False, "neutro", "Consultor homem relata caixa com etiqueta de outra pessoa colada por cima; conferiu, está tudo certo."),
    ("sint_019", "fraude_suspeita", True, False, "negativo", "Fita violada e sumiram justamente os três perfumes mais caros do pedido; a consultora diz que alguém abriu a caixa no caminho."),
    ("sint_020", "fraude_suspeita", True, True, "negativo", "Pela segunda vez a caixa chega com o peso certo, mas com produtos trocados por itens baratos; consultora acusa alguém da transportadora."),
    ("sint_021", "fraude_suspeita", True, False, "negativo", "Entregador insistiu para assinar sem conferir; ao abrir, a caixa tinha jornal no lugar de dois kits. Consultora tem foto e vídeo."),
    ("sint_022", "fraude_suspeita", True, False, "neutro", "Consultora calma relata lacre com fita de outra cor e faltam dois perfumes; ela mesma pergunta se pode ter sido desvio. O agente encaminha para a área de segurança."),
    ("sint_023", "fraude_suspeita", True, True, "negativo", "Vizinhas consultoras do mesmo bairro também receberam caixas abertas sem os itens de maior valor. É o terceiro caso dela."),
    ("sint_024", "outro", False, False, "negativo", "Um frasco de óleo trifásico chegou quebrado e vazou dentro da caixa; a caixa estava intacta. Consultora quer a troca."),
    ("sint_025", "outro", False, False, "neutro", "Consultora pergunta por que o boleto do ciclo veio com valor maior; o agente explica a mudança de pontuação."),
    ("sint_026", "outro", False, False, "positivo", "Consultora liga para mudar o endereço de entrega antes do faturamento; o agente resolve na hora."),
    ("sint_027", "outro", False, True, "negativo", "Cobrança em duplicidade no cartão pelo segundo mês; consultora irritada exige estorno."),
    ("sint_028", "outro", False, False, "neutro", "Consultor homem novo pergunta como funciona a promoção do ciclo e o prazo para passar o pedido."),
    ("sint_029", "outro", False, False, "negativo", "Produto chegou com validade vencendo em vinte dias; a consultora não consegue vender e quer trocar."),
    ("sint_030", "outro", False, False, "positivo", "Consultora liga para elogiar a entrega rápida e aproveita para tirar uma dúvida sobre o aplicativo."),
]

# Stays in Portuguese: the calls are in Portuguese and the measured results depend on its exact wording.
SISTEMA = """Você escreve roteiros realistas de ligações telefônicas para a Central de Relacionamento da Aurélia, entre uma consultora (ou consultor) de venda direta e um agente de atendimento, em português do Brasil.

Os roteiros viram áudio sintético para testar um pipeline de speech analytics. Regras:

1. Fala de telefone de verdade: frases curtas, hesitações ("é...", "então", "tipo assim", "hum", "peraí"), repetições, gente que corrige o que disse, agente que confirma dados e lê protocolo. Nada de texto de livro.
2. O agente abre a ligação se apresentando. Depois pede o código da consultora e o número do pedido, que vêm no pedido de cada ligação.
3. NUNCA escreva algarismos no texto das falas. Códigos, pedidos e protocolos são falados dígito a dígito, cada dígito por extenso; valores e quantidades também por extenso ("três perfumes", "cento e vinte reais").
4. Use produtos reais do portfólio Aurélia (Raízes, Diário, Âmbar, Maré, Aura, Tempo, Brilho, Riso, Lua, Memória, Colo, Tez) com nomes como uma consultora falaria.
5. Nomes de pessoas fictícios. Não fale CPF, cartão nem endereço completo.
6. Entre 14 e 22 falas. Marque interrompe=true em duas ou três falas em que a pessoa começa a falar antes da outra terminar.
7. O roteiro tem que sustentar sem ambiguidade a classificação pedida: o tipo do problema, se há suspeita de fraude (desvio intencional de produto), se a consultora diz que o problema já aconteceu antes (reincidência) e o sentimento predominante dela.
8. produtos_faltantes lista, exatamente como foram ditos na ligação, os produtos que não chegaram (lista vazia se nada faltou)."""

# Stays in Portuguese: the calls are in Portuguese and the measured results depend on its exact wording.
DEFINICOES = """Definições do tipo do problema:
- atraso: o pedido não chegou no prazo.
- falta_parcial: a caixa chegou íntegra, mas faltam itens (erro de separação), sem indício de desvio.
- violacao_caixa: a embalagem chegou danificada ou aberta, mas nenhum produto sumiu e ninguém sugere desvio.
- fraude_suspeita: há indício de desvio intencional (caixa violada com itens de valor sumidos, produtos trocados, enchimento no lugar de produto).
- outro: qualquer outro assunto (produto avariado com caixa intacta, cobrança, cadastro, dúvida)."""

ESQUEMA = {
    "type": "object",
    "properties": {
        "consultora_nome": {"type": "string"},
        "consultora_genero": {"type": "string", "enum": ["f", "m"]},
        "agente_nome": {"type": "string"},
        "agente_genero": {"type": "string", "enum": ["f", "m"]},
        "falas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quem": {"type": "string", "enum": ["agente", "consultora"]},
                    "texto": {"type": "string"},
                    "interrompe": {"type": "boolean"},
                },
                "required": ["quem", "texto", "interrompe"],
                "additionalProperties": False,
            },
        },
        "produtos_faltantes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "consultora_nome", "consultora_genero", "agente_nome", "agente_genero",
        "falas", "produtos_faltantes",
    ],
    "additionalProperties": False,
}

DIGITOS = "zero um dois três quatro cinco seis sete oito nove".split()


def _norm(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z ]+", " ", texto).split())


def _falado(numero: str) -> str:
    return _norm(" ".join(DIGITOS[int(d)] for d in numero))


def problemas(roteiro: dict, codigo: str, pedido: str) -> list[str]:
    erros = []
    falas = roteiro["falas"]
    texto = " ".join(f["texto"] for f in falas)
    if not 12 <= len(falas) <= 26:
        erros.append(f"{len(falas)} turns")
    if re.search(r"\d", texto):
        erros.append("digit in the turn text")
    if {f["quem"] for f in falas} != {"agente", "consultora"}:
        erros.append("one of the two speakers is missing")
    normal = _norm(texto)
    for nome, valor in (("consultant code", codigo), ("order", pedido)):
        if _falado(valor) not in normal:
            erros.append(f"{nome} {valor} is not spoken digit by digit")
    for produto in roteiro["produtos_faltantes"]:
        palavras = [p for p in _norm(produto).split() if len(p) > 3]
        if palavras and not any(p in normal for p in palavras):
            erros.append(f"product '{produto}' does not appear in the speech")
    return erros


def pedir(cliente: WorkspaceClient, cenario: tuple, tentativas: int = 3) -> dict:
    call_id, tipo, fraude, reinc, sentimento, descricao = cenario
    sorteio = random.Random(call_id)
    codigo = str(sorteio.randint(10000, 99999))
    numero_pedido = str(sorteio.randint(100000, 999999))
    # Stays in Portuguese: the calls are in Portuguese and the measured results depend on its exact wording.
    pedido = (
        f"{DEFINICOES}\n\n"
        f"Escreva a ligação {call_id}.\n"
        f"- O que acontece: {descricao}\n"
        f"- Código da consultora: {codigo}. Número do pedido: {numero_pedido}. Os dois falados dígito a dígito.\n"
        f"- tipo_problema: {tipo}\n"
        f"- suspeita_fraude: {'sim' if fraude else 'não'}\n"
        f"- reincidência (a consultora diz que já aconteceu antes): {'sim' if reinc else 'não'}\n"
        f"- sentimento predominante da consultora: {sentimento}\n"
        f"Se a descrição não disser o gênero, a consultora é mulher.\n\n"
        f"Responda só com um objeto JSON neste JSON Schema, sem texto antes ou depois:\n"
        f"{json.dumps(ESQUEMA, ensure_ascii=False)}"
    )
    url = f"{cliente.config.host}/serving-endpoints/{MODELO}/invocations"
    # Claude 5.x on the serving route rejects response_format with json_schema and
    # the temperature parameter; the JSON comes in the text and the validation
    # below rejects anything malformed.
    corpo = {
        "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": pedido}],
        "max_tokens": 6000,
    }
    ultimo = None
    for _ in range(tentativas):
        resposta = requests.post(url, headers=cliente.config.authenticate(), json=corpo, timeout=300)
        if not resposta.ok:
            raise RuntimeError(f"HTTP {resposta.status_code}: {resposta.text[:400]}")
        conteudo = resposta.json()["choices"][0]["message"]["content"]
        if isinstance(conteudo, list):
            conteudo = "".join(p.get("text", "") for p in conteudo)
        try:
            roteiro = json.loads(conteudo[conteudo.index("{") : conteudo.rindex("}") + 1])
            faltando = [k for k in ESQUEMA["required"] if k not in roteiro]
            ultimo = [f"missing fields {faltando}"] if faltando else problemas(roteiro, codigo, numero_pedido)
        except (ValueError, KeyError, TypeError) as exc:
            ultimo = [f"invalid JSON: {exc}"]
        if not ultimo:
            return {
                "call_id": call_id,
                "filename": f"{call_id}.wav",
                "scenario": descricao,
                **{k: roteiro[k] for k in ("consultora_nome", "consultora_genero", "agente_nome", "agente_genero")},
                "gabarito": {
                    "tipo_problema": tipo,
                    "suspeita_fraude": fraude,
                    "reincidencia": reinc,
                    "sentiment": sentimento,
                    "consultora_codigo": codigo,
                    "pedido_numero": numero_pedido,
                    "produtos_faltantes": roteiro["produtos_faltantes"],
                },
                "falas": roteiro["falas"],
                "script": " ".join(f["texto"] for f in roteiro["falas"]),
            }
    raise RuntimeError(f"{call_id}: script rejected after {tentativas} attempts: {ultimo}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None, help="Databricks CLI profile")
    parser.add_argument("--somente", default="", help="comma-separated ids to regenerate")
    args = parser.parse_args()

    cliente = WorkspaceClient(profile=args.profile)
    existentes = {r["call_id"]: r for r in json.loads(SAIDA.read_text())} if SAIDA.exists() else {}
    alvo = set(filter(None, args.somente.split(",")))
    pendentes = [c for c in CENARIOS if (c[0] in alvo) or (not alvo and c[0] not in existentes)]
    print(f"{len(pendentes)} scripts to generate with {MODELO}")

    falhas = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futuros = {pool.submit(pedir, cliente, c): c[0] for c in pendentes}
        for futuro in as_completed(futuros):
            try:
                roteiro = futuro.result()
                existentes[roteiro["call_id"]] = roteiro
                print(f"ok    {roteiro['call_id']}  {len(roteiro['falas'])} turns")
            except Exception as exc:
                falhas.append(futuros[futuro])
                print(f"fail  {futuros[futuro]}  {exc}")

    ordem = [c[0] for c in CENARIOS]
    roteiros = sorted(existentes.values(), key=lambda r: ordem.index(r["call_id"]))
    SAIDA.write_text(json.dumps(roteiros, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(roteiros)} scripts in {SAIDA}")
    if falhas:
        sys.exit(f"Run again with --somente {','.join(sorted(falhas))}")


if __name__ == "__main__":
    main()
