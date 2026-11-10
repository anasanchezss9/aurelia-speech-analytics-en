"""Synthetic operational world of Aurélia Cosméticos, a fictitious direct-selling company.

Generates the tables an order system and a customer-service system would have —
consultants, products, orders, deliveries, missing-product claims, investigations
and calls — and writes them as parquet, one folder per table, for the pipeline to
read with `read_files`.

What makes the data useful for training a model are the planted causes of missing
products, each with its own signature in the box weight, which the scale records
at two points (shipping at the DC and the last-mile hub):

    picking error      → the DC ships lighter than the expected weight; hub = shipped
    in-transit theft   → shipped = expected; the hub weighs less than shipped
    fraud              → all three weights match and the proof of delivery is signed

Two waves make the story concrete and mislead anyone who looks only at the claim
rate: a route with real theft (TransLeste in RJ, from cycle 14) and a DC with
picking errors during the WMS migration (CD Contagem, cycles 9 to 11). In both,
the claim rate rises without any fraud.

Labels exist only where there was an investigation, and investigations are not
random: the policy prioritizes expensive claims and repeat-offender consultants.
That is the selection bias a model trained on this data inherits, and it has to
be stated.

The 30 calls in `sintetico/roteiros.json` become recent, still-open claims, with
consultant, order, carrier and products consistent with each script's ground
truth.

Usage:
    python gerar_dados.py --saida ./saida
    python gerar_dados.py --saida ./saida --consultoras 5000 --semente 7
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_INICIO = date(2025, 5, 5)
CICLO_DIAS = 21
N_CICLOS = 24
DATA_CORTE = DATA_INICIO + timedelta(days=CICLO_DIAS * N_CICLOS)
JANELA_ABERTA_DIAS = 21

ROTA_FURTO = ("TransLeste", "RJ", 14)
CD_ERRO = ("CD Contagem", 9, 11)

CDS = {
    "CD Jundiaí": "SP",
    "CD Contagem": "MG",
    "CD Recife": "PE",
    "CD Canoas": "RS",
    "CD Goiânia": "GO",
}

# uf: (weight in the consultant base, region, DC, base lead time in days, [(carrier, weight)])
UFS = {
    "SP": (22, "Sudeste", "CD Jundiaí", 3, [("Via Rápida", 0.55), ("LogMais", 0.45)]),
    "RJ": (9, "Sudeste", "CD Jundiaí", 4, [("TransLeste", 0.6), ("Via Rápida", 0.4)]),
    "MG": (10, "Sudeste", "CD Contagem", 3, [("TransLeste", 0.5), ("LogMais", 0.5)]),
    "ES": (2, "Sudeste", "CD Contagem", 4, [("TransLeste", 1.0)]),
    "BA": (7, "Nordeste", "CD Recife", 5, [("Expresso Norte", 0.6), ("Correio Ágil", 0.4)]),
    "PE": (5, "Nordeste", "CD Recife", 3, [("Expresso Norte", 0.7), ("Correio Ágil", 0.3)]),
    "CE": (5, "Nordeste", "CD Recife", 4, [("Expresso Norte", 1.0)]),
    "MA": (3, "Nordeste", "CD Recife", 6, [("Correio Ágil", 1.0)]),
    "PB": (2, "Nordeste", "CD Recife", 4, [("Expresso Norte", 1.0)]),
    "RN": (2, "Nordeste", "CD Recife", 4, [("Expresso Norte", 1.0)]),
    "PA": (4, "Norte", "CD Recife", 8, [("Correio Ágil", 1.0)]),
    "AM": (2, "Norte", "CD Recife", 10, [("Correio Ágil", 1.0)]),
    "RS": (6, "Sul", "CD Canoas", 3, [("RotaSul", 1.0)]),
    "PR": (6, "Sul", "CD Canoas", 4, [("RotaSul", 0.6), ("LogMais", 0.4)]),
    "SC": (4, "Sul", "CD Canoas", 4, [("RotaSul", 1.0)]),
    "GO": (4, "Centro-Oeste", "CD Goiânia", 3, [("LogMais", 1.0)]),
    "DF": (2, "Centro-Oeste", "CD Goiânia", 3, [("LogMais", 1.0)]),
    "MT": (2, "Centro-Oeste", "CD Goiânia", 6, [("LogMais", 0.5), ("Correio Ágil", 0.5)]),
}

# name, line, category, price (R$), weight (kg), relative popularity
PRODUTOS = [
    ("Âmbar Exclusivo", "Âmbar", "perfumaria", 239.9, 0.42, 3),
    ("Âmbar Oud", "Âmbar", "perfumaria", 259.9, 0.42, 2),
    ("Âmbar feminino", "Âmbar", "perfumaria", 199.9, 0.40, 3),
    ("Âmbar masculino", "Âmbar", "perfumaria", 199.9, 0.40, 3),
    ("Âmbar Intenso", "Âmbar", "perfumaria", 219.9, 0.42, 1),
    ("Maré masculino", "Maré", "perfumaria", 149.9, 0.38, 5),
    ("Maré Oceano", "Maré", "perfumaria", 159.9, 0.38, 3),
    ("Maré Aventura", "Maré", "perfumaria", 154.9, 0.38, 2),
    ("Maré feminino", "Maré", "perfumaria", 149.9, 0.38, 3),
    ("Lua feminino", "Lua", "perfumaria", 169.9, 0.36, 3),
    ("Lua Radiante", "Lua", "perfumaria", 189.9, 0.36, 2),
    ("Memória masculino", "Memória", "perfumaria", 129.9, 0.36, 2),
    ("Memória feminino", "Memória", "perfumaria", 129.9, 0.36, 2),
    ("Riso feminino", "Riso", "perfumaria", 99.9, 0.30, 3),
    ("Riso masculino", "Riso", "perfumaria", 99.9, 0.30, 2),
    ("body splash Riso", "Riso", "perfumaria", 64.9, 0.28, 4),
    ("kit de presente Raízes castanha", "Raízes", "presentes", 149.9, 0.95, 2),
    ("creme de mãos Raízes", "Raízes", "corpo", 34.9, 0.10, 6),
    ("óleo trifásico Raízes maracujá", "Raízes", "corpo", 59.9, 0.22, 4),
    ("sabonete Raízes açaí", "Raízes", "corpo", 22.9, 0.12, 6),
    ("polpa hidratante Raízes castanha", "Raízes", "corpo", 54.9, 0.45, 4),
    ("Diário creme de ameixa", "Diário", "corpo", 44.9, 0.42, 7),
    ("hidratante Diário de macadâmia", "Diário", "corpo", 44.9, 0.42, 7),
    ("refil do Diário de nozes pecã e cacau", "Diário", "corpo", 34.9, 0.40, 5),
    ("sabonete em barra Diário", "Diário", "corpo", 19.9, 0.48, 8),
    ("desodorante Diário", "Diário", "corpo", 24.9, 0.15, 7),
    ("hidratante Diário de cereja", "Diário", "corpo", 44.9, 0.42, 5),
    ("batom Aura", "Aura", "maquiagem", 39.9, 0.05, 7),
    ("Aura Artisan", "Aura", "maquiagem", 79.9, 0.06, 3),
    ("base Aura", "Aura", "maquiagem", 69.9, 0.08, 4),
    ("rímel Aura", "Aura", "maquiagem", 49.9, 0.04, 4),
    ("pó compacto Aura", "Aura", "maquiagem", 59.9, 0.06, 3),
    ("creme Tempo", "Tempo", "rosto", 119.9, 0.12, 4),
    ("sérum Tempo", "Tempo", "rosto", 149.9, 0.08, 3),
    ("creme Tempo noite", "Tempo", "rosto", 129.9, 0.12, 3),
    ("protetor Tempo", "Tempo", "rosto", 89.9, 0.10, 3),
    ("kit Brilho", "Brilho", "cabelos", 109.9, 0.90, 2),
    ("shampoo Brilho", "Brilho", "cabelos", 34.9, 0.32, 6),
    ("condicionador Brilho", "Brilho", "cabelos", 36.9, 0.32, 5),
    ("máscara Brilho", "Brilho", "cabelos", 49.9, 0.28, 3),
    ("sabonete Colo", "Colo", "infantil", 24.9, 0.20, 3),
    ("colônia Colo", "Colo", "infantil", 69.9, 0.22, 2),
    ("hidratante Colo", "Colo", "infantil", 39.9, 0.22, 3),
    ("sabonete facial Tez", "Tez", "rosto", 42.9, 0.16, 4),
    ("hidratante facial Tez", "Tez", "rosto", 64.9, 0.08, 3),
]

# How the products are spoken in the scripts → (catalog name, quantity)
PRODUTOS_ROTEIRO = {
    "Maré masculino": ("Maré masculino", 1),
    "um Maré masculino": ("Maré masculino", 1),
    "dois perfumes Maré": ("Maré masculino", 2),
    "Maré Oceano": ("Maré Oceano", 1),
    "Diário creme de ameixa": ("Diário creme de ameixa", 1),
    "hidratante Diário de macadâmia": ("hidratante Diário de macadâmia", 1),
    "refil do Diário de nozes pecã e cacau": ("refil do Diário de nozes pecã e cacau", 1),
    "sabonete em barra Diário": ("sabonete em barra Diário", 1),
    "perfume Âmbar Oud": ("Âmbar Oud", 1),
    "Âmbar Exclusivo": ("Âmbar Exclusivo", 1),
    "perfume Âmbar": ("Âmbar feminino", 1),
    "um Âmbar feminino": ("Âmbar feminino", 1),
    "um Âmbar": ("Âmbar Exclusivo", 1),
    "kit de presente Raízes castanha": ("kit de presente Raízes castanha", 1),
    "kit Raízes castanha": ("kit de presente Raízes castanha", 1),
    "creme de mãos Raízes": ("creme de mãos Raízes", 1),
    "batom Aura": ("batom Aura", 1),
    "Aura Artisan": ("Aura Artisan", 1),
    "creme Tempo": ("creme Tempo", 1),
    "kit Brilho": ("kit Brilho", 1),
}

MOTIVO_POR_TIPO = {
    "atraso": "atraso",
    "falta_parcial": "falta_parcial",
    "violacao_caixa": "caixa_violada",
    "fraude_suspeita": None,
    "outro": "outro",
}

PESO_CAIXA_KG = 0.25


def inicio_ciclo(c: np.ndarray | int) -> np.ndarray:
    return np.datetime64(DATA_INICIO) + np.asarray(c) * CICLO_DIAS


def gerar_produtos() -> pd.DataFrame:
    df = pd.DataFrame(PRODUTOS, columns=["nome", "linha", "categoria", "preco", "peso_kg", "popularidade"])
    df.insert(0, "sku", [f"AUR-{10001 + i}" for i in range(len(df))])
    return df


def gerar_consultoras(rng, n: int, roteiros: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    codigos_roteiro = [r["gabarito"]["consultora_codigo"] for r in roteiros]
    livres = np.setdiff1d(np.arange(10000, 100000), np.array(codigos_roteiro, dtype=int))
    codigos = np.concatenate([np.array(codigos_roteiro, dtype=int), rng.choice(livres, n - len(roteiros), replace=False)])

    ufs = list(UFS)
    pesos_uf = np.array([UFS[u][0] for u in ufs], dtype=float)
    uf = rng.choice(ufs, n, p=pesos_uf / pesos_uf.sum())

    meses = np.clip(rng.exponential(30, n), 1, 180).astype(int)
    data_cadastro = np.datetime64(DATA_CORTE) - (meses * 30.4).astype("timedelta64[D]")

    produtividade = rng.lognormal(0, 0.45, n)
    nivel = pd.cut(produtividade, [0, 0.7, 1.0, 1.4, 2.0, np.inf],
                   labels=["semente", "bronze", "prata", "ouro", "diamante"]).astype(str)

    p_fraude = np.where(meses < 12, 0.022, 0.006)
    perfil_fraude = rng.random(n) < p_fraude

    for i, r in enumerate(roteiros):
        tipo = r["gabarito"]["tipo_problema"]
        perfil_fraude[i] = tipo == "fraude_suspeita"
        if tipo == "violacao_caixa":
            uf[i] = "RJ"
        if tipo == "fraude_suspeita":
            meses[i] = int(rng.integers(4, 11))
            data_cadastro[i] = np.datetime64(DATA_CORTE) - np.timedelta64(int(meses[i] * 30.4), "D")

    df = pd.DataFrame({
        "consultora_codigo": codigos.astype(str),
        "uf": uf,
        "regiao": [UFS[u][1] for u in uf],
        "data_cadastro": pd.to_datetime(data_cadastro).date,
        "nivel": nivel,
        "tem_app": rng.random(n) < 0.72,
        "_produtividade": produtividade,
    })
    return df, perfil_fraude


def gerar_pedidos(rng, consultoras: pd.DataFrame, perfil_fraude: np.ndarray, produtos: pd.DataFrame,
                  reservados: set[str]):
    n = len(consultoras)
    ciclo_cadastro = np.floor(
        (pd.to_datetime(consultoras["data_cadastro"]) - pd.Timestamp(DATA_INICIO)).dt.days / CICLO_DIAS
    ).to_numpy()
    p_ped = np.clip(0.30 + 0.25 * consultoras["_produtividade"].to_numpy(), 0.2, 0.92)
    ciclos = np.arange(N_CICLOS)
    pede = (rng.random((n, N_CICLOS)) < p_ped[:, None]) & (ciclos[None, :] >= ciclo_cadastro[:, None])
    idx_c, ciclo = np.nonzero(pede)
    m = len(idx_c)

    livres = np.setdiff1d(np.arange(100000, 1000000), np.array(sorted(reservados), dtype=int))
    numeros = rng.choice(livres, m, replace=False).astype(str)

    uf = consultoras["uf"].to_numpy()[idx_c]
    transp = np.empty(m, dtype=object)
    for u, (_, _, _, _, opcoes) in UFS.items():
        sel = np.nonzero(uf == u)[0]
        nomes, pesos = zip(*opcoes)
        transp[sel] = rng.choice(nomes, len(sel), p=np.array(pesos) / sum(pesos))

    data_pedido = inicio_ciclo(ciclo) + rng.integers(0, 15, m).astype("timedelta64[D]")
    pedidos = pd.DataFrame({
        "pedido_numero": numeros,
        "consultora_codigo": consultoras["consultora_codigo"].to_numpy()[idx_c],
        "ciclo": ciclo + 1,
        "data_pedido": data_pedido,
        "uf": uf,
        "cd": [UFS[u][2] for u in uf],
        "transportadora": transp,
        "canal": rng.choice(["app", "site", "revista"], m, p=[0.62, 0.23, 0.15]),
        "_fraude": perfil_fraude[idx_c],
    })

    n_itens = 2 + rng.poisson(np.where(pedidos["_fraude"], 6.5, 5.0))
    linha_pedido = np.repeat(np.arange(m), n_itens)
    pop = produtos["popularidade"].to_numpy(dtype=float)
    pop_fraude = pop * np.where(produtos["categoria"] == "perfumaria", 2.5, 1.0)
    eh_fraude = pedidos["_fraude"].to_numpy()[linha_pedido]
    sku_idx = np.where(
        eh_fraude,
        rng.choice(len(produtos), len(linha_pedido), p=pop_fraude / pop_fraude.sum()),
        rng.choice(len(produtos), len(linha_pedido), p=pop / pop.sum()),
    )
    qtd = rng.choice([1, 2, 3], len(linha_pedido), p=[0.84, 0.13, 0.03])
    itens = (pd.DataFrame({"pedido_numero": numeros[linha_pedido], "sku": produtos["sku"].to_numpy()[sku_idx],
                           "quantidade": qtd})
             .groupby(["pedido_numero", "sku"], as_index=False)["quantidade"].sum())
    itens = itens.merge(produtos[["sku", "preco", "peso_kg", "categoria"]], on="sku")
    itens = itens.rename(columns={"preco": "preco_unitario"})
    return pedidos, itens


def escolher_itens(rng, itens_do_pedido: pd.DataFrame, modo: str) -> list[tuple[str, int]]:
    """Items that go missing: cheap and random in a picking error, expensive in theft and fraud."""
    if itens_do_pedido.empty:
        return []
    k = int(rng.integers(1, 3)) if modo == "separacao" else int(rng.integers(1, 4))
    k = min(k, len(itens_do_pedido))
    if modo == "separacao":
        w = 1 / itens_do_pedido["preco_unitario"].to_numpy()
        escolha = rng.choice(len(itens_do_pedido), k, replace=False, p=w / w.sum())
    else:
        w = itens_do_pedido["preco_unitario"].to_numpy() ** 2
        escolha = rng.choice(len(itens_do_pedido), k, replace=False, p=w / w.sum())
    sel = itens_do_pedido.iloc[escolha]
    return [(s, int(rng.integers(1, q + 1))) for s, q in zip(sel["sku"], sel["quantidade"])]


def gerar_entregas(rng, pedidos: pd.DataFrame, itens: pd.DataFrame):
    m = len(pedidos)
    uf = pedidos["uf"].to_numpy()
    ciclo = pedidos["ciclo"].to_numpy() - 1
    transp = pedidos["transportadora"].to_numpy()
    cd = pedidos["cd"].to_numpy()

    envio = pedidos["data_pedido"].to_numpy() + rng.integers(1, 4, m).astype("timedelta64[D]")
    prazo = np.array([UFS[u][3] for u in uf]) + rng.integers(1, 4, m)
    prevista = envio + prazo.astype("timedelta64[D]")

    p_atraso = np.where(np.isin(transp, ["Correio Ágil", "TransLeste"]), 0.12, 0.07)
    atrasado = rng.random(m) < p_atraso
    dias_atraso = np.where(atrasado, rng.geometric(0.18, m), 0)

    rota_furto = (transp == ROTA_FURTO[0]) & (uf == ROTA_FURTO[1]) & (ciclo + 1 >= ROTA_FURTO[2])
    p_extravio = np.where(rota_furto, 0.015, 0.004)
    extraviado = rng.random(m) < p_extravio
    furtado = (~extraviado) & (rng.random(m) < np.where(rota_furto, 0.05, 0.003))
    onda_cd = (cd == CD_ERRO[0]) & (ciclo + 1 >= CD_ERRO[1]) & (ciclo + 1 <= CD_ERRO[2])
    erro_sep = rng.random(m) < np.where(onda_cd, 0.04, 0.009)

    itens_idx = itens.set_index("pedido_numero").sort_index()
    faltas: dict[str, tuple[str, list[tuple[str, int]]]] = {}
    peso_por_sku = dict(zip(itens["sku"], itens["peso_kg"]))
    for i in np.nonzero(erro_sep | furtado)[0]:
        num = pedidos["pedido_numero"].iat[i]
        do_pedido = itens_idx.loc[[num]]
        if erro_sep[i]:
            faltas[num] = ("erro_separacao", escolher_itens(rng, do_pedido, "separacao"))
        else:
            faltas[num] = ("furto_transporte", escolher_itens(rng, do_pedido, "furto"))

    peso_teorico = (itens.assign(p=itens["quantidade"] * itens["peso_kg"])
                    .groupby("pedido_numero")["p"].sum()
                    .reindex(pedidos["pedido_numero"]).to_numpy() + PESO_CAIXA_KG)
    falta_sep = np.zeros(m)
    falta_furto = np.zeros(m)
    pos = {n: i for i, n in enumerate(pedidos["pedido_numero"])}
    for num, (mec, lista) in faltas.items():
        kg = sum(peso_por_sku[s] * q for s, q in lista)
        if mec == "erro_separacao":
            falta_sep[pos[num]] = kg
        else:
            falta_furto[pos[num]] = kg

    ruido = lambda: 1 + rng.normal(0, 0.015, m) + np.where(rng.random(m) < 0.05, rng.normal(0, 0.10, m), 0)
    peso_expedido = (peso_teorico - falta_sep) * ruido()
    peso_hub = (peso_expedido - falta_furto) * ruido()
    peso_hub[(rng.random(m) < 0.45) | extraviado] = np.nan

    data_entrega = np.where(extraviado, np.datetime64("NaT"), prevista + dias_atraso.astype("timedelta64[D]"))
    assinado = (~extraviado) & (rng.random(m) < 0.92)

    entregas = pd.DataFrame({
        "pedido_numero": pedidos["pedido_numero"],
        "data_envio": envio,
        "data_prevista": prevista,
        "data_entrega": data_entrega,
        "status_entrega": np.where(extraviado, "extraviado", np.where(dias_atraso > 0, "entregue_com_atraso", "entregue")),
        "peso_teorico_kg": peso_teorico.round(3),
        "peso_expedido_kg": peso_expedido.round(3),
        "peso_hub_kg": np.round(peso_hub, 3),
        "comprovante_assinado": assinado,
    })
    entregas = entregas[pd.to_datetime(entregas["data_envio"]) < pd.Timestamp(DATA_CORTE)].copy()
    em_transito = pd.to_datetime(entregas["data_entrega"]) >= pd.Timestamp(DATA_CORTE)
    entregas.loc[em_transito, "data_entrega"] = pd.NaT
    entregas.loc[em_transito, "status_entrega"] = "em_transito"
    entregas.loc[em_transito, "comprovante_assinado"] = False
    estado = pd.DataFrame({"pedido_numero": pedidos["pedido_numero"], "extraviado": extraviado,
                           "dias_atraso": dias_atraso})
    return entregas, faltas, estado


def gerar_chamados(rng, pedidos, itens, entregas, faltas, estado):
    base = (pedidos[["pedido_numero", "consultora_codigo", "_fraude"]]
            .rename(columns={"_fraude": "perfil_fraude"})
            .merge(entregas, on="pedido_numero")
            .merge(estado, on="pedido_numero"))
    itens_idx = itens.set_index("pedido_numero").sort_index()
    registros, itens_ch = [], []

    def abrir(row, motivo, mecanismo, lista, dias_min, dias_max, referencia):
        abertura = pd.Timestamp(referencia) + pd.Timedelta(days=int(rng.integers(dias_min, dias_max + 1)))
        if abertura >= pd.Timestamp(DATA_CORTE):
            return
        canal = rng.choice(["telefone", "app", "chat"], p=[0.70, 0.18, 0.12] if mecanismo == "fraude" else [0.55, 0.30, 0.15])
        registros.append({"pedido_numero": row.pedido_numero, "consultora_codigo": row.consultora_codigo,
                          "data_abertura": abertura, "canal": canal, "motivo_declarado": motivo,
                          "_mecanismo": mecanismo})
        for sku, q in lista:
            itens_ch.append({"pedido_numero": row.pedido_numero, "data_abertura": abertura, "sku": sku,
                             "quantidade": q})

    sorteio = rng.random((len(base), 4))
    for k, row in enumerate(base.itertuples(index=False)):
        entregue = pd.notna(row.data_entrega)
        mec_falta = faltas.get(row.pedido_numero)
        if mec_falta and entregue:
            mec, lista = mec_falta
            if mec == "erro_separacao" and sorteio[k, 0] < 0.85:
                abrir(row, "falta_parcial", mec, lista, 1, 10, row.data_entrega)
            elif mec == "furto_transporte" and sorteio[k, 0] < 0.80:
                motivo = "caixa_violada" if sorteio[k, 1] < 0.7 else "falta_parcial"
                abrir(row, motivo, mec, lista, 0, 5, row.data_entrega)
            continue
        if row.extraviado and sorteio[k, 0] < 0.95:
            abrir(row, "nao_recebido", "extravio", [], 3, 12, row.data_prevista)
            continue
        if row.dias_atraso > 3 and sorteio[k, 0] < 0.35:
            abrir(row, "atraso", "atraso", [], 3, min(int(row.dias_atraso), 9), row.data_prevista)
            continue
        if row.perfil_fraude and entregue and row.comprovante_assinado and sorteio[k, 2] < 0.25:
            u = sorteio[k, 3]
            motivo = "falta_parcial" if u < 0.55 else ("nao_recebido" if u < 0.80 else "caixa_violada")
            lista = escolher_itens(rng, itens_idx.loc[[row.pedido_numero]], "fraude")
            if rng.random() < 0.4:
                lista = escolher_itens(rng, itens_idx.loc[[row.pedido_numero]], "separacao")[:1]
            if motivo == "nao_recebido":
                lista = list(itens_idx.loc[[row.pedido_numero], ["sku", "quantidade"]].itertuples(index=False, name=None))
            abrir(row, motivo, "fraude", lista, 0, 5, row.data_entrega)
            continue
        if entregue and not row.perfil_fraude and sorteio[k, 2] < 0.003:
            lista = escolher_itens(rng, itens_idx.loc[[row.pedido_numero]], "fraude")[:1]
            abrir(row, "falta_parcial", "fraude_oportunista", lista, 1, 7, row.data_entrega)
            continue
        if entregue and sorteio[k, 1] < 0.012:
            motivo = rng.choice(["produto_avariado", "outro"], p=[0.6, 0.4])
            lista = escolher_itens(rng, itens_idx.loc[[row.pedido_numero]], "separacao")[:1] if motivo == "produto_avariado" else []
            abrir(row, motivo, "outro", lista, 1, 15, row.data_entrega)

    return pd.DataFrame(registros), pd.DataFrame(itens_ch)


def injetar_roteiros(rng, roteiros, consultoras, produtos, pedidos, itens, entregas, chamados, itens_ch):
    """Each script becomes a recent order with the open claim the call reports."""
    sku_por_nome = dict(zip(produtos["nome"], produtos["sku"]))
    prod = produtos.set_index("sku")
    cons = consultoras.set_index("consultora_codigo")
    novos_ped, novos_itens, novas_ent, novos_ch, novos_ich, ligacoes = [], [], [], [], [], []

    for r in roteiros:
        g = r["gabarito"]
        codigo, numero, tipo = g["consultora_codigo"], g["pedido_numero"], g["tipo_problema"]
        uf = cons.at[codigo, "uf"]
        transp = ROTA_FURTO[0] if tipo == "violacao_caixa" else UFS[uf][4][-1][0]
        ciclo = N_CICLOS - 1
        data_pedido = pd.Timestamp(inicio_ciclo(ciclo)) + pd.Timedelta(days=int(rng.integers(0, 4)))

        faltantes = [PRODUTOS_ROTEIRO[p] for p in g["produtos_faltantes"]]
        extras = rng.choice(produtos["sku"], int(rng.integers(3, 8)), replace=False)
        cesta = {sku_por_nome[n]: q for n, q in faltantes}
        for s in extras:
            cesta.setdefault(s, 1)

        teorico = sum(prod.at[s, "peso_kg"] * q for s, q in cesta.items()) + PESO_CAIXA_KG
        falta_kg = sum(prod.at[sku_por_nome[n], "peso_kg"] * q for n, q in faltantes)
        envio = data_pedido + pd.Timedelta(days=2)
        prevista = envio + pd.Timedelta(days=UFS[uf][3] + 2)
        atraso = int(rng.integers(6, 11)) if tipo == "atraso" else 0
        entregue = None if tipo == "atraso" else prevista + pd.Timedelta(days=atraso)
        expedido = teorico - (falta_kg if tipo == "falta_parcial" else 0)
        hub = expedido - (falta_kg if tipo == "violacao_caixa" else 0)

        novos_ped.append({"pedido_numero": numero, "consultora_codigo": codigo, "ciclo": ciclo + 1,
                          "data_pedido": data_pedido, "uf": uf, "cd": UFS[uf][2], "transportadora": transp,
                          "canal": "app", "_fraude": tipo == "fraude_suspeita"})
        for s, q in cesta.items():
            novos_itens.append({"pedido_numero": numero, "sku": s, "quantidade": q,
                                "preco_unitario": prod.at[s, "preco"], "peso_kg": prod.at[s, "peso_kg"],
                                "categoria": prod.at[s, "categoria"]})
        novas_ent.append({"pedido_numero": numero, "data_envio": envio, "data_prevista": prevista,
                          "data_entrega": entregue,
                          "status_entrega": "em_transito" if tipo == "atraso" else "entregue",
                          "peso_teorico_kg": round(teorico, 3), "peso_expedido_kg": round(expedido, 3),
                          "peso_hub_kg": round(hub, 3), "comprovante_assinado": tipo != "atraso"})

        motivo = MOTIVO_POR_TIPO[tipo] or ("nao_recebido" if rng.random() < 0.4 else "falta_parcial")
        abertura = min(pd.Timestamp(DATA_CORTE) - pd.Timedelta(days=1),
                       (entregue or prevista) + pd.Timedelta(days=atraso or 1))
        mecanismo = {"atraso": "atraso", "falta_parcial": "erro_separacao", "violacao_caixa": "furto_transporte",
                     "fraude_suspeita": "fraude", "outro": "outro"}[tipo]
        novos_ch.append({"pedido_numero": numero, "consultora_codigo": codigo, "data_abertura": abertura,
                         "canal": "telefone", "motivo_declarado": motivo, "_mecanismo": mecanismo,
                         "_call_id": r["call_id"]})
        for n, q in faltantes:
            novos_ich.append({"pedido_numero": numero, "data_abertura": abertura, "sku": sku_por_nome[n],
                              "quantidade": q})

        if g["reincidencia"]:
            anterior = pedidos[(pedidos["consultora_codigo"] == codigo) & (pedidos["ciclo"] <= N_CICLOS - 6)]
            if not anterior.empty:
                p = anterior.iloc[-1]
                novos_ch.append({"pedido_numero": p["pedido_numero"], "consultora_codigo": codigo,
                                 "data_abertura": pd.Timestamp(p["data_pedido"]) + pd.Timedelta(days=12),
                                 "canal": "telefone", "motivo_declarado": motivo,
                                 "_mecanismo": "fraude" if tipo == "fraude_suspeita" else mecanismo,
                                 "_call_id": None})

    pedidos = pd.concat([pedidos, pd.DataFrame(novos_ped)], ignore_index=True)
    itens = pd.concat([itens, pd.DataFrame(novos_itens)], ignore_index=True)
    entregas = pd.concat([entregas, pd.DataFrame(novas_ent)], ignore_index=True)
    chamados = pd.concat([chamados.assign(_call_id=None), pd.DataFrame(novos_ch)], ignore_index=True)
    itens_ch = pd.concat([itens_ch, pd.DataFrame(novos_ich)], ignore_index=True)
    return pedidos, itens, entregas, chamados, itens_ch


def finalizar_chamados(rng, chamados, itens_ch, itens):
    chamados = chamados.sort_values("data_abertura", ignore_index=True)
    chamados.insert(0, "chamado_id", [f"CH-{1000001 + i}" for i in range(len(chamados))])
    itens_ch = itens_ch.merge(chamados[["chamado_id", "pedido_numero", "data_abertura"]],
                              on=["pedido_numero", "data_abertura"])
    precos = itens.drop_duplicates("sku").set_index("sku")["preco_unitario"]
    itens_ch["valor"] = itens_ch["sku"].map(precos) * itens_ch["quantidade"]
    valor = itens_ch.groupby("chamado_id")["valor"].sum()
    chamados["valor_reclamado"] = chamados["chamado_id"].map(valor).fillna(0).round(2)

    abertura = pd.to_datetime(chamados["data_abertura"])
    fechado = abertura <= pd.Timestamp(DATA_CORTE) - pd.Timedelta(days=JANELA_ABERTA_DIAS)
    anteriores = (chamados.assign(um=1).set_index("data_abertura")
                  .groupby("consultora_codigo")["um"].transform(lambda s: s.rolling("180D").sum() - 1))
    p_inv = (0.15 + 0.40 * (chamados["valor_reclamado"] > 200) + 0.25 * (anteriores.to_numpy() >= 2)
             + 0.30 * (chamados["motivo_declarado"] == "caixa_violada")).clip(upper=0.9)
    investigado = fechado & (rng.random(len(chamados)) < p_inv)

    mec = chamados["_mecanismo"]
    eh_fraude = mec.isin(["fraude", "fraude_oportunista"])
    resultado = np.select(
        [mec.isin(["erro_separacao"]), mec.isin(["furto_transporte", "extravio"]), mec.eq("atraso")],
        ["procedente_separacao", "procedente_transporte", "procedente_atraso"], "procedente_outro").astype(object)
    u = rng.random(len(chamados))
    resultado = np.where(eh_fraude & (u < 0.85), "fraude_confirmada",
                         np.where(eh_fraude, "improcedente", resultado))
    resultado = np.where(~eh_fraude & (u < 0.01), "fraude_confirmada", resultado)
    resultado = np.where(~eh_fraude & (u > 0.97), "improcedente", resultado)

    dias = rng.integers(5, 25, len(chamados))
    investigacoes = pd.DataFrame({
        "chamado_id": chamados["chamado_id"],
        "data_conclusao": (abertura + pd.to_timedelta(dias, "D")).dt.date,
        "resultado": resultado,
        "analista_id": rng.choice([f"AN-{i:02d}" for i in range(1, 13)], len(chamados)),
    })[investigado.to_numpy()].reset_index(drop=True)

    status = np.where(~fechado, "aberto", np.where(investigado, "investigado", "ressarcido_automatico"))
    negado = investigado & pd.Series(resultado).isin(["fraude_confirmada", "improcedente"]).to_numpy()
    chamados["status"] = status
    chamados["valor_ressarcido"] = np.where((status == "aberto") | negado, 0.0, chamados["valor_reclamado"])
    return chamados, itens_ch, investigacoes


def gerar_ligacoes(rng, chamados, n_texto: int):
    sint = chamados[chamados["_call_id"].notna()]
    ligacoes = [pd.DataFrame({"call_id": sint["_call_id"], "chamado_id": sint["chamado_id"],
                              "origem": "audio_sintetico", "arquivo": sint["_call_id"] + ".wav"})]
    recentes = chamados[(chamados["canal"] == "telefone") & chamados["_call_id"].isna()
                        & (pd.to_datetime(chamados["data_abertura"])
                           >= pd.Timestamp(inicio_ciclo(N_CICLOS - 8)))]
    amostra = recentes.sample(min(n_texto, len(recentes)), random_state=int(rng.integers(1 << 31)))
    ligacoes.append(pd.DataFrame({"call_id": [f"lig_{i:06d}" for i in range(1, len(amostra) + 1)],
                                  "chamado_id": amostra["chamado_id"].to_numpy(),
                                  "origem": "roteiro_texto", "arquivo": None}))
    lig = pd.concat(ligacoes, ignore_index=True)
    abertura = lig["chamado_id"].map(chamados.set_index("chamado_id")["data_abertura"])
    lig["data_hora"] = pd.to_datetime(abertura) + pd.to_timedelta(rng.integers(8 * 60, 20 * 60, len(lig)), "m")
    lig["duracao_s"] = rng.integers(150, 900, len(lig))
    return lig


def gravar(saida: Path, nome: str, df: pd.DataFrame) -> None:
    pasta = saida / nome
    pasta.mkdir(parents=True, exist_ok=True)
    for col in df.columns:
        if col.startswith("data_"):
            df[col] = pd.to_datetime(df[col])
            if col != "data_hora":
                df[col] = df[col].dt.date
    df.to_parquet(pasta / f"{nome}.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--consultoras", type=int, default=20000)
    ap.add_argument("--ligacoes-texto", type=int, default=1200)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--roteiros", type=Path, default=Path(__file__).parent.parent / "sintetico" / "roteiros.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.semente)
    roteiros = json.loads(args.roteiros.read_text(encoding="utf-8"))
    reservados = {r["gabarito"]["pedido_numero"] for r in roteiros}

    produtos = gerar_produtos()
    consultoras, perfil_fraude = gerar_consultoras(rng, args.consultoras, roteiros)
    pedidos, itens = gerar_pedidos(rng, consultoras, perfil_fraude, produtos, reservados)
    entregas, faltas, estado = gerar_entregas(rng, pedidos, itens)
    chamados, itens_ch = gerar_chamados(rng, pedidos, itens, entregas, faltas, estado)
    pedidos, itens, entregas, chamados, itens_ch = injetar_roteiros(
        rng, roteiros, consultoras, produtos, pedidos, itens, entregas, chamados, itens_ch)
    chamados, itens_ch, investigacoes = finalizar_chamados(rng, chamados, itens_ch, itens)
    ligacoes = gerar_ligacoes(rng, chamados, args.ligacoes_texto)

    gabarito_consultoras = pd.DataFrame({"consultora_codigo": consultoras["consultora_codigo"],
                                         "perfil_fraude": perfil_fraude})
    gabarito_chamados = chamados[["chamado_id", "_mecanismo"]].rename(columns={"_mecanismo": "mecanismo"})
    gabarito_chamados["eh_fraude"] = gabarito_chamados["mecanismo"].isin(["fraude", "fraude_oportunista"])

    pedidos = pedidos[pd.to_datetime(pedidos["data_pedido"]) < pd.Timestamp(DATA_CORTE)]
    itens = itens[itens["pedido_numero"].isin(pedidos["pedido_numero"])]
    saida = args.saida
    tabelas = {
        "centros_distribuicao": pd.DataFrame({"cd": list(CDS), "uf": list(CDS.values())}),
        "produtos": produtos.drop(columns="popularidade"),
        "consultoras": consultoras.drop(columns="_produtividade"),
        "pedidos": pedidos.drop(columns="_fraude"),
        "pedido_itens": itens[["pedido_numero", "sku", "quantidade", "preco_unitario"]],
        "entregas": entregas,
        "chamados": chamados.drop(columns=["_mecanismo", "_call_id"]),
        "chamado_itens": itens_ch[["chamado_id", "sku", "quantidade"]],
        "investigacoes": investigacoes,
        "ligacoes": ligacoes[["call_id", "chamado_id", "data_hora", "duracao_s", "origem", "arquivo"]],
    }
    for nome, df in tabelas.items():
        gravar(saida, nome, df.copy())
    gravar(saida / "_gabarito", "consultoras", gabarito_consultoras)
    gravar(saida / "_gabarito", "chamados", gabarito_chamados)

    print(f"Cutoff: {DATA_CORTE} · {N_CICLOS} cycles of {CICLO_DIAS} days since {DATA_INICIO}")
    for nome, df in tabelas.items():
        print(f"  {nome:22s} {len(df):>10,d} rows")


if __name__ == "__main__":
    main()
