"""Checks that the generated world tells the promised story before it goes to the Volume.

Four checks: both waves show up in the claim rate, the scripts became open claims
consistent with the ground truth, fraud is a plausible minority of the claims, and
a simple model learns fraud from the features the pipeline will compute — with
real lift over the prevalence, but without getting everything right (if it did,
the data would be leaking the label).

Usage:
    python checar_sinal.py --dados ./saida
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def ler(dados: Path, nome: str) -> pd.DataFrame:
    return pd.read_parquet(dados / nome)


def features(dados: Path) -> pd.DataFrame:
    ch = ler(dados, "chamados")
    ped = ler(dados, "pedidos")
    ent = ler(dados, "entregas")
    cons = ler(dados, "consultoras")
    itens = ler(dados, "chamado_itens").merge(ler(dados, "produtos")[["sku", "categoria"]], on="sku")

    perf = (itens.assign(perf=itens["categoria"].eq("perfumaria") * itens["quantidade"])
            .groupby("chamado_id").agg(n_itens=("quantidade", "sum"), n_perfumaria=("perf", "sum")))
    df = (ch.merge(ped[["pedido_numero", "uf", "transportadora", "cd", "ciclo"]], on="pedido_numero")
          .merge(ent, on="pedido_numero")
          .merge(cons[["consultora_codigo", "data_cadastro", "nivel"]], on="consultora_codigo")
          .merge(perf, left_on="chamado_id", right_index=True, how="left"))
    df[["n_itens", "n_perfumaria"]] = df[["n_itens", "n_perfumaria"]].fillna(0)
    abertura = pd.to_datetime(df["data_abertura"])
    df["dias_entrega_abertura"] = (abertura - pd.to_datetime(df["data_entrega"])).dt.days
    df["div_peso_cd"] = (df["peso_teorico_kg"] - df["peso_expedido_kg"]) / df["peso_teorico_kg"]
    df["div_peso_hub"] = (df["peso_expedido_kg"] - df["peso_hub_kg"]) / df["peso_expedido_kg"]
    df["meses_casa"] = (abertura - pd.to_datetime(df["data_cadastro"])).dt.days / 30.4
    df = df.sort_values("data_abertura")
    df["chamados_180d"] = (df.assign(um=1).set_index(pd.to_datetime(df["data_abertura"]))
                           .groupby("consultora_codigo")["um"].transform(lambda s: s.rolling("180D").sum() - 1)
                           .to_numpy())
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", type=Path, required=True)
    ap.add_argument("--roteiros", type=Path, default=Path(__file__).parent.parent / "sintetico" / "roteiros.json")
    args = ap.parse_args()
    d = args.dados

    ped = ler(d, "pedidos")
    ch = ler(d, "chamados").merge(ped[["pedido_numero", "uf", "transportadora", "cd", "ciclo"]], on="pedido_numero")

    print("1. The waves")
    for rotulo, filtro_ch, filtro_ped in [
        ("TransLeste·RJ", (ch.transportadora == "TransLeste") & (ch.uf == "RJ"),
         (ped.transportadora == "TransLeste") & (ped.uf == "RJ")),
        ("CD Contagem  ", ch.cd == "CD Contagem", ped.cd == "CD Contagem"),
        ("rest         ", ~((ch.transportadora == "TransLeste") & (ch.uf == "RJ")) & (ch.cd != "CD Contagem"),
         ~((ped.transportadora == "TransLeste") & (ped.uf == "RJ")) & (ped.cd != "CD Contagem")),
    ]:
        t = (ch[filtro_ch].groupby("ciclo").size() / ped[filtro_ped].groupby("ciclo").size() * 100).fillna(0)
        print(f"   {rotulo} claims/100 orders per cycle: " + " ".join(f"{v:4.1f}" for v in t.iloc[:23]))

    print("\n2. Scripts")
    roteiros = json.loads(args.roteiros.read_text(encoding="utf-8"))
    lig = ler(d, "ligacoes")
    sint = lig[lig.origem == "audio_sintetico"].merge(ler(d, "chamados"), on="chamado_id")
    gab = {r["call_id"]: r["gabarito"] for r in roteiros}
    ok = all(gab[r.call_id]["pedido_numero"] == r.pedido_numero and gab[r.call_id]["consultora_codigo"] == r.consultora_codigo
             for r in sint.itertuples())
    print(f"   {len(sint)}/30 calls with a claim · codes match: {ok} · status: {sint.status.value_counts().to_dict()}")

    print("\n3. Fraud and labels")
    gch = ler(d / "_gabarito", "chamados")
    inv = ler(d, "investigacoes")
    print(f"   real fraud among claims: {gch.eh_fraude.mean():.1%} · investigated: {len(inv) / len(gch):.1%}"
          f" · confirmed fraud among investigated: {(inv.resultado == 'fraude_confirmada').mean():.1%}")
    print(f"   results: {inv.resultado.value_counts().to_dict()}")

    print("\n4. Can it be learned?")
    df = features(d).merge(inv[["chamado_id", "resultado"]], on="chamado_id", how="left").merge(gch, on="chamado_id")
    cols_num = ["valor_reclamado", "n_itens", "n_perfumaria", "dias_entrega_abertura", "div_peso_cd",
                "div_peso_hub", "meses_casa", "chamados_180d", "comprovante_assinado"]
    cols_cat = ["motivo_declarado", "canal", "transportadora", "uf", "nivel"]
    X = pd.concat([df[cols_num].astype(float), pd.get_dummies(df[cols_cat]).astype(float)], axis=1)
    rotulado = df["resultado"].notna().to_numpy()
    y = (df["resultado"] == "fraude_confirmada").to_numpy()
    ordem = np.argsort(pd.to_datetime(df["data_abertura"]).to_numpy())
    idx_rot = ordem[rotulado[ordem]]
    corte = int(len(idx_rot) * 0.75)
    treino, teste = idx_rot[:corte], idx_rot[corte:]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X.iloc[treino], y[treino])
    p = m.predict_proba(X.iloc[teste])[:, 1]
    print(f"   temporal test: {len(teste)} claims, prevalence {y[teste].mean():.1%}"
          f" · PR-AUC {average_precision_score(y[teste], p):.2f} · ROC-AUC {roc_auc_score(y[teste], p):.2f}")
    aberto = (df["status"] == "aberto").to_numpy()
    pa = m.predict_proba(X[aberto])[:, 1]
    verdade = df.loc[aberto, "eh_fraude"].to_numpy()
    k = 50
    top = np.argsort(-pa)[:k]
    print(f"   open queue: {aberto.sum()} claims, real fraud {verdade.mean():.1%}"
          f" · precision in the top {k}: {verdade[top].mean():.0%}")
    sint_ids = set(sint.chamado_id)
    abertos = df[aberto].assign(score=pa)
    print("   scores of the fraud calls' claims:      "
          + ", ".join(f"{s:.2f}" for s in abertos[abertos.chamado_id.isin(sint_ids) & abertos.eh_fraude].score))
    print("   scores of the other calls' claims:      "
          + ", ".join(f"{s:.2f}" for s in abertos[abertos.chamado_id.isin(sint_ids) & ~abertos.eh_fraude].score))


if __name__ == "__main__":
    main()
