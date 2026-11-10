# Synthetic operational data — Aurélia Cosméticos

Aurélia is a fictitious direct-selling company. This folder generates what its
order system and its customer-service system would hold, so the build has real
volume around the 30 calls with ground truth in `sintetico/`: Genie needs data to
answer, and the fraud model needs history to learn from.

```bash
pip install numpy pandas pyarrow scikit-learn
python gerar_dados.py --saida ./saida          # ~15 s, 20 thousand consultants
python checar_sinal.py --dados ./saida         # checks the story before uploading
databricks fs cp -r ./saida dbfs:/Volumes/<catalog>/<schema>/operacional -p <profile>
```

`_gabarito/` holds the truth the real world does not have (who is a fraudster,
what caused each claim). It is there to measure the model and must **not** be
read by the pipeline.

## What comes out

| Table | Rows | What it is |
|---|---|---|
| `consultoras` | 20 thousand | state (UF), region, sign-up date, level |
| `produtos` | 45 | fictitious catalog (Âmbar, Maré, Raízes, Diário, Aura, Tempo…), price and weight |
| `pedidos` | ~211 thousand | 24 cycles of 21 days, with DC and carrier |
| `pedido_itens` | ~1.4 million | items and unit price |
| `entregas` | ~211 thousand | dates, status (includes `em_transito` for what had not arrived by the cutoff), expected weight, weight at shipping and at the hub, proof of delivery |
| `chamados` | ~9.9 thousand | declared motive, channel, claimed amount, status, reimbursed amount |
| `chamado_itens` | ~7.3 thousand | what the consultant says was missing |
| `investigacoes` | ~2 thousand | investigation result — the only label that exists |
| `ligacoes` | 1,230 | 30 with synthetic audio (`sint_NNN.wav`) and 1,200 to be transcribed as text |

The cutoff date is fixed (09/21/2026), not `datetime.now()`: the world is the same
on every run with the same seed, so the numbers in the deck do not change when the
data is regenerated.

## The planted story

Each cause of missing products leaves a signature in the weight of the box, which
is weighed at two points:

| Cause | Shipped weight | Hub weight | Other signals |
|---|---|---|---|
| Picking error | below expected | equal to shipped | cheap, random items |
| In-transit theft | equal to expected | below shipped | expensive items, specific route |
| Consultant fraud | equal to expected | equal to shipped | signed proof of delivery, fragrances, repeat claims, short tenure |

Two waves raise the claim rate **without** any fraud, and they are what trips up
anyone who looks only at the rate:

- **Theft at TransLeste in RJ** from cycle 14: from ~4 to ~11 claims per 100 orders.
- **WMS migration at CD Contagem** in cycles 9 to 11: from ~4.5 to ~8.

Fraud is ~10% of claims: 1.2% of consultants have a fraudster profile (more common
among those with less than a year of tenure), plus rare opportunistic fraud among
those without the profile. So it is not too easy, 45% of boxes are not weighed at
the hub, the scale is noisy, and four in ten frauds claim a single cheap item, to
stay below the amount that triggers an investigation.

## The label is biased, on purpose

Only ~20% of closed claims are investigated, and not at random: the policy
investigates claims above R$ 200, consultants with two or more claims in 180 days,
and tampered boxes more often. That is why confirmed fraud is 14.4% of investigated
claims and 10% of all claims. A model trained on this label learns the fraud the
investigation usually finds; the fraud that escapes the policy is underrepresented.
Claims from the last three weeks are open and unlabeled — they are the queue the
model ranks.

## `checar_sinal.py` results (seed 42)

| Measure | Value |
|---|---|
| Temporal test (most recent 25% of investigated claims) | PR-AUC 0.83 · ROC-AUC 0.96 · prevalence 14.2% |
| Open queue, top 50 by score | 88% real fraud, versus 10% across the whole queue |
| Claims from the 5 fraud calls | scores from 0.56 to 0.97; the other 25 below 0.05 |

The script's reference model is only a check that the pattern exists; the build's
model is trained and served on Databricks.
