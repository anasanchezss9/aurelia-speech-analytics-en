# Genie — Loss Prevention questions

Space **Aurélia · Chamados de falta e fraude** (`01f1d96b0d6f1bd9b132a3ea3ac0cf4d`),
over the pipeline's gold tables and the scored queue. For people who don't open the app:
loss prevention manager, customer service, logistics.

| Table | To answer |
|---|---|
| `gold_chamados_ciclo` | claims, reimbursement and confirmed fraud by cycle, DC, carrier, state (UF) and reason |
| `gold_pedidos_ciclo` | the denominator: orders by the same dimensions |
| `fila_priorizada` | what to investigate first, with the score and what the call said |
| `silver_chamados` | one claim per row, with the investigation outcome |
| `gold_call_insights`, `silver_ligacoes` | the LLM analysis of each call and its claim |

```bash
python genie/criar_genie.py --profile field-toolbox --warehouse <id>                 # creates
python genie/criar_genie.py --profile field-toolbox --warehouse <id> --space-id <id> # updates
python genie/testar_genie.py --space-id <id> "question"                               # converses
```

`criar_genie.py` executes every example SQL on the warehouse before publishing the
space: a wrong example teaches Genie to be wrong the same way.

## The instructions that make a difference

- **Claim rate aggregates before joining.** `gold_chamados_ciclo` has one row
  per reason and channel; joining with orders before aggregating multiplies the
  denominator and the rate comes out wrong without raising an error.
- **Confirmed fraud rate is over investigated claims**, never over the total.
- **The most recent cycle is incomplete** (claims still arriving); trends go
  up to cycle 23.

## Result — questions outside the examples (Conversation API, Nov 5, 2026)

| Question | What Genie did |
|---|---|
| Did the Contagem DC see an increase in partial-shortage claims? In which cycles? | found the jump in cycle 9 (from 10 to 43) — the WMS switch planted in the data |
| Did TransLeste in RJ get worse? Claims per 100 orders per cycle | aggregated claims and orders separately, stopped at cycle 23, found the turn in cycle 14 (from ~4 to 10–13) |
| How many high-risk claims are open per carrier and how much are they claiming? | used the prioritized queue; TransLeste, the route with real theft, has only 5 — the model doesn't confuse theft with fraud |
