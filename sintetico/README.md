# Synthetic calls with ground truth

30 fully synthetic missing-product claim calls, in the same format as a contact
center's phone recordings (G.711 μ-law WAV, 8 kHz, stereo, agent on one channel
and consultant on the other), each with the ground truth of what the gold layer
should extract.

The ground truth is what a real call does not have. With it you can measure the
pipeline end to end — transcription error rate and accuracy of each business
field — instead of only comparing one transcript with another.

## How to generate

```bash
pip install -r requirements.txt
brew install espeak-ng ffmpeg   # Kokoro uses espeak-ng to phonemize Portuguese

# 1. Scripts: 30 dialogues written by a workspace LLM, with the ground truth.
#    They are already in roteiros.json; only run this if you want other scenarios.
python gerar_roteiros.py --profile <profile>

# 2. Audio: two voices per call, background, signal dropouts and telephone band.
#    About one hour of audio, 20 to 40 minutes on a Mac with GPU (MPS).
python sintetizar.py --saida ./audios

# 3. Volume: the WAVs and _reference_scripts.json, which the comparison notebook reads.
databricks fs cp -r ./audios dbfs:/Volumes/<catalog>/<schema>/<volume> -p <profile>
```

Then run the pipeline (`01` → `02b` → `03`, the last one with the widgets
`table_silver=silver_transcripts_ait` and `table_gold=gold_call_insights_ait`) and measure:

```bash
python avaliar.py --profile <profile> --warehouse <id> \
  --silver <catalog>.<schema>.silver_transcripts_ait \
  --gold <catalog>.<schema>.gold_call_insights_ait
```

## What comes out

| File | What it is |
|---|---|
| `roteiros.json` | The 30 dialogues, turn by turn, with ground truth: `tipo_problema`, `suspeita_fraude`, `reincidencia`, `sentiment`, `consultora_codigo`, `pedido_numero`, `produtos_faltantes` |
| `audios/sint_NNN.wav` | The calls (outside git; they live in the Volume) |
| `audios/_reference_scripts.json` | The scripts with the voices and background drawn for each call |

The scenarios cover the five values of `tipo_problema` (5 to 7 calls each), with
repeat claims in 8 of them, all three sentiments, and 6 calls with a male
consultant.

## Results on fevm-field-toolbox (10/17/2026)

Pipeline with `ai_transcribe` and the gold layer on `databricks-claude-sonnet-4-6`,
on the 30 calls, in three gold runs over the same silver:

| Measure | Original prompt | + definitions and product lines | + "full name, never just the line" |
|---|---|---|---|
| Transcript word error rate | 5.2% (worst 8.5%) | 5.2% | 5.2% |
| `tipo_problema` | 23/30 | **30/30** | **30/30** |
| `suspeita_fraude` | 30/30 | 30/30 | 30/30 |
| `reincidencia` | 30/30 | 30/30 | 30/30 |
| `consultora_codigo` | 29/30 | 29/30 | 29/30 |
| `sentiment` | 25/30 | 25/30 | 25/30 |
| `pedido_numero` | 24/30 | 24/30 | 24/30 |
| `produtos_faltantes` | 21/22 | 8/22 | **22/22** |

The first run is kept in `gold_call_insights_ait_v1`.

What the runs teach:

- **A category without a definition becomes a guess.** With the original prompt
  the five fraud calls came out as `violacao_caixa`: the model got right that there
  was suspicion (`suspeita_fraude` 30/30), but had no way to know that "box opened
  with product missing" is fraud and "box opened with nothing missing" is
  tampering. The same definitions that generated the scripts, pasted into the
  prompt, brought it to 30/30.
- **A new instruction has side effects.** The list of product lines, added to fix
  spelling, made the model return only the line ("Diário") instead of the product,
  and missing products fell from 21 to 8. Saying what the list is for fixed it.
  That is the reason to measure every field on every prompt change, not only the
  one you meant to fix.
- **A number dictated digit by digit loses digits in transcription.** A repeated
  digit becomes one ("sete nove três três três sete", seven nine three three three
  seven, comes out as `7937`) and the leading "um" (both "one" and "a" in
  Portuguese) becomes an article. The error is already in the silver, not in the
  gold — a prompt cannot fix it. The way forward is to validate the number against
  the consultant's orders table.
- **Neutral sentiment tends toward negative.** A problem call with a calm tone is
  read as negative in most of the mismatches.

## Limits

- Synthetic audio is still easier than a real call. The telephone degradation
  matters, but the synthetic voice is regular and well articulated. Use this set to
  measure the gold layer and catch pipeline regressions; before promising accuracy
  to a customer, run transcription on a sample of their calls.
- Kokoro has only three Brazilian Portuguese voices (`pf_dora`, `pm_alex` and
  `pm_santa`). The others come from blends with Spanish and Italian voices and from
  pitch variations; two consultants may sound alike.
- Aurélia, the products, the consultants and the numbers are fictitious.

## Licenses

Kokoro-82M and the voices: Apache-2.0. espeak-ng (GPL-3.0) is called as an external
program for phonemization; it is not distributed with the asset.
