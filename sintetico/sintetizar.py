"""Turns the scripts into synthetic phone calls.

Each call comes out as G.711 μ-law WAV, 8 kHz, stereo, the same format as the
contact center's real recordings: the agent on the left channel and the
consultant on the right. The voice is Kokoro (Apache-2.0), which runs locally and
has Brazilian Portuguese voices. So it does not sound like a studio, each call
gets:

- two different voices, with pitch and speed variation per person;
- irregular pauses and overlap where the script marks an interruption;
- background on the consultant's side (home, street, car or silence), contact
  center murmur on the agent's side, and cell-signal dropouts;
- telephone band (300 to 3,400 Hz) and μ-law encoding at 8 kHz via ffmpeg.

Everything is drawn with the call id as the seed: running again produces the same
audio.

Usage:
    python sintetizar.py --saida ./audios
    python sintetizar.py --saida ./audios --somente sint_001,sint_019
"""

import argparse
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

AQUI = Path(__file__).parent
SR = 24000

# Only three Kokoro voices are Brazilian Portuguese. Blends with Spanish and
# Italian voices give new timbres without losing the pronunciation, which comes
# from the Portuguese phonemizer.
VOZES = {
    "f": ["pf_dora", "pf_dora,if_sara", "pf_dora,ef_dora"],
    "m": ["pm_alex", "pm_santa", "pm_alex,im_nicola", "pm_santa,em_alex"],
}

# Signal-to-noise ratio (dB) and whether there are signal dropouts, per consultant background.
AMBIENTES = {
    "silencio": (34, False),
    "casa": (26, False),
    "rua": (17, True),
    "carro": (14, True),
}


def ruido_colorido(n: int, expoente: float, sorteio: np.random.Generator) -> np.ndarray:
    """Noise with a 1/f^exponent spectrum: 1 is pink, 2 is brown."""
    espectro = np.fft.rfft(sorteio.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    sinal = np.fft.irfft(espectro / freqs ** (expoente / 2), n)
    return sinal / (np.std(sinal) + 1e-9)


def rms_falado(canal: np.ndarray) -> float:
    ativo = canal[np.abs(canal) > 0.01]
    return float(np.sqrt(np.mean(ativo**2))) if len(ativo) else 0.1


def murmurio(fonte: np.ndarray, n: int, sorteio: np.random.Generator) -> np.ndarray:
    """Background voices without words: the call itself, reversed and shifted."""
    if len(fonte) < SR:
        return np.zeros(n)
    base = fonte[::-1]
    saida = np.zeros(n)
    for _ in range(4):
        deslocamento = int(sorteio.integers(0, len(base)))
        trecho = np.roll(base, deslocamento)
        saida += np.resize(trecho, n) * sorteio.uniform(0.5, 1.0)
    return saida / (np.std(saida) + 1e-9)


class Locutor:
    def __init__(self, pipeline: KPipeline, voz: str, velocidade: float, semitons: float):
        self.pipeline = pipeline
        self.voz = voz
        self.fator = 2 ** (semitons / 12)
        # Pitch is raised by resampling, which also speeds up; the speed requested
        # from Kokoro already offsets that so the final duration matches the drawn speed.
        self.velocidade_kokoro = velocidade / self.fator

    def falar(self, texto: str) -> np.ndarray:
        partes = [
            np.asarray(audio, dtype=np.float32)
            for _, _, audio in self.pipeline(texto, voice=self.voz, speed=self.velocidade_kokoro)
        ]
        pausa = np.zeros(int(0.12 * SR), dtype=np.float32)
        audio = np.concatenate([x for p in partes for x in (p, pausa)][:-1]) if partes else pausa
        if abs(self.fator - 1) > 1e-3:
            n = int(len(audio) / self.fator)
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        return audio


def montar_ligacao(roteiro: dict, pipeline: KPipeline) -> tuple[np.ndarray, dict]:
    sorteio = random.Random(roteiro["call_id"])
    gerador = np.random.default_rng(sorteio.randrange(2**32))

    voz_agente = sorteio.choice(VOZES[roteiro["agente_genero"]])
    opcoes = [v for v in VOZES[roteiro["consultora_genero"]] if v != voz_agente]
    voz_consultora = sorteio.choice(opcoes)
    ambiente = sorteio.choice(list(AMBIENTES))
    locutores = {
        "agente": Locutor(pipeline, voz_agente, sorteio.uniform(1.0, 1.12), sorteio.uniform(-1.0, 1.0)),
        "consultora": Locutor(pipeline, voz_consultora, sorteio.uniform(0.92, 1.08), sorteio.uniform(-1.5, 1.5)),
    }
    canal = {"agente": 0, "consultora": 1}

    trechos = []
    cursor = sorteio.uniform(0.6, 1.2)
    inicio_anterior = fim_anterior = 0.0
    for fala in roteiro["falas"]:
        audio = locutores[fala["quem"]].falar(fala["texto"])
        duracao = len(audio) / SR
        if fala["interrompe"] and trechos:
            inicio = max(inicio_anterior + 0.5 * (fim_anterior - inicio_anterior), fim_anterior - sorteio.uniform(0.4, 1.0))
        else:
            longa = sorteio.random() < 0.12
            inicio = cursor + (sorteio.uniform(1.2, 2.4) if longa else sorteio.uniform(0.18, 0.75))
        trechos.append((canal[fala["quem"]], inicio, audio))
        inicio_anterior, fim_anterior = inicio, inicio + duracao
        cursor = max(cursor, fim_anterior)

    total = int((cursor + sorteio.uniform(0.5, 1.0)) * SR)
    estereo = np.zeros((total, 2), dtype=np.float64)
    for c, inicio, audio in trechos:
        i = int(inicio * SR)
        estereo[i : i + len(audio), c] += audio

    # Consultant's side: phone volume, background and signal dropouts.
    consultora = estereo[:, 1]
    consultora *= sorteio.uniform(0.5, 1.35)
    snr, com_quedas = AMBIENTES[ambiente]
    nivel = rms_falado(consultora) / 10 ** (snr / 20)
    expoente = 2.0 if ambiente in ("rua", "carro") else 1.0
    consultora += ruido_colorido(total, expoente, gerador) * nivel
    if ambiente in ("casa", "rua"):
        consultora += murmurio(estereo[:, 0], total, gerador) * nivel * 0.8
    if com_quedas:
        for _ in range(sorteio.randint(2, 5)):
            i = sorteio.randrange(total)
            consultora[i : i + int(sorteio.uniform(0.04, 0.12) * SR)] = 0
    estereo[:, 1] = np.tanh(consultora * 1.2) / np.tanh(1.2)

    # Agent's side: good headset, with the contact center murmur very low.
    agente = estereo[:, 0]
    nivel = rms_falado(agente) / 10 ** (32 / 20)
    agente += ruido_colorido(total, 1.0, gerador) * nivel
    agente += murmurio(agente.copy(), total, gerador) * nivel * 1.5

    estereo /= max(1.0, np.max(np.abs(estereo)) / 0.95)
    detalhes = {
        "voz_agente": voz_agente,
        "voz_consultora": voz_consultora,
        "ambiente": ambiente,
        "duracao_s": round(total / SR, 1),
    }
    return estereo.astype(np.float32), detalhes


def para_telefone(estereo: np.ndarray, destino: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, estereo, SR, subtype="PCM_16")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
                "-af", "highpass=f=300,lowpass=f=3400",
                "-ar", "8000", "-ac", "2", "-c:a", "pcm_mulaw", str(destino),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roteiros", default=str(AQUI / "roteiros.json"))
    parser.add_argument("--saida", default=str(AQUI / "audios"))
    parser.add_argument("--somente", default="", help="comma-separated ids")
    args = parser.parse_args()

    roteiros = json.loads(Path(args.roteiros).read_text(encoding="utf-8"))
    alvo = set(filter(None, args.somente.split(",")))
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    dispositivo = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = KPipeline(lang_code="p", repo_id="hexgrad/Kokoro-82M", device=dispositivo)
    print(f"Kokoro on {dispositivo}")

    referencia = []
    for roteiro in roteiros:
        if alvo and roteiro["call_id"] not in alvo:
            continue
        t0 = time.perf_counter()
        estereo, detalhes = montar_ligacao(roteiro, pipeline)
        para_telefone(estereo, saida / roteiro["filename"])
        referencia.append({**roteiro, "audio": detalhes})
        print(
            f"{roteiro['call_id']}  {detalhes['duracao_s']:5.1f}s of audio in {time.perf_counter() - t0:5.1f}s  "
            f"{detalhes['ambiente']:<8} {detalhes['voz_agente']} x {detalhes['voz_consultora']}"
        )

    # Same name and same keys (filename, scenario, script) that 00_setup_env
    # writes for the gTTS audio: the comparison notebook reads this file to
    # compute WER, and avaliar_gold.py uses the ground truth.
    caminho = saida / "_reference_scripts.json"
    anteriores = json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else []
    novos = {r["filename"] for r in referencia}
    juntos = [r for r in anteriores if r["filename"] not in novos] + referencia
    caminho.write_text(json.dumps(juntos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(referencia)} calls in {saida}; ground truth in {caminho}")


if __name__ == "__main__":
    main()
