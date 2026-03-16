"""
pipeline.py — Funções centrais de pré-processamento sísmico para o TCC.

Fluxo por trace:
  1. Selecionar canal desejado (ex: BHZ, HNZ)
  2. Detrend linear + demean  → remove offset/tendência lenta
  3. Taper cosine 5%          → suaviza bordas (evita artefato de Gibbs na FFT)
  4. Remoção de resposta instrumental (usando StationXML)
     → converte counts → velocidade (m/s) ou aceleração
  5. Filtro passa-banda 0.5–15 Hz → faixa útil para eventos regionais
  6. Reamostragem para TARGET_SR (40 Hz) se necessário
  7. Normalização z-score por janela (média=0, std=1)
"""

from typing import Tuple
import numpy as np
from obspy import read, read_inventory
import matplotlib.pyplot as plt

# ── Taxa de amostragem alvo do projeto ──────────────────────────────────
TARGET_SR = 40.0


def preprocessar_waveform(
    caminho_wave: str,
    caminho_inv: str,
    canal_alvo: str = "BHZ",
    output: str = "VEL",
    pre_filt: Tuple = (0.5, 1.0, 18.0, 20.0),
    water_level: float = 60,
    normalizar: bool = True,
    plot: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Pipeline completo para processar UMA waveform (.ms / .mseed).

    Parâmetros
    ----------
    caminho_wave : str
        Caminho para o arquivo de forma de onda (.ms).
    caminho_inv : str
        Caminho para o StationXML com a resposta instrumental da estação.
    canal_alvo : str
        Sufixo do canal desejado (ex: 'BHZ'). A busca usa endswith().
    output : str
        Tipo de saída após remoção de resposta: 'VEL', 'DISP' ou 'ACC'.
    pre_filt : tuple (f1, f2, f3, f4)
        Frequências de corte do pré-filtro (transição suave).
        - f1,f2 → corte inferior (remove microsissmos oceânicos)
        - f3,f4 → corte superior (deve ser ≤ Nyquist)
    water_level : float
        Estabiliza a deconvolução em frequências onde a resposta é fraca.
    normalizar : bool
        Se True, aplica z-score ao sinal final.
    plot : bool
        Se True, plota sinal antes/depois da normalização.

    Retorna
    -------
    dados : np.ndarray 1D (float32)
        Sinal processado.
    stats : dict
        Metadados úteis (rede, estação, canal, taxa, etc.).
    """

    # ── 1. Carregar stream e selecionar o canal ─────────────────────────
    st = read(caminho_wave)

    tr = None
    for trace in st:
        if trace.stats.channel.endswith(canal_alvo):
            tr = trace.copy()
            break

    # FIX: antes o raise estava dentro do try/except e nunca era alcançado
    # se nenhum trace batesse. Agora verificamos DEPOIS do loop.
    if tr is None:
        raise ValueError(
            f"Nenhum trace com canal terminando em '{canal_alvo}' "
            f"encontrado em {caminho_wave}. "
            f"Canais disponíveis: {[t.stats.channel for t in st]}"
        )

    # ── 2. Metadados ────────────────────────────────────────────────────
    stats = {
        "network": tr.stats.network,
        "station": tr.stats.station,
        "channel": tr.stats.channel,
        "sampling_rate": tr.stats.sampling_rate,
        "starttime": tr.stats.starttime,
        "npts": tr.stats.npts,
    }

    # ── 3. Pré-processamento básico ─────────────────────────────────────
    # detrend linear  → remove inclinação lenta (drift do sensor)
    # detrend demean  → centraliza em zero
    # taper           → suaviza 5% das bordas (reduz artefato espectral)
    tr.detrend(type="linear")
    tr.detrend(type="demean")
    tr.taper(max_percentage=0.05, type="cosine")

    # ── 4. Remoção de resposta instrumental ─────────────────────────────
    # Converte de counts (digital) → unidade física (ex: m/s para VEL).
    # O pre_filt evita amplificação excessiva nas bordas da banda.
    inv = read_inventory(caminho_inv)
    tr.remove_response(
        inventory=inv,
        output=output,
        pre_filt=pre_filt,
        water_level=water_level,
    )

    # ── 5. Filtro passa-banda ───────────────────────────────────────────
    # 0.5–15 Hz cobre a faixa de interesse para eventos regionais/locais.
    # Frequências < 0.5 Hz → microsissmos oceânicos (ruído de longo período)
    # Frequências > 15 Hz  → ruído antropogênico / aliasing próximo de Nyquist
    tr.filter("bandpass", freqmin=0.5, freqmax=15.0)

    # ── 6. Reamostragem para 40 Hz (se necessário) ─────────────────────
    # IMPORTANTE: garante que TODAS as janelas tenham a mesma escala temporal.
    # Sem isso, janelas de estações com 100 Hz teriam 2.5x mais pontos.
    if float(tr.stats.sampling_rate) != TARGET_SR:
        tr.resample(TARGET_SR)
        stats["sampling_rate"] = TARGET_SR  # atualiza metadado

    # ── 7. Normalização z-score ─────────────────────────────────────────
    # z = (x - média) / desvio_padrão
    # Faz com que TODOS os traces tenham escala comparável, independente
    # da magnitude original (estações próximas vs distantes do epicentro).
    dados = tr.data.astype(np.float32)

    if normalizar:
        std = float(np.std(dados))
        # Guard: se o sinal é quase constante (ex: gap/saturação),
        # o std ≈ 0 e a divisão explodiria. Retornamos zeros.
        if std < 1e-6:
            dados = np.zeros_like(dados)
        else:
            dados = (dados - float(np.mean(dados))) / std

    # ── 8. Plot opcional ────────────────────────────────────────────────
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        ax1.plot(tr.times(), tr.data, "b", lw=0.7)
        ax1.set_ylabel(output)
        ax1.set_title(f"Sinal processado — {tr.id}")
        ax1.grid(True, alpha=0.3)

        ax2.plot(tr.times(), dados, "r", lw=0.7)
        ax2.set_ylabel("Normalizado (z-score)")
        ax2.set_xlabel("Tempo (s)")
        ax2.set_title("Após normalização")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return dados, stats


def criar_janelas(
    dados: np.ndarray,
    sr: float,
    tamanho_seg: float,
    sobreposicao: float = 0.5,
) -> np.ndarray:
    """
    Divide um sinal 1D em janelas de tamanho fixo com sobreposição.

    Parâmetros
    ----------
    dados : np.ndarray (1D)
        Sinal de entrada já pré-processado.
    sr : float
        Taxa de amostragem em Hz (ex: 40.0).
    tamanho_seg : float
        Duração de cada janela em segundos (ex: 60).
    sobreposicao : float
        Fração de sobreposição entre janelas (0.0 a 1.0).
        0.5 = 50% de overlap → passo = metade da janela.

    Retorna
    -------
    janelas : np.ndarray shape (n_janelas, n_amostras_por_janela)

    Exemplo
    -------
    >>> sr = 40.0; seg = 60  # 60s @ 40Hz = 2400 amostras por janela
    >>> janelas = criar_janelas(dados, sr, seg, sobreposicao=0.5)
    >>> janelas.shape  # (N, 2400)
    """
    n_amostras_janela = int(tamanho_seg * sr)
    passo = int(n_amostras_janela * (1 - sobreposicao))

    janelas = []
    for inicio in range(0, len(dados) - n_amostras_janela + 1, passo):
        janelas.append(dados[inicio : inicio + n_amostras_janela])

    return np.array(janelas, dtype=np.float32)