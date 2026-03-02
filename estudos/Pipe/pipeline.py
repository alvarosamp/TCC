from typing import Tuple 
import numpy as np
import obspy 
from obspy import read, read_inventory, UTCDateTime
from obspy.clients.fdsn import Client
import numpy as np
import matplotlib.pyplot as plt
def preprocessar_waveform(caminho_wave : str, caminho_env : str, canal_alvo: str = 'BHZ', output: str = 'VEL', pre_filt: Tuple = (0.5, 1.0, 18.0, 20.0),
                       water_level : float = 60, normalizar : bool = True, plot: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Pipeline para processar uma waveform, incluindo remoção de resposta.
    Parametros : 
    - caminho_wave : str, caminho para o arquivo da waveform (.ms)
    - caminho_inv : str, caminho para o arquivo de inventário (STATIONXML)
    - canal_alvo : str, codigo do canal desejado (ex: 'BHZ')
    - output : str, tipo de saida desejada ('VEL', 'DISP', 'ACC')
    - pre_filt : tuple, frequencias de corte para o prefiltro (f1,f2,f3,f4)
    - water_level : float, valor do water level para estabilização da resposta
    - normalizar : bool, se True, normaliza o sinal corrigido
    - plot : bool, se True, plota os espectrogramas antes e depois da correção

    Retorna: 
    - dados: numpy array 1D com sinal processado
    - stats : dicionario com metadados uteis (taxa, estação, etc)

    """
    #Carregando o stream e selecionando o canal alvo
    st = read(caminho_wave)
    tr = None 
    for trace in st:
        try:
            if trace.stats.channel.endswith(canal_alvo):
                tr = trace.copy()
                break
        except AttributeError:        
            raise ValueError(f'Nenhum trace com canal {canal_alvo} encontrado em {caminho_wave}')
        
    #Extrair metaddados importantes
    stats = {
        'network': tr.stats.network,
        'station': tr.stats.station,
        'channel': tr.stats.channel,
        'sampling_rate': tr.stats.sampling_rate,
        'starttime': tr.stats.starttime,
        'npts': tr.stats.npts
    }
    
    #Pre processamento basico
    tr.detrend(type = 'linear')
    tr.detrend(type = 'demean')
    tr.taper(max_percentage = 0.05, type = 'cosine')
    
    #Carregand o o inventário
    inv = read_inventory(caminho_env)
    #Removendo resposta
    tr.remove_response(inventory = inv, output = output, pre_filt = pre_filt, water_level = water_level)
    tr.filter('bandpass', freqmin=0.5, freqmax=15.0)   
    
    #Normalizar (z-score)
    dados = tr.data
    if normalizar:
        dados = (dados - np.mean(dados)) / np.std(dados)
    # 7. Plot opcional
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        ax1.plot(tr.times(), tr.data, 'b', lw=0.7)
        ax1.set_ylabel(output)
        ax1.set_title(f'Sinal processado - {tr.id}')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(tr.times(), dados, 'r', lw=0.7)
        ax2.set_ylabel('Normalizado')
        ax2.set_xlabel('Tempo (s)')
        ax2.set_title('Após normalização')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return dados, stats    
   
def criar_janelas(dados, sr, tamanho_seg, sobreposicao=0.5):
    n_amostras_janela = int(tamanho_seg * sr)
    passo = int(n_amostras_janela * (1 - sobreposicao))
    janelas = []
    for i in range(0, len(dados) - n_amostras_janela + 1, passo):
        janelas.append(dados[i:i + n_amostras_janela])
    return np.array(janelas)