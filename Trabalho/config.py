import os
PASTA_RAIZ    = r"C:\Users\vish8\OneDrive\Documentos\TCC\data\scedc-pds"
PASTA_XML     = os.path.join(PASTA_RAIZ, "FDSNstationXML")
PASTA_PROJETO = r"C:\Users\vish8\OneDrive\Documentos\TCC\Trabalho\artefacts"
INVENTARIO    = os.path.join(PASTA_PROJETO, "data", "inventario_dados.csv")
PASTA_WINDOWS = os.path.join(PASTA_PROJETO, "data", "windows")
PASTA_RESULTS = os.path.join(PASTA_PROJETO, "results")
os.makedirs(PASTA_RESULTS, exist_ok=True)
os.makedirs(os.path.join(PASTA_PROJETO, "figures"), exist_ok=True)


SR_ALVO     = 40.0
CANAL_ALVO  = "BHZ"
OUTPUT_RESP = "VEL"
PRE_FILT    = (0.5, 1.0, 18.0, 20.0)
WATER_LEVEL = 60
FREQ_MIN    = 0.5
FREQ_MAX    = 15.0

# ── PARÂMETROS STA/LTA (valores iniciais — serão otimizados) ─────
STA_SEG     = 1.0    # duração da janela curta (segundos)
LTA_SEG     = 10.0   # duração da janela longa (segundos)
THRESH_ON   = 3.0    # threshold para disparar detecção
THRESH_OFF  = 1.5    # threshold para encerrar detecção
