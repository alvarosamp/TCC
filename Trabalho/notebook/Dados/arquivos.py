import os, platform, logging, time
from pathlib import Path

# --- Logging (console + opcional arquivo) ---
LOG_LEVEL = logging.INFO
LOG_TO_FILE = False  # mude para True se quiser salvar em .log
LOG_FILE = Path.cwd() / "diagnostico_caminhos.log"

logger = logging.getLogger("diagnostico_caminhos")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if LOG_TO_FILE:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

SO = platform.system()

# Raízes possíveis para buscar
if SO == 'Windows':
    raizes_busca = [
        r"C:\Users\vish8\OneDrive\Documentos\TCC",
        r"C:\Users\vish8\OneDrive\Documentos\GitHub\TCC",
        r"C:\Users\vish8\Documents\TCC",
        r"G:\Meu Drive\TCC",
        
    ]
else:
    raizes_busca = [
        "/Users/alvarosamp/Documents/Projetos/TCC",
        "/Users/alvarosamp/Documents/TCC",
    ]

logger.info("Buscando arquivos XML e .ms/.mseed no sistema...")

xmls_encontrados = []
ms_encontrados   = []

t0 = time.perf_counter()
for raiz in raizes_busca:
    if not os.path.exists(raiz):
        logger.warning(f"Raiz não existe (pulando): {raiz}")
        continue
    logger.info(f"Varrendo: {raiz}")
    for root, dirs, files in os.walk(raiz):
        # Pular pastas muito grandes para não demorar
        dirs[:] = [d for d in dirs if d not in
                   ['.git', '__pycache__', 'node_modules', '.venv',
                    'venv', '.idea', 'AppData']]
        for f in files:
            caminho = os.path.join(root, f)
            if f.endswith('.xml') and 'station' in f.lower() or (
               f.endswith('.xml') and any(x in f for x in ['AZ', 'CI', 'BZN', 'PASC'])):
                xmls_encontrados.append(caminho)
            elif f.endswith('.ms') or f.endswith('.mseed'):
                ms_encontrados.append(caminho)

dt = time.perf_counter() - t0
logger.info(f"Varredura concluída em {dt:.1f}s")

logger.info(f"XMLs encontrados: {len(xmls_encontrados)}")
for x in xmls_encontrados[:20]:
    logger.info(f"  {x}")
if len(xmls_encontrados) > 20:
    logger.info(f"  ... e mais {len(xmls_encontrados)-20}")

logger.info(f"Arquivos .ms/.mseed encontrados: {len(ms_encontrados)}")
# Mostrar as pastas únicas
pastas_ms = sorted(set(os.path.dirname(f) for f in ms_encontrados))
for p in pastas_ms[:10]:
    n = sum(1 for f in ms_encontrados if os.path.dirname(f) == p)
    logger.info(f"  {p}  ({n} arquivos)")
if len(pastas_ms) > 10:
    logger.info(f"  ... e mais {len(pastas_ms)-10} pastas")

if LOG_TO_FILE:
    logger.info(f"Log salvo em: {LOG_FILE}")