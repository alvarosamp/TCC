"""
Passo 2 v3 — Pipeline GENÉRICO de janelamento e preparação de dataset.

Princípio de design: DOMÍNIO-AGNÓSTICO.
  - Não usa TauP, IASP91, picks teóricos, catálogo de origin time.
  - Rótulo apenas em nível de arquivo: events/ = anômalo, continuous/ = normal.
  - Mesmo código rodaria em CWRU Bearing trocando as pastas.

Configuração (travada):
  - Janela: 20s (800 amostras @ 40 Hz), overlap 50%
  - N normal:  ~150.000 janelas (amostradas de ~570k potenciais)
  - N anômalo: 1 janela central por trace (~23.000)
  - Split:     DUPLO — por estação (70/15/15) e temporal (70/15/15)
  - Métricas principais: AUC-PR (primária), AUC-ROC, F1
  - Normalização: z-score por janela
  - Pré-processamento: detrend → demean → taper → remove_response → bandpass

Saída em G:\\Meu Drive\\TCC\\data2\\processed\\:
  - dataset_v3_split_estacao.npz    (X_train, X_val, X_test, y_*, meta_*)
  - dataset_v3_split_temporal.npz   (mesma estrutura)
  - dataset_v3_info.json            (parâmetros + estatísticas)
  - inventario_v3.csv               (log linha-a-linha)
"""
import csv
import json
import logging
import sys 
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import obspy
from obspy import UTCDateTime

warnings.filterwarnings("ignore")
logging.getLogger("obspy").setLevel(logging.ERROR)

log = logging.getLogger("passo_02_v3_pipeline")
if not log.handlers:
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
        )

# ---------- Caminhos ----------
DRIVE_BASE = Path(r"G:\Meu Drive\TCC\data2")
LOCAL_BASE = Path(r"C:\TCC_data")

DIR_EVENTS = DRIVE_BASE / "raw" / "events"
DIR_CONT = DRIVE_BASE / "raw" / "continuous"
DIR_XML = DRIVE_BASE / "raw" / "stationxml"
DIR_OUT = DRIVE_BASE / "processed"
DIR_LOGS = LOCAL_BASE / "logs"
DIR_OUT.mkdir(parents=True, exist_ok=True)
DIR_LOGS.mkdir(parents=True, exist_ok=True)

# ---------- Parâmetros ----------
SAMPLING_RATE = 40.0
JANELA_S = 20.0
JANELA_NPTS = int(JANELA_S * SAMPLING_RATE)  # 800
OVERLAP = 0.5
STEP_NPTS = int(JANELA_NPTS * (1 - OVERLAP))  # 400
N_NORMAL_ALVO = 150_000
PRE_FILT = (0.5, 1.0, 18.0, 20.0)
WATER_LEVEL = 60
BANDPASS = (0.5, 15.0)
SEED = 42

stats = defaultdict(int)
rng = np.random.default_rng(SEED)

def carregar_inventarios() -> dict:
    inv_map = {}
    for xml in DIR_XML.glob("*.xml"):
        try:
            inv = obspy.read_inventory(str(xml))
            net = inv[0].code
            sta = inv[0][0].code
            inv_map[f"{net}.{sta}"] = inv
        except Exception as e:
            log.warning(f"  XML inválido {xml.name}: {e}")
    return inv_map

def preprocessar(tr, inv):
    """detrend → demean → taper → remove_response → bandpass."""
    try:
        tr = tr.copy()
        if abs(tr.stats.sampling_rate - SAMPLING_RATE) > 0.01:
            tr.resample(SAMPLING_RATE)
        tr.detrend("linear")
        tr.detrend("demean")
        tr.taper(max_percentage=0.05)
        tr.remove_response(
            inventory=inv, output="VEL",
            pre_filt=PRE_FILT, water_level=WATER_LEVEL,
        )
        tr.filter("bandpass",
                  freqmin=BANDPASS[0], freqmax=BANDPASS[1],
                  zerophase=True)
        return tr
    except Exception:
        return None
def zscore(janela: np.ndarray):
    mu = janela.mean()
    sd = janela.std()
    if sd < 1e-12:
        return None
    return ((janela - mu) / sd).astype(np.float32)

def extrair_janela_central(dados: np.ndarray) -> np.ndarray:
    """1 janela de JANELA_NPTS amostras centrada no meio do trace."""
    n = len(dados)
    if n < JANELA_NPTS:
        return None
    meio = n // 2
    ini = meio - JANELA_NPTS // 2
    fim = ini + JANELA_NPTS
    if ini < 0 or fim > n:
        return None
    return dados[ini:fim]

def extrair_todas_janelas(dados: np.ndarray):
    """Sliding window com STEP_NPTS."""
    out = []
    i = 0
    while i + JANELA_NPTS <= len(dados):
        out.append(dados[i:i + JANELA_NPTS])
        i += STEP_NPTS
    return out

# ---------- Processamento ANÔMALO ----------
def processar_events(inventarios):
    log.info("─" * 70)
    log.info("FASE 1/3 — Processando EVENTS (anômalo)")
    log.info("─" * 70)
    arquivos = sorted(DIR_EVENTS.rglob("*.ms"))
    log.info(f"  {len(arquivos)} arquivos .ms encontrados")

    X, meta = [], []
    t0 = time.time()

    for i, path in enumerate(arquivos, 1):
        try:
            st = obspy.read(str(path), format="MSEED")
        except Exception:
            stats["ev_erro_read"] += 1
            continue

        # dedup por NET.STA (mantém primeiro location code)
        vistos = set()
        for tr in st:
            chave = f"{tr.stats.network}.{tr.stats.station}"
            if chave in vistos:
                continue
            vistos.add(chave)

            if chave not in inventarios:
                stats["ev_sem_xml"] += 1
                continue

            tr_proc = preprocessar(tr, inventarios[chave])
            if tr_proc is None:
                stats["ev_erro_preproc"] += 1
                continue

            dados = tr_proc.data.astype(np.float32)
            janela = extrair_janela_central(dados)
            if janela is None:
                stats["ev_trace_curto"] += 1
                continue

            janela_norm = zscore(janela)
            if janela_norm is None:
                stats["ev_janela_constante"] += 1
                continue

            X.append(janela_norm)
            meta.append({
                "evid": path.stem,
                "net_sta": chave,
                "timestamp": float(tr_proc.stats.starttime.timestamp),
            })
            stats["ev_ok"] += 1

        if i % 100 == 0 or i == len(arquivos):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(arquivos) - i) / rate if rate > 0 else 0
            log.info(f"  {i}/{len(arquivos)} | janelas={len(X)} "
                     f"| {rate:.1f} ev/s | ETA {eta/60:.1f}min")

    X = np.array(X, dtype=np.float32)
    log.info(f"  TOTAL anômalo: {X.shape}")
    return X, meta


# ---------- Processamento NORMAL ----------
def processar_continuous(inventarios):
    log.info("─" * 70)
    log.info("FASE 2/3 — Processando CONTINUOUS (normal)")
    log.info("─" * 70)
    arquivos = sorted(DIR_CONT.rglob("*.ms"))
    log.info(f"  {len(arquivos)} arquivos .ms encontrados")

    X, meta = [], []
    t0 = time.time()

    for i, path in enumerate(arquivos, 1):
        try:
            st = obspy.read(str(path), format="MSEED")
        except Exception:
            stats["no_erro_read"] += 1
            continue

        for tr in st:
            chave = f"{tr.stats.network}.{tr.stats.station}"
            if chave not in inventarios:
                stats["no_sem_xml"] += 1
                continue

            tr_proc = preprocessar(tr, inventarios[chave])
            if tr_proc is None:
                stats["no_erro_preproc"] += 1
                continue

            dados = tr_proc.data.astype(np.float32)
            janelas = extrair_todas_janelas(dados)
            ts_base = float(tr_proc.stats.starttime.timestamp)

            for k, j in enumerate(janelas):
                jn = zscore(j)
                if jn is None:
                    continue
                X.append(jn)
                meta.append({
                    "evid": f"cont_{path.stem}_{k}",
                    "net_sta": chave,
                    "timestamp": ts_base + k * STEP_NPTS / SAMPLING_RATE,
                })
                stats["no_ok"] += 1

        elapsed = time.time() - t0
        log.info(f"  {i}/{len(arquivos)} | janelas={len(X)} "
                 f"| {elapsed/60:.1f}min")

    log.info(f"  Total bruto normal: {len(X)}")

    # amostragem para N_NORMAL_ALVO
    if len(X) > N_NORMAL_ALVO:
        idx = rng.choice(len(X), size=N_NORMAL_ALVO, replace=False)
        X = [X[i] for i in idx]
        meta = [meta[i] for i in idx]
        log.info(f"  Após amostragem: {len(X)}")

    X = np.array(X, dtype=np.float32)
    return X, meta


# ---------- Splits ----------
def split_por_estacao(meta_all, y_all):
    estacoes = sorted({m["net_sta"] for m in meta_all})
    rng_est = np.random.default_rng(SEED)
    rng_est.shuffle(estacoes)
    n = len(estacoes)
    train_st = set(estacoes[:int(n * 0.70)])
    val_st = set(estacoes[int(n * 0.70):int(n * 0.85)])
    test_st = set(estacoes[int(n * 0.85):])

    splits = np.empty(len(meta_all), dtype=object)
    for i, m in enumerate(meta_all):
        if m["net_sta"] in train_st:
            splits[i] = "train"
        elif m["net_sta"] in val_st:
            splits[i] = "val"
        else:
            splits[i] = "test"
    return splits, {"train": sorted(train_st),
                    "val": sorted(val_st),
                    "test": sorted(test_st)}


def split_temporal(meta_all):
    timestamps = np.array([m["timestamp"] for m in meta_all])
    ordem = np.argsort(timestamps)
    n = len(timestamps)
    pos = np.empty(n, dtype=int)
    pos[ordem] = np.arange(n)
    splits = np.empty(n, dtype=object)
    for i in range(n):
        p = pos[i] / n
        if p < 0.70:
            splits[i] = "train"
        elif p < 0.85:
            splits[i] = "val"
        else:
            splits[i] = "test"
    return splits


# ---------- Salvamento ----------
def salvar_split(nome, X, y, meta, splits, info_extra=None):
    masks = {s: splits == s for s in ("train", "val", "test")}
    payload = {}
    for s in ("train", "val", "test"):
        payload[f"X_{s}"] = X[masks[s]]
        payload[f"y_{s}"] = y[masks[s]]
    out_path = DIR_OUT / f"dataset_v3_split_{nome}.npz"
    np.savez_compressed(out_path, **payload)
    log.info(f"  OK  {out_path.name}")
    for s in ("train", "val", "test"):
        n_norm = int((payload[f"y_{s}"] == 0).sum())
        n_anom = int((payload[f"y_{s}"] == 1).sum())
        log.info(f"    {s}: normal={n_norm}  anômalo={n_anom}")
    return {s: {"normal": int((payload[f"y_{s}"] == 0).sum()),
                "anomalo": int((payload[f"y_{s}"] == 1).sum())}
            for s in ("train", "val", "test")}


# ---------- Main ----------
def main():
    log.info("=" * 70)
    log.info("PASSO 2 v3 — Pipeline genérico de detecção de anomalias")
    log.info(f"  Janela: {JANELA_S}s ({JANELA_NPTS} amostras)")
    log.info(f"  Overlap: {OVERLAP*100:.0f}%")
    log.info(f"  N normal alvo: {N_NORMAL_ALVO}")
    log.info(f"  Saída: {DIR_OUT}")
    log.info("=" * 70)

    log.info("Carregando inventários...")
    inventarios = carregar_inventarios()
    log.info(f"  {len(inventarios)} estações")

    X_anom, meta_anom = processar_events(inventarios)
    X_norm, meta_norm = processar_continuous(inventarios)

    log.info("─" * 70)
    log.info("FASE 3/3 — Combinando, split e salvando")
    log.info("─" * 70)

    y_anom = np.ones(len(X_anom), dtype=np.int8)
    y_norm = np.zeros(len(X_norm), dtype=np.int8)

    X_all = np.concatenate([X_norm, X_anom], axis=0)
    y_all = np.concatenate([y_norm, y_anom], axis=0)
    meta_all = meta_norm + meta_anom
    log.info(f"  Total combinado: {X_all.shape}  "
             f"(normal={len(X_norm)}, anômalo={len(X_anom)})")

    log.info("Calculando split por estação...")
    splits_est, estacoes_info = split_por_estacao(meta_all, y_all)
    log.info(f"  train: {len(estacoes_info['train'])} estações")
    log.info(f"  val:   {len(estacoes_info['val'])} estações")
    log.info(f"  test:  {len(estacoes_info['test'])} estações")

    log.info("Calculando split temporal...")
    splits_tmp = split_temporal(meta_all)

    log.info("Salvando dataset_v3_split_estacao.npz...")
    counts_est = salvar_split("estacao", X_all, y_all, meta_all, splits_est)

    log.info("Salvando dataset_v3_split_temporal.npz...")
    counts_tmp = salvar_split("temporal", X_all, y_all, meta_all, splits_tmp)

    # inventário csv
    with open(DIR_OUT / "inventario_v3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "tipo", "evid", "net_sta", "timestamp",
                    "split_estacao", "split_temporal"])
        for i, m in enumerate(meta_all):
            w.writerow([i, "anomalo" if y_all[i] == 1 else "normal",
                        m["evid"], m["net_sta"], m["timestamp"],
                        splits_est[i], splits_tmp[i]])
    log.info(f"  OK  inventario_v3.csv")

    # info json
    info = {
        "parametros": {
            "sampling_rate": SAMPLING_RATE,
            "janela_s": JANELA_S, "janela_npts": JANELA_NPTS,
            "overlap": OVERLAP, "step_npts": STEP_NPTS,
            "pre_filt": PRE_FILT, "water_level": WATER_LEVEL,
            "bandpass": BANDPASS, "normalizacao": "zscore_por_janela",
            "pipeline": ["detrend_linear", "demean", "taper_5pct",
                         "remove_response_VEL", "bandpass_zerophase"],
            "n_normal_alvo": N_NORMAL_ALVO,
            "seed": SEED,
        },
        "shapes": {
            "X_anomalo": list(X_anom.shape),
            "X_normal": list(X_norm.shape),
            "X_combinado": list(X_all.shape),
        },
        "estatisticas_processamento": dict(stats),
        "split_estacao": {
            "estacoes": estacoes_info,
            "counts": counts_est,
        },
        "split_temporal": {"counts": counts_tmp},
        "metricas_recomendadas": {
            "primaria": "AUC-PR",
            "secundaria": "AUC-ROC",
            "ponto": "F1 no threshold ótimo (val)",
            "baseline_auc_pr": round(len(X_anom) / len(X_all), 4),
        },
    }
    with open(DIR_OUT / "dataset_v3_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    log.info(f"  OK  dataset_v3_info.json")

    log.info("=" * 70)
    log.info("RESUMO FINAL")
    log.info(f"  X_anomalo: {X_anom.shape}")
    log.info(f"  X_normal:  {X_norm.shape}")
    log.info(f"  Razão normal:anômalo = 1:{len(X_anom)/max(len(X_norm),1):.2f}")
    log.info(f"  Baseline AUC-PR (aleatório): {len(X_anom)/len(X_all):.4f}")
    log.info("  Estatísticas detalhadas:")
    for k, v in sorted(stats.items()):
        log.info(f"    {k:25s} = {v}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()