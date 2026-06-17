"""Gera slides do TCC em formato .pptx."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

AZUL_ESCURO = RGBColor(0x1A, 0x37, 0x5E)
AZUL_MEDIO  = RGBColor(0x1F, 0x6F, 0xAB)
VERDE       = RGBColor(0x27, 0xAE, 0x60)
VERMELHO    = RGBColor(0xC0, 0x39, 0x2B)
CINZA_CLARO = RGBColor(0xF2, 0xF6, 0xFC)
BRANCO      = RGBColor(0xFF, 0xFF, 0xFF)
PRETO       = RGBColor(0x1C, 0x1C, 0x1C)

W = Inches(13.33)
H = Inches(7.5)


def nova_slide_em_branco(prs):
    layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout)
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = BRANCO
    return slide


def caixa(slide, x, y, w, h, cor_fundo=None, cor_borda=None):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    if cor_fundo:
        shape.fill.solid()
        shape.fill.fore_color.rgb = cor_fundo
    else:
        shape.fill.background()
    if cor_borda:
        shape.line.color.rgb = cor_borda
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def texto(slide, txt, x, y, w, h, tamanho=18, negrito=False, cor=PRETO,
          alinhamento=PP_ALIGN.LEFT, italico=False):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf.text_frame.word_wrap = True
    p = tf.text_frame.paragraphs[0]
    p.alignment = alinhamento
    run = p.add_run()
    run.text = txt
    run.font.size = Pt(tamanho)
    run.font.bold = negrito
    run.font.italic = italico
    run.font.color.rgb = cor
    return tf


def faixa_topo(slide, titulo, subtitulo=""):
    caixa(slide, 0, 0, 13.33, 1.3, cor_fundo=AZUL_ESCURO)
    texto(slide, titulo, 0.3, 0.1, 12.5, 0.7, tamanho=28, negrito=True,
          cor=BRANCO, alinhamento=PP_ALIGN.LEFT)
    if subtitulo:
        texto(slide, subtitulo, 0.3, 0.75, 12.5, 0.45, tamanho=14,
              cor=RGBColor(0xB0, 0xC8, 0xE8), alinhamento=PP_ALIGN.LEFT)


def slide_capa(prs):
    slide = nova_slide_em_branco(prs)
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=AZUL_ESCURO)
    caixa(slide, 0, 5.5, 13.33, 2.0, cor_fundo=AZUL_MEDIO)
    texto(slide,
          "Pipeline Generico de TinyML/MLOps\npara Series Temporais",
          0.5, 1.2, 12.3, 2.0, tamanho=34, negrito=True,
          cor=BRANCO, alinhamento=PP_ALIGN.CENTER)
    texto(slide,
          "Deteccao de Anomalias em Dados Sismicos com Validacao em ESP32",
          0.5, 3.2, 12.3, 0.8, tamanho=18, cor=RGBColor(0xB0, 0xC8, 0xE8),
          alinhamento=PP_ALIGN.CENTER)
    texto(slide,
          "TCC — Engenharia / Ciencia de Dados\n2026",
          0.5, 5.7, 12.3, 1.0, tamanho=15, cor=BRANCO,
          alinhamento=PP_ALIGN.CENTER)


def slide_visao_geral(prs):
    slide = nova_slide_em_branco(prs)
    faixa_topo(slide, "Visao Geral do Sistema", "Do dado bruto ao dispositivo de borda")
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=CINZA_CLARO)
    faixa_topo(slide, "Visao Geral do Sistema", "Do dado bruto ao dispositivo de borda")

    etapas = [
        ("Dados\nBrutos", 0.2),
        ("Adapter\nDominio", 1.6),
        ("Dataset\nNPZ", 3.0),
        ("Treino\nMLflow", 4.4),
        ("Quality\nGate", 5.8),
        ("Export\nTFLite", 7.2),
        ("OTA\nHMAC", 8.6),
        ("ESP32\nTFLM", 10.0),
    ]
    for i, (nome, x) in enumerate(etapas):
        cor = AZUL_MEDIO if i % 2 == 0 else AZUL_ESCURO
        caixa(slide, x, 2.0, 1.1, 1.1, cor_fundo=cor)
        texto(slide, nome, x, 2.0, 1.1, 1.1, tamanho=10, negrito=True,
              cor=BRANCO, alinhamento=PP_ALIGN.CENTER)
        if i < len(etapas) - 1:
            texto(slide, "->", x + 1.1, 2.35, 0.5, 0.5, tamanho=14,
                  cor=AZUL_ESCURO, alinhamento=PP_ALIGN.CENTER)

    pontos = [
        "Pipeline generico: aplicavel a qualquer dominio de series temporais",
        "Preprocessing edge-aware: evita etapas inviáveis no microcontrolador",
        "Rastreabilidade completa: MLflow + manifestos JSON em cada etapa",
        "OTA com assinatura HMAC-SHA256: seguranca na atualizacao do modelo",
    ]
    for i, p in enumerate(pontos):
        texto(slide, "• " + p, 0.4, 3.5 + i * 0.7, 12.5, 0.6, tamanho=14, cor=PRETO)


def slide_dataset(prs):
    slide = nova_slide_em_branco(prs)
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=CINZA_CLARO)
    faixa_topo(slide, "Dataset Sismico", "MiniSEED — eventos e background continuo")

    col = [
        ("Origem", "Estacoes sismologicas — formato MiniSEED"),
        ("Classes", "Normal (background) e Anomalo (evento sismico)"),
        ("Janela", "800 amostras = 20 s a 40 Hz"),
        ("Overlap", "50% (step de 10 s)"),
        ("Split", "Por evento (sem vazamento temporal)"),
        ("Balanceamento", "~88% normal / ~12% anomalo"),
    ]
    for i, (k, v) in enumerate(col):
        y = 1.6 + i * 0.75
        caixa(slide, 0.4, y, 3.0, 0.6, cor_fundo=AZUL_ESCURO)
        texto(slide, k, 0.4, y, 3.0, 0.6, tamanho=13, negrito=True,
              cor=BRANCO, alinhamento=PP_ALIGN.CENTER)
        texto(slide, v, 3.6, y, 9.0, 0.6, tamanho=13, cor=PRETO)

    texto(slide, "Preprocessamento edge-aware:", 0.4, 6.2, 5.0, 0.4,
          tamanho=13, negrito=True, cor=AZUL_ESCURO)
    texto(slide,
          "resample 40 Hz -> detrend -> demean -> taper 5% -> bandpass 0.5-15 Hz -> zscore/janela",
          0.4, 6.6, 12.5, 0.5, tamanho=12, italico=True, cor=PRETO)


def slide_modelos(prs):
    slide = nova_slide_em_branco(prs)
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=CINZA_CLARO)
    faixa_topo(slide, "Comparativo de Modelos", "AUC-PR como metrica primaria (deteccao desbalanceada)")

    cabecalho = ["Modelo", "AUC-PR", "F1", "Precision", "Recall", "FP/h"]
    linhas = [
        ["Optuna Tiny CNN v4 (ref.)", "0.9127", "0.8526", "0.8982", "0.8114", "4.90"],
        ["Tiny CNN (atual)", "0.8775", "0.8109", "0.8431", "0.7810", "6.75"],
        ["Tiny CNN (baseline)", "0.8982", "0.7951", "0.7310", "0.8716", "16.94"],
        ["Tiny TCN", "0.8964", "0.7666", "0.6790", "0.8801", "21.98"],
        ["Random Forest (Optuna)", "0.8127", "0.7367", "0.7974", "0.6846", "9.26"],
        ["STA/LTA (baseline trad.)", "0.1662", "0.2760", "0.1773", "0.6230", "—"],
    ]
    larguras = [4.5, 1.4, 1.2, 1.4, 1.3, 1.3]
    xs = [0.3]
    for l in larguras[:-1]:
        xs.append(xs[-1] + l)

    y = 1.5
    for j, h in enumerate(cabecalho):
        caixa(slide, xs[j], y, larguras[j] - 0.05, 0.45, cor_fundo=AZUL_ESCURO)
        texto(slide, h, xs[j], y, larguras[j], 0.45, tamanho=11, negrito=True,
              cor=BRANCO, alinhamento=PP_ALIGN.CENTER)

    for i, linha in enumerate(linhas):
        y = 2.05 + i * 0.6
        destaque = i == 1
        cor_bg = AZUL_MEDIO if destaque else (BRANCO if i % 2 == 0 else CINZA_CLARO)
        for j, cel in enumerate(linha):
            caixa(slide, xs[j], y, larguras[j] - 0.05, 0.55, cor_fundo=cor_bg)
            texto(slide, cel, xs[j], y, larguras[j], 0.55, tamanho=11,
                  negrito=destaque, cor=BRANCO if destaque else PRETO,
                  alinhamento=PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT)

    texto(slide, "* Tiny CNN atual: quality gate aprovado em todos os criterios",
          0.3, 6.8, 12.5, 0.4, tamanho=11, italico=True, cor=AZUL_MEDIO)


def slide_validacao_esp32(prs):
    slide = nova_slide_em_branco(prs)
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=CINZA_CLARO)
    faixa_topo(slide, "Validacao Embarcada — ESP32", "Chip: ESP32-D0WD-V3 | TensorFlow Lite Micro")

    itens_ok = [
        "ESP32 detectado e identificado no WSL (/dev/ttyUSB0)",
        "Firmware compilado: RAM 31.0% | Flash 20.3%",
        "Upload via esptool: OK",
        "Modelo float32 carregado e invocado no ESP32",
        "Saida serial CSV com score, latencia e CPU% por inferencia",
    ]
    for i, item in enumerate(itens_ok):
        caixa(slide, 0.4, 1.55 + i * 0.7, 0.5, 0.5, cor_fundo=VERDE)
        texto(slide, "OK", 0.4, 1.55 + i * 0.7, 0.5, 0.5, tamanho=10,
              negrito=True, cor=BRANCO, alinhamento=PP_ALIGN.CENTER)
        texto(slide, item, 1.0, 1.55 + i * 0.7, 11.5, 0.5, tamanho=13, cor=PRETO)

    caixa(slide, 0.4, 5.2, 12.5, 1.5, cor_fundo=RGBColor(0xFF, 0xF3, 0xCD))
    texto(slide, "Diagnostico int8: kernel REDUCE_MAX quantizado incompativel com esta versao do TFLite Micro.",
          0.5, 5.3, 12.3, 0.45, tamanho=12, cor=RGBColor(0x7D, 0x60, 0x08))
    texto(slide, "Causa: head_pooling=avgmax -> GlobalMaxPooling -> REDUCE_MAX. Correcao: substituir por avg (GlobalAveragePooling).",
          0.5, 5.75, 12.3, 0.55, tamanho=11, italico=True, cor=RGBColor(0x7D, 0x60, 0x08))


def slide_ota(prs):
    slide = nova_slide_em_branco(prs)
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=CINZA_CLARO)
    faixa_topo(slide, "Fluxo OTA Simulado", "Atualizacao de modelo com integridade e assinatura")

    etapas = [
        ("production_manifest.json", "Modelo aprovado pelo quality gate"),
        ("ota_manifest.json", "Manifesto com versao, alvo e estrategia"),
        ("Pacote OTA", "artifact.tflite + SHA-256 + HMAC-SHA256"),
        ("validation_report.json", "Verificacao de integridade automatica"),
        ("releases/latest.json", "Publicacao local (simula repositorio)"),
        ("device_update_check", "Dispositivo consulta e confirma compatibilidade"),
        ("install_report.json", "Instalacao simulada com log completo"),
        ("rollback_report.json", "Reversao automatica em caso de falha"),
    ]
    for i, (titulo, desc) in enumerate(etapas):
        col = i // 4
        row = i % 4
        x = 0.3 + col * 6.5
        y = 1.6 + row * 1.3
        caixa(slide, x, y, 6.2, 1.1, cor_fundo=AZUL_ESCURO if col == 0 else AZUL_MEDIO)
        texto(slide, titulo, x + 0.1, y + 0.05, 6.0, 0.45, tamanho=11,
              negrito=True, cor=BRANCO)
        texto(slide, desc, x + 0.1, y + 0.5, 6.0, 0.5, tamanho=10,
              cor=RGBColor(0xD0, 0xE8, 0xFF))

    texto(slide, "Versao atual: seismic_edge_v1_tiny_cnn_20260614 | SHA-256 verificado | Assinatura HMAC valida",
          0.3, 6.9, 12.5, 0.4, tamanho=11, italico=True, cor=AZUL_ESCURO)


def slide_drift(prs):
    slide = nova_slide_em_branco(prs)
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=CINZA_CLARO)
    faixa_topo(slide, "Drift Detection", "Monitoramento de distribuicao e decisao de retreino")

    metricas = [
        ("Z-shift (max)", "0.0176", "Deslocamento de media", "Baixo"),
        ("PSI (max)", "0.3463", "Mudanca de distribuicao", "Alto"),
        ("KS p-value (min)", "0.000033", "Diferenca estatistica", "Significativa"),
    ]
    for i, (nome, valor, desc, nivel) in enumerate(metricas):
        x = 0.3 + i * 4.3
        cor = VERMELHO if nivel in ("Alto", "Significativa") else VERDE
        caixa(slide, x, 1.6, 4.0, 1.8, cor_fundo=BRANCO, cor_borda=cor)
        texto(slide, nome, x + 0.1, 1.65, 3.8, 0.5, tamanho=12, negrito=True, cor=AZUL_ESCURO)
        texto(slide, valor, x + 0.1, 2.1, 3.8, 0.6, tamanho=22, negrito=True, cor=cor)
        texto(slide, desc + " — " + nivel, x + 0.1, 2.7, 3.8, 0.45, tamanho=10, cor=PRETO)

    caixa(slide, 0.3, 3.7, 12.5, 1.2, cor_fundo=RGBColor(0xEB, 0xF5, 0xFB))
    texto(slide, "Interpretacao: medias globais com pequeno deslocamento, mas distribuicao interna\nmudou significativamente (PSI > 0.2, KS p-value << 0.05).",
          0.5, 3.8, 12.2, 1.0, tamanho=13, cor=PRETO)

    caixa(slide, 0.3, 5.1, 5.8, 0.7, cor_fundo=AZUL_MEDIO)
    texto(slide, "Politica: retrain_recommended", 0.4, 5.2, 5.6, 0.5,
          tamanho=13, negrito=True, cor=BRANCO)
    caixa(slide, 6.5, 5.1, 6.5, 0.7, cor_fundo=VERDE)
    texto(slide, "Decisao OTA: build_and_publish_ota", 6.6, 5.2, 6.3, 0.5,
          tamanho=13, negrito=True, cor=BRANCO)

    texto(slide, "Sistema so libera OTA quando existe candidato aprovado no quality gate.",
          0.3, 6.1, 12.5, 0.5, tamanho=12, italico=True, cor=AZUL_ESCURO)


def slide_proximos_passos(prs):
    slide = nova_slide_em_branco(prs)
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=CINZA_CLARO)
    faixa_topo(slide, "Proximos Passos", "Evolucao do sistema e novas direcoes")

    passos = [
        ("Correcao int8", "Substituir GlobalMaxPooling por GlobalAveragePooling e reexportar para inferencia int8 real no ESP32"),
        ("Servidor HTTP OTA", "Endpoint que expoe latest.json e o firmware para download real pelo ESP32"),
        ("Series Temporais Multivariadas", "Nova branch: expandir pipeline para multiplos canais de sensor simultaneos"),
        ("Validacao em Novo Dataset", "Aplicar o pipeline em dominio diferente do sismico (ex: vibracao industrial)"),
        ("Assinatura Assimetrica", "Substituir HMAC-SHA256 por RSA/ECDSA para OTA com chave publica no dispositivo"),
    ]
    for i, (titulo, desc) in enumerate(passos):
        y = 1.55 + i * 1.05
        caixa(slide, 0.3, y, 2.5, 0.85, cor_fundo=AZUL_ESCURO)
        texto(slide, titulo, 0.3, y, 2.5, 0.85, tamanho=11, negrito=True,
              cor=BRANCO, alinhamento=PP_ALIGN.CENTER)
        texto(slide, desc, 3.0, y + 0.1, 10.0, 0.65, tamanho=12, cor=PRETO)


def slide_series_multivariadas(prs):
    slide = nova_slide_em_branco(prs)
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=CINZA_CLARO)
    faixa_topo(slide, "Series Temporais Multivariadas", "Nova direcao — branch em desenvolvimento")

    texto(slide, "O que muda:", 0.4, 1.55, 5.0, 0.45, tamanho=14, negrito=True, cor=AZUL_ESCURO)

    mudancas = [
        ("Dataset", "Multiplos canais de sensor por janela (ex: X, Y, Z de acelerometro)"),
        ("Contrato NPZ", "X_train.shape = (n_janelas, n_amostras, n_canais)"),
        ("Arquitetura", "Convolucoes 2D ou TCN com entrada multicanal"),
        ("Preprocessing", "Normalizacao por canal ou global — decisao critica para edge"),
        ("Firmware", "Tensor de entrada 2D — ajuste no main.cpp e model_config.h"),
        ("Complexidade", "Mais parametros, mais RAM — validar viabilidade no ESP32"),
    ]
    for i, (k, v) in enumerate(mudancas):
        y = 2.1 + i * 0.75
        caixa(slide, 0.4, y, 2.2, 0.6, cor_fundo=AZUL_MEDIO)
        texto(slide, k, 0.4, y, 2.2, 0.6, tamanho=11, negrito=True,
              cor=BRANCO, alinhamento=PP_ALIGN.CENTER)
        texto(slide, v, 2.8, y, 10.0, 0.6, tamanho=12, cor=PRETO)

    caixa(slide, 0.4, 6.7, 12.5, 0.5, cor_fundo=AZUL_ESCURO)
    texto(slide, "Pipeline generico foi projetado para esta extensao: adapter de dominio isola a diferenca de contrato",
          0.5, 6.75, 12.3, 0.4, tamanho=11, cor=BRANCO)


def slide_conclusao(prs):
    slide = nova_slide_em_branco(prs)
    caixa(slide, 0, 0, 13.33, 7.5, cor_fundo=AZUL_ESCURO)
    caixa(slide, 0, 5.8, 13.33, 1.7, cor_fundo=AZUL_MEDIO)

    texto(slide, "Conclusao", 0.5, 0.4, 12.3, 0.8, tamanho=32, negrito=True,
          cor=BRANCO, alinhamento=PP_ALIGN.CENTER)

    conquistas = [
        "Pipeline MLOps completo: do dado bruto ao dispositivo de borda",
        "Modelo tiny_cnn: AUC-PR 0.877 | F1 0.811 | FP/h 6.75 — quality gate aprovado",
        "Validacao fisica real: build, flash e inferencia no ESP32 (chip ESP32-D0WD-V3)",
        "OTA com integridade: HMAC-SHA256 + rollback simulado",
        "Drift detection integrado ao ciclo de vida do modelo",
    ]
    for i, c in enumerate(conquistas):
        texto(slide, "✓  " + c, 0.8, 1.4 + i * 0.8, 11.7, 0.7,
              tamanho=14, cor=RGBColor(0xD5, 0xEA, 0xFF))

    texto(slide,
          "Proximo passo: series temporais multivariadas + OTA real via HTTP",
          0.5, 6.0, 12.3, 0.5, tamanho=13, cor=BRANCO,
          alinhamento=PP_ALIGN.CENTER)


def main():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    slide_capa(prs)
    slide_visao_geral(prs)
    slide_dataset(prs)
    slide_modelos(prs)
    slide_validacao_esp32(prs)
    slide_ota(prs)
    slide_drift(prs)
    slide_proximos_passos(prs)
    slide_series_multivariadas(prs)
    slide_conclusao(prs)

    saida = os.path.join(os.path.dirname(__file__), "..", "docs", "slides_tcc.pptx")
    saida = os.path.abspath(saida)
    prs.save(saida)
    print(f"Slides salvos em: {saida}")
    print(f"Total de slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
