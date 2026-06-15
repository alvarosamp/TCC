"""
fix_tflm.py  -  pre:scripts/fix_tflm.py

Corrige tres problemas ao compilar Arduino_TensorFlowLite no ESP32:

  1) Cria Windows Junctions de include/<dir> para cada subdiretorio
     da biblioteca (exceto peripherals/). O diretorio include/ e
     SEMPRE adicionado ao CPPPATH do projeto pelo PlatformIO, entao
     #include "tensorflow/lite/..." e resolvido automaticamente.

  2) Cria stub de tensorflow/lite/version.h se nao existir.
     O port tflite-micro-arduino-examples nao inclui esse arquivo
     (e especifico do TFLite full), mas main.cpp precisa de
     TFLITE_SCHEMA_VERSION para a verificacao de compatibilidade.

  3) Remove os arquivos de peripherals/ da compilacao.
     Esse codigo e especifico para Arduino Nano 33 BLE (nRF52840)
     e nao compila no ESP32 (#error "unsupported board").
"""
Import("env")
import os
import subprocess
from SCons.Script import GetOption

# TFLite Micro compila muitos arquivos C++ pesados. Em maquinas com muitos
# nucleos, o PlatformIO tenta usar todos eles e o cc1plus pode ficar sem RAM.
max_jobs = int(os.environ.get("TFLM_MAX_BUILD_JOBS", "1"))
current_jobs = GetOption("num_jobs")
if current_jobs > max_jobs:
    raise RuntimeError(
        "[fix_tflm] Compile com menos jobs para nao estourar a RAM do cc1plus.\n"
        f"[fix_tflm] Jobs atuais: {current_jobs}; maximo recomendado: {max_jobs}.\n"
        "[fix_tflm] Build : pio run -j1\n"
        "[fix_tflm] Upload: pio run -j1 --target upload\n"
        "[fix_tflm] Ou defina PLATFORMIO_RUN_JOBS=1 antes de compilar."
    )
print(f"[fix_tflm] Jobs SCons      : {current_jobs}")

project_dir = env.subst("$PROJECT_DIR")
pioenv      = env.subst("$PIOENV")

tflm_src    = os.path.join(project_dir, ".pio", "libdeps", pioenv,
                            "Arduino_TensorFlowLite", "src")
include_dir = os.path.join(project_dir, "include")

print("[fix_tflm] project_dir :", project_dir)
print("[fix_tflm] tflm_src    :", tflm_src)
print("[fix_tflm] src existe  :", os.path.isdir(tflm_src))

if os.path.isdir(tflm_src):

    # ----------------------------------------------------------
    # 1) Junctions em include/ -> subdiretorios da biblioteca
    # ----------------------------------------------------------
    SKIP_DIRS = {"peripherals"}
    for name in os.listdir(tflm_src):
        if name in SKIP_DIRS:
            continue
        src_path = os.path.join(tflm_src, name)
        dst_path = os.path.join(include_dir, name)
        if not os.path.isdir(src_path):
            continue
        if os.path.exists(dst_path):
            print(f"[fix_tflm] Junction ja existe : include/{name}")
            continue
        if os.name == "nt":
            cmd = f'mklink /J "{dst_path}" "{src_path}"'
            result = subprocess.run(
                f'cmd /c {cmd}', capture_output=True, text=True, shell=True
            )
            if result.returncode == 0:
                print(f"[fix_tflm] Junction criada    : include/{name}")
            else:
                err = (result.stderr or result.stdout).strip()
                print(f"[fix_tflm] FALHA junction {name}: {err}")
        else:
            try:
                os.symlink(src_path, dst_path, target_is_directory=True)
                print(f"[fix_tflm] Symlink criado     : include/{name}")
            except OSError as exc:
                print(f"[fix_tflm] FALHA symlink {name}: {exc}")

    # ----------------------------------------------------------
    # 2) Stub de version.h (ausente no tflite-micro-arduino-examples)
    # ----------------------------------------------------------
    version_h = os.path.join(tflm_src, "tensorflow", "lite", "version.h")
    version_stub = (
        "#pragma once\n"
        "// Stub: version.h nao incluido no tflite-micro-arduino-examples\n"
        "// Schema version 3 e usada por modelos TFLite Micro atuais.\n"
        "#ifndef TFLITE_SCHEMA_VERSION\n"
        "#define TFLITE_SCHEMA_VERSION 3\n"
        "#endif\n"
    )
    os.makedirs(os.path.dirname(version_h), exist_ok=True)
    current_version_h = ""
    if os.path.exists(version_h):
        with open(version_h, "r") as f:
            current_version_h = f.read()
    if current_version_h != version_stub:
        with open(version_h, "w") as f:
            f.write(version_stub)
        print("[fix_tflm] Stub atualizado: src/tensorflow/lite/version.h")
    else:
        print("[fix_tflm] version.h ja esta ok.")

else:
    print("[fix_tflm] AVISO: biblioteca nao encontrada:", tflm_src)
    print("[fix_tflm] Execute 'pio pkg install' e recompile.")

# ------------------------------------------------------------------
# 3) Exclui peripherals/ da compilacao
# ------------------------------------------------------------------
def skip_peripherals(node):
    path = str(node.path).replace("\\", "/")
    return None if "/peripherals/" in path else node

env.AddBuildMiddleware(skip_peripherals, "*.cpp")
print("[fix_tflm] Filtro de peripherals instalado.")
