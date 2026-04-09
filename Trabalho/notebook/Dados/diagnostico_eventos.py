r"""
Diagnóstico — verifica se os .ms baixados têm dados reais ou só headers.
"""
import random
from pathlib import Path
import obspy

DIR_EVENTS = Path(r"G:\Meu Drive\TCC\data2\raw\events")

todos = list(DIR_EVENTS.rglob("*.ms"))
print(f"Total de arquivos: {len(todos)}")

if not todos:
    print("NENHUM arquivo encontrado!")
    raise SystemExit

# tamanhos
tamanhos = [(p, p.stat().st_size) for p in todos]
tamanhos.sort(key=lambda x: x[1])

print(f"\nTamanho mínimo:  {tamanhos[0][1]/1024:.1f} KB  ({tamanhos[0][0].name})")
print(f"Tamanho mediano: {tamanhos[len(tamanhos)//2][1]/1024:.1f} KB")
print(f"Tamanho máximo:  {tamanhos[-1][1]/1024:.1f} KB  ({tamanhos[-1][0].name})")
print(f"Tamanho total:   {sum(s for _, s in tamanhos)/1024/1024:.1f} MB")

# inspeciona 5 aleatórios
print("\n" + "=" * 60)
print("INSPECIONANDO 5 ARQUIVOS ALEATÓRIOS:")
print("=" * 60)

for path in random.sample(todos, min(5, len(todos))):
    print(f"\n--- {path.name} ({path.stat().st_size/1024:.1f} KB) ---")
    try:
        st = obspy.read(str(path))
        print(f"  Streams: {len(st)}")
        for i, tr in enumerate(st):
            dur = tr.stats.endtime - tr.stats.starttime
            print(f"  [{i}] {tr.stats.network}.{tr.stats.station}.{tr.stats.channel} "
                  f"| fs={tr.stats.sampling_rate}Hz "
                  f"| npts={tr.stats.npts} "
                  f"| dur={dur:.1f}s "
                  f"| dtype={tr.data.dtype}")
    except Exception as e:
        print(f"  ERRO: {e}")