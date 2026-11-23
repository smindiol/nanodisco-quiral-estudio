"""Ejecuta las etapas A, B y C del análisis, con progreso en pantalla y reanudación.

Uso:
  python -u ejecutar.py --base ~/smindiola/computacion --procs 32 --B 50000
  python -u ejecutar.py --base ... --humo        # prueba corta: 1 estado, 3 temperaturas, B=200
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import dmi_analisis as da  # noqa: E402
import informe  # noqa: E402


def hms(s):
    s = int(s)
    return f"{s // 3600:d}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--salida", help="Directorio de resultados; por defecto, dentro de --base")
    ap.add_argument("--B", type=int, default=50000)
    ap.add_argument("--trozo", type=int, default=2500)
    ap.add_argument("--humo", action="store_true")
    a = ap.parse_args()
    base = os.path.expanduser(a.base)
    res = os.path.abspath(a.salida) if a.salida else os.path.join(base, "resultados_humo" if a.humo else "resultados")
    dA, dB = os.path.join(res, "etapaA"), os.path.join(res, "etapaB")
    for d in (dA, dB):
        os.makedirs(d, exist_ok=True)

    datos = da.descubrir(base)
    if not datos:
        ap.error("No se encontraron archivos crudos. Configure --base con la base externa; no viene incluida.")
    escalera = da.escalera_global(datos)
    B = 200 if a.humo else a.B
    if a.humo:
        e0 = sorted(datos)[0]
        datos = {e0: datos[e0][:3]}
        escalera = np.array([da.temperatura_de(r) for r in datos[e0]])
    print(f"[{time.strftime('%H:%M:%S')}] estados: {sorted(datos)} | temperaturas por estado: "
          f"{sorted({len(v) for v in datos.values()})} | escalera {escalera.max()}..{escalera.min()} K | B={B} | procesos={a.procs}",
          flush=True)
    faltan = {e: len(escalera) - len(v) for e, v in datos.items() if len(v) != len(escalera)}
    if faltan:
        print("AVISO: estados con temperaturas faltantes:", faltan, flush=True)

    # ---------------- Etapa A
    tareas = []
    for e, rutas in datos.items():
        os.makedirs(os.path.join(dA, f"estado_{e}"), exist_ok=True)
        for r in rutas:
            tareas.append((e, r, os.path.join(dA, f"estado_{e}", os.path.basename(r)), escalera))
    t0 = time.time()
    print(f"\n=== ETAPA A: {len(tareas)} tareas (estado x temperatura) ===", flush=True)
    errores = []
    with Pool(a.procs, maxtasksperchild=20) as pool:
        for i, (t, r) in enumerate(zip(tareas, pool.imap(da_seguro_A, tareas, chunksize=1)), 1):
            if r.startswith("ERROR"):
                errores.append(r)
            if i % 20 == 0 or i == len(tareas) or r.startswith("ERROR"):
                el = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] A {i}/{len(tareas)} ({100 * i / len(tareas):.0f} %) | "
                      f"transcurrido {hms(el)} | restante ~{hms(el / i * (len(tareas) - i))} | último: estado {t[0]} "
                      f"T={da.temperatura_de(t[1])} {r}", flush=True)
    if errores:
        print(f"ETAPA A con {len(errores)} errores; no sigo a la etapa B:\n" + "\n".join(errores[:20]), flush=True)
        sys.exit(1)

    # ---------------- Etapa B
    tareasB = []
    for e in sorted(datos):
        rA = sorted(os.path.join(dA, f"estado_{e}", os.path.basename(r)) for r in datos[e])
        tareasB.append((e, rA, os.path.join(dB, f"estado_{e}.npz"), B, 20261117 + e, min(a.trozo, B)))
    t1 = time.time()
    print(f"\n=== ETAPA B: bootstrap entre copias, {len(tareasB)} estados, B={B} ===", flush=True)
    with Pool(min(a.procs, len(tareasB))) as pool:
        for t, r in zip(tareasB, pool.imap(da_seguro_B, tareasB)):
            print(f"[{time.strftime('%H:%M:%S')}] B estado {t[0]}: {r} | transcurrido {hms(time.time() - t1)}", flush=True)
            if r.startswith("ERROR"):
                sys.exit(1)

    # ---------------- Etapa C
    print("\n=== ETAPA C: tablas, figuras y RESUMEN.md ===", flush=True)
    informe.generar(dB, res)
    open(os.path.join(res, "HECHO"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"\n[{time.strftime('%H:%M:%S')}] TERMINADO en {hms(time.time() - t0)}. Resultados en {res}", flush=True)


def da_seguro_A(t):
    try:
        return da.etapa_A(t)
    except Exception as ex:  # el error queda en el log con la ruta
        return f"ERROR {t[1]}: {type(ex).__name__}: {ex}"


def da_seguro_B(t):
    try:
        return da.etapa_B(t)
    except Exception as ex:
        import traceback
        return f"ERROR estado {t[0]}: {type(ex).__name__}: {ex}\n{traceback.format_exc()}"


if __name__ == "__main__":
    main()
