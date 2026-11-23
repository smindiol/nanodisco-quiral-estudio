"""Convergence versus Monte Carlo sweeps, between-path versus within-path variability.

Stage 1 (compute): for every state and temperature reads E and M (10000 sweeps x 100 copies)
and stores per-temperature diagnostics plus R-hat curves versus the number of sweeps.
Usage: python -I pasos_mc.py --datos ~/Descargas/lotes --salida <dir> [--procs 8]
"""
import argparse
import glob
import json
import os
import re
from multiprocessing import Pool

import numpy as np

N_GRID = np.unique(np.round(np.logspace(2, 4, 25)).astype(int))  # 100 ... 10000 sweeps
T_TRAZA = (20.0, 14.3, 10.0, 6.0)
COPIAS_TRAZA = 6
C_SOKAL = 6.0


def rhat(x):
    """Classic Gelman-Rubin potential scale reduction; x has shape (n, chains)."""
    n = x.shape[0]
    medias = x.mean(axis=0)
    W = x.var(axis=0, ddof=1).mean()
    B = n * medias.var(ddof=1)
    var_mas = (n - 1) / n * W + B / n
    return float(np.sqrt(var_mas / W)), float(W), float(B / n)


def tau_int(x):
    """Integrated autocorrelation time (Sokal window) of the chain-averaged autocorrelation."""
    n, m = x.shape
    y = x - x.mean(axis=0)
    f = np.fft.rfft(y, n=2 * n, axis=0)
    ac = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(axis=1)
    if ac[0] <= 0:
        return 1.0
    rho = ac / ac[0]
    tau = 1.0
    for w in range(1, n):
        tau += 2 * rho[w]
        if w >= C_SOKAL * tau:
            break
    return float(max(tau, 1.0))


def una_temperatura(ruta):
    with np.load(ruta, allow_pickle=False) as d:
        E = d["E"].astype(np.float64)
        M = d["M"].astype(np.float64)
        T = float(d["T"])
        par = {k: float(d[k]) for k in ("kDM", "Hex", "Kan1", "Eo")}
    rE, WE, BE = rhat(E)
    rM, WM, BM = rhat(M)
    curva = np.array([rhat(E[:n])[0] for n in N_GRID])
    curva_M = np.array([rhat(M[:n])[0] for n in N_GRID])
    medias = E.mean(axis=0)
    mitad = E.shape[0] // 2
    deriva = E[mitad:].mean(axis=0) - E[:mitad].mean(axis=0)  # within-path drift per copy
    out = dict(T=T, **par, rhat_E=rE, rhat_M=rM, W_E=WE, Bn_E=BE,
               icc_E=BE / (BE + WE / E.shape[0]) if WE > 0 else np.nan,
               sd_entre=float(medias.std(ddof=1)), sd_dentro=float(np.sqrt(WE)),
               deriva_rel=float(np.abs(deriva).mean() / np.sqrt(WE)) if WE > 0 else np.nan,
               tau_E=tau_int(E), tau_M=tau_int(M), E_med=float(E.mean()),
               curva=curva, curva_M=curva_M)
    if any(abs(T - t) < 1e-3 for t in T_TRAZA):
        out["traza"] = E[::20, :COPIAS_TRAZA]
        out["traza_M"] = M[::20, :COPIAS_TRAZA]
        out["medias_M"] = M.mean(axis=0)
        out["medias_copias"] = medias
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    base_datos = os.path.expanduser(a.datos)
    candidatos = glob.glob(os.path.join(base_datos, "lote*", "lote_*", "estado_*")) + glob.glob(os.path.join(base_datos, "lote_*", "estado_*")) + glob.glob(os.path.join(base_datos, "estado_*"))
    estados = sorted(set(candidatos),
                     key=lambda p: int(p.rsplit("_", 1)[1]))
    if not estados:
        ap.error("No se encontraron estados en la base externa indicada.")
    for dir_e in estados:
        n_e = int(dir_e.rsplit("_", 1)[1])
        destino = os.path.join(a.salida, f"estado_{n_e}.npz")
        if os.path.exists(destino):
            continue
        rutas = sorted(glob.glob(os.path.join(dir_e, "T_*.npz")),
                       key=lambda p: -float(re.search(r"T_([\d.]+)\.npz", p).group(1)))
        with Pool(a.procs) as pool:
            res = pool.map(una_temperatura, rutas)
        escalares = [k for k, v in res[0].items() if np.isscalar(v)]
        arr = {k: np.array([r[k] for r in res]) for k in escalares}
        arr["curva"] = np.stack([r["curva"] for r in res])
        arr["curva_M"] = np.stack([r["curva_M"] for r in res])
        for r in res:
            if "traza" in r:
                arr[f"traza_{r['T']:.1f}"] = r["traza"]
                arr[f"medias_{r['T']:.1f}"] = r["medias_copias"]
                arr[f"trazaM_{r['T']:.1f}"] = r["traza_M"]
                arr[f"mediasM_{r['T']:.1f}"] = r["medias_M"]
        arr["N_GRID"] = N_GRID
        np.savez(destino + ".tmp.npz", **arr)
        os.replace(destino + ".tmp.npz", destino)
        print(f"estado {n_e}: {len(res)} T, rhat max {arr['rhat_E'].max():.3f}", flush=True)
    with open(os.path.join(a.salida, "HECHO.json"), "w") as f:
        json.dump({"estados": len(estados)}, f)


if __name__ == "__main__":
    main()
