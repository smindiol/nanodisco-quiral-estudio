"""Specific heat c(T) of the eight states and T* versus kDM in the report style.

c(T) = Var(H) / (N kB T^2) with the pooled variance over sweeps and copies, rebuilt from the within-copy
variance W and the variance of copy means. T* and its 95 % percentile interval come from the official
bootstrap table (resultados/tablas/T_estrella.md).
Usage: python -I figura_calor.py --entrada resultados/pasos_mc --tabla resultados/tablas/T_estrella.md --fig <dir>
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figuras_informe as fi  # noqa: E402  (style and palette)
import matplotlib.pyplot as plt  # noqa: E402

KB = 0.08617333  # meV/K
N_ESPINES = 5425
N_BARR, N_COP = 10000, 100


def leer_tabla(ruta):
    filas = {}
    for linea in open(ruta, encoding="utf-8"):
        c = [x.strip() for x in linea.strip().strip("|").split("|")]
        if c and c[0].isdigit():
            lo, hi = map(float, re.findall(r"[-\d.]+", c[7]))
            filas[int(c[0])] = (float(c[4]), lo, hi)
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--tabla", required=True)
    ap.add_argument("--fig", required=True)
    a = ap.parse_args()
    est = fi.cargar(a.entrada)
    tstar = leer_tabla(a.tabla)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    for e, d in est.items():
        var = (N_BARR - 1) / N_BARR * d["W_E"] + (N_COP - 1) / N_COP * d["Bn_E"]
        c = d["Eo"] ** 2 * var / (N_ESPINES * KB * d["T"] ** 2)  # meV/K per spin
        ax[0].plot(d["T"], c, color=fi.PALETA[e - 1], lw=1.1, label=f"{e}: " + fi.etiqueta(d))
        T0 = tstar[e][0]
        ax[0].plot([T0], [np.interp(T0, d["T"][::-1], c[::-1])], marker="o", ms=4.5, color=fi.PALETA[e - 1],
                   markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    ax[0].set_xlabel("Temperatura $T$ (K)"); ax[0].set_ylabel(r"$c(T)$")
    ax[0].set_title(r"Calor específico; el punto marca $T^*$")
    for e, d in est.items():
        T0, lo, hi = tstar[e]
        x = d["kDM"][0]
        ax[1].errorbar(x, T0, yerr=[[T0 - lo], [hi - T0]], fmt="o" if d["Hex"][0] == 0 else "D", ms=5,
                       color=fi.PALETA[e - 1], ecolor=fi.PALETA[e - 1], elinewidth=1.2, capsize=0,
                       markeredgecolor="white", markeredgewidth=0.8)
        ax[1].annotate(str(e), (x, T0), textcoords="offset points", xytext=(5, 2), fontsize=7, color=fi.TINTA2)
    ax[1].set_xlabel(r"$k_{DM}$"); ax[1].set_ylabel(r"$T^*$ (K)")
    ax[1].set_title("IC 95 % percentil (bootstrap)")
    ax[0].legend(loc="lower left", fontsize=6.0, ncol=2, columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, "calor_especifico.pdf")); plt.close(fig)
    for e, d in est.items():
        var = (N_BARR - 1) / N_BARR * d["W_E"] + (N_COP - 1) / N_COP * d["Bn_E"]
        c = d["Eo"] ** 2 * var / (N_ESPINES * KB * d["T"] ** 2)
        print(e, f"argmax c en malla: {d['T'][np.argmax(c)]:.1f} K, c max {c.max():.4f}; T* tabla {tstar[e][0]}")


if __name__ == "__main__":
    main()
