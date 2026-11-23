"""Figures and correlation tables for the final report (reads resultados/pasos_mc/estado_*.npz).

Usage: python -I figuras_informe.py --entrada resultados/pasos_mc --fig <dir> --tablas <dir>
"""
import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
UMBRAL = 1.01
SEMILLA = 20261117

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "figure.dpi": 200, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def cargar(entrada):
    est = {}
    for e in range(1, 9):
        d = np.load(os.path.join(entrada, f"estado_{e}.npz"))
        est[e] = {k: d[k] for k in d.files}
    return est


def etiqueta(d):
    return rf"$k_{{DM}}={d['kDM'][0]:.2f}$, $H={d['Hex'][0]:.2f}$"


def tramo_continuo(T, ok):
    """Lowest temperature reached by an uninterrupted run of ok values starting at the hottest T."""
    orden = np.argsort(-T)
    ultimo = np.nan
    for i in orden:
        if not ok[i]:
            break
        ultimo = T[i]
    return ultimo


def barridos_necesarios(curva, grid):
    """Smallest n such that R-hat computed on the first n' sweeps stays below the threshold for every n' >= n."""
    ok = curva < UMBRAL
    if not ok[-1]:
        return np.inf
    j = len(ok) - 1
    while j > 0 and ok[j - 1]:
        j -= 1
    return grid[j]


def rangos(v):
    """Average ranks (ties share the mean rank)."""
    orden = np.argsort(v, kind="mergesort")
    r = np.empty(len(v)); r[orden] = np.arange(1, len(v) + 1)
    for u in np.unique(v):
        r[v == u] = r[v == u].mean()
    return r


def spearman(x, y):
    rx, ry = rangos(np.asarray(x, float)), rangos(np.asarray(y, float))
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def inferencia_spearman(x, y, rng, B=20000):
    rho = spearman(x, y)
    if np.isnan(rho):
        return np.nan, np.nan, np.nan, np.nan
    perm = np.array([spearman(x, np.array(p)) for p in itertools.permutations(y)])  # exact for n = 8
    p = float((np.abs(perm) >= abs(rho) - 1e-12).mean())
    n = len(x)
    boots = []
    for _ in range(B):
        idx = np.floor(n * rng.random(n)).astype(int)  # discrete inverse transform
        r = spearman(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(r)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return rho, p, float(lo), float(hi)


def ols(X, y):
    """Ordinary least squares via normal equations with HC0 sandwich standard errors."""
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    S = (X * r[:, None] ** 2).T @ X
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, se, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--tablas", required=True)
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True); os.makedirs(a.tablas, exist_ok=True)
    est = cargar(a.entrada)
    rng = np.random.default_rng(SEMILLA)
    grid = est[1]["N_GRID"]

    # ---------- Figure 1: R-hat versus temperature, energy and magnetization ----------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for e, d in est.items():
        for k, var in enumerate(("rhat_E", "rhat_M")):
            ax[k].plot(d["T"], np.maximum(d[var] - 1, 1e-5), color=PALETA[e - 1],
                       label=f"{e}: " + etiqueta(d) if k == 1 else None)
    for k, tit in enumerate((r"Energía $E$", r"Magnetización $m$")):
        ax[k].set_yscale("log"); ax[k].axhline(UMBRAL - 1, color=TINTA, lw=0.9, ls="--")
        ax[k].set_title(tit); ax[k].set_xlabel("Temperatura $T$ (K)"); ax[k].invert_xaxis()
    ax[0].set_ylabel(r"$\hat R-1$ entre las 100 copias")
    ax[0].text(19.6, 0.0125, r"umbral $\hat R=1.01$", fontsize=7, color=TINTA2)
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Estado")
    fig.savefig(os.path.join(a.fig, "rhat_T.pdf")); plt.close(fig)

    # ---------- Figure 2: traces, within-path stability versus between-path divergence ----------
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex=True)
    barr = np.arange(est[1]["trazaM_6.0"].shape[0]) * 20
    for fila, e in enumerate((1, 6)):
        for col, T in enumerate((20.0, 6.0)):
            tr = est[e][f"trazaM_{T:.1f}"]
            for c in range(tr.shape[1]):
                ax[fila, col].plot(barr, tr[:, c], color=PALETA[c], lw=0.8, alpha=0.9)
            ax[fila, col].set_title(f"Estado {e} ({etiqueta(est[e])}), $T={T:.0f}$ K")
            ax[fila, col].set_ylabel("$m$" if col == 0 else "")
    for c in range(2):
        ax[1, c].set_xlabel("Barrido de medición")
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, "trazas_m.pdf")); plt.close(fig)

    # ---------- Figure 3: R-hat versus number of sweeps at 6 K ----------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for e, d in est.items():
        i6 = int(np.argmin(np.abs(d["T"] - 6.0)))
        istar = int(np.argmin(np.abs(d["T"] - 14.3)))
        ax[0].plot(grid, np.maximum(d["curva_M"][istar] - 1, 1e-5), color=PALETA[e - 1])
        ax[1].plot(grid, np.maximum(d["curva_M"][i6] - 1, 1e-5), color=PALETA[e - 1], label=str(e))
    for k, tit in enumerate((r"$m$ cerca de $T^*$ ($T=14.3$ K)", r"$m$ en $T=6$ K")):
        ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_title(tit)
        ax[k].axhline(UMBRAL - 1, color=TINTA, lw=0.9, ls="--"); ax[k].set_xlabel("Barridos usados $n$")
    ax[0].set_ylabel(r"$\hat R(n)-1$")
    ax[1].legend(title="Estado", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(os.path.join(a.fig, "rhat_n.pdf")); plt.close(fig)

    # ---------- Figure 4: sweeps needed per (state, T), heatmap ----------
    T = est[1]["T"]
    mat = np.array([[barridos_necesarios(est[e]["curva_M"][i], grid) for i in range(len(T))] for e in est])
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    vis = np.log10(np.where(np.isinf(mat), np.nan, mat))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    cmap.set_bad("#bdbcb5")
    im = ax.imshow(vis, aspect="auto", cmap=cmap, extent=[T[0] + 0.05, T[-1] - 0.05, 8.5, 0.5],
                   vmin=2, vmax=4, interpolation="nearest")
    ax.set_yticks(range(1, 9)); ax.set_ylabel("Estado"); ax.set_xlabel("Temperatura $T$ (K)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label(r"$\log_{10}$ barridos para $\hat R_m<1.01$")
    ax.set_title("Gris: no se alcanza con 10 000 barridos de medición")
    fig.savefig(os.path.join(a.fig, "barridos_necesarios.pdf")); plt.close(fig)

    # ---------- Correlations across states ----------
    kdm = np.array([est[e]["kDM"][0] for e in est]); hex_ = np.array([est[e]["Hex"][0] for e in est])
    met = {
        "fracción de T con $\\hat R_m<1.01$": np.array([(est[e]["rhat_M"] < UMBRAL).mean() for e in est]),
        "T más baja del tramo convergido en $m$ (K)": np.array([tramo_continuo(est[e]["T"], est[e]["rhat_M"] < UMBRAL) for e in est]),
        "$\\tau_{int}$ máximo de $m$ (barridos)": np.array([est[e]["tau_M"].max() for e in est]),
        "$\\tau_{int}$ máximo de $E$ (barridos)": np.array([est[e]["tau_E"].max() for e in est]),
    }
    filas = []
    for nombre, y in met.items():
        for var, x in (("$k_{DM}$", kdm), ("$H$", hex_)):
            rho, p, lo, hi = inferencia_spearman(x, y, rng)
            filas.append((nombre, var, rho, lo, hi, p))
    with open(os.path.join(a.tablas, "correlaciones.tex"), "w") as f:
        for nombre, var, rho, lo, hi, p in filas:
            f.write(f"{nombre} & {var} & ${rho:+.2f}$ & $[{lo:+.2f},\\,{hi:+.2f}]$ & {p:.3f} \\\\\n")
    por_estado = {e: {k: float(v[i]) for k, v in met.items()} for i, e in enumerate(est)}
    with open(os.path.join(a.tablas, "metricas_estado.json"), "w") as f:
        json.dump(por_estado, f, indent=1, ensure_ascii=False)

    # ---------- Regression of log tau_int(m) on kDM, H and 1/T ----------
    X, y, yE = [], [], []
    for e, d in est.items():
        for i in range(len(d["T"])):
            X.append([d["kDM"][i], d["Hex"][i], 1.0 / d["T"][i]])
            y.append(np.log(d["tau_M"][i])); yE.append(np.log(d["tau_E"][i]))
    X, y, yE = np.array(X), np.array(y), np.array(yE)
    reg = {}
    for nom, yy in (("m", y), ("E", yE)):
        b, se, r2 = ols(X, yy)
        reg[nom] = dict(b=b.tolist(), se=se.tolist(), r2=r2)
    with open(os.path.join(a.tablas, "regresion_tau.json"), "w") as f:
        json.dump(reg, f, indent=1)

    # ---------- Figure 5: scatter of convergence metrics versus kDM ----------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ymet = (list(met.values())[0] * 100, list(met.values())[2])
    tits = (r"Temperaturas con $\hat R_m<1.01$ (%)", r"$\tau_{int}$ máximo de $m$ (barridos)")
    for k in range(2):
        for i, e in enumerate(est):
            ax[k].scatter(kdm[i], ymet[k][i], s=46, color=PALETA[e - 1], marker="o" if hex_[i] == 0 else "D",
                          edgecolor="white", linewidth=1.2, zorder=3)
            ax[k].annotate(str(e), (kdm[i], ymet[k][i]), textcoords="offset points", xytext=(5, 4),
                           fontsize=7, color=TINTA2)
        ax[k].set_xlabel(r"$k_{DM}$"); ax[k].set_title(tits[k])
    ax[1].set_yscale("log")
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=TINTA2, label="$H=0$"),
                          Line2D([], [], marker="D", ls="", color=TINTA2, label="$H>0$")], loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, "correlacion_kdm.pdf")); plt.close(fig)

    # ---------- Figure 6: between-copy spread of final <m> at 6 K versus 20 K ----------
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for e, d in est.items():
        for j, T in enumerate((20.0, 6.0)):
            v = d[f"mediasM_{T:.1f}"]
            xj = e + (-0.18 if j == 0 else 0.18) + 0.05 * (rng.random(len(v)) - 0.5)
            ax.scatter(xj, v, s=6, color=PALETA[e - 1], alpha=0.35 if j == 0 else 0.85, linewidth=0)
    ax.set_xticks(range(1, 9)); ax.set_xlabel("Estado (izquierda: 20 K, derecha: 6 K)")
    ax.set_ylabel(r"$\bar m$ de cada copia")
    ax.set_title("Cada punto es el promedio temporal de un camino independiente")
    fig.savefig(os.path.join(a.fig, "medias_copias.pdf")); plt.close(fig)

    print(json.dumps({"correlaciones": filas, "regresion": reg, "metricas": por_estado}, indent=1, default=str,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
