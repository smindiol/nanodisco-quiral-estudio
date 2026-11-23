"""Hidden populations of final states: Gaussian mixtures fitted by EM (from scratch) to the 100 copy means of m at 6 K.

For each state and k = 1, 2, 3 components: 20 random starts, EM until the log-likelihood gain is < 1e-10,
keep the best start, select k by BIC = -2 loglik + (3k - 1) log n.
Usage: python -I mezcla_em.py --entrada resultados/pasos_mc --fig <dir> --tablas <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figuras_informe as fi  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ARRANQUES, TOL, MAX_IT, VAR_MIN = 20, 1e-10, 5000, 1e-8


def log_normal(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def em(x, k, rng):
    n = len(x)
    mu = rng.choice(x, size=k, replace=False)
    var = np.full(k, x.var() + VAR_MIN)
    w = np.full(k, 1.0 / k)
    traza = []
    for _ in range(MAX_IT):
        lp = np.log(w)[None] + log_normal(x[:, None], mu[None], var[None])   # E step
        mx = lp.max(1, keepdims=True)
        ll = float((mx[:, 0] + np.log(np.exp(lp - mx).sum(1))).sum())
        r = np.exp(lp - mx); r /= r.sum(1, keepdims=True)
        traza.append(ll)
        nk = r.sum(0) + 1e-12                                                  # M step
        w = nk / n
        mu = (r * x[:, None]).sum(0) / nk
        var = np.maximum((r * (x[:, None] - mu[None]) ** 2).sum(0) / nk, VAR_MIN)
        if len(traza) > 1 and traza[-1] - traza[-2] < TOL:
            break
    return dict(w=w, mu=mu, var=var, ll=traza[-1], traza=traza)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True); ap.add_argument("--fig", required=True)
    ap.add_argument("--tablas", required=True)
    a = ap.parse_args()
    est = fi.cargar(a.entrada)
    rng = np.random.default_rng(fi.SEMILLA)
    res, monotono = {}, True
    for e, d in est.items():
        x = d["mediasM_6.0"].astype(float)
        n = len(x)
        mejor = {}
        for k in (1, 2, 3):
            ajustes = [em(x, k, rng) for _ in range(ARRANQUES)]
            for aj in ajustes:
                monotono &= bool(np.all(np.diff(aj["traza"]) > -1e-9))
            lls = np.array([aj["ll"] for aj in ajustes])
            b = ajustes[int(lls.argmax())]
            b["bic"] = -2 * b["ll"] + (3 * k - 1) * np.log(n)
            b["dispersion_ll"] = float(lls.max() - lls.min())
            mejor[k] = b
        kopt = min(mejor, key=lambda k: mejor[k]["bic"])
        res[e] = dict(kopt=kopt, bic={k: float(v["bic"]) for k, v in mejor.items()},
                      w=mejor[kopt]["w"].tolist(), mu=mejor[kopt]["mu"].tolist(),
                      sd=np.sqrt(mejor[kopt]["var"]).tolist(),
                      dispersion_ll={k: v["dispersion_ll"] for k, v in mejor.items()}, _mejor=mejor)
    # Table
    with open(os.path.join(a.tablas, "mezcla_em.tex"), "w") as f:
        for e, r in res.items():
            comp = "; ".join(f"{w:.2f}\\,$\\mathcal N({m:.3f},\\,{s:.3f}^2)$" for w, m, s in
                             sorted(zip(r["w"], r["mu"], r["sd"]), key=lambda t: -t[0]))
            f.write(f"{e} & {r['bic'][1]:.1f} & {r['bic'][2]:.1f} & {r['bic'][3]:.1f} & {r['kopt']} & {comp} \\\\\n")
    # Figure: histograms with fitted mixture for four states + EM log-likelihood trace
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 1]})
    for j, e in enumerate((2, 6)):
        x = est[e]["mediasM_6.0"]; r = res[e]
        ax[j].hist(x, bins=18, density=True, color=fi.PALETA[e - 1], alpha=0.35, edgecolor="white")
        g = np.linspace(x.min() - 0.03, x.max() + 0.03, 400)
        dens = sum(w * np.exp(log_normal(g, m, s ** 2)) for w, m, s in zip(r["w"], r["mu"], r["sd"]))
        ax[j].plot(g, dens, color=fi.PALETA[e - 1], lw=1.6)
        ax[j].set_title(f"Estado {e}: BIC elige $k={r['kopt']}$"); ax[j].set_xlabel(r"$\bar m$ por copia a 6 K")
    ax[0].set_ylabel("Densidad")
    tr = res[6]["_mejor"][2]["traza"]
    ax[2].plot(np.arange(1, len(tr) + 1), tr, color=fi.PALETA[5]); ax[2].set_xscale("log")
    ax[2].set_title("Log-verosimilitud por iteración (estado 6, $k=2$)"); ax[2].set_xlabel("Iteración EM")
    ax[2].title.set_fontsize(8)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, "mezcla_em.pdf")); plt.close(fig)
    salida = {e: {k: v for k, v in r.items() if k != "_mejor"} for e, r in res.items()}
    salida["loglik_nunca_decrece"] = monotono
    json.dump(salida, open(os.path.join(a.tablas, "mezcla_em.json"), "w"), indent=1)
    print(json.dumps(salida, indent=1))


if __name__ == "__main__":
    main()
