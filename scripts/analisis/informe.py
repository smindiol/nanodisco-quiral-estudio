"""Etapa C: comparación entre estados, tablas, figuras y RESUMEN.md a partir de las salidas de la etapa B."""
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dmi_analisis import holm  # noqa: E402

# Criterios de los veredictos (se informan en RESUMEN.md junto con los valores que los deciden)
CRIT = dict(rhat=1.01, ess=400, frac_T=0.90, mser_frac=0.05, ratio=(0.8, 1.25), ess_rw=0.1, grupo=0.80, acuerdo=0.90)


def cargar_B(dB):
    out = {}
    for f in sorted(glob.glob(os.path.join(dB, "estado_*.npz"))):
        d = dict(np.load(f, allow_pickle=False))
        d["meta"] = json.loads(str(d["meta"]))
        out[int(d["estado"])] = d
    return out


def _f(x, k=3):
    return f"{x:.{k}g}"


def veredictos(d):
    """Devuelve (dict de métricas, dict de veredictos) para un estado."""
    T = d["T"]
    R = d["conv_c_copia"].shape[1]
    rh = np.maximum(d["conv_rhat_H"], d["conv_rhat_m"])
    ess = np.minimum(d["conv_ess_H"], d["conv_ess_m"])
    gew_sig = (d["conv_gew_p_holm"] < 0.05).any(0)
    mser = np.maximum(np.median(d["conv_mser_H"], 1), np.median(d["conv_mser_m"], 1))
    homog_sig = (d["conv_homog_p_holm"] < 0.05).any(0)
    ratio = d["conv_ratio_int_ext"]
    met = dict(
        frac_rhat=float((rh < CRIT["rhat"]).mean()), rhat_max=float(rh.max()), T_rhat_max=float(T[rh.argmax()]),
        frac_ess=float((ess > CRIT["ess"]).mean()), ess_min=float(ess.min()),
        frac_gew=float((~gew_sig).mean()), frac_mser=float((mser <= CRIT["mser_frac"] * 10000).mean()),
        mser_max=float(mser.max()), frac_homog=float((~homog_sig).mean()),
        frac_ratio=float(((ratio > CRIT["ratio"][0]) & (ratio < CRIT["ratio"][1])).mean()),
        frac_jk=float(d["conv_jk_confiable"].mean()), ess_rw_min=float(d["ess_rw"].min()),
        acuerdo_c=float(d["c_acuerdo"]), frac_mayor_bajo=float(d["sp_frac_mayor"][-1]), ngrupos_bajo=int(d["sp_ngrupos"][-1]),
        tau_max=float(d["conv_tau_t"].max()), T_tau_max=float(T[d["conv_tau_t"].argmax()]))
    # Por temperatura: converge si cumple los tres diagnósticos; es representativa si además las copias son homogéneas;
    # es confiable si el ESS y el jackknife por bloques son suficientes.
    conv_T = (rh < CRIT["rhat"]) & ~gew_sig & (mser <= CRIT["mser_frac"] * 10000)
    rep_T = conv_T & ~homog_sig
    conf_T = (ess > CRIT["ess"]) & d["conv_jk_confiable"].astype(bool)

    def tramo(ok):
        """Porcentaje de temperaturas que cumplen y la T más baja alcanzada sin interrupción desde 20 K."""
        k = 0
        while k < len(T) and ok[k]:
            k += 1
        return float(ok.mean()), (float(T[k - 1]) if k > 0 else None)

    ver = dict(convergio=tramo(conv_T), representativo=tramo(rep_T), confiable=tramo(conf_T))
    return met, ver


def generar(dB, res):
    E = cargar_B(dB)
    dF = os.path.join(res, "figuras"); dT = os.path.join(res, "tablas")
    os.makedirs(dF, exist_ok=True); os.makedirs(dT, exist_ok=True)
    est = sorted(E)
    cmap = plt.get_cmap("viridis", max(len(est), 2))
    col = {e: cmap(i) for i, e in enumerate(est)}
    etiqueta = {e: f"{e}: kDM={E[e]['meta']['kDM']:g}, Hex={E[e]['meta']['Hex']:g}" for e in est}
    md = []

    # ---------------- Tabla 1: parámetros y T*
    filas = []
    for e in est:
        d = E[e]
        ts = d["c_Tstar_boot"]
        filas.append(f"| {e} | {d['meta']['kDM']:g} | {d['meta']['Hex']:g} | {d['meta']['Kan1']:g} | {float(d['c_Tstar']):.2f} | "
                     f"{np.median(ts):.2f} | {ts.std(ddof=1):.3f} | [{d['c_Tstar_ic_p'][0]:.2f}, {d['c_Tstar_ic_p'][1]:.2f}] | "
                     f"[{d['c_Tstar_ic_bca'][0]:.2f}, {d['c_Tstar_ic_bca'][1]:.2f}] | [{d['c_meseta'][0]:.2f}, {d['c_meseta'][1]:.2f}] | "
                     f"{_f(float(d['c_pico']))} ± {_f(float(d['c_pico_boot'].std(ddof=1)), 2)} | "
                     f"{float(d['chi_Tstar']):.2f}{' (borde)' if bool(d['chi_borde']) else ''} | {int(d['B'])} |")
    t1 = ("| Estado | kDM | Hex | Kan1 | T* de c (K) | Mediana bootstrap | se | IC 95 % percentil | IC 95 % BCa | Meseta del pico | "
          "c en el pico (meV/K por espín) | T* de χ (K) | B |\n|" + "---|" * 13 + "\n" + "\n".join(filas))
    open(os.path.join(dT, "T_estrella.md"), "w").write(t1)
    md += ["## 1. Temperatura del pico del calor específico", "", t1, ""]

    # ---------------- Tabla 2: convergencia y veredictos
    filas, V = [], {}
    for e in est:
        met, ver = veredictos(E[e]); V[e] = (met, ver)
        def s(v):
            f, tmin = v
            hasta = f"20.0–{tmin:.1f} K" if tmin is not None else "ninguna desde 20 K"
            return ("sí" if f == 1 else "**no**") + f": {f:.0%} de las T ({hasta})"
        filas.append(f"| {e} | {met['frac_rhat']:.0%} (máx {met['rhat_max']:.3f} en {met['T_rhat_max']:.1f} K) | {met['frac_gew']:.0%} | "
                     f"{met['frac_mser']:.0%} (máx {met['mser_max']:.0f}) | {met['tau_max']:.0f} en {met['T_tau_max']:.1f} K | "
                     f"{met['ess_min']:.0f} | {met['frac_homog']:.0%} | {met['frac_ratio']:.0%} | {met['frac_jk']:.0%} | "
                     f"{met['ess_rw_min']:.2f} | {met['acuerdo_c']:.0%} | {s(ver['convergio'])} | {s(ver['representativo'])} | {s(ver['confiable'])} |")
    t2 = ("| Estado | T con R̂<1.01 | T sin deriva Geweke (Holm) | T con MSER ≤ 500 barridos | τ_int máx (barridos) | ESS mín | "
          "T con copias homogéneas (Holm) | T con se_int/sd_ext en [0.8,1.25] | T con jackknife confiable | ESS mín del repesado | "
          "Acuerdo percentil/BCa de c | ¿Convergió? (% de T, tramo continuo desde 20 K) | ¿Representativo? | ¿Confiable? |\n|" + "---|" * 14 + "\n" + "\n".join(filas))
    open(os.path.join(dT, "convergencia.md"), "w").write(t2)
    md += ["## 2. Convergencia, representatividad y confiabilidad", "",
           "Cada temperatura se evalúa por separado. **Converge** en una T si se cumplen las tres condiciones: R̂ < 1.01 "
           "(H y m), sin exceso de rechazos de Geweke sobre el 5 % nominal (binomial, Holm sobre las T) y mediana de MSER-5 ≤ 500 "
           "barridos. Es **representativa** si además las 100 copias son homogéneas (χ² de sus medias, Holm). Es **confiable** "
           "si el ESS supera 400 y el jackknife por bloques tiene al menos 20 bloques de tamaño ≥ 10 τ_int. Un estado convergió "
           "solo si cumple en las 141 temperaturas; si no, se informa el porcentaje de temperaturas que cumplen y el tramo continuo "
           "desde 20 K hasta donde se cumple.", "", t2, ""]

    # ---------------- Tabla 3: estados finales
    filas = []
    for e in est:
        d = E[e]; k0 = -1; ks = int(d["k_tstar"])
        q = d["sp_q_bajo"]; iu = np.triu_indices(q.shape[0], 1)
        filas.append(f"| {e} | {d['T'][k0]:.1f} | {int(d['sp_Q_moda'][k0]):d} ({d['sp_Q_frac_moda'][k0]:.0%}) | "
                     f"{d['sp_Q_media'][k0]:.2f} [{d['sp_Q_media_ic'][0][k0]:.2f}, {d['sp_Q_media_ic'][1][k0]:.2f}] | "
                     f"{np.mean(q[iu]):.3f} / {np.mean(np.abs(q[iu])):.3f} | {int(d['sp_ngrupos'][k0])} ({d['sp_frac_mayor'][k0]:.0%}) | "
                     f"{d['sp_lambda'][k0]:.1f} | {d['sp_quir'][k0].mean():.3f} | "
                     f"{int(d['sp_Q_moda'][ks]):d} ({d['sp_Q_frac_moda'][ks]:.0%}) en {d['T'][ks]:.1f} K |")
    t3 = ("| Estado | T final (K) | Q modal (fracción de copias) | Q medio [IC 95 %] | q medio / \\|q\\| medio entre copias | "
          "Grupos de S(k) (fracción del mayor) | Período λ (sitios) | Quiralidad media | Q modal en T* |\n|" + "---|" * 9 + "\n" + "\n".join(filas))
    open(os.path.join(dT, "estados_finales.md"), "w").write(t3)
    md += ["## 3. Estados medidos: configuraciones finales", "", t3, ""]

    # ---------------- Tabla 4: diferencias de T* entre estados
    pares, pv = [], []
    for i, a in enumerate(est):
        for b in est[i + 1:]:
            n = min(len(E[a]["c_Tstar_boot"]), len(E[b]["c_Tstar_boot"]))
            dif = E[a]["c_Tstar_boot"][:n].astype(float) - E[b]["c_Tstar_boot"][:n]
            p = min(1.0, 2 * min((dif <= 0).mean(), (dif >= 0).mean()))
            pares.append((a, b, float(E[a]["c_Tstar"] - E[b]["c_Tstar"]), np.quantile(dif, [0.025, 0.975]))); pv.append(p)
    if pares:
        ph = holm(pv)
        filas = [f"| {a} − {b} | {d0:+.2f} | [{ic[0]:+.2f}, {ic[1]:+.2f}] | {p:.3g} | {'sí' if p < 0.05 else 'no'} |"
                 for (a, b, d0, ic), p in zip(pares, ph)]
        t4 = ("| Par | ΔT* (K) | IC 95 % | p (Holm) | ¿Difieren? |\n|---|---|---|---|---|\n" + "\n".join(filas))
        open(os.path.join(dT, "diferencias_T_estrella.md"), "w").write(t4)
        md += ["## 4. ¿Difieren los T* entre estados?", "",
               "Estados independientes: la diferencia bootstrap es T*ᵇ(a) − T*ᵇ(b) con réplicas independientes; p bilateral del "
               "bootstrap, corregido por Holm sobre todos los pares.", "", t4, ""]

    # ---------------- Figuras
    def panel(n):
        nc = min(4, n); nf = int(np.ceil(n / nc))
        fig, ax = plt.subplots(nf, nc, figsize=(4.2 * nc, 3.2 * nf), squeeze=False)
        return fig, ax.ravel()

    for nombre, ylab in (("c", "c (meV/K por espín)"), ("chi", "χ")):
        fig, ax = panel(len(est))
        for a, e in zip(ax, est):
            d = E[e]
            a.fill_between(d["Tf"], d[f"{nombre}_f_lo"], d[f"{nombre}_f_hi"], color=col[e], alpha=0.3, lw=0)
            a.plot(d["Tf"], d[f"{nombre}_f"], color=col[e], lw=1)
            a.errorbar(d["T"], d[nombre], yerr=[d[nombre] - d[f"{nombre}_lo_p"], d[f"{nombre}_hi_p"] - d[nombre]], fmt=".", ms=2, color="k", lw=0.5)
            a.axvline(float(d[f"{nombre}_Tstar"]), color="r", ls="--", lw=0.8)
            a.set_title(etiqueta[e], fontsize=9); a.set_xlabel("T (K)"); a.set_ylabel(ylab)
        for a in ax[len(est):]:
            a.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(dF, f"{nombre}_T.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8)); ax = ax.ravel()
    for e in est:
        d = E[e]; T = d["T"]
        ax[0].semilogy(T, d["conv_tau_t"], color=col[e], label=etiqueta[e])
        ax[1].plot(T, np.maximum(d["conv_rhat_H"], d["conv_rhat_m"]), color=col[e])
        ax[2].plot(T, d["conv_gew_frac"].max(0), color=col[e])
        ax[3].plot(T, np.maximum(np.median(d["conv_mser_H"], 1), np.median(d["conv_mser_m"], 1)), color=col[e])
        ax[4].semilogy(T, d["conv_homog_Q_H"] / (d["conv_c_copia"].shape[1] - 1), color=col[e])
        ax[5].plot(T, d["conv_ratio_int_ext"], color=col[e])
    ax[1].axhline(1.01, color="r", ls="--"); ax[2].axhline(0.05, color="r", ls="--"); ax[4].axhline(1, color="r", ls="--")
    ax[5].axhspan(0.8, 1.25, color="g", alpha=0.1)
    for a, t in zip(ax, ["τ_int máx de H, m, H², m² (barridos)", "R̂ (máx de H y m)", "Fracción de copias con |z Geweke| > 1.96",
                         "Mediana MSER-5 (barridos a descartar)", "Q/(R−1) de homogeneidad de H", "se interno / sd entre copias de c"]):
        a.set_title(t, fontsize=10); a.set_xlabel("T (K)")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(dF, "convergencia.png"), dpi=130); plt.close(fig)

    fig, ax = panel(len(est))
    for a, e in zip(ax, est):
        d = E[e]
        a.hist(d["c_Tstar_boot"], bins=60, color=col[e]); a.axvline(float(d["c_Tstar"]), color="r")
        a.axvspan(*d["c_Tstar_ic_p"], color="r", alpha=0.1); a.set_title(etiqueta[e], fontsize=9); a.set_xlabel("T*ᵇ (K)")
    for a in ax[len(est):]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(dF, "T_estrella_bootstrap.png"), dpi=130); plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 4))
    for e in est:
        d = E[e]; lo, hi = d["c_Tstar_ic_p"]; t = float(d["c_Tstar"])
        a.errorbar(d["meta"]["kDM"], t, yerr=[[t - lo], [hi - t]], fmt="o", color=col[e], capsize=3, label=etiqueta[e])
    a.set_xlabel("kDM"); a.set_ylabel("T* de c (K)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(dF, "T_estrella_vs_kDM.png"), dpi=130); plt.close(fig)

    for nombre in ("U", "e_exc", "e_dm", "e_an"):
        fig, a = plt.subplots(figsize=(7, 4))
        for e in est:
            d = E[e]
            a.fill_between(d["T"], d[f"{nombre}_lo_p"], d[f"{nombre}_hi_p"], color=col[e], alpha=0.3, lw=0)
            a.plot(d["T"], d[nombre], color=col[e], label=etiqueta[e])
        a.set_xlabel("T (K)"); a.set_title({"U": "Cumulante de Binder de |m|", "e_exc": "Energía de intercambio por espín (meV)",
                                            "e_dm": "Energía DMI por espín (meV)", "e_an": "Energía de anisotropía por espín (meV)"}[nombre])
        a.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(dF, f"{nombre}_T.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, len(est), figsize=(3 * len(est), 6), squeeze=False)
    for j, e in enumerate(est):
        d = E[e]
        Q = d["sp_Q"][-1].mean(1)
        ax[0, j].hist(Q, bins=np.arange(np.floor(Q.min()) - 0.5, np.ceil(Q.max()) + 1.5), color=col[e])
        ax[0, j].set_title(f"{etiqueta[e]}\nQ a {d['T'][-1]:.1f} K", fontsize=8)
        q = d["sp_q_bajo"]; iu = np.triu_indices(q.shape[0], 1)
        ax[1, j].hist(q[iu], bins=80, range=(-1, 1), color=col[e]); ax[1, j].set_title("P(q) entre copias", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(dF, "estados_finales_Q_q.png"), dpi=130); plt.close(fig)

    fig, a = plt.subplots(figsize=(7, 4))
    for e in est:
        d = E[e]; b = d["sp_sk_bordes"]; kc = 0.5 * (b[:-1] + b[1:]); sk = d["sp_sk"][-1]
        a.plot(kc[1:], sk.mean(0)[1:], color=col[e], label=etiqueta[e])
        a.fill_between(kc[1:], np.quantile(sk, 0.05, 0)[1:], np.quantile(sk, 0.95, 0)[1:], color=col[e], alpha=0.2, lw=0)
    a.set_yscale("log"); a.set_xlabel("|k| (1/sitio)"); a.set_title("Factor de estructura radial a la T más baja (media y 5–95 % de copias)")
    a.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(dF, "factor_estructura.png"), dpi=130); plt.close(fig)

    md = ["# Análisis estadístico de la simulación secuencial (estados 1–8)", "",
          f"Bootstrap entre copias (la copia completa es la unidad; mismas multiplicidades en toda la escalera), "
          f"B = {int(E[est[0]]['B'])}, α = 0.05, repesado de histograma simple a una rejilla de 0.01 K. "
          "Figuras en `figuras/`, tablas en `tablas/`.", ""] + md
    md += ["## 5. Veredicto por estado", ""]
    for e in est:
        met, ver = V[e]
        def frase(nombre, v):
            f, tmin = v
            if f == 1:
                return f"{nombre} en toda la escalera"
            tr = f"sin interrupción de 20.0 K a {tmin:.1f} K" if tmin is not None else "ya falla a 20 K"
            return f"no {nombre.split()[0]} en toda la escalera: solo en el {f:.0%} de las temperaturas ({tr})"
        md.append(f"- **Estado {etiqueta[e]}**: {frase('convergió', ver['convergio'])}; {frase('representativo', ver['representativo'])}; "
                  f"{frase('confiable', ver['confiable'])}. "
                  f"τ_int máximo {met['tau_max']:.0f} barridos en {met['T_tau_max']:.1f} K; MSER máximo {met['mser_max']:.0f} barridos; "
                  f"a la T más baja, {met['ngrupos_bajo']} tipo(s) de estado final, el mayor con {met['frac_mayor_bajo']:.0%} de las copias.")
    open(os.path.join(res, "RESUMEN.md"), "w").write("\n".join(md) + "\n")
    return E, V
