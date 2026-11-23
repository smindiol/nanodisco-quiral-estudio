"""Análisis estadístico de la simulación secuencial de nanodiscos con DMI (lotes 1-4, estados 1-8).

Etapa A (una tarea por estado y temperatura, independientes): verificación del archivo, sumas por copia,
autocorrelación de Sokal, jackknife por bloques, diagnósticos de convergencia, repesado de histograma hacia
una rejilla fina y análisis de las configuraciones de espines medidas.
Etapa B (una tarea por estado): bootstrap entre copias (la copia completa es la unidad de remuestreo, con las
mismas multiplicidades en toda la escalera), intervalos percentil y BCa, distribución de T*, resúmenes de
convergencia y de estados finales.
Etapa C: comparación entre estados, tablas, figuras y RESUMEN.md.

Todas las salidas se escriben primero en un archivo temporal y luego se renombran; una tarea cuyo archivo de
salida ya existe y se puede leer se salta, de modo que relanzar continúa desde donde quedó.
"""
import glob
import json
import math
import os
import re
import time

import numpy as np
from scipy import stats

FORMATO_A = "dmi-etapaA-v1"
FORMATO_B = "dmi-etapaB-v1"
N_MED = 10000
R_COPIAS = 100
PASO_FINO = 0.01
C_SOKAL = 10.0
LARGOS_BLOQUE = np.array([b for b in (1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500)])
NBINS_SK = 32
NBINS_Q = 100


# ----------------------------------------------------------------------------------------------------------
# Descubrimiento de datos
# ----------------------------------------------------------------------------------------------------------
def temperatura_de(ruta):
    return float(re.search(r"T_([0-9.]+)\.npz$", ruta).group(1))


def descubrir(base):
    """Devuelve {estado: [rutas ordenadas de mayor a menor T]}."""
    out = {}
    for d in sorted(set(glob.glob(os.path.join(base, "lote*", "lote_0*", "estado_*")) + glob.glob(os.path.join(base, "lote_*", "estado_*")) + glob.glob(os.path.join(base, "estado_*")))):
        e = int(d.rsplit("_", 1)[1])
        rutas = sorted(glob.glob(os.path.join(d, "T_*.npz")), key=temperatura_de, reverse=True)
        out[e] = rutas
    return out


def escalera_global(datos):
    ts = sorted({temperatura_de(r) for rs in datos.values() for r in rs}, reverse=True)
    return np.array(ts)


def guardar_atomico(ruta, **arrays):
    tmp = ruta + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, ruta)


def legible(ruta, formato):
    try:
        with np.load(ruta, allow_pickle=False) as d:
            return str(d["formato"]) == formato
    except Exception:
        return False


# ----------------------------------------------------------------------------------------------------------
# Autocorrelación, jackknife y diagnósticos de una serie (n, R)
# ----------------------------------------------------------------------------------------------------------
def acf(x):
    """Autocorrelación normalizada por columna con FFT; x: (n, R)."""
    n = x.shape[0]
    y = x - x.mean(0)
    f = np.fft.rfft(y, n=2 * n, axis=0)
    a = np.fft.irfft(f * np.conj(f), axis=0)[:n]
    a0 = a[0].copy()
    a0[a0 == 0] = 1.0
    return a / a0


def tau_sokal(x, c=C_SOKAL):
    """tau_int = 1/2 + sum_{l=1}^{W} rho(l), con la ventana mínima W >= c tau(W). Devuelve (tau, W, ok)."""
    rho = acf(x)
    n = x.shape[0]
    cum = 0.5 + np.cumsum(rho[1:], axis=0)            # cum[W-1] = tau(W)
    W = np.arange(1, n)[:, None]
    cond = W >= c * cum
    ok = cond.any(0)
    iw = np.where(ok, cond.argmax(0), n // 2 - 1)
    tau = np.maximum(cum[iw, np.arange(x.shape[1])], 0.5)
    return tau, iw + 1, ok & (iw + 1 < n // 4)


def jk_var(x, b):
    """Error jackknife por bloques de la varianza (divisor n) de cada columna; x: (n, R)."""
    n = (x.shape[0] // b) * b
    g = n // b
    xc = x[:n] - x[:n].mean(0)
    s1 = xc.reshape(g, b, -1).sum(1)
    s2 = (xc ** 2).reshape(g, b, -1).sum(1)
    m = (s1.sum(0)[None] - s1) / (n - b)
    v = (s2.sum(0)[None] - s2) / (n - b) - m ** 2
    return np.sqrt((g - 1) / g * ((v - v.mean(0)) ** 2).sum(0))


def geweke(x, tau):
    """z de Geweke por copia: primer 10 % contra último 50 %, varianzas espectrales 2 tau s^2 / n_j."""
    n = x.shape[0]
    a, b = x[: n // 10], x[n // 2:]
    va = 2 * tau * a.var(0, ddof=1) / a.shape[0]
    vb = 2 * tau * b.var(0, ddof=1) / b.shape[0]
    return (a.mean(0) - b.mean(0)) / np.sqrt(va + vb)


def mser5(x):
    """Truncamiento MSER-5 por copia (en barridos), buscado en la primera mitad."""
    n = (x.shape[0] // 5) * 5
    z = x[:n].reshape(n // 5, 5, -1).mean(1)        # (k, R) medias de lotes de 5
    k = z.shape[0]
    s1 = np.cumsum(z[::-1], 0)[::-1]                 # suma de z[d:]
    s2 = np.cumsum((z ** 2)[::-1], 0)[::-1]
    cnt = (k - np.arange(k))[:, None].astype(float)
    sse = s2 - s1 ** 2 / cnt
    stat = sse / cnt ** 2
    d = stat[: k // 2].argmin(0)
    return 5 * d


def _rhat(z):
    n = z.shape[0]
    W = z.var(0, ddof=1).mean()
    B = n * z.mean(0).var(ddof=1)
    return math.sqrt(((n - 1) / n * W + B / n) / W) if W > 0 else float("nan")


def rhat_rango(x):
    """R-hat dividido y normalizado por rangos (Vehtari et al. 2021): máximo entre bulk y folded."""
    n = (x.shape[0] // 2) * 2
    h = n // 2
    y = np.concatenate([x[:h], x[h:n]], axis=1)
    def normal_rangos(v):
        r = stats.rankdata(v, axis=None).reshape(v.shape)
        return stats.norm.ppf((r - 0.375) / (v.size + 0.25))
    bulk = _rhat(normal_rangos(y))
    fold = _rhat(normal_rangos(np.abs(y - np.median(y))))
    return max(bulk, fold), bulk


# ----------------------------------------------------------------------------------------------------------
# Configuraciones de espines
# ----------------------------------------------------------------------------------------------------------
def red_espines(spins, mascara):
    R = spins.shape[0]
    red = np.zeros((R, *mascara.shape, 3), dtype=np.float64)
    red[:, mascara] = spins
    return red


def carga_topologica(red, mascara):
    """Q por copia y capa (Berg-Lüscher) sobre plaquetas con los cuatro vértices dentro del disco."""
    S00, S10 = red[:, :, :-1, :-1], red[:, :, 1:, :-1]
    S11, S01 = red[:, :, 1:, 1:], red[:, :, :-1, 1:]
    ok = mascara[:, :-1, :-1] & mascara[:, 1:, :-1] & mascara[:, 1:, 1:] & mascara[:, :-1, 1:]

    def omega(a, b, c):
        num = np.einsum("...k,...k->...", a, np.cross(b, c))
        den = 1 + np.einsum("...k,...k->...", a, b) + np.einsum("...k,...k->...", b, c) + np.einsum("...k,...k->...", c, a)
        return 2 * np.arctan2(num, den)

    om = omega(S00, S01, S11) + omega(S00, S11, S10)
    return (om * ok[None]).sum((2, 3)) / (4 * np.pi)       # (R, capas)


def factor_estructura(red, mascara, nbins=NBINS_SK, pad=64):
    """Perfil radial de S(k) = sum_capas sum_comp |FFT|^2 / N por copia; devuelve (bordes, perfil (R, nbins))."""
    N = mascara.sum()
    f = np.fft.fft2(red, s=(pad, pad), axes=(2, 3))
    sk = (np.abs(f) ** 2).sum(axis=(1, 4)) / N               # (R, pad, pad)
    k = 2 * np.pi * np.fft.fftfreq(pad)
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    bordes = np.linspace(0, np.pi * math.sqrt(2), nbins + 1)
    ib = np.clip(np.digitize(kk.ravel(), bordes) - 1, 0, nbins - 1)
    cnt = np.bincount(ib, minlength=nbins).astype(float)
    cnt[cnt == 0] = 1
    prof = np.stack([np.bincount(ib, weights=s.ravel(), minlength=nbins) for s in sk]) / cnt
    return bordes, prof


def analizar_espines(spins, mascara):
    red = red_espines(spins.astype(np.float64), mascara)
    N = int(mascara.sum())
    Q = carga_topologica(red, mascara)
    mvec = red.sum((1, 2, 3)) / N                            # (R, 3)
    mz_capa = red[..., 2].sum((2, 3)) / mascara.sum((1, 2))  # (R, capas)
    S = spins.reshape(spins.shape[0], -1).astype(np.float64)
    q = (S @ S.T) / N
    bordes, prof = factor_estructura(red, mascara)
    # quiralidad media por enlace en x e y dentro del plano
    sx = red[:, :, :, 1:]; sxm = red[:, :, :, :-1]
    okx = mascara[:, :, 1:] & mascara[:, :, :-1]
    qx = np.cross(sxm, sx)[..., 0] * okx
    sy = red[:, :, 1:, :]; sym = red[:, :, :-1, :]
    oky = mascara[:, 1:, :] & mascara[:, :-1, :]
    qy = np.cross(sym, sy)[..., 1] * oky
    quir = (qx.sum((1, 2, 3)) + qy.sum((1, 2, 3))) / (okx.sum() + oky.sum())
    return dict(Q=Q.astype(np.float32), mvec=mvec.astype(np.float32), mz_capa=mz_capa.astype(np.float32),
                q=q.astype(np.float32), sk_bordes=bordes, sk=prof.astype(np.float32), quir=quir.astype(np.float32))


# ----------------------------------------------------------------------------------------------------------
# Etapa A
# ----------------------------------------------------------------------------------------------------------
def rejilla_fina(escalera):
    return np.round(np.arange(escalera.min(), escalera.max() + PASO_FINO / 2, PASO_FINO), 4)


def etapa_A(tarea):
    """tarea = (estado, ruta, ruta_salida, escalera). Devuelve un texto de estado."""
    estado, ruta, salida, escalera = tarea
    if legible(salida, FORMATO_A):
        return "saltada"
    t0 = time.time()
    with np.load(ruta, allow_pickle=False) as d:
        D = {k: d[k] for k in d.files}
    # --- verificación
    for k in ("E", "M", "E_exc", "E_an", "E_dm", "E_hex"):
        assert D[k].shape == (N_MED, R_COPIAS), f"{ruta}: {k} con forma {D[k].shape}"
        assert np.isfinite(D[k]).all(), f"{ruta}: {k} con valores no finitos"
    assert D["spins"].shape[0] == R_COPIAS and D["spins"].shape[2] == 3
    norma = np.linalg.norm(D["spins"], axis=-1)
    assert np.allclose(norma, float(D["So"]), atol=1e-3), f"{ruta}: espines fuera de la esfera"
    T = float(D["T"])
    assert abs(T - temperatura_de(ruta)) < 1e-6
    mascara = D["mascara"].astype(bool)
    N = int(mascara.sum())
    kB, Eo = float(D["kB"]), float(D["Eo"])
    H = D["E"].astype(np.float64) * Eo
    m = D["M"].astype(np.float64)
    n, R = H.shape

    out = dict(formato=np.array(FORMATO_A), estado=estado, T=T, N=N, kB=kB, Eo=Eo,
               kDM=float(D["kDM"]), Kan1=float(D["Kan1"]), Hex=float(D["Hex"]), gamma=float(D["gamma"]),
               Jex=float(D["Jex"]), semilla=int(D["semilla"]))

    # --- sumas por copia (centradas en la media global de esta T)
    cH, cm = H.mean(), m.mean()
    yH, ym = H - cH, m - cm
    out["cH"], out["cm"] = cH, cm
    out["SH"] = np.stack([np.full(R, float(n)), yH.sum(0), (yH ** 2).sum(0)])        # (3, R)
    out["Sm"] = np.stack([np.full(R, float(n)), ym.sum(0), (ym ** 2).sum(0)])
    out["Sm24"] = np.stack([(m ** 2).sum(0), (m ** 4).sum(0)])                         # Binder
    out["comp"] = np.stack([D[k].astype(np.float64).mean(0) * Eo / N for k in ("E_exc", "E_an", "E_dm", "E_hex")])
    out["H_media_copia"], out["m_media_copia"] = H.mean(0), m.mean(0)

    # --- autocorrelación
    tH, WH, okH = tau_sokal(H)
    tm, Wm, okm = tau_sokal(m)
    tH2, _, _ = tau_sokal(yH ** 2)
    tm2, _, _ = tau_sokal(ym ** 2)
    out.update(tau_H=tH, tau_m=tm, tau_H2=tH2, tau_m2=tm2, tau_ok=okH & okm)
    tau_max = np.maximum.reduce([tH, tm, tH2, tm2])
    tau_t = float(np.median(tau_max))
    ok_b = LARGOS_BLOQUE >= C_SOKAL * tau_t
    b = int(LARGOS_BLOQUE[ok_b.argmax()] if ok_b.any() else LARGOS_BLOQUE[-1])
    out["b"], out["tau_t"] = b, tau_t
    out["jk_confiable"] = bool(ok_b.any() and n // b >= 20)
    fc, fx = 1.0 / (N * kB * T ** 2), N / (kB * T)
    out["c_copia"] = H.var(0) * fc
    out["chi_copia"] = m.var(0) * fx
    out["se_c_int"] = jk_var(H, b) * fc
    out["se_chi_int"] = jk_var(m, b) * fx

    # --- convergencia
    out["gew_H"], out["gew_m"] = geweke(H, tH), geweke(m, tm)
    out["mser_H"], out["mser_m"] = mser5(H), mser5(m)
    out["rhat_H"], out["rhat_H_bulk"] = rhat_rango(H)
    out["rhat_m"], out["rhat_m_bulk"] = rhat_rango(m)
    out["ess_H"] = float((n / (2 * tH)).sum())
    out["ess_m"] = float((n / (2 * tm)).sum())
    se_H = np.sqrt(2 * tH * H.var(0, ddof=1) / n)
    se_m = np.sqrt(2 * tm * m.var(0, ddof=1) / n)
    for nombre, mu, se in (("H", H.mean(0), se_H), ("m", m.mean(0), se_m)):
        w = 1 / se ** 2
        mbar = (w * mu).sum() / w.sum()
        Qh = float(((mu - mbar) ** 2 * w).sum())
        out[f"homog_Q_{nombre}"] = Qh
        out[f"homog_p_{nombre}"] = float(stats.chi2.sf(Qh, R - 1))
        out[f"welch_p_{nombre}"] = float(stats.ttest_ind(mu[: R // 2], mu[R // 2:], equal_var=False).pvalue)

    # --- repesado hacia la rejilla fina (ranura más cercana en beta)
    Tf = rejilla_fina(escalera)
    bf, bt = 1.0 / (kB * Tf), 1.0 / (kB * escalera)
    ranura = np.abs(bf[:, None] - bt[None]).argmin(1)
    k_esta = int(np.argmin(np.abs(escalera - T)))
    idx = np.flatnonzero(ranura == k_esta)
    beta = 1.0 / (kB * T)
    SHf = np.zeros((len(idx), 3, R)); Smf = np.zeros_like(SHf); ess = np.zeros(len(idx))
    for j, i in enumerate(idx):
        lw = -(bf[i] - beta) * yH
        lw -= lw.max()
        w = np.exp(lw)
        SHf[j] = np.stack([w.sum(0), (w * yH).sum(0), (w * yH * yH).sum(0)])
        Smf[j] = np.stack([w.sum(0), (w * ym).sum(0), (w * ym * ym).sum(0)])
        ess[j] = w.sum() ** 2 / (w ** 2).sum() / w.size
    out.update(Tf=Tf[idx], SHf=SHf, Smf=Smf, ess_rw=ess)

    # --- configuraciones de espines medidas
    esp = analizar_espines(D["spins"], mascara)
    out.update({f"sp_{k}": v for k, v in esp.items()})
    out["seg"] = time.time() - t0
    guardar_atomico(salida, **out)
    return f"ok {out['seg']:.1f}s"


# ----------------------------------------------------------------------------------------------------------
# Etapa B: bootstrap entre copias
# ----------------------------------------------------------------------------------------------------------
def Phi(x):
    return stats.norm.cdf(x)


def Phi_inv(p):
    return stats.norm.ppf(np.clip(p, 1e-12, 1 - 1e-12))


def var_agrupada(W, S):
    """W: (B, R); S: (nT, 3, R) -> varianza agrupada (B, nT)."""
    P = np.einsum("br,tkr->btk", W, S, optimize=True)
    return P[..., 2] / P[..., 0] - (P[..., 1] / P[..., 0]) ** 2


def binder(W, S24):
    P = np.einsum("br,tkr->btk", W, S24, optimize=True)   # sumas ponderadas de m^2 y m^4
    Wn = W.sum(1)[:, None] * N_MED
    m2, m4 = P[..., 0] / Wn, P[..., 1] / Wn
    return 1 - m4 / (3 * m2 ** 2)


def media_ponderada(W, X):
    """X: (nT, R) medias por copia -> (B, nT)."""
    return (W @ X.T) / W.sum(1)[:, None]


def intervalos(est, boots, jk, alpha=0.05):
    lo_p, hi_p = np.quantile(boots, [alpha / 2, 1 - alpha / 2], axis=0)
    jbar = jk.mean(0)
    num = ((jbar - jk) ** 3).sum(0)
    den = 6.0 * (((jbar - jk) ** 2).sum(0)) ** 1.5
    a = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    z0 = Phi_inv(np.mean(boots < est[None], 0))
    zs = Phi_inv(np.array([alpha / 2, 1 - alpha / 2]))
    q = [Phi(z0 + (z0 + z) / (1 - a * (z0 + z))) for z in zs]
    lo_b = np.array([np.quantile(boots[:, t], q[0][t]) for t in range(boots.shape[1])])
    hi_b = np.array([np.quantile(boots[:, t], q[1][t]) for t in range(boots.shape[1])])
    return lo_p, hi_p, lo_b, hi_b


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def multiplicidades(rng, B, R):
    return rng.multinomial(R, np.full(R, 1.0 / R), size=B).astype(np.float64)


def cargar_A(rutas_A):
    A = [dict(np.load(r, allow_pickle=False)) for r in rutas_A]
    A.sort(key=lambda d: -float(d["T"]))
    return A


def etapa_B(tarea):
    """tarea = (estado, rutas_A, salida, B, semilla, trozo)."""
    estado, rutas_A, salida, B, semilla, trozo = tarea
    if legible(salida, FORMATO_B):
        return "saltada"
    t0 = time.time()
    A = cargar_A(rutas_A)
    T = np.array([float(a["T"]) for a in A])
    N, kB = float(A[0]["N"]), float(A[0]["kB"])
    R = A[0]["SH"].shape[1]
    SH = np.stack([a["SH"] for a in A]); Sm = np.stack([a["Sm"] for a in A]); S24 = np.stack([a["Sm24"] for a in A])
    comp = np.stack([a["comp"] for a in A])                       # (nT, 4, R)
    fc, fx = 1.0 / (N * kB * T ** 2), N / (kB * T)
    # rejilla fina ensamblada
    Tf = np.concatenate([a["Tf"] for a in A]); o = np.argsort(Tf)
    Tf = Tf[o]
    SHf = np.concatenate([a["SHf"] for a in A])[o]; Smf = np.concatenate([a["Smf"] for a in A])[o]
    ess_rw = np.concatenate([a["ess_rw"] for a in A])[o]
    ffc, ffx = 1.0 / (N * kB * Tf ** 2), N / (kB * Tf)

    rng = np.random.default_rng(semilla)
    uno = np.ones((1, R))
    Wjk = np.ones((R, R)) - np.eye(R)

    def curva(fun):
        est = fun(uno)[0]
        jk = fun(Wjk)
        bo = np.concatenate([fun(multiplicidades(np.random.default_rng([semilla, i]), min(trozo, B - i), R))
                             for i in range(0, B, trozo)])
        return est, bo, jk

    res = {}
    for nombre, fun in (("c", lambda W: var_agrupada(W, SH) * fc),
                        ("chi", lambda W: var_agrupada(W, Sm) * fx),
                        ("U", lambda W: binder(W, S24)),
                        ("e_exc", lambda W: media_ponderada(W, comp[:, 0])),
                        ("e_an", lambda W: media_ponderada(W, comp[:, 1])),
                        ("e_dm", lambda W: media_ponderada(W, comp[:, 2])),
                        ("e_hex", lambda W: media_ponderada(W, comp[:, 3]))):
        est, bo, jk = curva(fun)
        lo_p, hi_p, lo_b, hi_b = intervalos(est, bo, jk)
        se = bo.std(0, ddof=1)
        res[nombre] = est
        res[nombre + "_se"] = se
        res[nombre + "_lo_p"], res[nombre + "_hi_p"], res[nombre + "_lo_b"], res[nombre + "_hi_b"] = lo_p, hi_p, lo_b, hi_b
        res[nombre + "_acuerdo"] = float(np.mean((np.abs(lo_p - lo_b) < se) & (np.abs(hi_p - hi_b) < se)))

    # --- curvas repesadas, T* y meseta, por trozos de réplicas (mismas semillas que arriba)
    for nombre, S, f in (("c", SHf, ffc), ("chi", Smf, ffx)):
        est = var_agrupada(uno, S)[0] * f
        k_est = int(est.argmax())
        ts, pico, banda_lo, banda_hi = [], [], [], []
        acum = []
        for i in range(0, B, trozo):
            W = multiplicidades(np.random.default_rng([semilla, i]), min(trozo, B - i), R)
            v = var_agrupada(W, S) * f
            ts.append(Tf[v.argmax(1)]); pico.append(v.max(1)); acum.append(v.astype(np.float32))
        ts = np.concatenate(ts); pico = np.concatenate(pico); V = np.concatenate(acum)
        lo, hi = np.quantile(V, [0.025, 0.975], axis=0)
        del V
        jk = var_agrupada(Wjk, S) * f
        ts_jk = Tf[jk.argmax(1)]
        se_pico = float(pico.std(ddof=1))
        umbral = est[k_est] - se_pico
        lo_i = hi_i = k_est
        while lo_i > 0 and est[lo_i - 1] >= umbral:
            lo_i -= 1
        while hi_i + 1 < len(Tf) and est[hi_i + 1] >= umbral:
            hi_i += 1
        borde = k_est in (0, len(Tf) - 1)
        res.update({f"{nombre}_f": est, f"{nombre}_f_lo": lo, f"{nombre}_f_hi": hi,
                    f"{nombre}_Tstar": Tf[k_est], f"{nombre}_Tstar_boot": ts.astype(np.float32),
                    f"{nombre}_Tstar_jk": ts_jk, f"{nombre}_pico": est[k_est], f"{nombre}_pico_boot": pico.astype(np.float32),
                    f"{nombre}_meseta": np.array([Tf[lo_i], Tf[hi_i]]), f"{nombre}_borde": borde})
        # BCa de T* (orientativo: el estimador es discreto)
        lo_p, hi_p, lo_b, hi_b = intervalos(np.array([Tf[k_est]]), ts[:, None].astype(float), ts_jk[:, None].astype(float))
        res[f"{nombre}_Tstar_ic_p"] = np.array([lo_p[0], hi_p[0]])
        res[f"{nombre}_Tstar_ic_bca"] = np.array([lo_b[0], hi_b[0]])

    # --- convergencia por temperatura
    g = lambda k: np.stack([a[k] for a in A])
    conv = dict(tau_H=g("tau_H"), tau_m=g("tau_m"), tau_t=g("tau_t"), b=g("b"), jk_confiable=g("jk_confiable"),
                gew_H=g("gew_H"), gew_m=g("gew_m"), mser_H=g("mser_H"), mser_m=g("mser_m"),
                rhat_H=g("rhat_H"), rhat_m=g("rhat_m"), ess_H=g("ess_H"), ess_m=g("ess_m"),
                homog_p_H=g("homog_p_H"), homog_p_m=g("homog_p_m"), homog_Q_H=g("homog_Q_H"), homog_Q_m=g("homog_Q_m"),
                welch_p_H=g("welch_p_H"), welch_p_m=g("welch_p_m"), c_copia=g("c_copia"), se_c_int=g("se_c_int"),
                H_media_copia=g("H_media_copia"), m_media_copia=g("m_media_copia"))
    gew_frac = np.stack([(np.abs(conv["gew_H"]) > 1.96).mean(1), (np.abs(conv["gew_m"]) > 1.96).mean(1)])
    gew_p = np.array([[stats.binomtest(int(round(f * R)), R, 0.05, alternative="greater").pvalue for f in fila] for fila in gew_frac])
    conv["gew_frac"], conv["gew_p_holm"] = gew_frac, np.stack([holm(p) for p in gew_p])
    conv["homog_p_holm"] = np.stack([holm(conv["homog_p_H"]), holm(conv["homog_p_m"])])
    conv["ratio_int_ext"] = np.sqrt((conv["se_c_int"] ** 2).mean(1)) / conv["c_copia"].std(1, ddof=1)
    res.update({"conv_" + k: v for k, v in conv.items()})

    # --- estados medidos (espines): todas las T, con detalle en la más baja y en T* de c
    Qs = g("sp_Q")                                  # (nT, R, capas)
    res["sp_Q"] = Qs
    res["sp_quir"] = g("sp_quir")
    res["sp_mvec"] = g("sp_mvec")
    res["sp_sk"] = g("sp_sk")
    res["sp_sk_bordes"] = A[0]["sp_sk_bordes"]
    res["sp_q_bajo"] = A[-1]["sp_q"]
    k_tstar = int(np.argmin(np.abs(T - res["c_Tstar"])))
    res["sp_q_tstar"] = A[k_tstar]["sp_q"]
    res["k_tstar"] = k_tstar
    Qtot = np.rint(Qs.mean(2))                     # Q medio entre capas, redondeado
    mod = np.array([stats.mode(q, keepdims=False).mode for q in Qtot])
    res["sp_Q_moda"] = mod
    res["sp_Q_frac_moda"] = (Qtot == mod[:, None]).mean(1)
    res["sp_Q_media"] = Qs.mean(2).mean(1)
    bq = np.concatenate([media_ponderada(multiplicidades(np.random.default_rng([semilla, 7, i]), min(trozo, B - i), R),
                                         Qs.mean(2)) for i in range(0, B, trozo)])
    res["sp_Q_media_ic"] = np.quantile(bq, [0.025, 0.975], axis=0)
    # agrupamiento de copias por perfil de S(k) (invariante ante rotaciones globales)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    ngrupos, frac_mayor, medoide = [], [], []
    for sk in res["sp_sk"]:
        p = sk[:, 1:] / np.linalg.norm(sk[:, 1:], axis=1, keepdims=True)
        dist = pdist(p, "cosine")
        lab = fcluster(linkage(dist, "average"), t=0.05, criterion="distance")
        cnt = np.bincount(lab)
        ngrupos.append(int((cnt > 0).sum()))
        frac_mayor.append(cnt.max() / R)
        mayor = np.flatnonzero(lab == cnt.argmax())
        Dm = np.linalg.norm(p[mayor][:, None] - p[mayor][None], axis=-1).sum(1)
        medoide.append(int(mayor[Dm.argmin()]))
    res["sp_ngrupos"], res["sp_frac_mayor"], res["sp_medoide"] = np.array(ngrupos), np.array(frac_mayor), np.array(medoide)
    kpk = res["sp_sk"].mean(1)[:, 1:].argmax(1) + 1
    bordes = res["sp_sk_bordes"]
    kc = 0.5 * (bordes[:-1] + bordes[1:])
    res["sp_k_pico"] = kc[kpk]
    res["sp_lambda"] = 2 * np.pi / kc[kpk]

    meta = {k: float(A[0][k]) for k in ("kDM", "Kan1", "Hex", "gamma", "Jex", "N", "kB", "Eo")}
    guardar_atomico(salida, formato=np.array(FORMATO_B), estado=estado, T=T, Tf=Tf, ess_rw=ess_rw, B=B, semilla=semilla,
                    meta=json.dumps(meta), seg=time.time() - t0, **res)
    return f"ok {time.time() - t0:.0f}s"
