import os
import time
import math
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, pmap, lax

# ==========================================================================
# SIMULADOR v2: Metropolis con propuesta en cono adaptativo (transformada inversa
# en un casquete esferico) + intercambio de replicas (parallel tempering).
#  - La energia (intercambio, DMI, anisotropia cubica, campo) es la misma del
#    simulador original del semillero.
#  - Normalizacion: Eo usa el numero real de espines del disco (N_spins).
#  - Cada "copia" es una escalera completa de NT temperaturas; las copias son
#    independientes entre si (unidad del bootstrap). La mitad arranca desordenada
#    (caliente) y la otra mitad ordenada en +z (fria).
# ==========================================================================

jax.config.update("jax_enable_x64", False)   # FP32 para velocidad


def _env(nombre, defecto, tipo=float):
    return tipo(os.environ.get(nombre, defecto))


SMOKE = os.environ.get("SMOKE", "0") == "1"   # prueba corta de humo, no para resultados
Rd = _env("RD", 9.0)
COPIAS = _env("COPIAS", 20, int)              # copias independientes de la escalera
NT = _env("NT", 128, int)                     # temperaturas de la escalera
TMIN, TMAX = _env("TMIN", 0.5), _env("TMAX", 20.0)
MCS_TERM = _env("MCS_TERM", 20000, int)       # barridos de termalizacion (con adaptacion del cono)
MCS_MEAS = _env("MCS_MEAS", 50000, int)       # barridos de medicion (cono fijo)
CADA = 10           # barridos entre mediciones guardadas
PAR = 2             # barridos entre intentos de intercambio (pares pares/impares alternados)
ADAPT = 100         # barridos entre ajustes del angulo del cono
TROZOS = 10         # la corrida se parte en trozos para reportar avance
ACC_OBJ = 0.5       # aceptacion objetivo de la propuesta en cono
GANANCIA = 1.0      # paso del ajuste log(alfa) += GANANCIA * (aceptacion - objetivo)
ALFA_MIN = 0.01     # angulo minimo del cono (rad); alfa = pi es la esfera completa
N_BAJAS = 8         # temperaturas mas bajas para las que se guarda el perfil radial
SEMILLA = _env("SEMILLA", 42, int)
SALIDA = os.environ.get("SALIDA", "resultados")
if SMOKE:
    NT, COPIAS, MCS_TERM, MCS_MEAS, ADAPT, TROZOS = 16, 4, 200, 400, 20, 2
N_BAJAS = min(N_BAJAS, NT)
TROZO_TERM = MCS_TERM // TROZOS
TROZO_MED = MCS_MEAS // TROZOS
assert MCS_TERM % TROZOS == 0 and TROZO_TERM % ADAPT == 0 and ADAPT % PAR == 0
assert MCS_MEAS % TROZOS == 0 and TROZO_MED % CADA == 0 and CADA % PAR == 0
assert COPIAS % 2 == 0, "la mitad de las copias arranca caliente y la otra fria"

# ==========================================
# PARAMETROS FISICOS (iguales al simulador original)
# ==========================================
L = 5
So = 1.0
kB = 0.086173404
NUM_ESTADOS = 2
Jex_base = [1.0, 1.0]
kDM_base = [1.0, 0.6]
Kan1_base = [0.05, 0.4]
Hex_base = [0.0, 0.0]
gamma_base = [90.0, 90.0]

num_gpus = jax.device_count()
assert NUM_ESTADOS % num_gpus == 0, "NUM_ESTADOS debe ser multiplo del numero de GPUs"
EG = NUM_ESTADOS // num_gpus              # estados por GPU
B = EG * NT * COPIAS                      # redes por GPU; indice b = (e*NT + t)*COPIAS + r

# ==========================================
# GEOMETRIA DEL DISCO
# ==========================================
L2 = int(Rd) + 1
LADO = 2 * L2 + 1
x_c = np.arange(-L2, L2 + 1).reshape(1, 1, 1, -1)
y_c = np.arange(-L2, L2 + 1).reshape(1, 1, -1, 1)
z_c = np.arange(1, L + 1).reshape(1, -1, 1, 1)
disco_2d = (x_c**2 + y_c**2 <= Rd**2)
disco_3d = np.broadcast_to(disco_2d, (1, L, LADO, LADO))
tablero = (x_c + y_c + z_c) % 2
MASK_PAR = jnp.array(((tablero == 0) & disco_3d)[..., None])
MASK_IMPAR = jnp.array(((tablero == 1) & disco_3d)[..., None])
MASK = jnp.array(disco_3d.astype(np.float32))
N_spins = int(disco_3d.sum())

# plaquetas con sus cuatro esquinas dentro del disco (carga topologica)
d2 = disco_2d[0, 0]
PLAQ_np = d2 & np.roll(d2, -1, 1) & np.roll(d2, -1, 0) & np.roll(np.roll(d2, -1, 0), -1, 1)
PLAQ = jnp.array(PLAQ_np.astype(np.float32))[None, None]

# anillos para el perfil radial de S_z: r = redondeo de la distancia al centro
radio = np.sqrt(x_c[0, 0]**2 + y_c[0, 0]**2)
anillo_idx = np.where(d2, np.floor(radio + 0.5).astype(int), -1)
NR = int(anillo_idx.max()) + 1
ANILLOS_np = np.zeros((LADO * LADO, NR), np.float32)
ANILLOS_np[np.nonzero(anillo_idx.ravel() >= 0)[0], anillo_idx.ravel()[anillo_idx.ravel() >= 0]] = 1.0
ANILLOS_np /= ANILLOS_np.sum(axis=0, keepdims=True)
ANILLOS = jnp.array(ANILLOS_np)

# escalera geometrica de temperaturas: ranura 0 = la mas fria
T_ESC = (TMIN * (TMAX / TMIN) ** (np.arange(NT) / (NT - 1))).astype(np.float64)
BETAS_np = 1.0 / (kB * T_ESC)
BETAS = jnp.array(BETAS_np, dtype=jnp.float32)

# indices auxiliares para reordenar las redes despues de un intercambio
E_IDX = jnp.arange(EG)[:, None, None]
R_IDX = jnp.arange(COPIAS)[None, None, :]
IDX_BAJAS = jnp.array([(e * NT + t) * COPIAS + r for e in range(EG) for t in range(N_BAJAS) for r in range(COPIAS)])
IDX_FRIA = np.array([(e * NT + 0) * COPIAS + r for e in range(EG) for r in range(COPIAS)])

# Eo con el numero real de espines del disco (antes usaba los nodos de la malla)
Eo_base = [(Jex_base[i] * 6 * So**2 + kDM_base[i] * 6 * So**2) * N_spins + Hex_base[i] * N_spins * So
           for i in range(NUM_ESTADOS)]


# ==========================================
# GENERACION DE DIRECCIONES
# ==========================================
def esfera_uniforme(key, shape):
    """Uniforme en S^2 por transformada inversa: cos(theta) = 2u - 1, phi = 2 pi v."""
    u = random.uniform(key, shape + (2,))
    cz = 2.0 * u[..., 0] - 1.0
    phi = 2.0 * jnp.pi * u[..., 1]
    sz = jnp.sqrt(jnp.maximum(1.0 - cz * cz, 0.0))
    return jnp.stack([sz * jnp.cos(phi), sz * jnp.sin(phi), cz], axis=-1)


def propuesta_cono(key, s, cos_a):
    """Uniforme en el casquete {S': S.S' >= cos(alfa)} alrededor de s (vectores unitarios, forma (..., 3)).
    En el marco donde s es el polo: z' = 1 - U1 (1 - cos alfa), psi = 2 pi U2 (transformada inversa);
    luego se rota al marco (e1, e2, s), que conserva el area. Con cos_a = -1 es la esfera completa."""
    u = random.uniform(key, s.shape[:-1] + (2,))
    cz = 1.0 - u[..., 0] * (1.0 - cos_a)
    psi = 2.0 * jnp.pi * u[..., 1]
    sz = jnp.sqrt(jnp.maximum(1.0 - cz * cz, 0.0))
    aux = jnp.where(jnp.abs(s[..., 0:1]) < 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    e1 = jnp.cross(s, aux)
    e1 = e1 / jnp.sqrt(jnp.maximum(jnp.sum(e1 * e1, -1, keepdims=True), 1e-30))
    e2 = jnp.cross(s, e1)
    v = cz[..., None] * s + sz[..., None] * (jnp.cos(psi)[..., None] * e1 + jnp.sin(psi)[..., None] * e2)
    return v / jnp.sqrt(jnp.maximum(jnp.sum(v * v, -1, keepdims=True), 1e-30))


def permutacion_intercambio(H, betas, key, paridad):
    """Intercambio de replicas entre ranuras vecinas (t, t+1) con t de la paridad dada.
    H: (G, NT, R) energias; betas: (NT,). Acepta con probabilidad min{1, exp[(b_t - b_t+1)(H_t - H_t+1)]}.
    Devuelve perm (G, NT, R): para cada ranura, la ranura de donde viene su nueva configuracion."""
    G, nt, r = H.shape
    activo = (jnp.arange(nt - 1) % 2) == paridad
    logr = (betas[:-1] - betas[1:])[None, :, None] * (H[:, :-1] - H[:, 1:])
    u = random.uniform(key, logr.shape)
    acepta = (jnp.log(u) < logr) & activo[None, :, None]
    z = jnp.zeros((G, 1, r), bool)
    sube = jnp.concatenate([acepta, z], axis=1)
    baja = jnp.concatenate([z, acepta], axis=1)
    perm = jnp.arange(nt)[None, :, None] + sube.astype(jnp.int32) - baja.astype(jnp.int32)
    return perm, acepta


# ==========================================
# ENERGIA Y METROPOLIS (mismas formulas del simulador original)
# ==========================================
def vecinos(s):
    xp = jnp.roll(s, -1, axis=3); xm = jnp.roll(s, 1, axis=3)
    yp = jnp.roll(s, -1, axis=2); ym = jnp.roll(s, 1, axis=2)
    zp = jnp.roll(s, -1, axis=1).at[:, -1].set(0.0)
    zm = jnp.roll(s, 1, axis=1).at[:, 0].set(0.0)
    return xp, xm, yp, ym, zp, zm


def anisotropia(s, K):
    return (K / So**4) * ((s[..., 0] * s[..., 1])**2 + (s[..., 0] * s[..., 2])**2 + (s[..., 1] * s[..., 2])**2)


def energia(s, par):
    """Energia total H (meV) de cada red, forma (B,)."""
    xp, xm, yp, ym, zp, zm = vecinos(s)
    suma = xp + xm + yp + ym + zp + zm
    e_exc = -par["J"] * jnp.sum(s * suma, axis=-1) * 0.5
    dm_x = s[..., 1] * (xp[..., 2] - xm[..., 2]) - s[..., 2] * (xp[..., 1] - xm[..., 1])
    dm_y = s[..., 2] * (yp[..., 0] - ym[..., 0]) - s[..., 0] * (yp[..., 2] - ym[..., 2])
    e_dm = -par["D"] * (dm_x + dm_y) * 0.5
    e_h = -par["Hx"] * (s[..., 0] * jnp.cos(par["g"]) + s[..., 2] * jnp.sin(par["g"]))
    return jnp.sum((e_exc + anisotropia(s, par["K"]) + e_dm + e_h) * MASK, axis=(1, 2, 3))


def subpaso(s, key, par, mask, cos_lat):
    xp, xm, yp, ym, zp, zm = vecinos(s)
    kt, ka = random.split(key)
    trial = propuesta_cono(kt, s, cos_lat) * So
    trial = jnp.where(mask, trial, s)
    dS = trial - s
    suma = xp + xm + yp + ym + zp + zm
    dE_exc = -par["J"] * jnp.sum(dS * suma, axis=-1)
    dE_an = anisotropia(trial, par["K"]) - anisotropia(s, par["K"])
    d_dm_x = dS[..., 1] * (xp[..., 2] - xm[..., 2]) - dS[..., 2] * (xp[..., 1] - xm[..., 1])
    d_dm_y = dS[..., 2] * (yp[..., 0] - ym[..., 0]) - dS[..., 0] * (yp[..., 2] - ym[..., 2])
    dE_dm = -par["D"] * (d_dm_x + d_dm_y)
    dE_h = -par["Hx"] * (dS[..., 0] * jnp.cos(par["g"]) + dS[..., 2] * jnp.sin(par["g"]))
    dE = dE_exc + dE_an + dE_dm + dE_h
    u = random.uniform(ka, dE.shape)
    acepta = (dE < 0) | (u < jnp.exp(-dE * par["beta"]))
    am = acepta[..., None] & mask
    return jnp.where(am, trial, s), jnp.sum(am[..., 0], axis=(1, 2, 3)).astype(jnp.float32)


def barrido(s, key, par, cos_lat):
    k1, k2 = random.split(key)
    s, a1 = subpaso(s, k1, par, MASK_PAR, cos_lat)
    s, a2 = subpaso(s, k2, par, MASK_IMPAR, cos_lat)
    return s, a1 + a2


# ==========================================
# OBSERVABLES
# ==========================================
def carga_topologica(s):
    """Q por capa con la formula de Berg-Luscher (dos triangulos por plaqueta), promediada en las L capas."""
    a = s
    b = jnp.roll(s, -1, axis=3)
    c = jnp.roll(b, -1, axis=2)
    d = jnp.roll(s, -1, axis=2)

    def omega(p, q, r):
        num = jnp.sum(p * jnp.cross(q, r), -1)
        den = 1.0 + jnp.sum(p * q, -1) + jnp.sum(q * r, -1) + jnp.sum(r * p, -1)
        return 2.0 * jnp.arctan2(num, den)

    om = (omega(a, b, c) + omega(a, c, d)) * PLAQ
    return jnp.sum(om, axis=(1, 2, 3)) / (4.0 * jnp.pi * L)


def perfil(s):
    """Perfil radial de S_z promediado en capas: (b, NR)."""
    mz = jnp.mean(s[..., 2], axis=1).reshape(s.shape[0], -1)
    return mz @ ANILLOS


# ==========================================
# INTERCAMBIO DE REPLICAS DENTRO DE LA GPU
# ==========================================
def intercambiar(est, H, key):
    paridad = est["cnt"] % 2
    perm, acepta = permutacion_intercambio(H.reshape(EG, NT, COPIAS), BETAS, key, paridad)
    plano = ((E_IDX * NT + perm) * COPIAS + R_IDX).reshape(-1)
    wid = jnp.take_along_axis(est["wid"], perm, axis=1)
    # viajes redondos: un caminante que toca la ranura fria despues de haber tocado la caliente
    ranuras = jnp.arange(NT)[None, :, None]
    en_fria = ranuras == wid[:, 0, :][:, None, :]
    en_caliente = ranuras == wid[:, -1, :][:, None, :]
    llega = jnp.sum(en_fria & (est["marca"] == 2), axis=1).astype(jnp.int32)
    marca = jnp.where(en_fria, 1, jnp.where(en_caliente, 2, est["marca"]))
    activo = ((jnp.arange(NT - 1) % 2) == paridad).astype(jnp.float32)
    return dict(est, s=est["s"][plano], H=H[plano], wid=wid, marca=marca, viajes=est["viajes"] + llega,
                sw_acc=est["sw_acc"] + jnp.sum(acepta, axis=-1).astype(jnp.float32),
                sw_int=est["sw_int"] + activo, cnt=est["cnt"] + 1)


def pares(est, par, k):
    """k bloques de PAR barridos seguidos de un intento de intercambio. Devuelve la aceptacion acumulada por red."""
    cos_lat = jnp.repeat(jnp.cos(est["alfa"]).reshape(-1), COPIAS)[:, None, None, None]

    def cuerpo(j, carry):
        est, acc = carry
        key, k1, k2, k3 = random.split(est["key"], 4)
        s, a1 = barrido(est["s"], k1, par, cos_lat)
        s, a2 = barrido(s, k2, par, cos_lat)
        est = intercambiar(dict(est, s=s, key=key), energia(s, par), k3)
        return est, acc + a1 + a2

    return lax.fori_loop(0, k, cuerpo, (est, jnp.zeros(B, jnp.float32)))


@pmap
def termalizar(est, par):
    def bloque(i, est):
        est, acc = pares(est, par, ADAPT // PAR)
        tasa = (acc / (N_spins * ADAPT)).reshape(EG, NT, COPIAS).mean(axis=-1)
        alfa = jnp.clip(jnp.exp(jnp.log(est["alfa"]) + GANANCIA * (tasa - ACC_OBJ)), ALFA_MIN, jnp.pi)
        return dict(est, alfa=alfa)
    return lax.fori_loop(0, TROZO_TERM // ADAPT, bloque, est)


@pmap
def medir(est, par):
    def paso(est, _):
        est, acc = pares(est, par, CADA // PAR)
        est = dict(est, acc_med=est["acc_med"] + acc)
        s = est["s"]
        Mv = jnp.sum(s, axis=(1, 2, 3))
        m = jnp.sqrt(jnp.sum(Mv * Mv, axis=-1)) / N_spins
        return est, (est["H"], m, carga_topologica(s), perfil(s[IDX_BAJAS]))
    return lax.scan(paso, est, None, length=TROZO_MED // CADA)


def estado_inicial():
    G = num_gpus
    key = random.PRNGKey(SEMILLA)
    k_ini, k_run = random.split(key)
    s = np.array(esfera_uniforme(k_ini, (G, B, L, LADO, LADO)))
    fria = (np.arange(B) % COPIAS) >= COPIAS // 2          # copias r >= R/2 arrancan ordenadas en +z
    s[:, fria] = np.array([0.0, 0.0, 1.0], np.float32)
    s = (s * np.asarray(disco_3d)[0][None, None, ..., None] * So).astype(np.float32)
    est = dict(s=s, key=np.asarray(random.split(k_run, G)),
               alfa=np.full((G, EG, NT), np.pi, np.float32),
               wid=np.broadcast_to(np.arange(NT, dtype=np.int32)[None, None, :, None], (G, EG, NT, COPIAS)).copy(),
               marca=np.zeros((G, EG, NT, COPIAS), np.int32), viajes=np.zeros((G, EG, COPIAS), np.int32),
               H=np.zeros((G, B), np.float32), sw_acc=np.zeros((G, EG, NT - 1), np.float32),
               sw_int=np.zeros((G, NT - 1), np.float32), cnt=np.zeros(G, np.int32), acc_med=np.zeros((G, B), np.float32))

    def por_red(vals):
        a = np.zeros((G, B), np.float32)
        for g in range(G):
            for e in range(EG):
                a[g, e * NT * COPIAS:(e + 1) * NT * COPIAS] = vals[g * EG + e]
        return a[..., None, None, None]

    beta = np.zeros((G, B), np.float32)
    beta[:] = np.tile(np.repeat(BETAS_np, COPIAS), EG)[None]
    par = dict(J=por_red(Jex_base), D=por_red(kDM_base), K=por_red(Kan1_base), Hx=por_red(Hex_base),
               g=por_red([math.radians(v) for v in gamma_base]), beta=beta[..., None, None, None])
    return est, par


if __name__ == "__main__":
    print(f"Rd={Rd} | espines={N_spins} | NT={NT} ({TMIN}-{TMAX} K) | copias={COPIAS} | GPUs={num_gpus} | "
          f"termalizacion={MCS_TERM} | medicion={MCS_MEAS} (cada {CADA})", flush=True)
    t0 = time.time()
    est, par = estado_inicial()
    for i in range(TROZOS):
        est = termalizar(est, par)
        jax.block_until_ready(est["s"])
        print(f"termalizacion {i + 1}/{TROZOS}  {(time.time() - t0) / 60:.1f} min", flush=True)
    # los diagnosticos de la cadena se cuentan solo durante la medicion
    est = dict(est, marca=jnp.zeros_like(est["marca"]), viajes=jnp.zeros_like(est["viajes"]),
               sw_acc=jnp.zeros_like(est["sw_acc"]), sw_int=jnp.zeros_like(est["sw_int"]),
               acc_med=jnp.zeros_like(est["acc_med"]))
    trozos = []
    for i in range(TROZOS):
        est, out = medir(est, par)
        trozos.append([np.asarray(o) for o in out])
        print(f"medicion {i + 1}/{TROZOS}  {(time.time() - t0) / 60:.1f} min", flush=True)

    G = num_gpus
    H, m, Q, prof = [np.concatenate([tr[k] for tr in trozos], axis=1) for k in range(4)]
    n = H.shape[1]
    H = H.reshape(G, n, EG, NT, COPIAS); m = m.reshape(G, n, EG, NT, COPIAS); Q = Q.reshape(G, n, EG, NT, COPIAS)
    prof = prof.reshape(G, n, EG, N_BAJAS, COPIAS, NR)
    est = {k: np.asarray(v) for k, v in est.items()}
    acc = (est["acc_med"] / (N_spins * MCS_MEAS)).reshape(G, EG, NT, COPIAS)
    foto = est["s"][:, IDX_FRIA].reshape(G, EG, COPIAS, L, LADO, LADO, 3)

    os.makedirs(SALIDA, exist_ok=True)
    for gl in range(NUM_ESTADOS):
        g, e = gl // EG, gl % EG
        np.savez_compressed(
            f"{SALIDA}/PT_est{gl}.npz",
            H=H[g, :, e], m=m[g, :, e], Q=Q[g, :, e], perfil=prof[g, :, e],
            T=T_ESC, beta=BETAS_np, alfa=est["alfa"][g, e], aceptacion=acc[g, e],
            sw_acc=est["sw_acc"][g, e], sw_int=est["sw_int"][g], viajes=est["viajes"][g, e],
            foto=foto[g, e], anillo_idx=anillo_idx, arranque_frio=np.arange(COPIAS) >= COPIAS // 2,
            n_spins=N_spins, Eo=Eo_base[gl], kB=kB, Rd=Rd, L=L, cada=CADA, mcs_term=MCS_TERM, mcs_meas=MCS_MEAS,
            Jex=Jex_base[gl], kDM=kDM_base[gl], Kan=Kan1_base[gl], Hex=Hex_base[gl], gamma=gamma_base[gl])
        sw = est["sw_acc"][g, e] / np.maximum(est["sw_int"][g] * COPIAS, 1)
        print(f"estado {gl}: aceptacion local [{acc[g, e].mean(-1).min():.2f}, {acc[g, e].mean(-1).max():.2f}] | "
              f"intercambio minimo {sw.min():.2f} | viajes redondos por caminante (mediana) "
              f"{np.median(est['viajes'][g, e]) / NT:.1f}", flush=True)
    print(f"listo en {(time.time() - t0) / 3600:.2f} h", flush=True)
