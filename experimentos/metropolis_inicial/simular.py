import jax
import jax.numpy as jnp
from jax import random, pmap, jit, lax
from jax.scipy.special import gammaincc
from functools import partial
import numpy as np
import time
import math
import os

# ==========================================================================
# BLOQUES DEL CURSO CUBIERTOS POR ESTE CODIGO
# --------------------------------------------------------------------------
# [B4] MCMC: Metropolis de un solo espin con propuesta simetrica y actualizacion
#      en tablero de ajedrez. NO se modifica (mismas formulas, misma energia,
#      misma normalizacion por Eo).
# [B1] Generacion de variables aleatorias: la direccion de propuesta y el estado
#      inicial se generan por TRANSFORMADA INVERSA a partir de U(0,1):
#      cos(theta)=2u-1, phi=2*pi*v. Prueba de uniformidad chi-cuadrado opcional.
# [B2] Monte Carlo + remuestreo: jackknife por bloques, tiempo de autocorrelacion
#      y bootstrap entre replicas para Cv, Chi, E, M y temperaturas de pico.
#      Se activa/desactiva con la bandera CALCULAR_ERRORES.
#
# Con CALCULAR_ERRORES = False el flujo es el de siempre (solo Datos_*.npz).
# Con CALCULAR_ERRORES = True se genera ademas Stats_*.npz por estado.
# ==========================================================================

# Verificación de entorno Multi-GPU
num_gpus = jax.device_count()
SMOKE = os.environ.get("SMOKE", "0") == "1"   # prueba corta de humo (CPU o GPU), no para resultados
if not SMOKE:
    assert num_gpus >= 2, f"Se requieren al menos 2 GPUs, detectadas: {num_gpus}"
jax.config.update("jax_enable_x64", False) # Mantener FP32 para velocidad

# ==========================================
# BANDERAS DE EJECUCION
# ==========================================
CALCULAR_ERRORES = True        # True: jackknife + tau + bootstrap (Stats_*.npz). False: flujo original.
PRUEBA_UNIFORMIDAD = False     # True: chi-cuadrado de uniformidad del generador de direcciones (B1)
METODO_ESFERA = "inversa"      # "inversa" (transformada inversa) | "gauss" (normalizar gaussianas)

# ==========================================
# PARÁMETROS FIJOS (GLOBALES)
# ==========================================
NUM_ESTADOS = 16
REPLICAS_POR_ESTADO = 100
MCS_TERM = 16000
MCS_MEAS = 2000
MCS_MAX = MCS_TERM + MCS_MEAS

Rd = 9.0   # [protocolo v01] disco reducido: 1265 espines (antes 20.3, 6525 espines)
L = 5
So = 1.0
Ti = 20.0
Tf = 0.0
dT = -0.1
kB = 0.086173404
L2 = int(Rd) + 1
Ntot_grid = (2*L2 + 1) * (2*L2 + 1) * (L + 2) 

# ==========================================
# PARÁMETROS DE LA ESTADISTICA DE ERRORES (solo se usan si CALCULAR_ERRORES)
# ==========================================
N_BLOQUES_JK = 10             # bloques del jackknife (MCS_MEAS debe ser divisible)
SOKAL_C = 5.0                 # constante de la ventana automatica de Sokal
NBOOT = 2000                  # remuestras bootstrap entre replicas
UNIF_N_POR_GPU = 2_000_000    # muestras por GPU en la prueba de uniformidad
UNIF_BINS = 64

SALIDA = os.environ.get("SALIDA", "resultados")
NOMBRES_SERIES = ("E", "M", "E_exc", "E_an", "E_dm", "E_hex")
NOMBRES_BOOT = NOMBRES_SERIES + ("Cv", "Chi")
# [E3] Tamaños de bloque para la meseta del jackknife (todos dividen MCS_MEAS); b* se elige despues con b >= 10 tau, g >= 20
LARGOS_BLOQUE = (1, 2, 5, 10, 20, 50, 100)
SOKAL_C = 10.0   # [r01 Ec. 9] convencion tau = 1 + 2 sum rho, equivale a c ~ 5 de Sokal
T_SERIES = 1.0   # [E3] se guardan las series completas de E y M cada T_SERIES K
if SMOKE:
    MCS_TERM, MCS_MEAS, REPLICAS_POR_ESTADO, NBOOT, Ti = 100, 200, 8, 200, 1.0
    MCS_MAX = MCS_TERM + MCS_MEAS
    LARGOS_BLOQUE = (1, 2, 5, 10)
assert all(MCS_MEAS % b == 0 for b in LARGOS_BLOQUE), "cada largo de bloque debe dividir MCS_MEAS"
assert MCS_MEAS % N_BLOQUES_JK == 0, "MCS_MEAS debe ser multiplo de N_BLOQUES_JK"

# # Genera una semilla basada en hardware/OS
# np_seed = int.from_bytes(os.urandom(4), 'little')
# np.random.seed(np_seed)
# # Generación de parámetros (se mantiene tu código, pero ahora será distinto cada vez)
# Jex_base  = [1.0 for _ in range(NUM_ESTADOS)]
# kDM_base  = [round(np.random.uniform(0.0, 1.5), 2) for _ in range(NUM_ESTADOS)]
# Kan1_base = [round(np.random.uniform(0.0, 0.6), 2) for _ in range(NUM_ESTADOS)]
# Hex_base  = [round(np.random.uniform(0.0, 1.5), 2) for _ in range(NUM_ESTADOS)]
# gamma_base= [90.0 for _ in range(NUM_ESTADOS)] 
# gamma_rad_base = [math.radians(g) for g in gamma_base]

NUM_ESTADOS = 2

# Valores fijos para los dos estados de control
Jex_base  = [1.0, 1.0]
kDM_base  = [1.0, 0.6]
Kan1_base = [0.05, 0.4]
Hex_base  = [0, 0]
gamma_base= [90, 90] 
gamma_rad_base = [math.radians(g) for g in gamma_base]

assert NUM_ESTADOS % num_gpus == 0, "NUM_ESTADOS debe ser multiplo del numero de GPUs"

# Preparación de tensores divididos para jax.pmap (Num_GPUs, Replicas_por_GPU, ...)
estados_por_gpu = NUM_ESTADOS // num_gpus
batch_por_gpu = estados_por_gpu * REPLICAS_POR_ESTADO

def format_param(base_list):
    arr = np.array(base_list).repeat(REPLICAS_POR_ESTADO)
    return arr.reshape(num_gpus, batch_por_gpu, 1, 1, 1)

Jex_pmap = format_param(Jex_base)
kDM_pmap = format_param(kDM_base)
Kan1_pmap = format_param(Kan1_base)
Hex_pmap = format_param(Hex_base)
gamma_pmap = format_param(gamma_rad_base)
Eo_pmap = (Jex_pmap * Ntot_grid * 6 * So**2 + kDM_pmap * Ntot_grid * 6 * So**2 + Hex_pmap * Ntot_grid * So).reshape(num_gpus, batch_por_gpu)

# ==========================================
# PRECALCULO DE MÁSCARAS (CPU -> JAX)
# ==========================================
x_coords = np.arange(-L2, L2 + 1).reshape(1, 1, 1, -1)
y_coords = np.arange(-L2, L2 + 1).reshape(1, 1, -1, 1)
z_coords = np.arange(1, L + 1).reshape(1, -1, 1, 1)

radii_sq = x_coords**2 + y_coords**2
disco_mask_2d = (radii_sq <= Rd**2)

# Equivalente exacto de .expand(1, L, 2*L2 + 1, 2*L2 + 1)
disco_mask_3d = np.broadcast_to(disco_mask_2d, (1, L, 2*L2 + 1, 2*L2 + 1))

checkerboard = (x_coords + y_coords + z_coords) % 2

mask_even_np = ((checkerboard == 0) & disco_mask_3d)[..., np.newaxis]
mask_odd_np  = ((checkerboard == 1) & disco_mask_3d)[..., np.newaxis]
disco_mask_np = disco_mask_3d[..., np.newaxis]

N_spins = int(disco_mask_3d.sum()) 

# Transferir a dispositivos JAX
mask_even_jnp = jnp.array(mask_even_np)
mask_odd_jnp = jnp.array(mask_odd_np)
disco_mask_jnp = jnp.array(disco_mask_np)

# ==========================================
# [B1] GENERACION DE DIRECCIONES ALEATORIAS UNIFORMES EN LA ESFERA
# ==========================================
def esfera_uniforme(key, shape, metodo=METODO_ESFERA):
    """Vectores unitarios uniformes en S^2 con forma shape + (3,).
    'inversa': dOmega = sin(theta) dtheta dphi  =>  cos(theta) ~ U[-1,1], phi ~ U[0,2pi).
               Transformada inversa aplicada a dos U(0,1) independientes.
    'gauss'  : normalizar una gaussiana 3D (metodo anterior, se conserva para comparar)."""
    if metodo == "inversa":
        u = random.uniform(key, shape + (2,))
        cos_t = 2.0 * u[..., 0] - 1.0
        phi = 2.0 * jnp.pi * u[..., 1]
        sin_t = jnp.sqrt(jnp.maximum(1.0 - cos_t * cos_t, 0.0))
        return jnp.stack([sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), cos_t], axis=-1)
    elif metodo == "gauss":
        v = random.normal(key, shape + (3,))
        return v / jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True))
    raise ValueError(f"metodo de esfera desconocido: {metodo}")

@pmap
def _muestras_esfera_pmap(key):
    """Cada GPU genera su propia muestra (claves distintas) con ambos metodos y devuelve
    histogramas de cos(theta) y phi, y momentos de primer/segundo orden."""
    ks = random.split(key, 2)
    salida = []
    for i, metodo in enumerate(("inversa", "gauss")):
        v = esfera_uniforme(ks[i], (UNIF_N_POR_GPU,), metodo)
        z = v[:, 2]
        phi = jnp.arctan2(v[:, 1], v[:, 0])
        iz = jnp.clip(jnp.floor((z + 1.0) / 2.0 * UNIF_BINS).astype(jnp.int32), 0, UNIF_BINS - 1)
        ip = jnp.clip(jnp.floor((phi + jnp.pi) / (2.0 * jnp.pi) * UNIF_BINS).astype(jnp.int32), 0, UNIF_BINS - 1)
        salida.append((jnp.bincount(iz, length=UNIF_BINS),
                       jnp.bincount(ip, length=UNIF_BINS),
                       jnp.mean(v, axis=0),
                       jnp.mean(v * v, axis=0)))
    return tuple(salida)

def prueba_uniformidad_esfera(key):
    """Chi-cuadrado de bondad de ajuste (cos(theta) y phi uniformes) + momentos. Se ejecuta una vez."""
    res = _muestras_esfera_pmap(random.split(key, num_gpus))
    informe = {}
    for metodo, r in zip(("inversa", "gauss"), res):
        cz = np.asarray(r[0]).sum(axis=0).astype(np.float64)
        cp = np.asarray(r[1]).sum(axis=0).astype(np.float64)
        m1 = np.asarray(r[2]).mean(axis=0)
        m2 = np.asarray(r[3]).mean(axis=0)
        n = cz.sum()
        esperado = n / UNIF_BINS
        chi_z = float(((cz - esperado)**2 / esperado).sum())
        chi_p = float(((cp - esperado)**2 / esperado).sum())
        df = UNIF_BINS - 1
        p_z = float(gammaincc(df / 2.0, chi_z / 2.0))
        p_p = float(gammaincc(df / 2.0, chi_p / 2.0))
        informe[metodo] = dict(chi2_cos=chi_z, p_cos=p_z, chi2_phi=chi_p, p_phi=p_p,
                               media_xyz=m1, segundo_momento_xyz=m2, n=n)
        print(f"[B1] Uniformidad '{metodo}': chi2(cos)={chi_z:.1f} p={p_z:.3f} | chi2(phi)={chi_p:.1f} p={p_p:.3f} "
              f"| <v>={np.round(m1, 4)} | <v^2>={np.round(m2, 4)} (esperado 0 y 0.3333) | N={int(n)}")
    return informe

# ==========================================
# FUNCIONES JAX COMPILADAS  [B4]
# ==========================================
def compute_energy_components(spins, mask_vec, kDM, Jex, Kan1, Hex_mag, gamma_rad, So):
    S_xp = jnp.roll(spins, -1, axis=3); S_xm = jnp.roll(spins, 1, axis=3)
    S_yp = jnp.roll(spins, -1, axis=2); S_ym = jnp.roll(spins, 1, axis=2)
    S_zp = jnp.roll(spins, -1, axis=1); S_zm = jnp.roll(spins, 1, axis=1)
    S_zp = S_zp.at[:, -1, :, :, :].set(0.0)
    S_zm = S_zm.at[:, 0, :, :, :].set(0.0)
    sum_vecinos = S_xp + S_xm + S_yp + S_ym + S_zp + S_zm
    
    E_exc = -Jex * jnp.sum(spins * sum_vecinos, axis=-1) * 0.5
    E_an1 = (Kan1 / So**4) * ((spins[...,0]*spins[...,1])**2 + (spins[...,0]*spins[...,2])**2 + (spins[...,1]*spins[...,2])**2)
    dm_x = spins[...,1]*(S_xp[...,2] - S_xm[...,2]) - spins[...,2]*(S_xp[...,1] - S_xm[...,1])
    dm_y = spins[...,2]*(S_yp[...,0] - S_ym[...,0]) - spins[...,0]*(S_yp[...,2] - S_ym[...,2])
    E_dm = -kDM * (dm_x + dm_y) * 0.5
    E_hex = -Hex_mag * (spins[...,0]*jnp.cos(gamma_rad) + spins[...,2]*jnp.sin(gamma_rad))
    
    mask = jnp.squeeze(mask_vec, axis=-1)
    return (jnp.sum(E_exc * mask, axis=(1,2,3)),
            jnp.sum(E_an1 * mask, axis=(1,2,3)),
            jnp.sum(E_dm * mask, axis=(1,2,3)),
            jnp.sum(E_hex * mask, axis=(1,2,3)))

def metropolis_substep(spins, key, beta, mask_vec, kDM, Jex, Kan1, Hex_mag, gamma_rad):
    S_xp = jnp.roll(spins, -1, axis=3); S_xm = jnp.roll(spins, 1, axis=3)
    S_yp = jnp.roll(spins, -1, axis=2); S_ym = jnp.roll(spins, 1, axis=2)
    S_zp = jnp.roll(spins, -1, axis=1); S_zm = jnp.roll(spins, 1, axis=1)
    S_zp = S_zp.at[:, -1, :, :, :].set(0.0)
    S_zm = S_zm.at[:, 0, :, :, :].set(0.0)
    
    key_t, key_a = random.split(key)
    # [B1] propuesta: direccion uniforme en la esfera por transformada inversa (antes: normal + normalizar)
    spins_trial = esfera_uniforme(key_t, spins.shape[:-1]) * So
    spins_trial = jnp.where(mask_vec, spins_trial, spins)
    dS = spins_trial - spins
    
    sum_vecinos = S_xp + S_xm + S_yp + S_ym + S_zp + S_zm
    dE_exc = -Jex * jnp.sum(dS * sum_vecinos, axis=-1)
    E_an1_t = (Kan1 / So**4) * ((spins_trial[...,0]*spins_trial[...,1])**2 + (spins_trial[...,0]*spins_trial[...,2])**2 + (spins_trial[...,1]*spins_trial[...,2])**2)
    E_an1_s = (Kan1 / So**4) * ((spins[...,0]*spins[...,1])**2 + (spins[...,0]*spins[...,2])**2 + (spins[...,1]*spins[...,2])**2)
    dE_an1 = E_an1_t - E_an1_s
    d_dm_x = dS[...,1]*(S_xp[...,2] - S_xm[...,2]) - dS[...,2]*(S_xp[...,1] - S_xm[...,1])
    d_dm_y = dS[...,2]*(S_yp[...,0] - S_ym[...,0]) - dS[...,0]*(S_yp[...,2] - S_ym[...,2])
    dE_dm = -kDM * (d_dm_x + d_dm_y)
    dE_hex = -Hex_mag * (dS[...,0]*jnp.cos(gamma_rad) + dS[...,2]*jnp.sin(gamma_rad))
    delta_E = dE_exc + dE_an1 + dE_dm + dE_hex
    
    rand_tensor = random.uniform(key_a, delta_E.shape)
    accept = (delta_E < 0) | (rand_tensor < jnp.exp(-delta_E * beta))
    accept_mask = jnp.expand_dims(accept, -1) & mask_vec
    # [E3] numero de aceptaciones por replica (solo sitios del subtablero)
    n_acc = jnp.sum(jnp.squeeze(accept_mask, -1), axis=(1, 2, 3))
    return jnp.where(accept_mask, spins_trial, spins), n_acc

# Bucle compilado en GPU de termalización pura
def term_loop(spins, key, beta, mask_even, mask_odd, kDM, Jex, Kan1, Hex_mag, gamma_rad):
    def body_fn(i, val):
        s, k = val
        k, k1, k2 = random.split(k, 3)
        s, _ = metropolis_substep(s, k1, beta, mask_even, kDM, Jex, Kan1, Hex_mag, gamma_rad)
        s, _ = metropolis_substep(s, k2, beta, mask_odd, kDM, Jex, Kan1, Hex_mag, gamma_rad)
        return s, k
    return jax.lax.fori_loop(0, MCS_TERM, body_fn, (spins, key))

# Bucle compilado en GPU que acumula mediciones paso a paso sin escribir en disco
def meas_loop(spins, key, beta, mask_even, mask_odd, mask_disco, kDM, Jex, Kan1, Hex_mag, gamma_rad, Eo_batch):
    def step_fn(val, _):
        s, k = val
        k, k1, k2 = random.split(k, 3)
        s, a1 = metropolis_substep(s, k1, beta, mask_even, kDM, Jex, Kan1, Hex_mag, gamma_rad)
        s, a2 = metropolis_substep(s, k2, beta, mask_odd, kDM, Jex, Kan1, Hex_mag, gamma_rad)
        acc = (a1 + a2) / N_spins   # [E3] tasa de aceptacion del barrido
        
        e_jex, e_kan, e_kdm, e_hex = compute_energy_components(s, mask_disco, kDM, Jex, Kan1, Hex_mag, gamma_rad, So)
        Ene_batch = e_jex + e_kan + e_kdm + e_hex 
        M_vec = jnp.sum(s * mask_disco, axis=(1,2,3))
        M_norm = jnp.sqrt(jnp.sum(M_vec * M_vec, axis=-1)) / N_spins   # misma norma euclidea, sin linalg.norm
        
        # Guardar en el historial (gestionado internamente por lax.scan)
        metrics = (Ene_batch / Eo_batch, M_norm, e_jex / Eo_batch, e_kan / Eo_batch, e_kdm / Eo_batch, e_hex / Eo_batch, acc)
        return (s, k), metrics
    return jax.lax.scan(step_fn, (spins, key), None, length=MCS_MEAS)

# ==========================================
# PARALELISMO NATIVO MULTI-GPU (PMAP)
# ==========================================
# Se mapean las variables con prefijo pmap a lo largo del eje 0 (num_gpus).
@partial(pmap, in_axes=(0, 0, None, None, None, None, 0, 0, 0, 0, 0, 0))
def step_temperature_multi_gpu(spins, key, beta, mask_even, mask_odd, mask_disco, kDM, Jex, Kan1, Hex_mag, gamma_rad, Eo_batch):
    spins, key = term_loop(spins, key, beta, mask_even, mask_odd, kDM, Jex, Kan1, Hex_mag, gamma_rad)
    (spins, key), metrics = meas_loop(spins, key, beta, mask_even, mask_odd, mask_disco, kDM, Jex, Kan1, Hex_mag, gamma_rad, Eo_batch)
    return spins, key, metrics

# ==========================================
# [B2] ESTADISTICA DE LA MEDICION EN GPU (jackknife por bloques + autocorrelacion)
# Solo se llama si CALCULAR_ERRORES = True
# ==========================================
def jackknife_bloques(x):
    """x: (n, B) serie temporal por replica. Devuelve (media, se_media, var, sesgo_var, se_var).
    Estimador de varianza identico al usado en el pipeline (E[x^2]-E[x]^2, divisor N), pero
    calculado sobre datos centrados para evitar cancelacion en FP32.
    Jackknife leave-one-block-out: bias=(g-1)(tbar-theta), se=sqrt((g-1)/g * sum (theta_j-tbar)^2)."""
    n, B = x.shape
    nb = N_BLOQUES_JK
    bl = n // nb
    N = nb * bl
    x = x[:N]
    mu0 = jnp.mean(x, axis=0)
    xc = (x - mu0).reshape(nb, bl, B)
    s1 = jnp.sum(xc, axis=1)                         # (nb,B)
    s2 = jnp.sum(xc * xc, axis=1)
    S1 = jnp.sum(s1, axis=0)
    S2 = jnp.sum(s2, axis=0)
    media_c = S1 / N
    media = mu0 + media_c
    var = S2 / N - media_c**2
    m_j = (S1[None, :] - s1) / (N - bl)              # media centrada sin el bloque j
    v_j = (S2[None, :] - s2) / (N - bl) - m_j**2     # varianza sin el bloque j
    def _jk(theta, theta_j):
        tbar = jnp.mean(theta_j, axis=0)
        sesgo = (nb - 1) * (tbar - theta)
        se = jnp.sqrt((nb - 1) / nb * jnp.sum((theta_j - tbar)**2, axis=0))
        return sesgo, se
    _, se_m = _jk(media_c, m_j)
    sesgo_v, se_v = _jk(var, v_j)
    return media, se_m, var, sesgo_v, se_v

def jackknife_bloques_b(x, bl):
    """Error jackknife de la media y de la varianza con largo de bloque bl fijo (g = n // bl bloques)."""
    n, B = x.shape
    nb = n // bl
    N = nb * bl
    x = x[:N]
    mu0 = jnp.mean(x, axis=0)
    xc = (x - mu0).reshape(nb, bl, B)
    s1 = jnp.sum(xc, axis=1); s2 = jnp.sum(xc * xc, axis=1)
    S1 = jnp.sum(s1, axis=0); S2 = jnp.sum(s2, axis=0)
    m_j = (S1[None, :] - s1) / (N - bl)
    v_j = (S2[None, :] - s2) / (N - bl) - m_j**2
    def _se(theta_j):
        tbar = jnp.mean(theta_j, axis=0)
        return jnp.sqrt((nb - 1) / nb * jnp.sum((theta_j - tbar)**2, axis=0))
    return _se(m_j), _se(v_j)

def tau_int_sokal(x):
    """Tiempo de autocorrelacion integrado tau=1+2*sum rho(t) con ventana automatica de Sokal
    (primer M con M >= c*tau(M)). ACF por FFT (convolucion circular con relleno de ceros). x: (n,B)."""
    n, B = x.shape
    xc = x - jnp.mean(x, axis=0)
    f = jnp.fft.rfft(xc, n=2 * n, axis=0)
    acf = jnp.fft.irfft(f * jnp.conj(f), n=2 * n, axis=0)[:n]
    acf0 = acf[0]
    rho = acf / jnp.where(acf0 > 1e-24, acf0, 1.0)   # serie constante -> rho=0 -> tau=1
    cum = 1.0 + 2.0 * jnp.cumsum(rho[1:], axis=0)    # tau(M), M=1..n-1
    t = jnp.arange(1, n, dtype=x.dtype)[:, None]
    cond = t >= SOKAL_C * cum
    primero = jnp.argmax(cond, axis=0)
    hay = jnp.any(cond, axis=0)
    idx = jnp.where(hay, primero, n - 2)
    tau = jnp.take_along_axis(cum, idx[None, :], axis=0)[0]
    return jnp.maximum(tau, 1.0)

# Post-proceso por temperatura: una GPU, su bloque de replicas, sin comunicacion entre GPUs
@pmap
def post_temperature_multi_gpu(metrics):
    series = jnp.stack(metrics[:6], axis=0)                  # (6, MCS_MEAS, batch): E, M, E_exc, E_an, E_dm, E_hex
    media, media_se, var, var_sesgo, var_se = jax.vmap(jackknife_bloques)(series)   # cada uno (6,batch)
    tau = jax.vmap(tau_int_sokal)(series)                    # (6,batch)
    # [E3] tau de las series de fluctuacion (A - Abar)^2, relevantes para Cv y Chi
    cuad = (series[:2] - jnp.mean(series[:2], axis=1, keepdims=True))**2
    tau_cuad = jax.vmap(tau_int_sokal)(cuad)                 # (2,batch)
    # [E3] error jackknife de media y varianza de E y M para cada largo de bloque (meseta)
    se_m_b, se_v_b = [], []
    for b in LARGOS_BLOQUE:
        sm, sv = jax.vmap(lambda x: jackknife_bloques_b(x, b))(series[:2])
        se_m_b.append(sm); se_v_b.append(sv)
    acc = jnp.mean(metrics[6], axis=0)                       # (batch,)
    return media, media_se, var, var_sesgo, var_se, tau, tau_cuad, jnp.stack(se_m_b), jnp.stack(se_v_b), acc

# ==========================================
# [B2] BOOTSTRAP ENTRE REPLICAS (cada GPU procesa sus estados, claves independientes)
# ==========================================
@partial(pmap, in_axes=(0, None, 0))
def bootstrap_replicas_pmap(curvas, T_arr, key):
    """curvas: (estados_por_gpu, nT, 8, R) con canales = NOMBRES_BOOT (6 medias, Cv, Chi) por replica.
    Remuestrea replicas con reemplazo (indices = floor(U*R)); la media de cada remuestra se obtiene
    con una matriz de multiplicidades W (NBOOT,R) -> un unico producto matricial en GPU."""
    R = curvas.shape[-1]
    ks = random.split(key, curvas.shape[0])
    def un_estado(args):
        c, k = args
        u = random.uniform(k, (NBOOT, R))
        idx = jnp.minimum(jnp.floor(u * R).astype(jnp.int32), R - 1)
        W = jnp.sum(jax.nn.one_hot(idx, R, dtype=jnp.float32), axis=1) / R
        boots = jnp.einsum('br,tcr->btc', W, c)              # (NBOOT,nT,8)
        est = jnp.mean(c, axis=-1)                           # (nT,8)
        bmean = jnp.mean(boots, axis=0)
        bse = jnp.std(boots, axis=0, ddof=1)
        lo = jnp.quantile(boots, 0.025, axis=0)
        hi = jnp.quantile(boots, 0.975, axis=0)
        ipk = jnp.argmax(boots[:, :, 6:8], axis=1)           # (NBOOT,2): pico de Cv y de Chi
        tpk = T_arr[ipk]
        tpk_est = T_arr[jnp.argmax(est[:, 6:8], axis=0)]
        return est, bmean, bse, lo, hi, tpk, tpk_est
    return lax.map(un_estado, (curvas, ks))

# ==========================================
# AUXILIAR: (num_gpus, S, batch) -> (NUM_ESTADOS, S, R), conserva el orden estado-mayor
# ==========================================
def _a_estados(a):
    ns = a.shape[1]
    return (a.reshape(num_gpus, ns, estados_por_gpu, REPLICAS_POR_ESTADO)
              .transpose(0, 2, 1, 3).reshape(NUM_ESTADOS, ns, REPLICAS_POR_ESTADO))

# ==========================================
# BUCLE PRINCIPAL (PYTHON / CPU)
# ==========================================
if __name__ == '__main__':
    print(f"Total Estados: {NUM_ESTADOS} | Átomos: {N_spins} | Replicas por estado: {REPLICAS_POR_ESTADO} | GPUs detectadas: {num_gpus}")
    print(f"CALCULAR_ERRORES={CALCULAR_ERRORES} | PRUEBA_UNIFORMIDAD={PRUEBA_UNIFORMIDAD} | METODO_ESFERA={METODO_ESFERA}")
    start_time = time.time()
    
    # Inicialización de Espines y Claves Aleatorias para cada GPU
    key = random.PRNGKey(42)
    key_init, key_run, key_unif = random.split(key, 3)
    keys_pmap = random.split(key_run, num_gpus)

    # [B1] Prueba de uniformidad del generador de direcciones (una muestra independiente por GPU)
    if PRUEBA_UNIFORMIDAD:
        informe_unif = prueba_uniformidad_esfera(key_unif)
        np.savez_compressed("Prueba_Uniformidad_Esfera.npz",
                            **{f"{m}_{k}": np.asarray(v) for m, dct in informe_unif.items() for k, v in dct.items()})

    # [B1] Estado inicial aleatorio: direcciones uniformes por transformada inversa
    spins_pmap = esfera_uniforme(key_init, (num_gpus, batch_por_gpu, L, 2*L2 + 1, 2*L2 + 1)) * So
    spins_pmap = spins_pmap * disco_mask_jnp # Enmascarar base
    

    # Almacenamiento RAM (Diccionarios listos para NumPy)
    datos_globales = {i: {"resultados": [], "energias": [], "evolucion": []} for i in range(NUM_ESTADOS)}
    Cv_prev = np.zeros(NUM_ESTADOS)
    Chi_prev = np.zeros(NUM_ESTADOS)

    # [B2] Almacenamiento de estadisticas por replica (solo si CALCULAR_ERRORES): nT x estados x 6 x R
    T_list = []
    claves_stats = ("media", "media_se", "var", "var_sesgo", "var_se", "tau",
                    "Cv_rep", "Cv_sesgo", "Cv_se", "Chi_rep", "Chi_sesgo", "Chi_se")
    stats_list = {k: [] for k in claves_stats + ("tau_cuad", "se_media_b", "se_var_b", "aceptacion")}
    series_guardadas = []
    prev_rep = {k: np.zeros((NUM_ESTADOS, REPLICAS_POR_ESTADO), np.float32)
                for k in ("Cv_rep", "Cv_sesgo", "Cv_se", "Chi_rep", "Chi_sesgo", "Chi_se")}

    T_current = Ti
    while T_current >= Tf:
        beta = 1.0 / (kB * T_current) if T_current > 0 else 1e9
        
        # LANZAMIENTO A TODAS LAS GPUS SIMULTANEAMENTE
        spins_pmap, keys_pmap, metrics = step_temperature_multi_gpu(
            spins_pmap, keys_pmap, beta, mask_even_jnp, mask_odd_jnp, disco_mask_jnp, 
            kDM_pmap, Jex_pmap, Kan1_pmap, Hex_pmap, gamma_pmap, Eo_pmap
        )

        # [B2] Jackknife y tau_int en GPU (series ya residentes en cada dispositivo)
        if CALCULAR_ERRORES:
            stats_gpu = post_temperature_multi_gpu(metrics)
        
        # Recuperar datos a CPU: la tupla `metrics` contiene arrays de forma (num_gpus, MCS_MEAS, batch_por_gpu)
        E_hist, M_hist, ej_hist, eka_hist, ekd_hist, eh_hist = [np.array(m) for m in metrics[:6]]
        # [E3] series completas de E y M cada T_SERIES K: (MCS_MEAS, estados, R)
        if CALCULAR_ERRORES and abs(T_current / T_SERIES - round(T_current / T_SERIES)) < 1e-6:
            def _ser(h):
                return (h.transpose(1, 0, 2).reshape(MCS_MEAS, NUM_ESTADOS, REPLICAS_POR_ESTADO).astype(np.float32))
            series_guardadas.append((T_current, _ser(E_hist), _ser(M_hist)))
        
        # Procesamiento Estadístico en CPU
        E_mean = np.mean(E_hist, axis=1).reshape(NUM_ESTADOS, REPLICAS_POR_ESTADO)
        M_mean = np.mean(M_hist, axis=1).reshape(NUM_ESTADOS, REPLICAS_POR_ESTADO)
        E_sq_mean = np.mean(E_hist**2, axis=1).reshape(NUM_ESTADOS, REPLICAS_POR_ESTADO)
        M_sq_mean = np.mean(M_hist**2, axis=1).reshape(NUM_ESTADOS, REPLICAS_POR_ESTADO)
        
        E_j_mean = np.mean(ej_hist, axis=1).reshape(NUM_ESTADOS, REPLICAS_POR_ESTADO).mean(axis=1)
        E_ka_mean = np.mean(eka_hist, axis=1).reshape(NUM_ESTADOS, REPLICAS_POR_ESTADO).mean(axis=1)
        E_kd_mean = np.mean(ekd_hist, axis=1).reshape(NUM_ESTADOS, REPLICAS_POR_ESTADO).mean(axis=1)
        E_h_mean = np.mean(eh_hist, axis=1).reshape(NUM_ESTADOS, REPLICAS_POR_ESTADO).mean(axis=1)

        E_promedio = E_mean.mean(axis=1)
        M_promedio = M_mean.mean(axis=1)
        
        if T_current > 0.1:
            Cv = np.mean((E_sq_mean - E_mean**2) / (kB * T_current**2), axis=1)
            Chi = np.mean((M_sq_mean - M_mean**2) / (kB * T_current), axis=1)
            Cv_prev, Chi_prev = Cv, Chi
        else:
            Cv, Chi = Cv_prev, Chi_prev

        # [B2] Reordenar estadisticas por estado y escalar varianzas -> Cv, Chi por replica (misma regla T>0.1)
        if CALCULAR_ERRORES:
            media_g, media_se_g, var_g, var_sesgo_g, var_se_g, tau_g, tauc_g, sem_b_g, sev_b_g, acc_g = [np.asarray(a) for a in stats_gpu]
            # [E3] reordenar: (num_gpus, nb, 2, batch) -> (estados, nb, 2, R); (num_gpus, batch) -> (estados, R)
            stats_list["tau_cuad"].append(_a_estados(tauc_g))
            stats_list["se_media_b"].append(np.stack([_a_estados(sem_b_g[:, k]) for k in range(len(LARGOS_BLOQUE))], axis=1))
            stats_list["se_var_b"].append(np.stack([_a_estados(sev_b_g[:, k]) for k in range(len(LARGOS_BLOQUE))], axis=1))
            stats_list["aceptacion"].append(_a_estados(acc_g[:, None, :])[:, 0, :])
            media_e = _a_estados(media_g); media_se_e = _a_estados(media_se_g)
            var_e = _a_estados(var_g); var_sesgo_e = _a_estados(var_sesgo_g); var_se_e = _a_estados(var_se_g)
            tau_e = _a_estados(tau_g)
            if T_current > 0.1:
                fE = 1.0 / (kB * T_current**2)
                fM = 1.0 / (kB * T_current)
                prev_rep = dict(Cv_rep=var_e[:, 0, :] * fE, Cv_sesgo=var_sesgo_e[:, 0, :] * fE, Cv_se=var_se_e[:, 0, :] * fE,
                                Chi_rep=var_e[:, 1, :] * fM, Chi_sesgo=var_sesgo_e[:, 1, :] * fM, Chi_se=var_se_e[:, 1, :] * fM)
            T_list.append(T_current)
            for k, v in (("media", media_e), ("media_se", media_se_e), ("var", var_e), ("var_sesgo", var_sesgo_e),
                         ("var_se", var_se_e), ("tau", tau_e)):
                stats_list[k].append(v)
            for k in prev_rep:
                stats_list[k].append(prev_rep[k])
            
        print(f"Completado T: {T_current:.2f} K")
        
        # Guardar resultados e imágenes reducidas de espines en RAM
        spins_cpu = np.array(spins_pmap).reshape(NUM_ESTADOS, REPLICAS_POR_ESTADO, L, 2*L2+1, 2*L2+1, 3)
        for i_estado in range(NUM_ESTADOS):
            datos_globales[i_estado]["resultados"].append([T_current, E_promedio[i_estado], Cv[i_estado], M_promedio[i_estado], Chi[i_estado]])
            datos_globales[i_estado]["energias"].append([T_current, E_ka_mean[i_estado], E_h_mean[i_estado], E_j_mean[i_estado], E_kd_mean[i_estado]])
            
            # Guardar muestra evolutiva (ej. de la réplica 0 y 1 para cada estado)
            for idx_rep in range(2):
                datos_globales[i_estado]["evolucion"].append({
                    'T': T_current, 'estado': idx_rep + 1,
                    'spins': spins_cpu[i_estado, idx_rep][disco_mask_np[0, ..., 0]] # Mantiene forma plana
                })
        
        T_current = round(T_current + dT, 2)

    # ==========================================
    # [B2] BOOTSTRAP ENTRE REPLICAS (GPU, independiente por dispositivo)
    # ==========================================
    if CALCULAR_ERRORES:
        nT = len(T_list)
        T_arr = np.array(T_list, dtype=np.float32)
        S = {k: np.stack(v, axis=0) for k, v in stats_list.items()}          # (nT, NUM_ESTADOS, ...)
        curvas = np.concatenate([S["media"], S["Cv_rep"][:, :, None, :], S["Chi_rep"][:, :, None, :]], axis=2)  # (nT,S,8,R)
        curvas = curvas.transpose(1, 0, 2, 3).astype(np.float32)             # (S,nT,8,R)
        curvas_pmap = curvas.reshape(num_gpus, estados_por_gpu, nT, len(NOMBRES_BOOT), REPLICAS_POR_ESTADO)
        keys_boot = random.split(random.PRNGKey(2024), num_gpus)
        boot = bootstrap_replicas_pmap(curvas_pmap, jnp.asarray(T_arr), keys_boot)
        boot = [np.asarray(a).reshape((NUM_ESTADOS,) + a.shape[2:]) for a in boot]
        b_est, b_mean, b_se, b_lo, b_hi, b_tpk, b_tpk_est = boot            # (S,nT,8) x5, (S,NBOOT,2), (S,2)

        # Error jackknife (intra-replica) del promedio sobre replicas: sqrt(sum se^2)/R
        Cv_jk_se_estado = np.sqrt((S["Cv_se"]**2).sum(axis=-1)) / REPLICAS_POR_ESTADO     # (nT,S)
        Chi_jk_se_estado = np.sqrt((S["Chi_se"]**2).sum(axis=-1)) / REPLICAS_POR_ESTADO

    # ==========================================
    # ESCRITURA EN DISCO AL FINALIZAR
    # ==========================================
    print("Simulación terminada. Volcando a disco (.npz)...")
    os.makedirs(SALIDA, exist_ok=True)
    for i_estado in range(NUM_ESTADOS):
        nombre = f"est{i_estado}_jex={Jex_base[i_estado]:.2f}_kdm={kDM_base[i_estado]:.2f}_kan={Kan1_base[i_estado]:.2f}_hex={Hex_base[i_estado]:.2f}_gamma={gamma_base[i_estado]:.0f}"
        
        # Archivo original: mismas claves y formato que antes
        np.savez_compressed(
            f"{SALIDA}/Datos_{nombre}.npz",
            resultadosVsT=np.array(datos_globales[i_estado]["resultados"]),
            energiasVsT=np.array(datos_globales[i_estado]["energias"]),
            evolucion=np.array(datos_globales[i_estado]["evolucion"], dtype=object)
        )

        # Archivo de errores (solo con CALCULAR_ERRORES): jackknife, tau y bootstrap
        if CALCULAR_ERRORES:
            tpk_muestras = b_tpk[i_estado]                                   # (NBOOT,2) -> [Cv, Chi]
            np.savez_compressed(
                f"{SALIDA}/Stats_{nombre}.npz",
                T=T_arr,
                nombres_series=np.array(NOMBRES_SERIES),
                nombres_boot=np.array(NOMBRES_BOOT),
                n_bloques_jk=N_BLOQUES_JK, largo_bloque=MCS_MEAS // N_BLOQUES_JK, nboot=NBOOT,
                media_rep=S["media"][:, i_estado], media_se_rep=S["media_se"][:, i_estado],
                var_rep=S["var"][:, i_estado], var_sesgo_rep=S["var_sesgo"][:, i_estado], var_se_rep=S["var_se"][:, i_estado],
                tau_rep=S["tau"][:, i_estado],
                Cv_rep=S["Cv_rep"][:, i_estado], Cv_sesgo_rep=S["Cv_sesgo"][:, i_estado], Cv_se_rep=S["Cv_se"][:, i_estado],
                Chi_rep=S["Chi_rep"][:, i_estado], Chi_sesgo_rep=S["Chi_sesgo"][:, i_estado], Chi_se_rep=S["Chi_se"][:, i_estado],
                Cv_jk_se_estado=Cv_jk_se_estado[:, i_estado], Chi_jk_se_estado=Chi_jk_se_estado[:, i_estado],
                boot_est=b_est[i_estado], boot_media=b_mean[i_estado], boot_se=b_se[i_estado],
                boot_ic_lo=b_lo[i_estado], boot_ic_hi=b_hi[i_estado],
                boot_Tpico_muestras=tpk_muestras, boot_Tpico_est=b_tpk_est[i_estado],
                boot_Tpico_se=np.std(tpk_muestras, axis=0, ddof=1),
                boot_Tpico_ic=np.percentile(tpk_muestras, [2.5, 97.5], axis=0),
                tau_cuad_rep=S["tau_cuad"][:, i_estado], se_media_b_rep=S["se_media_b"][:, i_estado],
                se_var_b_rep=S["se_var_b"][:, i_estado], aceptacion_rep=S["aceptacion"][:, i_estado],
                largos_bloque=np.array(LARGOS_BLOQUE), mcs_meas=MCS_MEAS, n_spins=N_spins, kB=kB,
                T_series=np.array([s[0] for s in series_guardadas]),
                E_series=np.stack([s[1][:, i_estado] for s in series_guardadas]),
                M_series=np.stack([s[2][:, i_estado] for s in series_guardadas]),
                Jex=Jex_base[i_estado], kDM=kDM_base[i_estado], Kan=Kan1_base[i_estado],
                Hex=Hex_base[i_estado], gamma=gamma_base[i_estado]
            )
        
    print(f"Pipeline multi-GPU completado. Tiempo total: {(time.time() - start_time)/3600:.2f} hrs.")
