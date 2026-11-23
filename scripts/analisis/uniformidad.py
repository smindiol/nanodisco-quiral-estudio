"""Chi-square uniformity test of the inverse-transform sphere generator (z = 2U1 - 1, phi = 2 pi U2).

For each seed: n directions, 8 x 8 equiprobable cells on [-1,1] x [0, 2pi), X2 = sum (O-E)^2/E with 63 df,
plus the marginal tests of z and phi with 64 cells (63 df). Under H0 the number of rejections at
alpha = 0.05 among R seeds is Binomial(R, 0.05) and the p-values are U(0,1).
Usage: python -I uniformidad.py --salida <json>
"""
import argparse, json
import numpy as np
from scipy.special import gammaincc  # survival function of the chi-square (regularized gamma), check only

R, N, K, ALFA = 100, 4_000_000, 8, 0.05

def pval(x2, gl):
    return float(gammaincc(gl / 2.0, x2 / 2.0))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--salida", required=True); a = ap.parse_args()
    p_conj, p_z, p_phi, seg = [], [], [], []
    for s in range(R):
        rng = np.random.default_rng(1000 + s)
        u1, u2 = rng.random(N), rng.random(N)
        z, phi = 2 * u1 - 1, 2 * np.pi * u2
        iz = np.minimum((u1 * K).astype(int), K - 1); ip = np.minimum((u2 * K).astype(int), K - 1)
        O = np.bincount(iz * K + ip, minlength=K * K); E = N / K**2
        p_conj.append(pval(((O - E) ** 2 / E).sum(), K * K - 1))
        for v, lst in ((u1, p_z), (u2, p_phi)):
            O1 = np.bincount(np.minimum((v * 64).astype(int), 63), minlength=64); E1 = N / 64
            lst.append(pval(((O1 - E1) ** 2 / E1).sum(), 63))
        st = np.sqrt(1 - z ** 2)
        x, y = st * np.cos(phi), st * np.sin(phi)
        seg.append([float((x ** 2).mean()), float((y ** 2).mean()), float((z ** 2).mean())])
    def resumen(p):
        p = np.sort(np.array(p)); k = int((p < ALFA).sum())
        from scipy.stats import binom
        ks = float(np.max(np.maximum(np.arange(1, R + 1) / R - p, p - np.arange(R) / R)))
        return dict(rechazos=k, p_binomial=float(binom.sf(k - 1, R, ALFA)), ks=ks)
    out = dict(R=R, n=N, celdas=K * K, gl=K * K - 1, alfa=ALFA, conjunta=resumen(p_conj),
               marginal_z=resumen(p_z), marginal_phi=resumen(p_phi),
               segundo_momento=np.mean(seg, axis=0).tolist())
    json.dump(out, open(a.salida, "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
