"""Auditoría local acotada: momentos energéticos, regresión agrupada y texturas.
No modifica las simulaciones ni vuelve a ejecutar el bootstrap original.
"""
from pathlib import Path
import sys, json, itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.stats import rankdata, norm, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuracion import ROOT, RESULTADOS, AUDITORIA, BUILD, archivo_crudo


def rh(x):
    n, r = x.shape
    w = x.var(axis=0, ddof=1).mean()
    return float(np.sqrt(((n-1)/n*w+x.mean(0).var(ddof=1))/w)) if w else 1.

def rank_rh(x):
    n = len(x)//2
    x = np.concatenate([x[:n], x[-n:]], axis=1)
    def z(a):
        return norm.ppf((rankdata(a).reshape(a.shape)-.375)/(a.size+.25))
    return max(rh(z(x)), rh(z(np.abs(x-np.median(x)))))

def rawpath(e,t):
    return archivo_crudo(e, t)

TSTAR=[14.35,14.14,14.34,14.15,14.55,14.45,14.35,13.64]
SNAP=sorted(set([float(t) for t in range(6,21)]+[14.3]),reverse=True)

def audit_one(task):
    e,t = task
    with np.load(rawpath(e,t)) as d:
        h = d['E'].astype(float)*float(d['Eo'])
        m = d['M'].astype(float)
        y = (h-h.mean())**2
        return dict(estado=e,T=t,rhat_E=rank_rh(h),rhat_E2_centrado=rank_rh(y),
                    rhat_m=rank_rh(m),media_E=float(h.mean()),
                    c=float(h.var()/(5425*.086173404*t*t)))

def snapshot(task):
    e,t=task
    with np.load(rawpath(e,t)) as d:
        mask=d['mascara'].astype(bool)
        s=d['spins'][:3]
        grid=np.zeros((3,)+mask.shape+(3,))
        grid[:,mask,:]=s
        # Capa central, tres caminos fijos, no selección por aspecto.
        return dict(estado=e,T=t,sz=np.round(grid[:,2,:,:,2],3).tolist(),
                    mascara=mask[2].astype(int).tolist(),
                    m_instantanea=np.linalg.norm(s.sum(1),axis=1).round(5).tolist())

def main():
    AUDITORIA.mkdir(parents=True, exist_ok=True)
    tasks=[]
    for e,ts in enumerate(TSTAR,1):
        # Dos temperaturas que encierran T* y sus vecinas; evita redondeo bancario.
        lower=np.floor(ts*10+1e-8)/10
        for t in sorted(set([6.,10.,20.]+[round(lower+d,1) for d in [-.1,0,.1,.2]])):
            tasks.append((e,t))
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(audit_one,tasks):
            rows.append(row)
            print(f"Momentos: estado {row['estado']}, T={row['T']}, R E²={row['rhat_E2_centrado']:.4f}",flush=True)
    allstates=[]
    for e in range(1,9):
        with np.load(RESULTADOS/'pasos_mc'/f'estado_{e}.npz') as d:
            entry={k: d[k].tolist() for k in ['T','rhat_E','rhat_M','tau_E','tau_M','E_med','curva_M','N_GRID']}
            entry.update(estado=e,kDM=float(d['kDM'][0]),H=float(d['Hex'][0]))
            Eo=d['Eo']; T=d['T']; n=10000; r=100
            v=((n-1)/n*d['W_E']+(r-1)/r*d['Bn_E'])*Eo**2
            entry['c_recalculado']=(v/(5425*.086173404*T*T)).tolist()
            entry['energia_meV_por_espin']=(d['E_med']*Eo/5425).tolist()
            allstates.append(entry)
    X=np.array([[1,s['kDM'],s['H'],1/t] for s in allstates for t in s['T']])
    y=np.log(np.array([v for s in allstates for v in s['tau_M']]))
    groups=np.repeat(np.arange(8),141)
    bread=np.linalg.inv(X.T@X); coef=bread@X.T@y; residual=y-X@coef
    meat=np.zeros((4,4))
    for g in range(8):
        score=X[groups==g].T@residual[groups==g]; meat+=np.outer(score,score)
    cr1=8/7*(len(y)-1)/(len(y)-4)*bread@meat@bread
    hc0=bread@((X*residual[:,None]).T@(X*residual[:,None]))@bread
    loo=[]
    for g in range(8):
        keep=groups!=g; loo.append(np.linalg.lstsq(X[keep],y[keep],rcond=None)[0].tolist())
    kd=np.array([s['kDM'] for s in allstates]); ts=np.array(TSTAR)
    # Permutación bilateral, empates conservados.
    ranks=rankdata(kd); ranks-=ranks.mean(); v=rankdata(ts); v-=v.mean()
    rho=float(ranks@v/np.linalg.norm(ranks)/np.linalg.norm(v))
    perm=np.array([ranks@np.array(p)/np.linalg.norm(ranks)/np.linalg.norm(v) for p in itertools.permutations(v)])
    p=float(np.mean(np.abs(perm)>=abs(rho)-1e-12))
    reg=dict(coef=coef.tolist(),se_HC0=np.sqrt(np.diag(hc0)).tolist(),
             se_cluster_CR1=np.sqrt(np.diag(cr1)).tolist(),leave_one_state_out=loo,
             n_observaciones=len(y),n_grupos=8)
    snaps=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,s in enumerate(pool.map(snapshot,[(e,t) for e in range(1,9) for t in SNAP])):
            snaps.append(s)
            if i%16==15: print(f'Texturas: estado {s["estado"]} completo',flush=True)
    out=dict(metodo='Auditoría local: 56 puntos de momentos, no toda la escalera. Curvas agregadas originales 141 temperaturas. Texturas de 3 copias fijas en 16 temperaturas.',
             momentos=rows,regresion=reg,spearman_Tstar=dict(rho=rho,p_exacto=p),curvas=allstates,
             temperaturas_texturas=SNAP,texturas=snaps)
    (AUDITORIA/'auditoria.json').write_text(json.dumps(out,ensure_ascii=False,allow_nan=False))
    print('AUDITORÍA GUARDADA',reg,flush=True)

if __name__=='__main__': main()
