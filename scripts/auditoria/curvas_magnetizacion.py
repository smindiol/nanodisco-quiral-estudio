"""Describe todos los estados y temperaturas sin suponer equilibrio."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from revisar_datos import rawpath, AUDITORIA

def one(task):
    e,t=task
    with np.load(rawpath(e,t)) as d:
        m=d['M'].astype(float)
        means=m.mean(0)
        within=float(m.var(0).mean()); between=float(means.var())
        return dict(estado=e,T=t,m_media=float(m.mean()),m_se_caminos=float(means.std(ddof=1)/10),
                    chi_protocolo=float(5425*(within+between)/(.086173404*t)),
                    fraccion_var_entre=between/(within+between),
                    m_p10=float(np.quantile(means,.1)),m_p90=float(np.quantile(means,.9)))

def main():
    AUDITORIA.mkdir(parents=True, exist_ok=True)
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,row in enumerate(pool.map(one,[(e,round(t,1)) for e in range(1,9) for t in np.arange(20,5.95,-.1)])):
            rows.append(row)
            if (i+1)%141==0: print(f"Magnetización: estado {row['estado']}, 141 temperaturas",flush=True)
    (AUDITORIA/'magnetizacion.json').write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False))

if __name__ == '__main__':
    main()
