"""Rutas portables. NANODISCO_DATOS puede apuntar al futuro dataset externo."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
DATOS = Path(os.environ.get('NANODISCO_DATOS', str(ROOT / 'datos/crudos'))).expanduser()
RESULTADOS = ROOT / 'resultados'
AUDITORIA = RESULTADOS / 'auditoria'
BUILD = ROOT / 'build'

def archivo_crudo(estado, temperatura):
    lote = (estado + 1) // 2
    filename = f'T_{temperatura:.1f}.npz'
    candidates = [DATOS / f'lote{lote}' / f'lote_{lote:02d}' / f'estado_{estado}' / filename,
                  DATOS / f'lote_{lote:02d}' / f'estado_{estado}' / filename,
                  DATOS / f'estado_{estado}' / filename]
    for p in candidates:
        if p.is_file(): return p
    raise FileNotFoundError(f'No se encontró estado {estado}, T={temperatura:.1f} K en {DATOS}. '
                            'La base cruda no está incluida; configure NANODISCO_DATOS al disponer de ella.')
