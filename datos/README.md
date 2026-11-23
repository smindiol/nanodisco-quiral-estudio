# Datos externos y diccionario de archivos

La base cruda queda fuera de Git. Su incorporación desde Kaggle está pendiente. Este repositorio no incluye un dataset remoto ficticio ni publica los archivos pesados.

El código usa `NANODISCO_DATOS` como raíz de búsqueda. Sin esa variable busca en `datos/crudos/`, carpeta excluida mediante `.gitignore`. Un archivo corresponde a un estado y una temperatura:

```text
lote1/
  lote_01/
    estado_1/
      T_20.0.npz
      T_19.9.npz
      ...
    estado_2/
      ...
```

También se admite empezar directamente en `lote_01/` o en `estado_1/`, siempre que los directorios de los estados cuelguen de la raíz configurada. La escalera de producción contiene 141 temperaturas entre 20.0 y 6.0 K.

| Clave | Forma o tipo | Contenido |
|---|---|---|
| `E` | 10000 × 100 | Energía total normalizada por Eo, por barrido y camino |
| `M` | 10000 × 100 | Norma de la magnetización por espín |
| `E_exc`, `E_an`, `E_dm`, `E_hex` | 10000 × 100 | Componentes de energía, también normalizadas |
| `spins` | 100 × 5425 × 3 | Configuración final de cada camino a esa temperatura |
| `mascara` | 5 × 39 × 39 | Sitios que pertenecen al disco |
| `T`, `Jex`, `kDM`, `Kan1`, `Hex`, `gamma` | Escalares | Temperatura y parámetros físicos |
| `Rd`, `So`, `kB`, `Eo` | Escalares | Geometría, norma del espín y unidades |
| `term`, `med`, `semilla`, `keys`, `formato` | Metadatos | Barridos y datos de continuación |

La energía física se recupera mediante `E_fisica = E * Eo`. Su varianza requiere el factor Eo². No se puede comparar la magnitud de energías normalizadas de protocolos diferentes sin revisar esa constante.

Para reconstruir la configuración de la primera copia:

```python
import numpy as np
with np.load("T_20.0.npz", allow_pickle=False) as d:
    mask = d["mascara"].astype(bool)
    grid = np.zeros((*mask.shape, 3), dtype=d["spins"].dtype)
    grid[mask] = d["spins"][0]
```

Los archivos de `resultados/pasos_mc/` son derivados pequeños de diagnósticos, trazas seleccionadas y medias. `resultados/auditoria/auditoria.json` contiene los 56 puntos revisados, coeficientes, sensibilidad, curvas y texturas seleccionadas. `magnetizacion.json` contiene 1128 resúmenes por estado y temperatura. Ninguno equivale a la base temporal completa.
