# Análisis estadístico de la simulación secuencial (estados 1–8)

Bootstrap entre copias (la copia completa es la unidad; mismas multiplicidades en toda la escalera), B = 50000, α = 0.05, repesado de histograma simple a una rejilla de 0.01 K. Figuras en `figuras/`, tablas en `tablas/`.

## 1. Temperatura del pico del calor específico

| Estado | kDM | Hex | Kan1 | T* de c (K) | Mediana bootstrap | se | IC 95 % percentil | IC 95 % BCa | Meseta del pico | c en el pico (por espín) | T* de χ (K) | B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.05 | 0 | 0.05 | 14.35 | 14.24 | 0.117 | [14.00, 14.37] | [14.14, 14.45] | [14.35, 14.40] | 0.17 ± 0.00073 | 6.75 | 50000 |
| 2 | 0.15 | 0 | 0.05 | 14.14 | 14.14 | 0.068 | [14.04, 14.35] | [13.95, 14.14] | [14.11, 14.14] | 0.17 ± 0.00089 | 6.45 | 50000 |
| 3 | 0.33 | 0 | 0.05 | 14.34 | 14.34 | 0.164 | [14.04, 14.65] | [13.85, 14.55] | [14.25, 14.34] | 0.163 ± 0.00059 | 6.45 | 50000 |
| 4 | 0.6 | 0 | 0.05 | 14.15 | 14.15 | 0.127 | [14.04, 14.55] | [13.64, 14.15] | [14.15, 14.24] | 0.155 ± 0.0006 | 6.54 | 50000 |
| 5 | 1 | 0 | 0.05 | 14.55 | 14.35 | 0.445 | [13.14, 14.55] | [13.55, 14.65] | [14.55, 14.58] | 0.151 ± 0.00056 | 8.35 | 50000 |
| 6 | 0.3 | 0.05 | 0.05 | 14.45 | 14.35 | 0.166 | [14.05, 14.65] | [14.05, 14.65] | [14.45, 14.49] | 0.158 ± 0.00052 | 6.04 | 50000 |
| 7 | 0.6 | 0.1 | 0.05 | 14.35 | 14.35 | 0.201 | [13.87, 14.54] | [13.94, 14.55] | [14.35, 14.44] | 0.155 ± 0.00051 | 6.14 | 50000 |
| 8 | 0.75 | 0.3 | 0.05 | 13.64 | 13.64 | 0.127 | [13.25, 13.74] | [13.14, 13.64] | [13.60, 13.64] | 0.143 ± 0.00051 | 6.04 | 50000 |

## 2. Convergencia, representatividad y confiabilidad

Cada temperatura se evalúa por separado. **Converge** en una T si se cumplen las tres condiciones: R̂ < 1.01 (H y m), sin exceso de rechazos de Geweke sobre el 5 % nominal (binomial, Holm sobre las T) y mediana de MSER-5 ≤ 500 barridos. Es **representativa** si además las 100 copias son homogéneas (χ² de sus medias, Holm). Es **confiable** si el ESS supera 400 y el jackknife por bloques tiene al menos 20 bloques de tamaño ≥ 10 τ_int. Un estado convergió solo si cumple en las 141 temperaturas; si no, se informa el porcentaje de temperaturas que cumplen y el tramo continuo desde 20 K hasta donde se cumple.

| Estado | T con R̂<1.01 | T sin deriva Geweke (Holm) | T con MSER ≤ 500 barridos | τ_int máx (barridos) | ESS mín | T con copias homogéneas (Holm) | T con se_int/sd_ext en [0.8,1.25] | T con jackknife confiable | ESS mín del repesado | Acuerdo percentil/BCa de c | ¿Convergió? (% de T, tramo continuo desde 20 K) | ¿Representativo? | ¿Confiable? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 30% (máx 1.992 en 6.1 K) | 34% | 89% (máx 1068) | 461 en 7.3 K | 1315 | 34% | 100% | 30% | 0.65 | 100% | **no**: 30% de las T (20.0–15.9 K) | **no**: 29% de las T (20.0–16.1 K) | **no**: 30% de las T (20.0–15.9 K) |
| 2 | 31% (máx 1.919 en 6.5 K) | 40% | 84% (máx 1820) | 453 en 6.0 K | 1314 | 35% | 100% | 31% | 0.66 | 100% | **no**: 31% de las T (20.0–15.7 K) | **no**: 31% de las T (20.0–15.7 K) | **no**: 31% de las T (20.0–15.7 K) |
| 3 | 38% (máx 1.221 en 6.5 K) | 70% | 100% (máx 132) | 283 en 6.7 K | 2160 | 43% | 100% | 38% | 0.66 | 100% | **no**: 38% de las T (20.0–14.7 K) | **no**: 38% de las T (20.0–15.1 K) | **no**: 38% de las T (20.0–14.7 K) |
| 4 | 55% (máx 1.037 en 6.5 K) | 96% | 100% (máx 0) | 90 en 6.2 K | 6347 | 47% | 99% | 72% | 0.66 | 100% | **no**: 54% de las T (20.0–13.8 K) | **no**: 46% de las T (20.0–14.5 K) | **no**: 72% de las T (20.0–11.4 K) |
| 5 | 56% (máx 1.034 en 6.0 K) | 91% | 100% (máx 0) | 44 en 6.2 K | 12036 | 38% | 99% | 100% | 0.66 | 100% | **no**: 55% de las T (20.0–13.7 K) | **no**: 38% de las T (20.0–17.5 K) | sí: 100% de las T (20.0–6.0 K) |
| 6 | 30% (máx 1.634 en 6.2 K) | 52% | 100% (máx 65) | 294 en 6.0 K | 2122 | 38% | 99% | 32% | 0.66 | 100% | **no**: 30% de las T (20.0–15.8 K) | **no**: 28% de las T (20.0–18.0 K) | **no**: 32% de las T (20.0–15.6 K) |
| 7 | 38% (máx 1.186 en 6.1 K) | 91% | 100% (máx 15) | 187 en 6.3 K | 3169 | 39% | 99% | 40% | 0.66 | 100% | **no**: 38% de las T (20.0–15.1 K) | **no**: 36% de las T (20.0–17.9 K) | **no**: 40% de las T (20.0–14.5 K) |
| 8 | 40% (máx 1.442 en 6.0 K) | 96% | 100% (máx 5) | 151 en 11.6 K | 3797 | 48% | 100% | 41% | 0.66 | 100% | **no**: 40% de las T (20.0–14.7 K) | **no**: 40% de las T (20.0–14.7 K) | **no**: 41% de las T (20.0–14.3 K) |

## 3. Estados medidos: configuraciones finales

| Estado | T final (K) | Q modal (fracción de copias) | Q medio [IC 95 %] | q medio / \|q\| medio entre copias | Grupos de S(k) (fracción del mayor) | Período λ (sitios) | Quiralidad media | Q modal en T* |
|---|---|---|---|---|---|---|---|---|
| 1 | 6.0 | 0 (100%) | 0.01 [-0.01, 0.02] | -0.005 / 0.519 | 1 (100%) | 30.2 | 0.021 | 0 (35%) en 14.3 K |
| 2 | 6.0 | 0 (100%) | 0.00 [-0.03, 0.03] | 0.012 / 0.254 | 1 (100%) | 30.2 | 0.064 | 0 (27%) en 14.1 K |
| 3 | 6.0 | 0 (90%) | 0.02 [-0.04, 0.08] | 0.005 / 0.166 | 1 (100%) | 18.1 | 0.135 | 1 (23%) en 14.3 K |
| 4 | 6.0 | 0 (92%) | -0.00 [-0.06, 0.06] | -0.004 / 0.199 | 1 (100%) | 10.1 | 0.228 | 0 (30%) en 14.2 K |
| 5 | 6.0 | 0 (95%) | 0.02 [-0.03, 0.07] | 0.005 / 0.222 | 1 (100%) | 7.0 | 0.340 | -1 (24%) en 14.6 K |
| 6 | 6.0 | -1 (78%) | -0.78 [-0.86, -0.70] | 0.613 / 0.613 | 2 (78%) | 18.1 | 0.079 | -1 (27%) en 14.4 K |
| 7 | 6.0 | -4 (85%) | -4.25 [-4.31, -4.19] | 0.496 / 0.496 | 1 (100%) | 12.9 | 0.182 | -4 (33%) en 14.3 K |
| 8 | 6.0 | -6 (50%) | -6.43 [-6.56, -6.30] | 0.513 / 0.513 | 2 (89%) | 10.1 | 0.163 | -5 (33%) en 13.6 K |

## 4. ¿Difieren los T* entre estados?

Estados independientes: la diferencia bootstrap es T*ᵇ(a) − T*ᵇ(b) con réplicas independientes; p bilateral del bootstrap, corregido por Holm sobre todos los pares.

| Par | ΔT* (K) | IC 95 % | p (Holm) | ¿Difieren? |
|---|---|---|---|---|
| 1 − 2 | +0.21 | [-0.20, +0.29] | 1 | no |
| 1 − 3 | +0.01 | [-0.50, +0.31] | 1 | no |
| 1 − 4 | +0.20 | [-0.40, +0.31] | 1 | no |
| 1 − 5 | -0.20 | [-0.51, +1.10] | 1 | no |
| 1 − 6 | -0.10 | [-0.50, +0.30] | 1 | no |
| 1 − 7 | +0.00 | [-0.40, +0.41] | 1 | no |
| 1 − 8 | +0.71 | [+0.31, +1.04] | 0.018 | sí |
| 2 − 3 | -0.20 | [-0.51, +0.20] | 1 | no |
| 2 − 4 | -0.01 | [-0.41, +0.21] | 1 | no |
| 2 − 5 | -0.41 | [-0.43, +1.08] | 1 | no |
| 2 − 6 | -0.31 | [-0.51, +0.19] | 1 | no |
| 2 − 7 | -0.21 | [-0.40, +0.30] | 1 | no |
| 2 − 8 | +0.50 | [+0.30, +0.90] | 0.00972 | sí |
| 3 − 4 | +0.19 | [-0.30, +0.50] | 1 | no |
| 3 − 5 | -0.21 | [-0.51, +1.20] | 1 | no |
| 3 − 6 | -0.11 | [-0.41, +0.49] | 1 | no |
| 3 − 7 | -0.01 | [-0.40, +0.60] | 1 | no |
| 3 − 8 | +0.70 | [+0.37, +1.19] | 0.0114 | sí |
| 4 − 5 | -0.40 | [-0.51, +1.10] | 1 | no |
| 4 − 6 | -0.30 | [-0.50, +0.30] | 1 | no |
| 4 − 7 | -0.20 | [-0.39, +0.40] | 1 | no |
| 4 − 8 | +0.51 | [+0.29, +1.01] | 0.0365 | sí |
| 5 − 6 | +0.10 | [-1.21, +0.50] | 1 | no |
| 5 − 7 | +0.20 | [-1.10, +0.61] | 1 | no |
| 5 − 8 | +0.91 | [-0.50, +1.22] | 1 | no |
| 6 − 7 | +0.10 | [-0.39, +0.60] | 1 | no |
| 6 − 8 | +0.81 | [+0.40, +1.20] | 0.00224 | sí |
| 7 − 8 | +0.71 | [+0.21, +1.10] | 0.168 | no |

## 5. Veredicto por estado

- **Estado 1: kDM=0.05, Hex=0**: no convergió en toda la escalera: solo en el 30% de las temperaturas (sin interrupción de 20.0 K a 15.9 K); no representativo en toda la escalera: solo en el 29% de las temperaturas (sin interrupción de 20.0 K a 16.1 K); no confiable en toda la escalera: solo en el 30% de las temperaturas (sin interrupción de 20.0 K a 15.9 K). τ_int máximo 461 barridos en 7.3 K; MSER máximo 1068 barridos; a la T más baja, 1 tipo(s) de estado final, el mayor con 100% de las copias.
- **Estado 2: kDM=0.15, Hex=0**: no convergió en toda la escalera: solo en el 31% de las temperaturas (sin interrupción de 20.0 K a 15.7 K); no representativo en toda la escalera: solo en el 31% de las temperaturas (sin interrupción de 20.0 K a 15.7 K); no confiable en toda la escalera: solo en el 31% de las temperaturas (sin interrupción de 20.0 K a 15.7 K). τ_int máximo 453 barridos en 6.0 K; MSER máximo 1820 barridos; a la T más baja, 1 tipo(s) de estado final, el mayor con 100% de las copias.
- **Estado 3: kDM=0.33, Hex=0**: no convergió en toda la escalera: solo en el 38% de las temperaturas (sin interrupción de 20.0 K a 14.7 K); no representativo en toda la escalera: solo en el 38% de las temperaturas (sin interrupción de 20.0 K a 15.1 K); no confiable en toda la escalera: solo en el 38% de las temperaturas (sin interrupción de 20.0 K a 14.7 K). τ_int máximo 283 barridos en 6.7 K; MSER máximo 132 barridos; a la T más baja, 1 tipo(s) de estado final, el mayor con 100% de las copias.
- **Estado 4: kDM=0.6, Hex=0**: no convergió en toda la escalera: solo en el 54% de las temperaturas (sin interrupción de 20.0 K a 13.8 K); no representativo en toda la escalera: solo en el 46% de las temperaturas (sin interrupción de 20.0 K a 14.5 K); no confiable en toda la escalera: solo en el 72% de las temperaturas (sin interrupción de 20.0 K a 11.4 K). τ_int máximo 90 barridos en 6.2 K; MSER máximo 0 barridos; a la T más baja, 1 tipo(s) de estado final, el mayor con 100% de las copias.
- **Estado 5: kDM=1, Hex=0**: no convergió en toda la escalera: solo en el 55% de las temperaturas (sin interrupción de 20.0 K a 13.7 K); no representativo en toda la escalera: solo en el 38% de las temperaturas (sin interrupción de 20.0 K a 17.5 K); confiable en toda la escalera. τ_int máximo 44 barridos en 6.2 K; MSER máximo 0 barridos; a la T más baja, 1 tipo(s) de estado final, el mayor con 100% de las copias.
- **Estado 6: kDM=0.3, Hex=0.05**: no convergió en toda la escalera: solo en el 30% de las temperaturas (sin interrupción de 20.0 K a 15.8 K); no representativo en toda la escalera: solo en el 28% de las temperaturas (sin interrupción de 20.0 K a 18.0 K); no confiable en toda la escalera: solo en el 32% de las temperaturas (sin interrupción de 20.0 K a 15.6 K). τ_int máximo 294 barridos en 6.0 K; MSER máximo 65 barridos; a la T más baja, 2 tipo(s) de estado final, el mayor con 78% de las copias.
- **Estado 7: kDM=0.6, Hex=0.1**: no convergió en toda la escalera: solo en el 38% de las temperaturas (sin interrupción de 20.0 K a 15.1 K); no representativo en toda la escalera: solo en el 36% de las temperaturas (sin interrupción de 20.0 K a 17.9 K); no confiable en toda la escalera: solo en el 40% de las temperaturas (sin interrupción de 20.0 K a 14.5 K). τ_int máximo 187 barridos en 6.3 K; MSER máximo 15 barridos; a la T más baja, 1 tipo(s) de estado final, el mayor con 100% de las copias.
- **Estado 8: kDM=0.75, Hex=0.3**: no convergió en toda la escalera: solo en el 40% de las temperaturas (sin interrupción de 20.0 K a 14.7 K); no representativo en toda la escalera: solo en el 40% de las temperaturas (sin interrupción de 20.0 K a 14.7 K); no confiable en toda la escalera: solo en el 41% de las temperaturas (sin interrupción de 20.0 K a 14.3 K). τ_int máximo 151 barridos en 11.6 K; MSER máximo 5 barridos; a la T más baja, 2 tipo(s) de estado final, el mayor con 89% de las copias.
