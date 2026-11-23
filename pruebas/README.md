# Pruebas ejecutadas y comprobaciones reproducibles

Las pruebas estadísticas del estudio y las verificaciones del repositorio responden preguntas distintas. Las primeras evalúan el generador, los estimadores o la mezcla. Las segundas comprueban que los archivos compartidos están íntegros y que las visualizaciones utilizan datos disponibles.

| Prueba | Código | Datos y evidencia conservada | Procedencia |
|---|---|---|---|
| Uniformidad de la esfera | `scripts/analisis/uniformidad.py` | `resultados/informe/uniformidad.json` | Análisis original |
| Autocorrelación, jackknife, Geweke, MSER y R-hat por rangos | `scripts/analisis/dmi_analisis.py` | `resultados/tablas/convergencia.md` | Análisis original |
| Bootstrap, repesado e intervalos T* | Mismo módulo y `ejecutar.py` | `resultados/tablas/T_estrella.md` | Análisis original, 50000 réplicas |
| Comparaciones de T* con Holm | `scripts/analisis/informe.py` | `resultados/tablas/diferencias_T_estrella.md` | Análisis original |
| R-hat según longitud y variable | `scripts/analisis/pasos_mc.py` | Ocho NPZ en `resultados/pasos_mc/` | Análisis original |
| Correlaciones, regresión y costo | `scripts/analisis/figuras_informe.py` | JSON y tablas de `resultados/informe/` | Análisis original |
| Mezclas gaussianas y BIC | `scripts/analisis/mezcla_em.py` | `resultados/informe/mezcla_em.json` | Análisis original |
| Carga y factor de estructura | `scripts/analisis/dmi_analisis.py` | `resultados/tablas/estados_finales.md` | Análisis original |
| Momentos energéticos en 56 puntos | `scripts/auditoria/revisar_datos.py` | `resultados/auditoria/auditoria.json` | Revisión adicional |
| Regresión agrupada y retirada de estados | Mismo script | Mismo JSON, apartado `regresion` | Revisión adicional |
| Spearman exacto kDM–T* | Mismo script | Mismo JSON, apartado `spearman_Tstar` | Revisión adicional |
| Magnetización y varianza entre caminos | `scripts/auditoria/curvas_magnetizacion.py` | `resultados/auditoria/magnetizacion.json` | Revisión adicional, 1128 archivos |
| Integridad del HTML | `pruebas/validar_html.py` | Registro de validación | Verificación de la entrega |
| Controles del HTML | `pruebas/validar_controles.cjs` | Registro de validación | DOM simulado; sin validación visual de navegador |

La revisión adicional no volvió a ejecutar las simulaciones ni el bootstrap completo. Las comprobaciones del repositorio tampoco lo hacen. La auditoría de momentos cubre 56 puntos; las curvas de magnetización cubren las 141 temperaturas de cada estado.

Los cuadernos de `cuadernos/` conservan verificaciones de salidas y pruebas de humo. La prueba corta del primer lote dispone además de tablas y figuras en `resultados/prueba_salida_lote01/`. Una prueba corta verifica ejecución y formato; no reemplaza la producción ni demuestra mezcla de equilibrio.

`test_metodos.py` comprueba propiedades que ayudan a detectar errores de normalización o diagnóstico: invariancia de R-hat ante un cambio afín, detección de cadenas desplazadas, descomposición de varianza y reconstrucción de las curvas energéticas a partir de los momentos guardados. Usa datos sintéticos o derivados incluidos; no necesita la base externa.

Los registros históricos de la revisión están en `registros/`. Los registros de las comprobaciones del repositorio limpio se generan allí al prepararlo y en la salida de la integración continua. La procedencia de los archivos y el entorno observado se documentan por separado.
