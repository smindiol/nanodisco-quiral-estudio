# Presentación de resultados

[Abrir la presentación completa](presentacion.pdf).

La exposición tiene 22 diapositivas principales y seis de respaldo. El recorrido parte de una familia de nanodiscos parametrizada por radio y número de capas, define el objetivo de estimación de observables y fija el caso de 5425 espines. La convergencia se estudia como parte de la evaluación numérica de Metropolis para ese sistema. Después se presentan el protocolo, los resultados y las herramientas del curso. El anexo conserva fórmulas y tablas para la discusión.

Se utiliza la plantilla Beamer Oxygen y el encabezado institucional de `presentacion_formato`, adaptados al contenido del nanodisco. Los archivos necesarios están incluidos; la compilación no depende de aquella carpeta.

## Reproducir las figuras y el PDF

Desde la raíz del repositorio:

```bash
make presentacion
```

Se necesitan las dependencias de `requirements-analisis.txt` y Tectonic con sus recursos disponibles o una distribución TeX con pdfLaTeX y Beamer. El script intenta Tectonic sin descargas y usa pdfLaTeX si no hay un bundle local. Los registros quedan en `build/presentacion/`.

La fuente es `presentacion.tex`. Las notas incluidas mediante `\note{...}` orientan la explicación oral y no se muestran en las diapositivas normales.

## Procedencia de las figuras

`generar_figuras.py` usa los datos de `resultados/` para reconstruir las curvas y los diagnósticos. No ejecuta nuevas cadenas ni un nuevo bootstrap. Los intervalos pertenecen al análisis original de 50000 réplicas por caminos. La figura de distribuciones bootstrap del anexo se toma del informe final.

Las texturas de enfriamiento muestran la misma copia en cada fila y una escala común de la componente Sz. El panel de ocho estados muestra la copia 1 a 6 K. Son ejemplos de configuraciones, no una estimación visual de sus frecuencias.

Para ilustrar las dos poblaciones del estado 6, se extrajeron las capas centrales de las copias 63 y 44 a 6 K, seleccionadas por sus medias de magnetización próximas a los centros EM. Este pequeño conjunto derivado está en `datos/ejemplos_estado6.json`. Las etiquetas indican medias temporales; los mapas son configuraciones finales. Su selección es ilustrativa, no aleatoria.

`datos/procedencia_figuras.json` resume estas decisiones. La presentación se puede regenerar con los derivados incluidos sin descargar la base pesada.

## Criterios de interpretación

La temperatura T* es característica del disco finito; no demuestra una singularidad de Curie. Los diagnósticos favorables de energía y fluctuación cuadrática refuerzan los momentos observados, pero no prueban exploración global. Los pesos de las mezclas finales describen el protocolo de enfriamiento, sin establecer pesos de equilibrio. La asociación con la interacción quiral no elimina las limitaciones del diseño de ocho combinaciones.

## Vista tridimensional y contexto físico

`figuras_3d.py` dibuja un esquema geométrico y una configuración completa del estado 6, copia 63 a 6 K. El archivo `datos/estado6_copia63_3d.npz` conserva los 5425 vectores y la máscara de cinco capas, sin series temporales. La vista muestra todos los sitios y solo una selección de flechas para facilitar la lectura; el eje vertical se amplía gráficamente cuatro veces. La geometría numérica y los datos no se modifican.

La motivación sobre estados magnéticos de interés para memorias y spintrónica se apoya en Nagaosa y Tokura (2013), [Topological properties and dynamics of magnetic skyrmions](https://doi.org/10.1038/nnano.2013.243). El objetivo del proyecto es caracterizar un modelo térmico y la calidad de sus estimaciones. El algoritmo de Metropolis no se interpreta como dinámica temporal calibrada de un dispositivo.

El tamaño se mantiene fijo al comparar las ocho combinaciones de interacción quiral y campo. El problema general permite situar el caso, pero los resultados de este proyecto no constituyen un análisis de escalado con radio o espesor ni una prueba universal de desempeño del algoritmo.
