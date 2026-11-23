# Presentación de resultados

[Abrir la presentación completa](presentacion.pdf).

La exposición tiene 19 diapositivas principales y seis de respaldo. Parte de las texturas magnéticas y su motivación experimental. Después presenta el modelo de espines, el sistema de 5425 sitios, la simulación y la evaluación de sus estimaciones. La [revisión de figuras](REVISION_FIGURAS.md) vincula cada resultado con la pregunta que responde.

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

Las tres filas del esquema de enfriamiento muestran las copias 1 a 3 del estado 6 en cuatro temperaturas, con una escala común de la componente Sz. El panel de ocho estados muestra la copia 1 a 6 K. Son ejemplos de configuraciones, no una estimación visual de sus frecuencias.

Para ilustrar las dos poblaciones del estado 6, se extrajeron las capas centrales de las copias 63 y 44 a 6 K, seleccionadas por sus medias de magnetización próximas a los centros EM. Este pequeño conjunto derivado está en `datos/ejemplos_estado6.json`. Las etiquetas indican medias temporales; los mapas son configuraciones finales. Su selección es ilustrativa, no aleatoria.

`datos/procedencia_figuras.json` resume estas decisiones. La presentación se puede regenerar con los derivados incluidos sin descargar la base pesada.

## Criterios de interpretación

La temperatura T* es característica del disco finito; no demuestra una singularidad de Curie. Los diagnósticos favorables de energía y fluctuación cuadrática refuerzan los momentos observados, pero no prueban exploración global. Los pesos de las mezclas finales describen el protocolo de enfriamiento, sin establecer pesos de equilibrio. La asociación con la interacción quiral no elimina las limitaciones del diseño de ocho combinaciones.

## Contexto físico y procedencia visual

`figuras_3d.py` dibuja la apertura con una textura analítica ilustrativa. Los resultados térmicos y las texturas posteriores se construyen a partir de las simulaciones. El archivo `datos/estado6_copia63_3d.npz` conserva una configuración de cinco capas para comprobar la geometría de la malla.

El contexto se apoya en los estudios de creación y desplazamiento en nanoestructuras y de formación controlada con imagen magnética. [Sampaio et al. (2013)](https://doi.org/10.1038/nnano.2013.210), [Kern et al. (2025)](https://doi.org/10.1002/adma.202501250).
