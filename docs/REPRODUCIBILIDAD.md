# Cómo reproducir el estudio

Hay dos recorridos. El primero usa los resultados derivados incluidos y permite reconstruir la explicación y el PDF sin la base cruda. El segundo vuelve a calcular estadísticas desde las matrices externas y tiene un costo mucho mayor. Ningún comando de verificación inicia ese segundo recorrido automáticamente.

## Lectura, reconstrucción y comprobaciones

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements-analisis.txt
make verificar
make html
make informe
```

`make html` necesita Poppler (`pdftoppm`) para convertir las figuras PDF a imágenes incrustadas. `make informe` necesita una distribución TeX con los paquetes declarados en `informe/latex/main.tex`, pdfLaTeX y BibTeX. El PDF se actualiza después de completar las pasadas de compilación. Los registros quedan en `build/latex/`.

La prueba funcional del HTML usa un DOM simulado en Node.js. Recorre las variables y las temperaturas, verifica las descargas y ejecuta los dibujos de texturas. No sustituye la revisión visual de un navegador. La captura con Firefox falló en el entorno de preparación; esa limitación se conserva en el registro de validación.

## Simulación de producción

Los cuatro cuadernos de `simulacion/lotes/` contienen el código que generó los ocho estados del informe. Se ejecutan en un entorno con dos GPU, una configuración por GPU y 100 caminos por configuración. Cada lote produce dos estados. El archivo de cada temperatura conserva parámetros, semilla y claves de continuación.

Los cuadernos comprueban dimensiones y normas antes de cerrar los archivos. Las salidas de temperaturas anteriores persisten mientras exista el almacenamiento de la sesión. Aunque se guardan claves, no se implementa reanudación automática de la simulación. Para reconstruir una corrida concreta deben conservarse su semilla y metadatos; generar una nueva semilla produce otra realización del protocolo.

Los scripts de `experimentos/` implementan diseños exploratorios distintos. Sus resultados se incluyen por trazabilidad y no sustituyen las corridas de producción.

## Conectar los datos externos

La base no está incluida y todavía no se ha publicado un identificador de dataset de Kaggle. No hay una descarga automática que apunte a un recurso inexistente. Al disponer de los archivos:

```bash
export NANODISCO_DATOS=/ruta/a/los/lotes
```

Los lectores admiten los árboles `lote1/lote_01/estado_1/T_20.0.npz`, `lote_01/estado_1/T_20.0.npz` o `estado_1/T_20.0.npz` bajo esa raíz. El [diccionario de datos](../datos/README.md) explica las claves y la normalización energética.

## Análisis original completo

La etapa A calcula estadísticas y diagnósticos por temperatura. La etapa B remuestrea caminos completos y obtiene intervalos y máximos. La etapa C produce las tablas, las figuras y el resumen. Para una comprobación corta sobre la base externa:

```bash
python scripts/analisis/ejecutar.py \
  --base "$NANODISCO_DATOS" --salida build/analisis_humo \
  --procs 2 --humo
```

Para repetir el análisis completo, con 50000 réplicas bootstrap:

```bash
python scripts/analisis/ejecutar.py \
  --base "$NANODISCO_DATOS" --salida build/analisis_completo \
  --procs 4 --B 50000
```

Las etapas A y B omiten salidas existentes que tengan el formato esperado. Esa comprobación no detecta cambios de parámetros del análisis; para comparar configuraciones distintas debe usarse otro directorio de salida. Los resultados completos pueden ocupar bastante más que los resúmenes publicados. No se copian automáticamente a los archivos que sustentan el informe.

## Barridos, figuras y mezclas

Los siguientes comandos reconstruyen diagnósticos y figuras sin sobrescribir las figuras de la entrega:

```bash
python scripts/analisis/pasos_mc.py \
  --datos "$NANODISCO_DATOS" --salida build/pasos_mc --procs 4
mkdir -p build/figuras build/tablas
python scripts/analisis/figuras_informe.py \
  --entrada resultados/pasos_mc --fig build/figuras --tablas build/tablas
python scripts/analisis/figura_calor.py \
  --entrada resultados/pasos_mc --tabla resultados/tablas/T_estrella.md \
  --fig build/figuras
python scripts/analisis/mezcla_em.py \
  --entrada resultados/pasos_mc --fig build/figuras --tablas build/tablas
```

El primer comando requiere la base externa. Los otros usan los derivados incluidos. Los programas originales utilizan una aproximación de kB ligeramente diferente en algunas figuras de calor; la auditoría usa el valor guardado por el diseño, 0.086173404 meV/K. Esa diferencia no se corrige retroactivamente en los resultados originales.

## Uniformidad del generador

El programa original prueba 100 semillas con cuatro millones de direcciones cada una. Su ejecución es independiente de la base:

```bash
mkdir -p build/tablas
python scripts/analisis/uniformidad.py --salida build/tablas/uniformidad.json
```

El archivo `resultados/informe/uniformidad.json` conserva el resultado original. Este contraste no se ejecuta en `make verificar`, para evitar convertir una comprobación rápida del repositorio en un experimento de 400 millones de propuestas.

## Auditoría adicional

```bash
make auditoria
make magnetizacion
make html
```

Las dos primeras órdenes requieren la base externa y actualizan `resultados/auditoria/`. La primera comprueba 56 puntos de momentos, calcula la regresión agrupada y su sensibilidad, reproduce la permutación exacta y extrae texturas de caminos fijos. La segunda lee las 1128 matrices de magnetización. No ejecutan nuevas simulaciones ni repiten el bootstrap original.

Los derivados publicados permiten evaluar las conclusiones y reconstruir las visualizaciones. No permiten rehacer análisis arbitrarios que necesiten datos no conservados, como tasas de cambio topológico durante cada barrido. El entorno de la revisión está documentado en `pruebas/registros/entorno_revision.json`; no se presenta como un bloqueo exacto del entorno histórico de Kaggle.
