# Nanodisco quiral: simulación, inferencia y convergencia

Proyecto de Computación Estadística de Sebastián Mindiola Garizado, Universidad Nacional de Colombia, sede La Paz, 2026-II.

El estudio analiza ocho combinaciones de interacción quiral y campo externo en un nanodisco de 5425 espines. Cada combinación tiene 100 caminos de enfriamiento, desde 20 hasta 6 K, en 141 temperaturas. La pregunta física es cómo cambian el orden y las texturas. La pregunta estadística es qué resultados pueden interpretarse como equilibrio y cuáles describen la memoria del protocolo de enfriamiento.

Para leer el estudio, conviene empezar por la [explicación completa en HTML](docs/explicacion_completa.html) y continuar con el [informe final en PDF](docs/informe_final.pdf). GitHub muestra el código de un HTML; para usar los gráficos y selectores, descargue el repositorio y abra ese archivo en el navegador. Funciona sin conexión.

El HTML desarrolla el problema, las hipótesis y la matemática paso a paso. Incluye las figuras del informe, curvas comparativas de los ocho estados, pruebas estadísticas, tablas completas y un explorador de texturas durante el enfriamiento. Es una guía extensa que complementa al informe.

## Contenido del repositorio

```text
simulacion/
  lotes/                 Cuatro cuadernos que produjeron los ocho estados
scripts/
  analisis/              Momentos, autocorrelación, bootstrap, EM y figuras
  auditoria/             Comprobaciones adicionales sobre los datos
  informe/               Generador del HTML y compilación del informe
  configuracion.py       Rutas relativas y acceso a la base externa
pruebas/
  cuadernos/             Verificación de salidas y pruebas de humo
  registros/             Evidencia de ejecuciones y entorno de la revisión
  validar_html.py        Integridad del HTML, recursos y datos
  validar_controles.cjs  Prueba funcional de sus controles
  test_metodos.py        Comprobaciones de propiedades estadísticas
experimentos/            Métodos exploratorios, separados de la producción
resultados/
  auditoria/             Momentos revisados, curvas y texturas extraídas
  pasos_mc/              Diagnósticos y trazas derivadas de los ocho estados
  tablas/                Tablas originales del análisis
  informe/               BIC, regresión, uniformidad y otras tablas del informe
  figuras/               Figuras del análisis original
  experimentos/          Salidas conservadas de pruebas exploratorias
informe/latex/           main.tex, secciones, bibliografía y figuras
docs/                    HTML completo, PDF y documentación
datos/                   Instrucciones de conexión; no incluye datos crudos
build/                   Archivos temporales generados, excluidos de Git
```

Los archivos se ordenan por su función. Los experimentos exploratorios se conservan para documentar lo que se probó; sus resultados no se mezclan con las ocho configuraciones del informe final. El [registro de procedencia](resultados/procedencia.json) conserva la correspondencia con los archivos de origen y sus huellas SHA-256.

## Empezar sin la base pesada

Se necesita Python 3.10 o posterior. Node.js permite ejecutar la prueba de los controles. El informe requiere pdfLaTeX y BibTeX; reconstruir el HTML requiere además `pdftoppm`, del paquete Poppler. La instalación del análisis no instala JAX ni inicia simulaciones.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analisis.txt
make verificar
```

`make verificar` comprueba los archivos publicables, la integridad de datos e imágenes del HTML, sus controles y propiedades de los métodos estadísticos. No ejecuta las simulaciones ni el bootstrap completo.

Los resultados derivados incluidos permiten reconstruir la explicación:

```bash
make html
make informe
```

El primer comando actualiza `docs/explicacion_completa.html`. El segundo compila `informe/latex/main.tex` y actualiza `docs/informe_final.pdf`. Los temporales quedan en `build/`. La [guía de reproducción](docs/REPRODUCIBILIDAD.md) describe también cómo repetir el análisis desde los archivos crudos cuando estén disponibles.

## Qué se simuló

El disco tiene cinco capas, radio de 18.5 sitios y bordes libres. El hamiltoniano combina intercambio, interacción de Dzyaloshinskii y Moriya en el plano, anisotropía cúbica y campo perpendicular. En todos los estados, J = 1 meV y K = 0.05 meV. El parámetro H es el coeficiente energético del término de Zeeman; no se expresa directamente en teslas.

| Lote | Estado | kDM | H |
|---|---:|---:|---:|
| 1 | 1 | 0.05 | 0 |
| 1 | 2 | 0.15 | 0 |
| 2 | 3 | 0.33 | 0 |
| 2 | 4 | 0.60 | 0 |
| 3 | 5 | 1.00 | 0 |
| 3 | 6 | 0.30 | 0.05 |
| 4 | 7 | 0.60 | 0.10 |
| 4 | 8 | 0.75 | 0.30 |

Cada temperatura tiene 25000 barridos de termalización y 10000 de medición. Cada camino continúa su configuración al pasar a la siguiente temperatura. Las temperaturas de un camino están relacionadas; los caminos son las unidades de remuestreo independientes.

Los cuadernos de [simulación](simulacion/README.md) mantienen el código de producción y tienen las salidas de ejecución retiradas. Requieren el entorno de dos GPU previsto por el diseño. No se deben ejecutar los experimentos como sustitutos de esos cuadernos: cambian parámetros, normalización o algoritmo.

## Resultados y límites de interpretación

El máximo de fluctuación energética se sitúa alrededor de 14 K. El estado 8 tiene el menor T* reportado, 13.64 K. La ausencia de asociación significativa entre kDM y T* no demuestra independencia ni equivalencia entre temperaturas.

Ningún estado pasa el criterio conjunto de convergencia en toda la escalera. La magnetización conserva diferencias entre caminos al enfriar. Esto permite estudiar memoria y diversidad de texturas, aunque no se hayan demostrado sus pesos de equilibrio.

La revisión adicional comprobó energía, fluctuación cuadrática y magnetización en 56 puntos. En los 32 puntos próximos al máximo energético, el diagnóstico de la fluctuación cuadrática es menor que 1.002. Es evidencia favorable para los momentos observados, sin garantía de exploración de sectores ausentes. La revisión también recalculó errores agrupados por estado y sensibilidad al retirar un estado: la magnitud del efecto ajustado del campo depende del estado 8.

La [documentación de las pruebas](pruebas/README.md) distingue las ejecuciones originales de las comprobaciones nuevas. En esta revisión no se repitieron las simulaciones ni el bootstrap original de 50000 réplicas. El entorno registrado corresponde a la revisión local; no reconstruye retroactivamente la versión exacta de todas las dependencias usadas en Kaggle.

## Datos externos y publicación

La base cruda, de aproximadamente 25 GB, está excluida. No se descarga ni se publica mediante este repositorio. La incorporación desde Kaggle queda para una etapa posterior. Cuando esté disponible, se indicará su ubicación mediante `NANODISCO_DATOS`; el [diccionario de datos](datos/README.md) documenta las dimensiones y claves esperadas.

El repositorio conserva resultados derivados suficientes para leer el estudio y reconstruir el HTML. Estos archivos no sustituyen las matrices crudas para recalcular cualquier análisis. Tampoco contienen las series temporales completas de carga topológica, que no se registraron durante cada barrido.

La bibliografía está en [refs.bib](informe/latex/refs.bib). Las fuentes del informe conservan las citas por autor y año. La redacción sigue las pautas de claridad académica del marco Morfeo. El [HOOK](HOOK.md) registra el estado de preparación y publicación.

## Publicar la copia preparada

El repositorio local ya tiene su commit. Desde una terminal con acceso a GitHub y la sesión del usuario iniciada:

```bash
bash scripts/publicar_github.sh
```

La orden crea `nanodisco-quiral-estudio` como repositorio público en la cuenta autenticada y sube la rama `main`. No guarda credenciales ni cambia un remoto existente. En el entorno de preparación, la publicación quedó bloqueada por conexión a `api.github.com`; el código y los resultados locales están completos.
