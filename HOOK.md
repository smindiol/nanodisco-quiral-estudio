# Estado del repositorio para compartir

## Completado

- Repositorio organizado por simulación, análisis, auditoría, pruebas, resultados e informe.
- Cuatro cuadernos de producción incorporados sin salidas ni metadatos de sesión.
- Experimentos exploratorios y sus resultados separados del estudio final.
- HTML completo incluido y reconstruido desde las nuevas rutas relativas.
- Informe LaTeX y PDF incorporados; rutas del anexo adaptadas a esta estructura.
- Base pesada excluida. Acceso externo mediante `NANODISCO_DATOS`, sin cargar datos a Kaggle.
- README, diccionario de datos, catálogo de pruebas y guía de reproducción redactados.
- Verificaciones de HTML, controles y métodos incorporadas con integración continua.
- Procedencia de archivos conservada con huellas de los originales.
- Script de publicación preparado para usar la sesión de GitHub del usuario, sin copiar credenciales.

## Comprobación final

- Generación del HTML: completada, 15 secciones, alrededor de 9500 palabras de texto y tablas.
- Recursos del HTML: 54 imágenes y fórmulas válidas, sin recursos externos.
- Controles: 18 combinaciones de variable y escala, 141 tablas, 48 selecciones de textura y tres descargas comprobadas con DOM simulado.
- El resultado del conjunto final de pruebas se conserva en `pruebas/registros/verificacion_repositorio.log`.
- Los entornos y límites de las ejecuciones científicas anteriores se documentan por separado. No se volvieron a ejecutar la simulación ni el bootstrap de 50000 réplicas al organizar el repositorio.

## Publicación

El repositorio Git local tiene un commit en la rama `main`. El intento de publicación con `scripts/publicar_github.sh` terminó con `error connecting to api.github.com` antes de crear el remoto. No se creó ningún repositorio de GitHub desde esta tarea. El fallo es de conexión desde el entorno y no demuestra que la sesión del usuario haya caducado.

La publicación puede completarse desde una terminal con acceso a GitHub mediante `bash scripts/publicar_github.sh`, desde esta carpeta. El programa consulta la cuenta autenticada y crea el repositorio público con el contenido confirmado. Se comprobó una alternativa de conexión con GitHub, todavía sin conexión confirmada.

## Fuera de esta entrega

- Carga de la base cruda a Kaggle, reservada para una etapa posterior.
- Nuevas simulaciones factoriales, estudios de tamaño y medición de transiciones de carga por barrido.
- Revisión visual del HTML en otro navegador: la captura con Firefox falló en el entorno de la revisión previa. La validación funcional con DOM simulado no sustituye esa inspección.
