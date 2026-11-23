# Simulación que sustenta el informe

Los cuatro cuadernos de `lotes/` conservan el código de producción. Cada uno ejecuta dos estados en dos GPU, con 100 caminos por estado. Las salidas de ejecución y los metadatos de sesión se retiraron para compartir el código; los resultados científicos están en `resultados/`.

El disco de radio 18.5 sitios tiene cinco capas y 5425 espines. La actualización de Metropolis utiliza una propuesta uniforme sobre la esfera obtenida por transformada inversa. Las subredes de un tablero de ajedrez permiten actualizar en paralelo sitios que no son vecinos entre sí.

El enfriamiento recorre 141 temperaturas de 20 a 6 K. En cada una descarta 25000 barridos y registra 10000. El estado final inicia la temperatura siguiente del mismo camino. Los barridos son pasos del algoritmo, no una unidad de tiempo físico.

Para ejecutar un lote, abra su cuaderno en el entorno de dos GPU previsto y revise los parámetros al comienzo. El programa genera archivos bajo `/kaggle/working/lote_XX/estado_N/`. Conviene conservar todos los metadatos y persistir las salidas fuera de la sesión antes de cerrarla.

La escritura comprueba dimensiones, finitud y norma de los espines antes de renombrar el archivo temporal. Se guardan claves de continuación, pero el programa no reanuda automáticamente una simulación interrumpida. Una nueva ejecución no equivale a continuar las mismas historias salvo que se restaure el estado necesario.

Las claves y semillas del archivo definen la realización concreta. Los comentarios del código sobre generación de semillas no sustituyen el metadato guardado. La reproducción estadística del diseño y la reproducción exacta de una realización deben distinguirse.

Los cuadernos de `pruebas/cuadernos/` verifican salidas o ejecutan pruebas cortas. Los métodos de `experimentos/` cambian el protocolo; su inclusión documenta los ensayos y no los convierte en fuentes de los ocho estados del informe.
