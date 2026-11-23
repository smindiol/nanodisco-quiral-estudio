# Experimentos exploratorios conservados

Estos programas pertenecen a ensayos del proyecto. Se conservan separados de la producción para permitir su inspección. Sus parámetros y normalizaciones no son intercambiables con los de los ocho estados del informe final. No se ejecutan al importar las pruebas ni durante la integración continua.

| Carpeta | Método | Diferencia respecto de la producción |
|---|---|---|
| `metropolis_inicial/` | Actualización local con diagnósticos y remuestreo integrados | Geometría, número de estados y barridos de un ensayo distinto |
| `tempering_cono/` | Cono adaptativo e intercambio de réplicas | Escaleras simultáneas, adaptación durante termalización y otras longitudes |
| `tempering_multigpu/` | Variante posterior de intercambio y verificación | Otro diseño y convenciones propias, conservados como experimento |

Los resultados de ensayos que estaban disponibles están en `resultados/experimentos/tempering/`, con sus tablas y figuras. La correspondencia con las ubicaciones anteriores se registra en `resultados/procedencia.json`. No se atribuyen todas esas salidas a una sola variante sin una identificación adicional de su corrida.

Los scripts pueden iniciar cómputo al ejecutarse o importarse. Antes de correrlos hay que revisar sus variables de entorno, parámetros y directorio de salida. Este repositorio no los lanza sobre la GPU remota ni altera trabajos activos.
