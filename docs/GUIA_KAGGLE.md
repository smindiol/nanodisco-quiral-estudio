# Ejecutar la simulación en Kaggle con dos GPU T4

La simulación se divide en cuatro lotes. Cada cuaderno ejecuta dos estados, uno por GPU, con 100 caminos por estado. Para obtener los ocho estados se ejecutan los cuatro cuadernos por separado. No hace falta subir la base de resultados anteriores: el simulador genera sus propias configuraciones iniciales.

## 1. Crear y verificar la cuenta

Entre en [Kaggle](https://www.kaggle.com/), cree una cuenta y complete las verificaciones que aparezcan en su configuración antes de iniciar una sesión de cómputo. Compruebe la verificación del correo, la del teléfono si se solicita y el estado de acceso a aceleradores.

Si Kaggle solicita **verificación de identidad mediante Persona**, complete el proceso desde el enlace que muestra la propia plataforma y espere la confirmación. Esa verificación se realiza en la cuenta; no se introduce ningún documento de identidad ni código personal en los cuadernos o en este repositorio.

Conviene distinguir los requisitos: el anuncio oficial de Kaggle que presenta Persona se refiere expresamente al acceso a **TPU**. No permite afirmar que Persona sea un requisito universal de todas las GPU T4. Si la cuenta exige ese paso para habilitar recursos, debe completarse; si no lo exige, la comprobación práctica es que permita seleccionar y arrancar una sesión GPU. Véase el [anuncio de verificación de identidad](https://www.kaggle.com/discussions/product-announcements/670573).

Tener la cuenta verificada no garantiza cuota restante ni disponibilidad inmediata. Revise las opciones y la cuota que muestre su cuenta. Esta simulación necesita **GPU T4 ×2**, no TPU. Kaggle mantiene la opción T4×2 según su [anuncio sobre aceleradores](https://www.kaggle.com/product-announcements/735239).

## 2. Descargar el cuaderno del lote

Desde GitHub puede descargar el repositorio completo mediante **Code → Download ZIP**, descomprimirlo y entrar en `simulacion/lotes/`. Los archivos son:

| Cuaderno | Estados | Parámetros (kDM, H) |
|---|---|---|
| `lote_01.ipynb` | 1 y 2 | (0.05, 0), (0.15, 0) |
| `lote_02.ipynb` | 3 y 4 | (0.33, 0), (0.60, 0) |
| `lote_03.ipynb` | 5 y 6 | (1.00, 0), (0.30, 0.05) |
| `lote_04.ipynb` | 7 y 8 | (0.60, 0.10), (0.75, 0.30) |

Empiece por un solo lote. Ejecutar cuatro cuadernos simultáneamente no es necesario y puede superar los recursos o la cuota de la cuenta.

## 3. Importarlo en un notebook de Kaggle

Cree un notebook desde **Code → New Notebook** o la opción equivalente de creación. En el editor, abra **File → Import Notebook**, seleccione el archivo `.ipynb` del lote y espere a que aparezcan sus celdas. El nombre exacto de algún menú puede cambiar; la operación necesaria es importar un cuaderno local, no añadirlo como dataset.

Revise el título y compruebe que corresponde al lote elegido. Los cuadernos del repositorio no incluyen las salidas de sesiones anteriores. Eso es normal: el código y los parámetros permanecen en sus celdas.

## 4. Activar las dos GPU

En el editor abra **Settings → Accelerator** y seleccione **GPU T4 ×2**. Esa opción asigna las dos T4 al mismo notebook. No se activan por separado ni hace falta abrir dos notebooks para un lote. La selección de acelerador se describe en la [documentación de Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

Cambiar el acelerador puede reiniciar la sesión. Compruebe la configuración antes de ejecutar las celdas de producción. Si la opción no aparece o está bloqueada, revise la verificación, los mensajes de la cuenta, la cuota y la disponibilidad. No continúe con CPU esperando que el código reparta automáticamente el trabajo entre GPU inexistentes.

Puede comprobar el hardware en una celda nueva, antes de iniciar la producción:

```python
import jax

gpus = [device for device in jax.devices() if device.platform == "gpu"]
print("GPU visibles para JAX:", gpus)
assert len(gpus) == 2, "Seleccione GPU T4 ×2 antes de ejecutar este lote."
```

También puede ejecutar `!nvidia-smi -L` en otra celda. Deben aparecer dos dispositivos. La comprobación de JAX es necesaria porque el simulador usa JAX: que `nvidia-smi` vea GPU no basta si el entorno de Python solo tiene disponible el backend CPU. Si ocurre eso, revise el entorno de Kaggle y la instalación GPU de JAX antes de continuar; no cambie a ciegas las versiones del cuaderno. La [documentación de instalación de JAX](https://docs.jax.dev/en/latest/installation.html) describe los entornos compatibles.

## 5. Revisar los parámetros de producción

Los cuadernos contienen el disco de 5425 espines, cinco capas y radio de 18.5 sitios. Cada estado tiene 100 caminos. La escalera va de 20.0 a 6.0 K en pasos de 0.1 K; en cada temperatura hay 25000 barridos de termalización y 10000 de medición.

Para repetir el diseño del informe, conserve esos parámetros. Una prueba con menos espines, caminos o barridos sirve para comprobar ejecución, pero no reproduce los resultados publicados. Los experimentos de `experimentos/` tampoco son sustitutos de estos cuatro cuadernos.

No se necesitan credenciales de GitHub, claves de Kaggle ni la base cruda para ejecutar la simulación desde el editor. Si falta alguna dependencia y hay que instalarla, el acceso a Internet dependerá de los permisos y ajustes de la sesión. Revise los mensajes antes de iniciar el lote largo.

## 6. Ejecutar y seguir el avance

Ejecute las celdas en orden. Una comprobación inicial debe mostrar dos GPU, dos estados, 100 copias y 5425 espines. La compilación inicial de JAX puede tardar más que los pasos posteriores; no equivale por sí sola a que la sesión esté bloqueada.

Cada camino continúa su configuración al bajar a la temperatura siguiente. El avance debe recorrer 20.0, 19.9, 19.8 K y así sucesivamente. Los mensajes de progreso y los archivos de temperatura permiten comprobar que el lote sigue avanzando.

Para una corrida guardada de principio a fin, use **Save Version → Save & Run All**, si esa es la opción disponible en el editor. Ese modo ejecuta el cuaderno como una corrida nueva; no debe suponerse que conserva el estado de una ejecución interactiva ya iniciada. Elija el modo antes de la producción y revise que la versión use las dos GPU. Conserve la versión, los parámetros y la semilla de la corrida cuyo resultado vaya a analizar.

## 7. Guardar las salidas antes de cerrar

El lote 1 escribe, por ejemplo:

```text
/kaggle/working/lote_01/
  estado_1/
    T_20.0.npz
    T_19.9.npz
    ...
    T_6.0.npz
  estado_2/
    ...
```

Cada archivo incluye las mediciones de 10000 barridos por 100 caminos, la configuración final y los metadatos. Al finalizar un estado completo deben existir 141 archivos de temperatura. El código verifica los archivos antes de cerrar su escritura, pero eso no evita perderlos si desaparece el almacenamiento de la sesión.

Guarde una versión que conserve las salidas y descargue los archivos desde el panel de salida correspondiente. Compruebe el espacio disponible antes de crear un ZIP adicional: comprimir toda la carpeta mientras conserva los originales requiere almacenamiento extra. Las cuotas y los límites de sesión se consultan en la cuenta y en la documentación de Kaggle; no se asume que todas las cuentas tengan el mismo presupuesto disponible.

Los archivos incluyen claves de continuación, pero estos cuadernos **no implementan reanudación automática**. Volver a ejecutar todo el cuaderno no garantiza continuar desde la última temperatura terminada. Si la sesión se interrumpe, conserve los archivos ya escritos y documente el punto de interrupción.

## 8. Analizar los resultados

Después de descargar las salidas, mantenga los directorios de lote, estado y temperatura. El [diccionario de datos](../datos/README.md) explica sus claves. La [guía de reproducción](REPRODUCIBILIDAD.md) muestra cómo configurar `NANODISCO_DATOS` y ejecutar los análisis.

La base pesada continúa fuera de este repositorio. Su publicación como dataset de Kaggle es una operación posterior, distinta de ejecutar los cuadernos. Antes de comparar los nuevos resultados con el informe, compruebe los parámetros, las dimensiones, la normalización Eo, las semillas y los diagnósticos de convergencia.

## Problemas frecuentes

| Situación | Qué comprobar |
|---|---|
| GPU deshabilitada | Verificaciones pendientes, mensajes de la cuenta, cuota y disponibilidad |
| Solo aparece una GPU o JAX muestra CPU | Acelerador seleccionado y backend GPU de JAX |
| Error al activar TPU | Este proyecto requiere dos GPU T4; no se ejecuta en TPU sin adaptar el código |
| Persona no termina de verificar | Estado del proceso en Kaggle y soporte de la plataforma; el cuaderno no puede resolverlo |
| Sesión terminada antes de 6 K | Archivos persistidos y temperatura alcanzada; no asumir reanudación automática |
| Resultados distintos del informe | Diseño, semilla, número de pasos y mezcla de las cadenas; una nueva realización tiene variabilidad |

Los enlaces oficiales se consultaron el 23 de noviembre de 2026. Los menús, permisos y cuotas pueden cambiar; prevalece lo que indique la cuenta al iniciar la ejecución.
