# Lista de comprobación final

Repaso de los catorce puntos de la consigna, con la evidencia de cada uno.

| # | Punto de la consigna | Estado | Dónde verificarlo |
|---:|---|---|---|
| 1 | El problema y el usuario de la predicción están definidos | Cumplido | Informe, sección 1; README, `Problema` |
| 2 | Las variables están disponibles al momento de predecir | Cumplido | Informe, secciones 1.3 y 2.4; fecha de corte por cultivo |
| 3 | El umbral está justificado | Cumplido | Informe, secciones 1.4, 3.4 y 5.2 |
| 4 | El test no se utilizó para seleccionar modelos | Cumplido | Informe, sección 3.1; `tests/test_regresion.py` y `tests/test_clasificacion.py` verifican que la búsqueda solo recibe train y validación |
| 5 | Se incluyeron los dos baselines | Cumplido | Informe, sección 3.3; `datos/procesado/baselines_*.csv` |
| 6 | Se compararon los modelos obligatorios | Cumplido | Informe, secciones 4.1 y 5.1: Dummy, árbol, KNN y GaussianNB, con y sin clima |
| 7 | Las métricas están interpretadas | Cumplido | Informe, secciones 4.2, 5.3 y 7; MAE traducido a kg/ha y a porcentaje del rinde mediano |
| 8 | Se analizaron errores y casos cercanos al umbral | Cumplido | Informe, secciones 4.5 y 6.4; seis casos concretos con su consecuencia |
| 9 | El notebook se ejecuta desde cero | Cumplido | `notebooks/entrenamiento.ipynb`, 49 celdas, ejecutado sin errores; controla los conteos contra `metadata_preparacion.json` |
| 10 | La aplicación valida las entradas | Cumplido | `app.py` y `tests/test_app.py`; rangos y mensajes en lenguaje claro |
| 11 | La URL de Vercel funciona desde otro dispositivo | Parcial | Verificado por HTTP desde una máquina fuera del entorno de build: `GET /health` devuelve `{"status":"ok","entrega":5,"regresor":true,"clasificador":true}`. Falta la confirmación visual desde un teléfono con datos móviles |
| 12 | El repositorio contiene instrucciones | Cumplido | README: instalación, ejecución, estructura, notebooks, tests y despliegue |
| 13 | El informe reconoce limitaciones | Cumplido | Informe, sección 9; además el bloque fijo de advertencias de la app |
| 14 | Todos los integrantes pueden defender el trabajo | Cumplido | `informe/distribucion_tareas.md`, sección 3, con el repaso y las preguntas cruzadas |

## Entregables de la sección 7

| Entregable | Estado | Dónde está |
|---|---|---|
| Notebook ejecutado desde cero | Cumplido | `notebooks/entrenamiento.ipynb` |
| Dataset o instrucciones para obtenerlo | Cumplido | `datos/raw/` con hash, `src/descargar_datos.py` y `src/clima.py` |
| Diccionario de datos | Cumplido | `datos/diccionario_variables.md` |
| Modelos y pipelines guardados | Cumplido | `modelos/regresor.joblib`, `modelos/clasificador.joblib` y sus metadatos |
| Código de la aplicación | Cumplido | `app.py` y `templates/` |
| `requirements.txt` | Cumplido | Versiones fijas para producción; `requirements-dev.txt` para desarrollo |
| README con instalación y ejecución | Cumplido | `README.md` |
| Informe final en PDF | Cumplido | `informe/informe_final.pdf` |
| Presentación | Cumplido | `presentacion/presentacion.html` y `presentacion/presentacion.pdf` |
| Bitácora de decisiones | Cumplido | `informe/bitacora.md`, 21 entradas |
| Distribución de tareas | Cumplido | `informe/distribucion_tareas.md` |
| URL del repositorio | Cumplido | README y portada del informe |
| URL de Vercel | Cumplido | README y portada del informe |

## Causas de devolución de la sección 8.1

| Riesgo | Situación |
|---|---|
| No se puede ejecutar el notebook | Se ejecutó de punta a punta en Python 3.14.2 con las versiones de `requirements-dev.txt` |
| Se usa el test para elegir el modelo | El test se mide una sola vez; los hiperparámetros y el umbral salen de validación |
| Fuga de información | Sin `rendimiento`, `ratio`, `campania_inicio` ni variables posteriores a la cosecha; transformaciones dentro del Pipeline |
| No se compara contra baselines | Dos baselines, recalculados con train + validación antes de predecir el test |
| Métricas sin interpretación | Cada tabla tiene su lectura, incluida la derrota en MAE contra el promedio de 5 campañas |
| La app no usa los modelos presentados | Carga `modelos/*.joblib` y ejecuta `predict`; el notebook pasa el mismo caso por el mismo código |
| Faltan integrantes en la defensa | Los tres exponen un bloque y responden preguntas |
| Se reproduce una solución externa | El problema, los objetivos y los análisis son propios; el clima y las decisiones están documentados |

## Pendiente

Queda una sola cosa abierta: abrir https://tp-integrador-rendimiento-cultivos.vercel.app desde un teléfono con datos móviles y confirmar que carga el formulario y responde una predicción. La disponibilidad del servicio ya está verificada por HTTP desde fuera del entorno de build. El resto de la lista está cerrado y verificado en el repositorio.
