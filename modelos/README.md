# Modelos

Pipelines entrenados con joblib. La aplicación solo carga estos archivos y ejecuta `predict`; no entrena.

| Archivo | Contenido | Generado por |
|---|---|---|
| `regresor.joblib` | Pipeline de regresión: KNN con clima, k = 300 y pesos por distancia. Predice el ratio. | `python -m src.regresion` |
| `regresor_metadata.json` | Variables de entrada, hiperparámetros, métricas de test y versiones | `python -m src.regresion` |
| `clasificador.joblib` | Clasificador de campaña mala: árbol de decisión con clima (profundidad 5, hoja ≥ 50) dentro de un `FixedThresholdClassifier` con umbral 0,65. | `python -m src.clasificacion` |
| `clasificador_metadata.json` | Variables de entrada, hiperparámetros, umbral, costos, métricas de validación y de test, y versiones | `python -m src.clasificacion` |

## Uso del regresor

El Pipeline recibe las columnas originales y aplica internamente el one-hot, el logaritmo y el escalado:

```python
import joblib

modelo = joblib.load("modelos/regresor.joblib")
ratio = modelo.predict(df[list(modelo.feature_names_in_)])
rendimiento_kg_ha = ratio * df["prev5"]
alerta = ratio < 0.85
```

## Uso del clasificador

`predict` ya aplica el umbral elegido en validación. La probabilidad se obtiene del Pipeline interno:

```python
clasificador = joblib.load("modelos/clasificador.joblib")
X = df[list(clasificador.feature_names_in_)]
alerta = clasificador.predict(X)  # 1 = campaña mala esperada
proba = clasificador.estimator_.predict_proba(X)[:, 1]
umbral = clasificador.threshold
```

Variables de entrada de ambos modelos: `cultivo`, `superficie_sembrada`, `prev5`, `prev5_ceros` y las nueve variables climáticas `clima_*`.

Los archivos `.joblib` se cargan con la misma versión de scikit-learn con la que se guardaron, que figura en la metadata. No se deben cargar archivos joblib o pickle de origen externo.
