# Modelos

Pipelines entrenados con joblib. La aplicación solo carga estos archivos y ejecuta `predict`; no entrena.

| Archivo | Contenido | Generado por |
|---|---|---|
| `regresor.joblib` | Pipeline de regresión: KNN con clima, k = 300 y pesos por distancia. Predice el ratio. | `python -m src.regresion` |
| `regresor_metadata.json` | Variables de entrada, hiperparámetros, métricas de test y versiones | `python -m src.regresion` |

## Uso del regresor

El Pipeline recibe las columnas originales y aplica internamente el one-hot, el logaritmo y el escalado:

```python
import joblib

modelo = joblib.load("modelos/regresor.joblib")
ratio = modelo.predict(df[list(modelo.feature_names_in_)])
rendimiento_kg_ha = ratio * df["prev5"]
alerta = ratio < 0.85
```

Variables de entrada: `cultivo`, `superficie_sembrada`, `prev5`, `prev5_ceros` y las nueve variables climáticas `clima_*`.

Los archivos `.joblib` se cargan con la misma versión de scikit-learn con la que se guardaron, que figura en la metadata. No se deben cargar archivos joblib o pickle de origen externo.
