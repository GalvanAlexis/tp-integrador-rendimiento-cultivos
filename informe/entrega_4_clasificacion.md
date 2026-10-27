# Entrega 4: Clasificacion directa y aplicacion

Oct 27, 2026 · Francisco Calahorra, Federico Valdes, Alexis Galvan

---

## 1. Objetivo de la entrega

La Entrega 4 cierra el ciclo de modelado del TP Integrador con tres productos:

1. **Clasificadores directos** que predicen `campana_mala = 1` sin pasar por la regresion.
2. **Comparacion de sistemas**: clasificador directo vs. alerta por regresion + umbral (Entrega 3).
3. **Aplicacion Flask** operativa con formulario de prediccion, autocompletado de variables y visualizacion del resultado.

---

## 2. Protocolo de clasificacion

El protocolo sigue las mismas reglas que la Entrega 3:

| Paso | Detalle |
|---|---|
| Datos | `dataset_modelado_con_clima.csv` (15.179 filas) |
| Target | `mala85 = 1` si `ratio < 0,85` |
| Split | Train 1981-2013, validacion 2014-2018, test 2019-2024 |
| Busqueda | Solo con train; se mide en validacion |
| Criterio de seleccion | Menor costo total en validacion (FN = 2 × FP) |
| Evaluacion final | Reentrena con train + validacion; se mide una vez en test |

El test no interviene ni en la busqueda ni en la seleccion.

### Familias comparadas

| Familia | Variantes | Conjuntos de variables |
|---|---|---|
| DecisionTreeClassifier | 10 profundidades × 5 hojas minimas | sin_clima, con_clima |
| KNeighborsClassifier | 14 valores de k × 2 pesos | sin_clima, con_clima |

Total: 228 configuraciones evaluadas.

---

## 3. Esquema de costos

Para el productor o el asesor agronómico, el error más costoso es **una campaña mala que el modelo no anticipa** (falso negativo): obliga a financiarse a tasas desfavorables, pierde la cobertura de seguro y queda sin venta anticipada en el peor momento.

| Error | Descripcion | Costo relativo |
|---|---|---|
| Falso negativo (FN) | Campana mala sin alerta | 2 |
| Falso positivo (FP) | Alerta en campana normal | 1 |

Costo total = 2 × FN + 1 × FP. El denominador de referencia es el numero de campanas malas reales (positivos).

---

## 4. Mejor configuracion por familia (validacion)

| Modelo | Variables | Parametros | Recall | Precision | Costo |
|---|---|---|---:|---:|---:|
| KNeighborsClassifier | con_clima | k=500, pesos=distance | 0,007 | 1,000 | 536 |
| KNeighborsClassifier | sin_clima | k=800, pesos=distance | 0,007 | 0,400 | 539 |
| DecisionTreeClassifier | con_clima | prof=5, hoja>=10 | 0,456 | 0,313 | 564 |
| DecisionTreeClassifier | sin_clima | prof=6, hoja>=10 | 0,319 | 0,181 | 756 |

**Clasificador seleccionado por criterio de validacion:** KNeighborsClassifier con clima (k=500, pesos por distancia), costo 536.

El KNN en validacion emite casi ninguna alerta pero con precision perfecta, por eso su costo es el minimo en ese periodo. El arbol tiene un recall muy superior y seria preferible si el objetivo es minimizar FN.

---

## 5. Comparacion de sistemas (test)

| Sistema | Recall | Precision | F1 | Costo | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (clase mayoritaria) | 0,00 | — | 0,00 | 940 | 0 | 0 | 470 | 1.734 |
| Regresion + umbral (E3) | 0,23 | 0,77 | 0,35 | **758** | 107 | 32 | 363 | 1.702 |
| KNeighborsClassifier (con_clima) | 0,08 | 0,95 | 0,14 | 870 | 36 | 2 | 434 | 1.732 |
| DecisionTreeClassifier (con_clima) | **0,67** | 0,42 | 0,51 | **752** | 315 | 442 | 155 | 1.292 |

Campanas malas reales en test: 470 (21,3% del total).

### Lectura de los resultados

- El **arbol con_clima** es el que mas campanas malas detecta (recall 67%) y tiene el menor costo total (752), apenas por debajo de la regresion (758).
- El **KNN seleccionado** es extremadamente conservador: emite 38 alertas con precision 95%, pero deja sin detectar el 92% de las campanas malas (FN=434). En test, su costo (870) supera al del arbol y al de la regresion.
- La **regresion + umbral** sigue siendo competitiva: recall 23%, precision 77%, costo 758. Es un punto de referencia solido.
- Ninguno de los clasificadores supera al arbol en recall ni en costo.

### Sesgo de validacion-test

En validacion el KNN gana por costo porque el periodo 2014-2018 fue relativamente normal (pocos positivos). El periodo de test 2019-2024 incluye la sequia 2022/23, que concentra el 60% de los casos malos, y el KNN no generaliza bien a esa magnitud.

Este sesgo es el mismo que se observo en la regresion (E3): el test fue mas malo que cualquier periodo de entrenamiento.

---

## 6. Comparacion de sistemas

La consigna pide comparar clasificador directo vs. regresion + umbral sobre los mismos casos de prueba:

| Criterio | Clasificador directo (arbol) | Regresion + umbral (E3) | Ventaja |
|---|---|---|---|
| Recall | **0,67** | 0,23 | Arbol |
| Precision | 0,42 | **0,77** | Regresion |
| F1 | **0,51** | 0,35 | Arbol |
| Costo total | **752** | 758 | Arbol (marginal) |
| Tasa de alertas | 34% | 6% | Regresion (mas conservadora) |

**Conclusion:** el clasificador directo (arbol con_clima, profundidad 5) reduce el costo total y tiene un recall muy superior. La regresion tiene mayor precision: cuando alerta, casi siempre acierta. Para el caso de uso del productor (donde el FN es mas costoso), el arbol directo es preferible. Para una alerta de alta precision (ej. aseguradora que quiere minimizar falsos positivos), la regresion es mejor.

---

## 7. Aplicacion Flask

La app carga los pipelines al arrancar y ejecuta `predict`; no entrena.

| Endpoint | Metodo | Descripcion |
|---|---|---|
| `/` | GET | Pagina principal con estado de los modelos |
| `/predecir` | GET | Formulario de prediccion con autocompletado |
| `/predecir` | POST | Ejecuta prediccion y muestra resultado |
| `/health` | GET | JSON con estado y entrega |
| `/api/datos` | GET | JSON con partidos, cultivos y anos disponibles |

### Funcionamiento del formulario

1. El usuario selecciona partido, cultivo y campana.
2. Al presionar "Autocompletar desde historico", la app consulta el dataset y rellena `prev5`, `prev5_ceros`, superficie sembrada y las 9 variables climaticas.
3. El usuario puede editar las variables para simular escenarios (ej. reducir precipitacion).
4. Al enviar, la app ejecuta el regresor y el clasificador y muestra:
   - Ratio predicho y rendimiento estimado en kg/ha.
   - Alerta de cada sistema (regresion + clasificador).
   - Alerta combinada (si alguno alerta, se muestra alerta).
   - Valor real historico (si existe en el dataset).

### Arranque local

```bash
py -m flask --app app run --debug
```

---

## 8. Limitaciones

- **Sesgo de nivel.** El periodo de test fue mas malo que los de entrenamiento. Cualquier modelo entrenado con datos hasta 2018 subestima el impacto de la sequia 2022/23.
- **KNN seleccionado por validacion.** En validacion el KNN tiene menor costo, pero en test el arbol es mejor. Esto refleja la variabilidad interanual y la dificultad de elegir modelos cuando el periodo de prueba es estructuralmente diferente al de entrenamiento.
- **Trigo y girasol.** El clima tiene poca relacion con el ratio en estos cultivos.
- **Autocompletado limitado.** La app solo autocompleta hasta campana 2024. Para campanas futuras habria que descargar nuevo clima de NASA POWER.

---

## 9. Reproducibilidad

```bash
py -m src.clasificacion
py -m flask --app app run
py -m pytest
```

| Archivo | Contenido |
|---|---|
| `datos/procesado/clasificacion_busqueda_validacion.csv` | Las 228 configuraciones evaluadas |
| `datos/procesado/clasificacion_comparacion.csv` | Tabla comparativa en test |
| `datos/procesado/clasificacion_por_cultivo_test.csv` | Metricas por cultivo |
| `datos/procesado/clasificacion_por_campania_test.csv` | Metricas por campana |
| `datos/procesado/clasificacion_predicciones_test.csv` | Predicciones y errores |
| `modelos/clasificador.joblib` | Pipeline entrenado |
| `modelos/clasificador_metadata.json` | Metadata del clasificador |
