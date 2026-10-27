# Entrega 4: clasificación e integración

**TP Integrador de Aprendizaje Automático**  
**Proyecto:** rendimiento de cultivos por partido en la Provincia de Buenos Aires  
**Repositorio:** https://github.com/GalvanAlexis/tp-integrador-rendimiento-cultivos

## Datos del trabajo

| Dato | Completar |
|---|---|
| Integrantes | Francisco Calahorra, Federico Valdés, Alexis Galván |
| Curso | Aprendizaje Automático 2026 |
| Fecha de inicio | dd/mm/aaaa |
| Fecha de entrega | dd/mm/aaaa |

---

## Resumen ejecutivo

Se compararon los cuatro clasificadores obligatorios (DummyClassifier, DecisionTreeClassifier, KNeighborsClassifier con StandardScaler en un Pipeline y GaussianNB), con y sin variables climáticas: **268 configuraciones**, cada una medida con 19 umbrales de decisión. El umbral y los hiperparámetros se eligieron en validación con un **costo asimétrico** (un falso negativo cuesta el doble que un falso positivo).

El **clasificador seleccionado es un árbol de decisión con clima** (profundidad 5, hojas de al menos 50 filas, umbral 0,65). Es el único modelo que se separa con claridad del Dummy en validación: costo 443 contra 540.

En test (2019/20 a 2024/25, 470 campañas malas reales):

| Sistema | Recall | Precisión | F1 | Costo |
|---|---:|---:|---:|---:|
| Dummy | 0,00 | — | 0,00 | 940 |
| Regresión + umbral (Entrega 3) | 0,23 | 0,77 | 0,35 | 758 |
| **Árbol con clima (seleccionado)** | **0,37** | 0,65 | **0,47** | **687** |

El clasificador directo **detecta 66 campañas malas más que la regresión** (173 contra 107) a cambio de 61 falsas alarmas más (93 contra 32), y reduce el costo un 9%. La razón es la que anticipó la Entrega 3: la regresión subestima la magnitud de las campañas muy malas (regresión hacia la media) y casi nunca predice un ratio por debajo de 0,85. El clasificador aprende directamente la frontera que importa.

Los errores tienen un patrón claro. La mitad de los falsos negativos son campañas **apenas malas** (ratio de 0,75 a 0,85), y casi todos los aciertos ocurren en la **sequía 2022/23**. Los falsos positivos son campañas **secas que terminaron bien**. Cuando el clasificador y la regresión coinciden en alertar, el 90% de las alertas son correctas.

La **aplicación Flask** funciona localmente. Carga los dos pipelines guardados, autocompleta las variables desde el histórico y muestra la alerta del clasificador, su probabilidad y la opinión de la regresión.

---

## 1. Objetivo y protocolo

**Objetivo de clasificación:** `mala85 = 1` si el ratio real es menor a 0,85, es decir, si el rendimiento queda más de un 15% por debajo del promedio de las cinco campañas anteriores. Es el mismo umbral justificado en la Entrega 2 y aplicado a la regresión en la Entrega 3.

**Protocolo** (el mismo de la Entrega 3):

| Paso | Datos | Uso |
|---|---|---|
| Búsqueda | Train (1981/82 a 2013/14, 11.147 filas) → Validación (2014/15 a 2018/19, 1.828 filas) | Elegir hiperparámetros, umbral y modelo |
| Selección | Validación | Menor costo total |
| Evaluación final | Train + validación → Test (2019/20 a 2024/25, 2.204 filas) | Una sola medición |

El test no interviene en la búsqueda ni en la selección. Un test automático verifica que la función de búsqueda solo recibe train y validación.

**Variables:** las mismas de la Entrega 3. `cultivo` (one-hot), `superficie_sembrada` (log), `prev5`, `prev5_ceros` y, en el conjunto `con_clima`, las nueve variables climáticas de la ventana de cada cultivo. Ninguna usa información posterior a la fecha de corte.

## 2. Esquema de costos

Para el productor o el asesor, el error más caro es **una campaña mala que no se anticipa** (falso negativo): no se contrata cobertura, no se ajusta la venta anticipada y se llega a la cosecha sin margen financiero. Una falsa alarma (falso positivo) cuesta una cobertura o una consulta que no hacían falta.

| Error | Descripción | Costo relativo |
|---|---|---|
| Falso negativo (FN) | Campaña mala sin alerta | 2 |
| Falso positivo (FP) | Alerta en una campaña normal | 1 |

**Costo total = 2 × FN + FP.** Con este esquema, una alerta conviene si la probabilidad de que la campaña sea mala supera 1/3. Por eso el umbral de decisión no se fija en 0,5: se trata como un hiperparámetro y se elige en validación.

El **Dummy** (clase mayoritaria) no alerta nunca, así que su costo es 2 × cantidad de campañas malas: 540 en validación y 940 en test. Es el piso que cualquier modelo útil tiene que superar.

## 3. Búsqueda de hiperparámetros

| Familia | Grilla | Configuraciones por conjunto |
|---|---|---:|
| DecisionTreeClassifier | `max_depth` ∈ {2, 3, 4, 5, 6, 8, 10, 12, 15, sin límite} × `min_samples_leaf` ∈ {1, 10, 30, 50, 100} × `class_weight` ∈ {None, balanced} | 100 |
| KNeighborsClassifier (StandardScaler en el Pipeline) | `n_neighbors` ∈ {1, 3, 5, 10, 15, 25, 50, 75, 100, 150, 200, 300, 500, 800} × `weights` ∈ {uniform, distance} | 28 |
| GaussianNB | `var_smoothing` ∈ {1e-9, 1e-6, 1e-3, 0,01, 0,1, 0,3} | 6 |

Son 134 configuraciones por conjunto de variables, **268 en total**. Cada una se entrena una sola vez con train y se mide en validación con los umbrales 0,05, 0,10, …, 0,95: 5.092 combinaciones.

### 3.1 Mejor configuración por familia

| Modelo | Variables | Parámetros | Umbral | Recall val. | Precisión val. | Costo val. |
|---|---|---|---:|---:|---:|---:|
| **DecisionTreeClassifier** | con clima | depth = 5, leaf = 50, sin balanceo | 0,65 | 0,248 | 0,644 | **443** |
| KNeighborsClassifier | con clima | k = 500, distance | 0,40 | 0,026 | 0,583 | 531 |
| GaussianNB | con clima | var_smoothing = 0,3 | 0,50 | 0,041 | 0,393 | 535 |
| KNeighborsClassifier | sin clima | k = 150, uniform | 0,40 | 0,011 | 0,750 | 535 |
| GaussianNB | sin clima | var_smoothing = 0,001 | 0,60 | 0,037 | 0,357 | 538 |
| DecisionTreeClassifier | sin clima | depth = 2, leaf = 1, balanced | 0,90 | 0,000 | — | 540 |
| *Dummy (referencia)* | — | clase mayoritaria | — | 0,000 | — | *540* |

- **Sin clima no hay señal útil.** Ningún modelo sin clima baja de 535: el mejor árbol empata al Dummy con un umbral que no alerta nunca. Con solo cultivo, superficie y la historia del partido no se puede anticipar una campaña mala.
- **Con clima, el árbol es el único que se separa del Dummy** (443 contra 540). KNN y GaussianNB quedan entre 5 y 9 puntos por debajo del piso, con muy pocas alertas en validación.
- El período de validación tuvo pocas campañas malas (14,8%, contra 18,0% en train y 21,3% en test). Eso favorece en validación a los modelos que alertan poco.

### 3.2 Umbral de decisión

![Costo en validación según el umbral](figuras/clasificacion_umbral_costo.png)

- Con probabilidades bien calibradas, el umbral óptimo sería 1/3. El árbol tiene su mínimo en **0,65 a 0,70**, porque sus probabilidades no están calibradas y la validación tuvo menos campañas malas que el entrenamiento.
- La selección no depende de un umbral afinado. Con el umbral por defecto (0,5), el mismo árbol cuesta 473, todavía muy por debajo del Dummy y de las otras familias.
- Entre 0,70 y 0,75 el costo salta de 443 a 530: el umbral elegido está en el borde de una meseta. Un umbral algo menor es más robusto que uno mayor.

### 3.3 Peso de las clases

| `class_weight` del árbol (con clima) | Mejor costo con umbral 0,5 | Mejor costo con umbral elegido |
|---|---:|---:|
| None | 465 | **443** |
| balanced | 564 | 474 |

Balancear las clases y bajar el umbral son dos formas de mover la misma frontera. Cuando el umbral se elige en validación, el balanceo no aporta. Con umbral 0,5, el balanceo empeora el costo, porque produce demasiadas alertas.

## 4. Tabla comparativa en test

| Sistema | Umbral | Costo val. | Accuracy | Precisión | Recall | F1 | Costo test | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (clase mayoritaria) | — | 540 | 0,787 | — | 0,000 | 0,000 | 940 | 0 | 0 | 470 | 1.734 |
| Regresión + umbral (Entrega 3) | ratio < 0,85 | — | 0,821 | 0,770 | 0,228 | 0,351 | 758 | 107 | 32 | 363 | 1.702 |
| **DecisionTreeClassifier, con clima** | 0,65 | **443** | **0,823** | 0,650 | 0,368 | 0,470 | 687 | 173 | 93 | 297 | 1.641 |
| KNeighborsClassifier, con clima | 0,40 | 531 | 0,800 | 0,603 | 0,187 | 0,286 | 822 | 88 | 58 | 382 | 1.676 |
| GaussianNB, con clima | 0,50 | 535 | 0,822 | 0,618 | 0,428 | 0,506 | 662 | 201 | 124 | 269 | 1.610 |
| KNeighborsClassifier, sin clima | 0,40 | 535 | 0,787 | — | 0,000 | 0,000 | 940 | 0 | 0 | 470 | 1.734 |
| GaussianNB, sin clima | 0,60 | 538 | 0,785 | 0,421 | 0,017 | 0,033 | 935 | 8 | 11 | 462 | 1.723 |
| DecisionTreeClassifier, sin clima | 0,90 | 540 | 0,787 | 0,500 | 0,004 | 0,008 | 938 | 2 | 2 | 468 | 1.732 |
| Clasificador o regresión (alerta combinada) | — | — | 0,816 | 0,612 | 0,379 | 0,468 | 697 | 178 | 113 | 292 | 1.621 |

Campañas malas reales en test: 470 de 2.204 (21,3%). El costo de la regresión en validación no se informa porque el regresor final se entrenó con train + validación.

![Costo normalizado en validación y en test](figuras/clasificacion_costo_val_test.png)

**Lectura:**

- **El accuracy no sirve para elegir.** El Dummy tiene 0,787 sin detectar ninguna campaña mala, y todos los sistemas útiles quedan entre 0,80 y 0,82. La diferencia está en el recall y en el costo.
- **El árbol seleccionado supera a la regresión** en recall (0,37 contra 0,23), F1 (0,47 contra 0,35) y costo (687 contra 758). La regresión mantiene una precisión mayor (0,77 contra 0,65): alerta poco, pero cuando alerta suele acertar.
- **GaussianNB con clima tuvo el menor costo en test** (662). En validación quedó casi empatado con el Dummy (535), así que el protocolo no lo podía elegir. Elegirlo ahora sería usar el test para seleccionar. Se informa como resultado, no como cambio de decisión.
- **KNN** generaliza peor. Con k = 500 promedia vecinos de campañas muy distintas y su probabilidad casi nunca supera el umbral.
- **La alerta combinada** (alertar si cualquiera de los dos alerta) agrega 5 aciertos y 20 falsas alarmas al árbol solo: el costo sube de 687 a 697. Por eso la app no la usa como alerta principal (sección 7).

### 4.1 Matrices de confusión

![Matrices de confusión en test](figuras/clasificacion_matrices_confusion.png)

| | Árbol: predicho normal | Árbol: predicho mala | Regresión: predicho normal | Regresión: predicho mala |
|---|---:|---:|---:|---:|
| **Real normal** | 1.641 (VN) | 93 (FP) | 1.702 (VN) | 32 (FP) |
| **Real mala** | 297 (FN) | 173 (VP) | 363 (FN) | 107 (VP) |

## 5. Comparación de los dos sistemas

La consigna pide comparar el clasificador directo con la alerta obtenida aplicando el umbral sobre la regresión, sobre los mismos casos de test.

| Criterio | Clasificador directo (árbol) | Regresión + umbral (E3) | Ventaja |
|---|---:|---:|---|
| Campañas malas detectadas | **173 de 470** | 107 de 470 | Clasificador |
| Falsas alarmas | 93 | **32** | Regresión |
| Recall | **0,368** | 0,228 | Clasificador |
| Precisión | 0,650 | **0,770** | Regresión |
| F1 | **0,470** | 0,351 | Clasificador |
| Accuracy | **0,823** | 0,821 | Empate |
| Costo total (2·FN + FP) | **687** | 758 | Clasificador (−9%) |
| Tasa de alertas | 12,1% | 6,3% | — |
| Salida adicional | Probabilidad de campaña mala | Rendimiento estimado en kg/ha | — |

**Por qué gana el clasificador directo:** la regresión minimiza el error medio del ratio y, como mostró la Entrega 3, comprime sus predicciones hacia la media. Una campaña con ratio real de 0,60 recibe una predicción cercana a 0,85 o mayor, y la alerta no se dispara. El clasificador optimiza directamente la frontera de 0,85 con el costo de los errores, que es lo que importa para la alerta.

**Cuándo conviene cada uno:** con el esquema de costos del proyecto (FN = 2 × FP), el clasificador directo es preferible. Para un usuario al que le pesan más las falsas alarmas (por ejemplo, una aseguradora que solo quiere actuar cuando la señal es muy firme), la regresión o la coincidencia de ambos sistemas son mejores.

## 6. Análisis de falsos positivos y falsos negativos

### 6.1 Según el ratio real

![Tasa de alerta según el ratio real](figuras/clasificacion_errores_por_ratio.png)

| Ratio real | Casos | Árbol: alertas | Árbol: FN | Árbol: FP | Regresión: alertas | Regresión: FN | Regresión: FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 0,50 | 84 | 57 (68%) | 27 | — | 27 (32%) | 57 | — |
| 0,50 a 0,75 | 217 | 98 (45%) | 119 | — | 78 (36%) | 139 | — |
| 0,75 a 0,85 | 169 | 18 (11%) | 151 | — | 2 (1%) | 167 | — |
| 0,85 a 0,95 | 452 | 22 (5%) | — | 22 | 6 (1%) | — | 6 |
| 0,95 a 1,05 | 556 | 36 (6%) | — | 36 | 13 (2%) | — | 13 |
| ≥ 1,05 | 726 | 35 (5%) | — | 35 | 13 (2%) | — | 13 |

**Falsos negativos.** Son el error más caro y el más frecuente (297).

- **La mitad son campañas apenas malas.** 151 de los 297 FN del árbol tienen un ratio real entre 0,75 y 0,85. Para el productor, una caída del 15% al 25% es menos grave que una del 50%, así que este tipo de FN pesa menos de lo que indica el costo uniforme.
- **En las campañas muy malas el árbol funciona mucho mejor.** Detecta el 68% de las que tienen ratio menor a 0,50, contra el 32% de la regresión.

**Falsos positivos.** Son pocos (93, el 5% de las campañas normales) y están repartidos de forma pareja.

- Solo 22 de los 93 son campañas "casi malas" (ratio de 0,85 a 0,95). La tasa de falsas alarmas es igual en campañas normales y en campañas muy buenas, así que los FP **no se explican por cercanía al umbral**.

### 6.2 Perfil de cada tipo de resultado

| Árbol (test) | Casos | Ratio mediano | Precipitación media (mm) | Sequía máxima media (días) | Días > 30 °C | Casos de 2022/23 |
|---|---:|---:|---:|---:|---:|---:|
| VP | 173 | 0,56 | 210 | 23,6 | 61,5 | 86% |
| FN | 297 | 0,76 | 306 | 20,8 | 36,8 | 25% |
| FP | 93 | 1,01 | 257 | 24,1 | 41,0 | 12% |
| VN | 1.641 | 1,03 | 375 | 18,6 | 38,8 | 8% |

- **Los VP son la sequía.** El 86% ocurre en 2022/23, con poca lluvia y muchos días de calor extremo. Es el patrón que el árbol aprendió de la sequía 2008/09, la peor del entrenamiento (77% de campañas malas).
- **Los FN no tienen firma climática.** Su lluvia y sus días de calor se parecen más a los de las campañas normales que a los de los VP. Son campañas malas por poco margen o por causas que las variables de la ventana no registran (por ejemplo, granizo, plagas o excesos de agua puntuales). Ningún modelo con estas variables las puede anticipar; hacen falta otras fuentes.
- **Los FP son campañas secas que terminaron bien.** Tienen poca lluvia (257 mm) y la sequía máxima más larga de los cuatro grupos. El clima de la ventana anticipaba una campaña mala, pero el cultivo la compensó o la lluvia llegó después de la fecha de corte.

### 6.3 Por cultivo

| Cultivo | Campañas malas | Árbol: recall | Árbol: precisión | Regresión: recall | Regresión: precisión |
|---|---:|---:|---:|---:|---:|
| Maíz | 143 | 0,43 | 0,63 | 0,37 | 0,67 |
| Soja total | 149 | 0,42 | 0,78 | 0,36 | 0,90 |
| Trigo total | 139 | 0,33 | 0,54 | 0,00 | — |
| Girasol | 39 | 0,05 | 1,00 | 0,00 | — |

- En **maíz y soja**, los cultivos de verano donde el clima explica más, los dos sistemas detectan alrededor del 40% de las campañas malas. El árbol detecta un poco más; la regresión, con más precisión.
- En **trigo**, la regresión no emite ninguna alerta en todo el test. El árbol detecta un tercio de las campañas malas. Es la mayor diferencia entre los dos sistemas.
- En **girasol** ninguno funciona: el árbol detecta 2 de 39. La Entrega 3 ya mostró que el clima de la ventana casi no explica su ratio.

### 6.4 Por campaña

![Recall por campaña](figuras/clasificacion_recall_por_campania.png)

| Campaña | Campañas malas | Árbol: VP | Árbol: FP | Regresión: VP | Regresión: FP |
|---|---:|---:|---:|---:|---:|
| 2019/20 | 58 | 7 | 9 | 0 | 0 |
| 2020/21 | 70 | 9 | 6 | 0 | 0 |
| 2021/22 | 73 | 4 | 12 | 0 | 0 |
| 2022/23 | 222 | 148 | 11 | 107 | 10 |
| 2023/24 | 34 | 5 | 10 | 0 | 2 |
| 2024/25 | 13 | 0 | 45 | 0 | 20 |

- **2022/23 concentra el 47% de las campañas malas del test** (222 de 470) y el 86% de los aciertos del árbol. El árbol detecta el 67% de las campañas malas de ese año; la regresión, el 48%.
- **Fuera de la sequía, la regresión no detecta ninguna campaña mala.** El árbol detecta pocas (entre 4 y 9 por año, ninguna en 2024/25), con una precisión cercana al 40%.
- **2024/25 es el año con más falsas alarmas** (45 del árbol y 20 de la regresión). Fue un año con pocas campañas malas (13) y un clima que los dos sistemas leyeron como adverso.

### 6.5 Acuerdo entre los dos sistemas

| Clasificador | Regresión | Casos | Campañas malas reales | Precisión |
|---|---|---:|---:|---:|
| alerta | alerta | 114 | 102 | **0,89** |
| alerta | normal | 152 | 71 | 0,47 |
| normal | alerta | 25 | 5 | 0,20 |
| normal | normal | 1.913 | 292 | 0,15 |

- La regresión casi no agrega alertas propias: 114 de sus 139 alertas coinciden con el clasificador.
- **Cuando los dos alertan, 9 de cada 10 alertas son correctas.** Es la señal más confiable del sistema.
- Cuando solo alerta la regresión, la precisión (20%) apenas supera la tasa base (15%). Sumar esas alertas al clasificador empeora el costo.

## 7. Aplicación Flask

La app carga los dos pipelines guardados al arrancar y solo ejecuta `predict`; no entrena.

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Página principal con el estado de los modelos |
| `/predecir` | GET | Formulario; con `partido`, `cultivo` y `anio` autocompleta desde el histórico |
| `/predecir` | POST | Ejecuta la predicción y muestra el resultado |
| `/health` | GET | JSON con el estado de la app y de los modelos |
| `/api/datos` | GET | JSON con partidos, cultivos y campañas disponibles |

**Funcionamiento:**

1. El usuario elige partido, cultivo y campaña. Con "Autocompletar desde histórico", la app completa `prev5`, `prev5_ceros`, la superficie sembrada y las nueve variables climáticas.
2. El usuario puede editar los valores para simular escenarios (por ejemplo, menos lluvia).
3. Al enviar, la app ejecuta los dos modelos y muestra:
   - **La alerta principal, que es la del clasificador**, con la probabilidad estimada de campaña mala y el umbral a partir del cual alerta (65%).
   - El nivel de la alerta según el acuerdo con la regresión: **alta confianza** si ambos alertan, **alerta** si solo alerta el clasificador y **señal débil** si solo alerta la regresión (sección 6.5).
   - El ratio predicho por la regresión y el rendimiento estimado en kg/ha.
   - El valor real, si la campaña está en el histórico.

**Arranque local:**

```bash
python -m flask --app app run --debug
```

Se verificó con el cliente de prueba de Flask. Por ejemplo, para soja en Pergamino 2022/23 (ratio real 0,52), la app da P(campaña mala) = 100% y alerta. Para trigo en Tandil 2023/24 (ratio real 1,23), da 12% y no alerta.

## 8. Modelos guardados

| Archivo | Contenido |
|---|---|
| `modelos/clasificador.joblib` | `FixedThresholdClassifier` con el umbral 0,65 sobre el Pipeline del árbol (preprocesamiento + modelo). `predict` ya aplica el umbral. |
| `modelos/clasificador_metadata.json` | Variables de entrada, hiperparámetros, umbral, costos, métricas de validación y de test, semilla y versiones |
| `modelos/regresor.joblib` | Regresor de la Entrega 3 (sin cambios de modelo; guardado de nuevo con scikit-learn 1.9.0, la versión de `requirements.txt`) |

Un test verifica que `predict` del modelo guardado coincide con la regla usada en la búsqueda (`P(mala) ≥ umbral`). Otro verifica que su costo en validación es menor que el del Dummy.

## 9. Preguntas del informe

**¿El clasificador directo es mejor que la regresión con umbral?** Sí, para el objetivo del proyecto. Reduce el costo un 9% y detecta un 62% más de campañas malas. La regresión es más precisa, pero se pierde tres de cada cuatro campañas malas.

**¿Qué aportan las variables climáticas?** Todo. Sin clima, ningún clasificador supera al Dummy en validación ni en test.

**¿Qué errores comete y por qué?** Los FN son campañas apenas malas o sin señal climática. Los FP son campañas secas que terminaron bien. Los dos tipos de error indican que las variables de la ventana capturan la sequía, pero no otras causas de pérdida.

**¿El resultado es estable?** No del todo. El 86% de los aciertos viene de un solo año (2022/23). En años sin un evento climático fuerte, el clasificador detecta poco (0% a 15% de recall).

## 10. Limitaciones

- **Un solo evento domina el test.** La sequía 2022/23 aporta casi la mitad de las campañas malas y la mayoría de los aciertos. Las métricas globales dicen sobre todo qué tan bien se detecta una gran sequía.
- **Cambio de prevalencia.** La validación tuvo 14,8% de campañas malas y el test, 21,3%. El umbral elegido en validación (0,65) resultó conservador para el test. GaussianNB, que en validación casi empataba con el Dummy, fue el mejor en test.
- **Probabilidades sin calibrar.** La probabilidad que muestra la app es la fracción de campañas malas en la hoja del árbol. Sirve para ordenar casos, no como probabilidad exacta.
- **Girasol** no tiene una alerta útil con estas variables.
- **Autocompletado limitado.** La app autocompleta hasta la campaña 2024/25. Para campañas futuras hay que descargar el clima nuevo de NASA POWER.

## 11. Próximos pasos (Clase 5)

1. Desplegar la app (Vercel) con los mismos pipelines.
2. Calibrar las probabilidades del clasificador (por ejemplo, con `CalibratedClassifierCV` sobre validación) para que la probabilidad mostrada se pueda leer como tal.
3. Evaluar modelos por grupo de cultivos (verano e invierno), donde el clima tiene un efecto distinto.

## 12. Reproducibilidad

```bash
python -m src.clasificacion
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_clasificacion.ipynb
python -m flask --app app run
python -m pytest
```

`src.clasificacion` corre la búsqueda, la evaluación final y el análisis de errores. Guarda las tablas en `datos/procesado/clasificacion_*.csv` y el clasificador en `modelos/`. El notebook ejecuta el mismo protocolo y genera las figuras. La semilla es 42 y las versiones están en `modelos/clasificador_metadata.json`.

| Archivo | Contenido |
|---|---|
| `datos/procesado/clasificacion_busqueda_validacion.csv` | Las 268 configuraciones × 19 umbrales, con métricas de train y validación, más el Dummy |
| `datos/procesado/clasificacion_comparacion.csv` | Tabla comparativa en test |
| `datos/procesado/clasificacion_por_cultivo_test.csv` | Métricas por cultivo |
| `datos/procesado/clasificacion_por_campania_test.csv` | Métricas por campaña |
| `datos/procesado/clasificacion_errores_por_ratio_test.csv` | FN y FP por banda de ratio real |
| `datos/procesado/clasificacion_errores_perfil_test.csv` | Perfil de VP, FP, FN y VN |
| `datos/procesado/clasificacion_acuerdo_sistemas_test.csv` | Cruce de las alertas de los dos sistemas |
| `datos/procesado/clasificacion_predicciones_test.csv` | Predicción, probabilidad y tipo de error de cada caso |
