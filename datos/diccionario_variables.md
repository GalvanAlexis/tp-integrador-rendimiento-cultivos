# Diccionario de variables

Unidad de análisis: **partido × cultivo × campaña**.  
Variable numérica: **rendimiento (kg/ha)**.  
Variable de modelado: **ratio = rendimiento / promedio de las 5 campañas anteriores**.  
Clase: **campaña_mala = 1 si ratio < 0,85**.

## 1. Variables originales

| Variable | Tipo | Unidad | Descripción | Disponible al predecir | Uso |
|---|---|---|---|---|---|
| `cultivo` | texto | — | Nombre del cultivo | Sí | Predictor categórico |
| `campania` | texto | — | Campaña agrícola (`2015/16`) | Sí | Identificador |
| `campania_inicio` | entero | año | Año de inicio de la campaña | Sí | Predictor temporal y split |
| `campania_fin` | entero | año | Año de fin de la campaña | Sí | Auditoría |
| `municipio_id` | entero | código | Código de municipio de la cartografía censal 2010 | Sí | Identificador y unión con clima |
| `municipio_nombre` | texto | — | Nombre del partido | Sí | Identificador y muestra en la app |
| `superficie_sembrada` | entero | ha | Superficie sembrada | Sí | Predictor y peso de costos |
| `superficie_cosechada` | entero | ha | Superficie cosechada | No usar como predictor | Auditoría; fuga |
| `produccion` | entero | t | Producción total | No usar como predictor | Auditoría; fuga |
| `rendimiento` | entero | kg/ha | Rendimiento estimado | No; es el objetivo | Target de regresión |

## 2. Variables derivadas

| Variable | Tipo | Unidad | Fórmula | Descripción | Uso |
|---|---|---|---|---|---|
| `prev5` | float | kg/ha | Media de las 5 campañas calendario anteriores del mismo cultivo y municipio | Rendimiento esperado de referencia | Predictor y baseline |
| `prev5_ceros` | entero (0 a 5) | campañas | Cantidad de campañas con rendimiento 0 entre las 5 anteriores | Indica un promedio deprimido por campañas sin cosecha | Predictor |
| `ratio` | float | adimensional | `rendimiento / prev5` | Rendimiento relativo al histórico reciente | Target de regresión |
| `mala80` | 0/1 | — | `ratio < 0,80` | Campaña mala con umbral 80% | Sensibilidad |
| `mala85` | 0/1 | — | `ratio < 0,85` | Campaña mala, umbral principal | Target de clasificación |
| `mala90` | 0/1 | — | `ratio < 0,90` | Campaña mala con umbral 90% | Sensibilidad |
| `perdida_total` | booleano | — | `rendimiento == 0 & superficie_cosechada == 0` | Campaña sin cosecha | Auditoría y análisis |
| `peso_superficie` | float | ha | `superficie_sembrada` | Peso para costo ponderado por superficie | Evaluación |
| `split` | texto | — | `train`, `validacion`, `test` | Partición temporal | Evaluación |

### Cómo se calcula `prev5`

Se construye una grilla completa de cultivo × municipio × campaña entre 1969 y 2024. Sobre esa grilla:

```python
prev5 = rendimiento.shift(1).rolling(5, min_periods=5).mean()
```

Con `min_periods=5` se exigen cinco campañas calendario consecutivas con dato. Si falta un año, el promedio queda nulo y la fila no entra al modelado. Esto evita que el promedio saltee huecos y use campañas más viejas de lo que corresponde.

`prev5_ceros` cuenta, sobre la misma ventana, cuántas campañas tuvieron rendimiento 0. Los ceros se incluyen en `prev5` porque son campañas reales sin cosecha, pero lo bajan: las 12 filas con `ratio > 3` tienen al menos un cero previo. La lista está en `datos/procesado/casos_ratio_extremo.csv`. Hay 190 filas de modelado (1,25%) con al menos un cero previo.

## 3. Variables climáticas

Se agregan por municipio, cultivo y campaña, usando la ventana desde la siembra hasta la fecha de corte.

| Variable | Tipo | Unidad | Descripción | Disponible al predecir | Fuga |
|---|---|---|---|---|---|
| `clima_precip_total_mm` | float | mm | Precipitación acumulada en la ventana | Sí | No, si la ventana termina en el corte |
| `clima_dias_tmax_30` | entero | días | Días con temperatura máxima mayor a 30 °C | Sí | No |
| `clima_tmax_media_c` | float | °C | Temperatura máxima media | Sí | No |
| `clima_tmin_media_c` | float | °C | Temperatura mínima media | Sí | No |
| `clima_radiacion_kwh_m2` | float | kWh/m² | Radiación solar acumulada. Se conserva en la tabla cruda, pero se excluye del dataset de modelado por faltantes en 1981-1983 | Sí | No |
| `clima_humedad_media_pct` | float | % | Humedad relativa media | Sí | No |
| `clima_sequia_max_dias` | entero | días | Máxima racha de días con menos de 1 mm | Sí | No |
| `clima_precip_pre_siembra_mm` | float | mm | Precipitación de los 60 días previos a la siembra | Sí | No |
| `clima_dias_helada` | entero | días | Días con temperatura mínima menor a 0 °C | Sí | No |
| `clima_dias_lluvia` | entero | días | Días con precipitación mayor a 1 mm | Sí | No |

### Ventanas por cultivo

| Cultivo | Inicio | Corte |
|---|---|---|
| Trigo total | 01/06 del año de campaña | 15/11 del año de campaña |
| Maíz | 01/09 del año de campaña | 31/01 del año siguiente |
| Girasol | 01/09 del año de campaña | 31/01 del año siguiente |
| Soja total | 01/11 del año de campaña | 28/02 del año siguiente |

El corte se ubica cuando el período crítico de cada cultivo ya empezó para las fechas de siembra típicas: espigazón y floración en trigo, floración en maíz y girasol, y R3–R6 en soja. Las fechas son únicas para toda la provincia; la justificación y la limitación están en la sección 1.4 del informe de la Entrega 2.

El dataset de modelado con clima usa **nueve** de las diez variables climáticas. `clima_radiacion_kwh_m2` queda fuera porque NASA POWER no tiene radiación para las ventanas de 1981-1983; las restantes tienen cobertura completa en las 15.179 filas.

## 4. Variables excluidas y por qué

| Variable | Motivo |
|---|---|
| `produccion` | El rendimiento se calcula como producción / superficie cosechada. Incluirla deja la respuesta entre las entradas. |
| `superficie_cosechada` | Ídem anterior. |
| `rendimiento` | Es el objetivo de regresión. No puede ser predictor. |
| `ratio` | Es el objetivo transformado. No puede ser predictor. |
| `mala80`, `mala85`, `mala90` | Son los objetivos de clasificación. No pueden ser predictores. |
| `municipio_nombre` | Se usa para mostrar, no como predictor. Para modelar se usa `municipio_id` o una codificación categórica. |
| Marca de ratio extremo | No existe como columna. Se calcula con el objetivo y sería fuga; los casos se listan en un archivo aparte. |
| `campania` | Se usa `campania_inicio` numérico. La etiqueta de texto no aporta. |
| `superficie_cosechada` | Además de la fuga, no está disponible antes de la cosecha. |

## 5. Controles de fuga de información

1. **`prev5` y `prev5_ceros` usan solo campañas anteriores.** Se calculan con `shift(1)` y una grilla calendario.
2. **El clima termina en la fecha de corte.** No se usa clima posterior.
3. **No se usan producción ni superficie cosechada.**
4. **El split es temporal por campaña.** El test no se usa para elegir modelos ni hiperparámetros.
5. **No se usan codificaciones de target calculadas con el test.** Si en la Clase 3 se agrega una codificación por partido, se ajusta solo con train.
6. **La superficie sembrada es una estimación final.** En un uso operativo habría que usar la estimación disponible al corte. Se documenta como limitación menor.

## 6. Variables de la app

La aplicación de la Clase 4 pedirá o autocompletará:

- cultivo;
- partido;
- campaña;
- superficie sembrada;
- promedio de las 5 campañas anteriores y cantidad de campañas sin cosecha entre ellas (autocompletados desde una tabla histórica);
- variables climáticas acumuladas (autocompletadas desde la tabla de clima y editables para simular escenarios).

La app no entrena: solo carga los pipelines guardados y ejecuta `predict`.
