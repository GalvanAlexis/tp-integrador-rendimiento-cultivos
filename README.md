# Rendimiento de cultivos por partido: ¿se viene una campaña mala?

Trabajo Práctico Integrador de **Aprendizaje Automático**.

**Repositorio:** https://github.com/GalvanAlexis/tp-integrador-rendimiento-cultivos

**Estado:** Entrega 2 terminada. Dataset preparado, diccionario de variables, protocolo de evaluación y baselines.

---

## Problema

Los productores y asesores agronómicos tienen que decidir insumos, cobertura de seguro, financiamiento y venta anticipada antes de conocer el rendimiento de la campaña. El clima explica una parte importante de la variación entre campañas, pero no toda.

El proyecto predice el rendimiento de un cultivo en un partido bonaerense antes de la cosecha y genera una alerta si la campaña se espera mala.

| Tarea | Pregunta | Salida |
|---|---|---|
| Regresión | ¿Qué rendimiento se espera? | kg/ha |
| Clasificación | ¿La campaña será mala? | Alerta sí/no |

La alerta se genera de dos maneras y luego se comparan:

1. **Clasificador directo:** predice la clase.
2. **Regresión + umbral:** predice el rendimiento y aplica el umbral.

## Decisión y usuario

- **Usuario:** productores, asesores agronómicos, cooperativas, aseguradoras y áreas municipales de producción.
- **Decisión:** contratar o reforzar cobertura, vender una parte de la cosecha a futuro, gestionar financiamiento o priorizar asistencia en los partidos con alerta.
- **Momento de la predicción:** fecha de corte fija por cultivo, al final del período crítico y antes de la cosecha.

| Cultivo | Corte provisorio |
|---|---|
| Trigo | 15/11 |
| Maíz | 31/01 |
| Girasol | 31/01 |
| Soja | 28/02 |

Las fechas se validan con material del INTA o con un asesor agronómico.

## Variable numérica y umbral

- **Variable numérica:** rendimiento en kg/ha.
- **Variable de modelado:** `ratio = rendimiento / promedio de las 5 campañas anteriores`.
- **Clase:** `campaña_mala = 1` si `ratio < 0,85`.

El umbral relativo absorbe diferencias de suelo y tendencia tecnológica entre partidos. El 0,85 es provisorio: se analizan también 0,80 y 0,90.

## Fuente de datos

**Estimaciones Agrícolas**, Datos Abiertos de la Provincia de Buenos Aires.

- Organismo: Ministerio de Economía. Secretaría de Agricultura, Ganadería y Pesca. Dirección de Estimaciones Agrícolas.
- Período: campañas 1969/70 a 2025/26.
- Licencia: CC BY 4.0.
- Catálogo: https://catalogo.datos.gba.gob.ar/dataset/estimaciones-agro
- Archivo: `datos/raw/estimaciones-agricolas-1969_2026.csv`

El clima se obtiene de **NASA POWER** (datos diarios desde 1981, reanálisis en grilla de aproximadamente 0,5°).

## Alcance de la Entrega 2

- Cultivos: **maíz, trigo total, girasol y soja total**.
- Se excluyen sorgo (alta proporción de rendimiento cero) y cebada cervecera (serie corta, termina en 2015/16).
- Se excluye la campaña 2025/26 por estar incompleta.
- Campañas de modelado: 1981/82 a 2024/25.
- Unidad: partido × cultivo × campaña.

## Estructura del repositorio

```text
proyecto/
├── README.md
├── requirements.txt
├── app.py
├── modelos/
├── templates/
├── public/
├── notebooks/
│   └── 02_preparacion_y_baselines.ipynb
├── src/
│   ├── config.py
│   ├── descargar_datos.py
│   ├── preparar_datos.py
│   ├── baselines.py
│   └── clima.py
├── datos/
│   ├── README.md
│   ├── diccionario_variables.md
│   ├── raw/
│   └── procesado/
├── informe/
│   ├── entrega_2_preparacion_y_baselines.md
│   └── figuras/
└── tests/
```

## Inicio rápido

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.descargar_datos
python -m src.preparar_datos
python -m src.baselines
```

Notebook:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/02_preparacion_y_baselines.ipynb
```

Clima:

```bash
python -m src.clima --sample
python -m src.clima --full
python -m src.clima --merge
```

La descarga completa ya se ejecutó: 107 partidos, 18.832 filas de clima y 15.179 filas de modelado con clima completo. La radiación se conserva en la tabla cruda pero se excluye del modelado por faltantes en 1981-1983; quedan nueve variables climáticas.

## Resultados principales de la Entrega 2

Dataset de modelado: **15.179 filas** (11.147 train, 1.828 validación, 2.204 test).

| Baseline (test) | MAE ratio | MAE kg/ha | Accuracy | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Dummy | 0,177 | 756 | 0,787 | 0,000 | 0,000 |
| Promedio 5 campañas | **0,154** | **626** | 0,787 | 0,000 | 0,000 |

Lectura:

- El promedio de 5 campañas es el baseline real del problema y supera al Dummy en test.
- En validación el Dummy gana por poco; en test pierde. El período de prueba es más malo que el de entrenamiento: el ratio medio pasa de 1,079 en train a 0,960 en test.
- Ambos baselines tienen **recall = 0**. Aciertan cerca del 79% y no detectan ninguna campaña mala. Es el ejemplo de "buen MAE, mala clasificación".
- La campaña 2022/23 concentra el **60,7%** de los casos malos del test. El modelo de la Clase 3 tiene que demostrar que la detecta con clima.

## Próximos pasos

1. Usar el dataset con clima (`dataset_modelado_con_clima.csv`) en el modelado.
2. Entrenar Dummy, DecisionTree y KNN con StandardScaler en Pipeline.
3. Evaluar MAE, RMSE y R², residuos y errores por campaña.
4. Entrenar los clasificadores y comparar los dos sistemas de alerta.
5. Construir la app Flask y desplegarla en Vercel.

## Equipo

Completar con nombre y apellido de los cuatro integrantes, curso y comisión.

## Licencia

Código: MIT. Datos: CC BY 4.0, con atribución a la Dirección de Estimaciones Agrícolas de la Provincia de Buenos Aires.
