# Datos

## Fuente principal

**Estimaciones Agrícolas**  
Datos Abiertos de la Provincia de Buenos Aires.  
Ministerio de Economía. Secretaría de Agricultura, Ganadería y Pesca.  
Subsecretaría de Agricultura. Dirección Nacional de Agricultura.  
Dirección de Estimaciones Agrícolas.

- Catálogo: https://catalogo.datos.gba.gob.ar/dataset/estimaciones-agro
- Descarga directa: https://catalogo.datos.gba.gob.ar/dataset/4ee50a8f-6866-4191-9bf8-aebb30b0d718/resource/d3e23031-164a-43e0-b104-01ced63e4abd/download/estimaciones-agricolas-1969_2026.csv
- Archivo local: `datos/raw/estimaciones-agricolas-1969_2026.csv`
- Formato: CSV UTF-8, 2.959.265 bytes.
- Licencia: **CC BY 4.0**.
- Cobertura: campañas 1969/70 a 2025/26.
- Última actualización del catálogo: 14/08/2026.
- SHA-256 del archivo descargado: `a3b8e1295978343847324b2c3dff4d41b985f093372b34b1290cbc7ad75934e0`.

### Columnas originales

| Columna | Tipo | Descripción |
|---|---|---|
| `cultivo` | texto | Nombre del cultivo |
| `campania` | texto | Campaña agrícola, por ejemplo `2015/16` |
| `municipio_id` | entero | Código del municipio según la cartografía censal 2010 |
| `municipio_nombre` | texto | Nombre del partido o municipio |
| `superficie_sembrada` | entero | Hectáreas sembradas |
| `superficie_cosechada` | entero | Hectáreas cosechadas |
| `produccion` | entero | Toneladas producidas |
| `rendimiento` | entero | Kilogramos por hectárea |

### Controles de calidad realizados

- 53.905 filas, 31 cultivos, 152 nombres de municipio y 57 campañas.
- Sin duplicados por cultivo, campaña y municipio.
- Sin casos de superficie cosechada mayor que superficie sembrada.
- `rendimiento` coincide con `produccion × 1000 / superficie_cosechada` con una diferencia menor al 1% en todas las filas comparables.
- Los valores `0` de rendimiento coinciden con `superficie_cosechada = 0`; se interpretan como campaña sin cosecha, no como un rendimiento medido.

## Fuente de clima

**NASA POWER** — Prediction Of Worldwide Energy Resources.

- API: https://power.larc.nasa.gov/api/temporal/daily/point
- Variables: `PRECTOTCORR`, `T2M_MAX`, `T2M_MIN`, `ALLSKY_SFC_SW_DWN`, `RH2M`.
- Cobertura usada: 01/01/1981 a 31/03/2025.
- Resolución: reanálisis en grilla de aproximadamente 0,5°.
- Los valores `-999` se tratan como faltantes.
- La caché diaria se guarda en `datos/cache/clima/` y no se versiona por tamaño.

## Alcance del dataset de modelado

- Cultivos: maíz, trigo total, girasol y soja total.
- Campañas: 1981/82 a 2024/25.
- Se excluye la campaña 2025/26 porque está incompleta: 588 filas en total, de las cuales solo 102 son trigo.
- Se excluyen sorgo y cebada cervecera del alcance principal:
  - Sorgo tiene 418 filas con rendimiento cero sobre 4.106 (14,5%).
  - Cebada cervecera termina en la campaña 2015/16.
- Se excluye el municipio `Ignorado`.
- Se conservan los rendimientos cero como casos de pérdida total. En el dataset de modelado son pocos: 27 en train, 9 en validación y 10 en test.

## Decisiones de limpieza

1. **No usar producción ni superficie cosechada como predictoras.** El rendimiento es producción dividida por superficie cosechada; incluirlas sería fuga de información.
2. **No usar clima posterior a la fecha de corte.** Cada cultivo tiene una ventana climática distinta.
3. **Calcular el promedio de 5 campañas solo con campañas anteriores.** Se usa una grilla calendario y `shift(1).rolling(5, min_periods=5)`. Si falta un año, el promedio queda nulo y la fila no entra al modelado.
4. **Separar por campañas, no al azar.** Los partidos de una misma campaña comparten el mismo shock climático.
5. **No usar el test para elegir modelos o hiperparámetros.**

## Advertencias

- Los rendimientos son estimaciones oficiales por partido, no mediciones por lote.
- El clima de NASA POWER es reanálisis, no una estación local: suaviza extremos.
- El centroide del partido es una aproximación del área agrícola.
- La superficie sembrada del dataset es una estimación final; en un uso operativo habría que usar la estimación disponible a la fecha de corte.
- Los datos son agregados por partido: no contienen datos personales.

## Atribución

> Fuente: Ministerio de Economía de la Provincia de Buenos Aires, Secretaría de Agricultura, Ganadería y Pesca, Dirección de Estimaciones Agrícolas. Datos Abiertos de la Provincia de Buenos Aires, licencia CC BY 4.0.

> Fuente climática: NASA POWER (Prediction Of Worldwide Energy Resources), NASA Langley Research Center.
