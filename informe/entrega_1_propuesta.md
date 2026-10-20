# Propuesta de proyecto - TP Integrador de Aprendizaje Automático (Entrega 1)

Oct 13, 2026 · @Francisco Calahorra

## Datos del trabajo

| Dato | A completar |
| --- | --- |
| Integrantes | Francisco Calahorra, Federico Valdés, Alexis Galván |
| Curso | Aprendizaje Automático 2026 |
| Fecha de inicio | 13/10/2026 |
| Fecha de entrega final | dd/mm/aaaa |

## Resumen de las alternativas

Las tres alternativas siguen el esquema de la consigna: una variable numérica, una alerta derivada de un umbral y dos sistemas (clasificador directo y regresión más umbral) que se comparan sobre los mismos casos de prueba.

| Alternativa | Variable numérica | Alerta y umbral | Fuente principal | Filas estimadas |
| --- | --- | --- | --- | --- |
| 1. Rendimiento de cultivos por partido | Rendimiento (kg/ha) | Campaña mala: menos del 85 % del promedio de las 5 campañas anteriores | Estimaciones agrícolas + clima NASA POWER | Cerca de 10.000 |
| 2. Demanda de una línea de colectivo | Transacciones SUBE por día | Alta demanda: por encima del percentil 90 de la línea | SUBE, Ministerio de Transporte | Más de 1.800 por línea |
| 3. Ocupación de alojamientos en Chascomús | Ocupación de plazas del fin de semana (%) | Ocupación alta: 80 % o más | A obtener de la Municipalidad o de alojamientos | A conseguir: 2.600 de ejemplo |

## Alternativa 1: Rendimiento de cultivos por partido, ¿campaña mala?

Se predice el rendimiento (kg/ha) de un cultivo en un partido bonaerense antes de la cosecha y se alerta si la campaña será mala. Es la alternativa con la fuente más sólida: serie pública de las campañas 1969/70 a 2025/26 con licencia CC BY 4.0.

| Elemento | Propuesta |
| --- | --- |
| Problema | Productores y asesores deben decidir insumos, cobertura, financiamiento y venta anticipada sin saber si la campaña se desviará a la baja. El clima es uno de los factores que más variación genera entre campañas. |
| Usuario de la predicción | Productores, asesores agronómicos, cooperativas, aseguradoras, bancos agropecuarios y áreas de producción de los municipios. |
| Decisión que se tomaría | Contratar o reforzar una cobertura de seguro, vender de forma anticipada una parte de la cosecha, gestionar financiamiento, y priorizar asistencia en los partidos con alerta. |
| Fila del dataset | Una combinación partido × cultivo × campaña (por ejemplo, soja en un partido, campaña 2015/16). |
| Variable numérica | Rendimiento (kg/ha). |
| Clase derivada y umbral | campaña\_mala = 1 si el rendimiento es menor al 85 % del promedio de las 5 campañas anteriores del mismo partido y cultivo. Es un umbral relativo: absorbe la tendencia tecnológica y las diferencias de suelo entre partidos. El 85 % es un valor inicial; en la Clase 2 se prueba con 80 % y 90 % y se elige uno que deje una clase positiva con suficientes casos. |
| Fuente | [Estimaciones agrícolas](https://catalogo.datos.gba.gob.ar/dataset/estimaciones-agro) (Datos Abiertos PBA, origen en la Secretaría de Agricultura, Ganadería y Pesca), licencia CC BY 4.0. Clima: [NASA POWER](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), datos diarios desde 1981. |
| Tamaño estimado | Del orden de 10.000 filas, a confirmar en la Clase 2. Estimación propia: unos 80 partidos × 3 a 4 cultivos × unas 40 campañas desde 1981, el primer año con clima. |

### Momento de la predicción

La predicción se hace en una fecha de corte fija por cultivo, al final de su período crítico y antes de la cosecha. Fechas provisorias, a validar con material del INTA o con un asesor: trigo 15/11, maíz 31/01, girasol 31/01, soja 28/02. Solo se usan datos disponibles hasta esa fecha.

### Variables predictoras

| Variable | Origen | Disponible al predecir |
| --- | --- | --- |
| Cultivo | Estimaciones agrícolas | Desde la siembra |
| Partido (categoría) y coordenadas del centroide | Estimaciones agrícolas y datos geográficos oficiales | Siempre |
| Campaña (año de inicio) | Estimaciones agrícolas | Siempre |
| Superficie sembrada (ha) | Estimaciones agrícolas | Desde la siembra |
| Rendimiento de la campaña anterior y promedio de las 5 anteriores | Estimaciones agrícolas (derivadas) | Antes de la siembra |
| Precipitación acumulada hasta la fecha de corte (mm) | NASA POWER | A la fecha de corte |
| Temperatura máxima media y días con máxima mayor a 30 °C | NASA POWER | A la fecha de corte |
| Radiación solar acumulada | NASA POWER | A la fecha de corte |
| Humedad relativa media | NASA POWER | A la fecha de corte |

Cumple el mínimo de cinco predictores. La app pedirá al usuario los valores climáticos acumulados en un formulario con rangos validados, para no depender de una consulta externa en cada predicción.

### Fuga de información a evitar

- No usar producción ni superficie cosechada: el rendimiento es producción dividida por superficie cosechada, así que la respuesta quedaría en las entradas.
- No usar clima posterior a la fecha de corte.
- Calcular el promedio de 5 campañas solo con campañas anteriores a la fila.
- Separar entrenamiento, validación y prueba por campañas (por ejemplo, el test con las últimas seis). Un reparto al azar mezclaría partidos vecinos de la misma campaña, que comparten casi el mismo clima.

### Costos de los errores

| Error | Significado | Consecuencia |
| --- | --- | --- |
| Falso negativo | La campaña fue mala y no hubo alerta | Productor sin cobertura o con ventas anticipadas comprometidas por encima de lo que cosecha. Pérdida alta. |
| Falso positivo | Hubo alerta y la campaña fue normal | Prima de cobertura o venta anticipada innecesarias. Costo moderado. |

Hipótesis de trabajo: un falso negativo cuesta 3 veces un falso positivo. En la Clase 2 el grupo la justifica con valores de referencia (costo de una cobertura frente al margen bruto perdido por hectárea) y analiza la sensibilidad con relaciones 2:1 y 5:1.

### Riesgos y limitaciones

- Los rendimientos son estimaciones oficiales por partido, no mediciones por lote.
- El clima de NASA POWER es de reanálisis sobre una grilla de aproximadamente 0,5°, no de estación: suaviza extremos locales.
- Puede haber partidos o cultivos con huecos o pocas campañas; se revisa en la Clase 2.
- Las campañas malas son minoría: hay que medir el desbalance y no confiar en la accuracy sola.
- Causas no climáticas (plagas, precios, decisiones de manejo) no están en el modelo.
- La app no reemplaza el asesoramiento agronómico ni sirve para lotes individuales.
- Los datos son agregados por partido: no hay datos personales.

## Alternativa 2: Transporte, ¿una línea de colectivo tendrá alta demanda?

Se predice cuántas transacciones SUBE tendrá una línea de colectivo al día siguiente y se alerta si la demanda superará un nivel alto, para reforzar frecuencias a tiempo. La fuente es oficial, diaria y de gran volumen; lo que hay que cerrar en la Clase 2 es qué líneas usar y qué columnas trae realmente cada archivo.

| Elemento | Propuesta |
| --- | --- |
| Problema | Las empresas y los municipios planifican frecuencias con poca anticipación. Un día de demanda alta sin refuerzo deja unidades saturadas y esperas largas; un refuerzo sin demanda es costo operativo desperdiciado. |
| Usuario de la predicción | Empresa operadora de la línea y área de transporte del municipio. |
| Decisión que se tomaría | Despachar o no unidades adicionales en la línea al día siguiente. |
| Fila del dataset | Una línea × un día. |
| Variable numérica | Transacciones SUBE de la línea en el día. |
| Clase derivada y umbral | alta\_demanda = 1 si las transacciones superan el percentil 90 de esa línea, calculado solo con el período de entrenamiento. Alternativa a evaluar: 110 % de la mediana del mismo día de la semana en las últimas 8 semanas. Se elige en la Clase 2. |
| Fuente | [SUBE: cantidad de transacciones (usos) por fecha](https://datos.transporte.gob.ar/dataset/sube-cantidad-de-transacciones-usos-por-fecha), Ministerio de Transporte (Dirección Nacional de Desarrollo Tecnológico). Un CSV por año (2020 a 2026), actualización diaria, CC BY 4.0. |
| Tamaño estimado | Más de 1.800 filas por línea desde 2021 (estimación propia). Con 5 a 10 líneas de una misma ciudad se supera ampliamente el mínimo de 2.000. |

La página oficial no lista las columnas. El [espejo de Alphacast](https://preview.alphacast.io/datasets/activity-production-argentina-mt-number-of-transactions-sube-42530) las describe como empresa, línea, tipo de transporte, jurisdicción, provincia y municipio, de enero de 2020 a noviembre de 2024. Lo primero que se hace es abrir un CSV original y confirmar que tiene esas columnas.

### Momento de la predicción

La tarde del día anterior, con datos hasta el cierre de ese día, para decidir el refuerzo del día siguiente.

### Variables predictoras

| Variable | Origen | Disponible al predecir |
| --- | --- | --- |
| Día de la semana | Calendario | Siempre |
| Feriado o día puente (sí/no) | Calendario oficial | Siempre |
| Mes y período de vacaciones | Calendario | Siempre |
| Transacciones del mismo día de la semana anterior | SUBE | D−7 |
| Promedio de transacciones de los últimos 7 días | SUBE | Hasta D−1 |
| Transacciones del día anterior | SUBE | D−1 |
| Línea (categoría) | SUBE | Siempre |
| Lluvia y temperatura | NASA POWER | Se evalúa; ver limitaciones |

Cumple el mínimo de cinco predictores sin necesidad del clima.

### Fuga de información a evitar

- No usar transacciones del propio día ni de otras líneas del mismo día.
- No calcular el percentil del umbral con validación o prueba.
- Separar por fechas y no al azar: los días consecutivos son muy parecidos. Por ejemplo, entrenamiento 2021 a 2024, validación 2025 y prueba 2026.

### Costos de los errores

| Error | Significado | Consecuencia |
| --- | --- | --- |
| Falso negativo | Día de alta demanda sin alerta | Unidades saturadas, esperas largas, pasajeros que no logran viajar. |
| Falso positivo | Alerta en un día normal | Unidades adicionales con combustible y chofer sin pasajeros que las justifiquen. |

Hipótesis de trabajo: un falso negativo cuesta el doble que un falso positivo. En la Clase 2 se justifica con el costo de un servicio adicional frente al impacto sobre los pasajeros, y se prueban relaciones 1:1 y 3:1.

### Riesgos y limitaciones

- Quiebre del año 2020 por la pandemia: se excluye o se marca con una variable.
- Cambios de tarifa, de recorridos, líneas dadas de baja o renumeradas y huecos por fallas de validadoras.
- Paros de transporte y eventos puntuales que no se pueden anticipar con estas variables.
- Las transacciones SUBE no equivalen a pasajeros: no incluyen a quienes no validan.
- La demanda de ayer y la de la semana pasada ya explican gran parte de la de mañana. El informe compara contra el baseline Dummy y también contra la regla “igual que el mismo día de la semana anterior”, para mostrar qué aporta realmente el modelo.
- El clima de NASA POWER es observado, no pronóstico. Usarlo como si se conociera de antemano sobrestima el desempeño. Se declara como limitación o se lo reemplaza por la climatología del mes.
- Datos agregados por línea y día: no hay datos personales.

## Alternativa 3: Turismo y ocupación hotelera en Chascomús

Se predice la ocupación de los alojamientos de Chascomús en un fin de semana y se alerta si será alta, con dos semanas de anticipación. Es la más cercana a la realidad local, pero hoy es la más débil en datos: no se encontró una serie pública de ocupación de Chascomús, así que depende de conseguirla.

| Elemento | Propuesta |
| --- | --- |
| Problema | El turismo en Chascomús se concentra en fines de semana largos y vacaciones. En el fin de semana largo de octubre de 2016 la Municipalidad informó [ocupación casi total](https://www.lacapitalmdp.com/ocupacion-plena-en-chascomus-durante-el-fin-de-semana-largo/): cabañas 100 %, hoteles 98 %, estancias 90 % y campings 80 %. Alojamientos, comercios y servicios municipales deben dimensionar personal, stock y operativos con poca anticipación. |
| Usuario de la predicción | Dueños y gerentes de alojamientos, Dirección de Turismo, comercios gastronómicos y áreas municipales de tránsito, limpieza y seguridad. |
| Decisión que se tomaría | Dos semanas antes: contratar personal, reforzar stock, ajustar tarifas o cerrar reservas, y planificar operativos municipales. |
| Fila del dataset | Un alojamiento × un fin de semana (viernes a domingo). Variante: un alojamiento × un día. |
| Variable numérica | Tasa de ocupación de plazas del fin de semana (%): plazas ocupadas sobre plazas disponibles. |
| Clase derivada y umbral | ocupación\_alta = 1 si la tasa es de 80 % o más. Valor provisorio: se justifica con el criterio del sector (nivel desde el cual un alojamiento sube tarifas o cierra reservas) y con la distribución real de los datos. |
| Fuente | A obtener. Rutas posibles: (1) pedido formal a la Dirección de Turismo municipal, que ya difunde ocupación por tipo de alojamiento; (2) acuerdo con alojamientos o con una asociación del sector; (3) como alternativa de respaldo, la [Encuesta de Ocupación Hotelera del INDEC](https://www.indec.gob.ar/uploads/informesdeprensa/eoh_08_19286493D896.pdf), mensual desde 2006, que no incluye Chascomús. |
| Tamaño necesario | Mínimo 2.000 filas. Ejemplo: 25 alojamientos × 104 semanas (2 años) = 2.600 filas. |

### Estado de los datos

- La EOH del INDEC relevó 48 localidades en su informe de agosto de 2019. En la provincia de Buenos Aires solo figuran Bahía Blanca, Mar del Plata, Pinamar, Tandil y Villa Gesell. Chascomús no está.
- En la búsqueda realizada, lo único público y específico de Chascomús fueron informes de prensa con cifras puntuales de la Municipalidad, sin serie descargable.
- Antes de pedir la aprobación final, el grupo necesita una respuesta concreta de la Dirección de Turismo o de al menos 20 alojamientos sobre qué datos pueden entregar y con qué frecuencia.

### Momento de la predicción

Catorce días antes del viernes del fin de semana a predecir. Solo se usan datos conocidos a esa fecha.

### Variables predictoras

| Variable | Origen | Disponible al predecir |
| --- | --- | --- |
| Tipo de alojamiento (hotel, cabaña, estancia, camping) | Alojamiento o Dirección de Turismo | Siempre |
| Plazas disponibles | Alojamiento | Siempre |
| Fin de semana largo o feriado puente | Calendario oficial | Siempre |
| Vacaciones de invierno o verano | Calendario | Siempre |
| Mes y semana del año | Calendario | Siempre |
| Eventos programados en la ciudad | Agenda municipal | Si ya están anunciados |
| Ocupación del mismo fin de semana del año anterior | Histórico del alojamiento | Siempre |
| Ocupación promedio de las 4 semanas previas a la fecha de corte | Histórico del alojamiento | Hasta la fecha de corte |
| Reservas confirmadas a 14 días (opcional) | Alojamiento | A la fecha de corte |

Si el clima se incorpora, se usa pronóstico o climatología del mes: el clima observado del fin de semana no se conoce 14 días antes.

### Fuga de información a evitar

- No usar plazas ocupadas, ocupación del propio fin de semana ni reservas posteriores a la fecha de corte.
- Medir las reservas siempre a la misma distancia de la fecha (14 días).
- Separar por fechas: entrenamiento con los primeros meses, validación y prueba con los fines de semana más recientes.

### Costos de los errores

| Error | Significado | Consecuencia |
| --- | --- | --- |
| Falso negativo | Ocupación alta sin alerta | Falta de personal y stock, tarifas por debajo de lo posible, servicios municipales saturados. |
| Falso positivo | Alerta y ocupación normal | Personal y stock de más, descuentos o reservas cerradas sin necesidad. |

Hipótesis de trabajo: un falso negativo cuesta el doble que un falso positivo. Se justifica en la Clase 2 con datos de tarifa y costo de personal que aporten los alojamientos.

### Riesgos y limitaciones

- Riesgo principal: sin acuerdo de datos, no se llega a 2.000 registros ni a cinco predictores. Con esta alternativa aprobada y sin datos, el proyecto queda bloqueado.
- Información comercial sensible. Hace falta acuerdo escrito, códigos en lugar de nombres de alojamientos y solo ocupación agregada. Sin datos de huéspedes, para cumplir la condición de privacidad.
- Pocos fines de semana largos por año: la clase positiva puede concentrarse en pocas fechas y el modelo puede limitarse a aprender el calendario.
- La pandemia de 2020 y 2021 distorsiona el histórico.
- Criterios dispares entre alojamientos para medir plazas y ocupación.
- Un modelo de una ciudad pequeña no se generaliza a otros destinos.

## Comparación y recomendación

Se recomienda continuar con la Alternativa 1: combina fuente pública verificada, licencia abierta, volumen suficiente y una decisión concreta, y completa los cinco predictores con un clima disponible desde 1981. La Alternativa 2 también tiene fuente pública y queda como segunda opción; la 3 depende de datos que hoy no son públicos.

| Condición de la consigna | Alternativa 1 | Alternativa 2 | Alternativa 3 |
| --- | --- | --- | --- |
| Necesidad y decisión concreta | Cumple | Cumple | Cumple |
| Variable numérica y clase derivada | Cumple | Cumple | Cumple |
| Al menos 2.000 registros | Cumple (unos 10.000, a confirmar) | Cumple con varias líneas | En riesgo: depende de conseguir datos |
| Al menos 5 predictores disponibles al predecir | Cumple | Cumple | A verificar con los datos recibidos |
| Umbral justificable | Cumple, a calibrar | Cumple, a elegir | Cumple, 80 % provisorio |
| Fuente documentada | Cumple, verificada | Cumple, columnas a confirmar | No resuelta |
| Datos que requieren preparación | Unión con clima, centroides y ventanas de 5 campañas | Calendario, rezagos, huecos y 2020 | Depende de lo que se reciba |
| Falsos positivos y negativos con significado | Cumple | Cumple | Cumple |
| Sin datos personales | Cumple, agregado por partido | Cumple, agregado por línea y día | A resguardar con acuerdo y códigos |
| Proyecto permitido (no repite los casos ya trabajados) | Cumple | Cumple, distinto de entregas | Cumple |

### Por qué la Alternativa 1

- Es la única cuyas variables, licencia y cobertura (campañas 1969/70 a 2025/26) quedaron comprobadas en las páginas oficiales, y el clima de NASA POWER cubre desde 1981.
- El test por campañas puede incluir años de sequía y campañas normales, lo que permite estudiar los casos cercanos al umbral.
- Tiene un usuario y una decisión claros, con un costo asimétrico fácil de explicar en la defensa.
- Para la defensa se puede mostrar un partido cercano a Chascomús, o Chascomús mismo si el dataset tiene registros de cultivos (a verificar).

### Por qué la Alternativa 2 es segunda

La fuente es oficial y diaria, pero el aporte del modelo es más difícil de mostrar: la regla “igual que el mismo día de la semana anterior” ya es un baseline fuerte. Además, las columnas del archivo original todavía no están confirmadas.

### Por qué la Alternativa 3 depende de un acuerdo

Es la de mayor valor local, pero solo se puede aprobar si antes se consigue el compromiso de datos. Como el trabajo continúa únicamente con el proyecto aprobado, aprobarla sin datos dejaría al grupo bloqueado.

## Verificaciones pendientes y fuentes

Antes de la Clase 2 hay que comprobar con los datos reales lo que esta propuesta solo estima.

- [ ] Alternativa 1: descargar el CSV de estimaciones agrícolas, contar filas de la provincia de Buenos Aires por cultivo y campaña, y revisar huecos.
- [ ] Alternativa 1: verificar si hay registros de cultivos del partido de Chascomús.
- [ ] Alternativa 1: descargar el clima de NASA POWER para un partido de prueba. Se pide una vez por centroide y se guarda el resultado, porque el servicio advierte que puede bloquear pedidos repetidos del mismo lugar.
- [ ] Alternativa 1: validar con material del INTA o con un asesor las fechas de corte por cultivo.
- [ ] Alternativa 2: abrir un CSV anual de SUBE, confirmar columnas (línea, municipio, tipo de transporte) y días disponibles hasta 2026.
- [ ] Alternativa 2: elegir de 5 a 10 líneas de una misma ciudad.
- [ ] Alternativa 3: pedir por escrito a la Dirección de Turismo de Chascomús y a los alojamientos ocupación semanal o diaria de al menos dos años, sin datos de huéspedes.
- [ ] Completar la tabla de datos del trabajo al comienzo de este documento.

### Fuentes consultadas el 13/10/2026

- [Estimaciones agrícolas, Datos Abiertos de la Provincia de Buenos Aires](https://catalogo.datos.gba.gob.ar/dataset/estimaciones-agro)
- [Estimaciones agrícolas, Secretaría de Agricultura, Ganadería y Pesca](https://datosestimaciones.magyp.gob.ar)
- [NASA POWER, API de datos diarios](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
- [API Georef, partidos de Buenos Aires con centroides](https://apis.datos.gob.ar/georef/api/departamentos?provincia=06&max=3&campos=nombre,centroide.lat,centroide.lon)
- [SUBE: cantidad de transacciones (usos) por fecha, Ministerio de Transporte](https://datos.transporte.gob.ar/dataset/sube-cantidad-de-transacciones-usos-por-fecha)
- [Transacciones SUBE, espejo de Alphacast](https://preview.alphacast.io/datasets/activity-production-argentina-mt-number-of-transactions-sube-42530)
- [INDEC, Encuesta de Ocupación Hotelera, informe de agosto de 2019](https://www.indec.gob.ar/uploads/informesdeprensa/eoh_08_19286493D896.pdf)
- [La Capital MdP: ocupación plena en Chascomús en el fin de semana largo (2016)](https://www.lacapitalmdp.com/ocupacion-plena-en-chascomus-durante-el-fin-de-semana-largo/)
