# Distribución de tareas

**TP Integrador de Aprendizaje Automático**
**Proyecto:** rendimiento de cultivos por partido en la Provincia de Buenos Aires
**Repositorio:** https://github.com/GalvanAlexis/tp-integrador-rendimiento-cultivos

## Datos del trabajo

| Dato | Completar |
|---|---|
| Integrantes | Francisco Calahorra, Federico Valdés, Alexis Galván |
| Curso | Aprendizaje Automático 2026 |
| Fecha de inicio | 13/10/2026 |
| Fecha de entrega final | 28/10/2026 |

---

## 1. Cómo se organizó el trabajo

El grupo tiene **tres integrantes** y resolvió las cinco etapas **en conjunto**: cada entrega se discutió, se escribió y se revisó entre los tres, y las decisiones quedaron registradas en `informe/bitacora.md` con la alternativa descartada y el motivo.

Por eso este documento no reparte el proyecto en bloques cerrados por persona. La consigna pide que **todos puedan explicar cualquier parte del trabajo**, y el proyecto se armó con ese criterio: no hay una sección que solo entienda quien la escribió.

| Etapa | Qué se hizo | Dónde está |
|---|---|---|
| 1. Propuesta | Tres alternativas de problema, elección y justificación del umbral | `informe/entrega_1_propuesta.md`, issue #1 |
| 2. Datos y baselines | Descarga, limpieza, diccionario, clima y los dos baselines | `informe/entrega_2_preparacion_y_baselines.md`, PR #2 |
| 3. Regresión | Búsqueda de hiperparámetros, informe y análisis de residuos | `informe/entrega_3_regresion.md`, PR #3 |
| 4. Clasificación | Comparación de los cuatro clasificadores, umbral, errores y app local | `informe/entrega_4_clasificacion.md`, PR #7 a #9 |
| 5. Cierre | Despliegue, notebook integrador, informe final y presentación | PR #10 a #14, issue #6 |

La evidencia del trabajo conjunto está en el repositorio: los diez pull requests con sus descripciones, las cuatro issues y la bitácora, que tiene una entrada por cada decisión con las alternativas que se descartaron.

## 2. Reparto de la exposición

Los diez minutos se dividen en tres bloques de tres minutos y una demostración de un minuto. Cada bloque lo expone un integrante distinto y todos responden preguntas de cualquier parte.

| Bloque | Diapositivas | Tiempo | Contenido |
|---|---|---:|---|
| 1. Problema, datos y objetivo | 1 a 5 | 3 min | usuario y decisión, fuentes, split temporal, ratio, umbral 0,85, protocolo y baselines |
| 2. Modelos y métricas | 6 a 9 | 3 min | KNN y compresión hacia la media, árbol clasificador, tabla integradora y costo de los errores |
| 3. Errores, límites y cierre | 10 a 14 | 3 min | errores por banda de ratio, acuerdo entre sistemas, limitaciones, tres ideas finales |
| Demostración | 12 | 1 min | caso Pergamino / soja 2022-23 en la URL publicada |

La presentación está en `presentacion/presentacion.html`, con notas del orador en cada diapositiva (tecla `S`) y el tiempo estimado de cada bloque.

## 3. Preparación para la defensa individual

Antes de la exposición el grupo hizo dos pasadas de repaso:

1. **Recorrido del notebook integrador.** `notebooks/entrenamiento.ipynb` se ejecutó de punta a punta y cada uno siguió las cinco etapas sobre el mismo archivo.
2. **Preguntas cruzadas.** Cada integrante preparó dos preguntas de la sección 4.3 y de la lista de comprobación para hacérselas a otro, con el informe final abierto como respaldo.

| Tema que puede preguntar el docente | Dónde está la respuesta |
|---|---|
| Por qué se modela el ratio y no los kilos | Informe, sección 2.4 |
| Por qué el umbral es 0,85 y por qué el costo es 2 a 1 | Informe, secciones 1.2 y 3.4 |
| Cómo se evita usar el test para elegir | Informe, secciones 3.1 y 2.5 |
| Por qué el MAE es bueno y la clasificación mala | Informe, sección 7, pregunta 6 |
| Por qué no se eligió GaussianNB, que tuvo menor costo en test | Informe, sección 5.3 y sección 7 |
| Qué pasa con los casos cerca del umbral | Informe, secciones 4.5 y 6.4 |
| Cómo se despliega y qué carga la app al arrancar | Informe, sección 8, y `app.py` |
| Qué harían con más tiempo | Informe, sección 9, mejoras pendientes |

## 4. Uso del repositorio

- Un solo repositorio para las cinco entregas, con rama por issue y merge por squash: nunca se hizo push directo a `main`.
- Cada entrega tiene su tag: `entrega-1-propuesta`, `entrega-2-datos`, `entrega-3-regresion`, `entrega-4-clasificacion` y `entrega-5-final`.
- Los commits llevan la referencia al issue y los pull request describen qué se cambió y cómo se verificó.
