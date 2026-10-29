# Mejoras post-Entrega 5: calibración, CI y cierre del checklist

Rama: `chore/ISS-7-cierre-tecnico`. Entrega objetivo: 5 (cierre técnico sin cambiar el alcance académico).
Runner: `python -m pytest` (Python 3.14.2, scikit-learn 1.9.0). Flujo git: rama por issue, commits convencionales, sin push ni PR sin decisión del usuario.

## Specs

- **S1** El repositorio sigue verde: `python -m pytest` pasa completo después de cada cambio.
- **S2** Antes de adoptar una calibración se mide su efecto en validación. Solo se adopta si el costo de validación no empeora; si no mejora, `modelos/clasificador.joblib` queda igual, la evidencia se guarda en `datos/procesado/clasificacion_calibracion_comparacion.csv` y el texto publicado explica el resultado. El test no se vuelve a medir.
- **S3** La app y la presentación no afirman que la calibración está pendiente: dicen que se evaluó y qué pasó, con la advertencia de que la probabilidad sigue sin calibrar y sirve para ordenar casos.
- **S4** Los números publicados (informe final, entrega 4, README, presentación) coinciden con los artefactos regenerados, y el PDF del informe y el de la presentación se regeneran.
- **S5** Existe integración continua: `.github/workflows/ci.yml` instala `requirements-dev.txt` y ejecuta `python -m pytest` en cada push y pull request sobre cualquier rama.
- **S6** Existe un experimento documentado de modelos por grupo de cultivos (verano: maíz, soja, girasol; invierno: trigo), evaluado con validación. No reemplaza el modelo entregado salvo que mejore el costo.
- **S7** El checklist final refleja lo efectivamente verificado: la URL pública responde desde una máquina distinta del entorno de build, y la prueba desde un teléfono queda marcada como pendiente de confirmación del usuario, sin declararla cumplida.
- **S8** Las ramas locales residuales se eliminan sin perder trabajo: las que existen en `origin` se recuperan con `git branch <nombre> origin/<nombre>`, y la única que no estaba publicada se archiva en un `git bundle` fuera del repositorio antes de borrarla.

## Tareas

| ID | Spec | Ruta | Estado | Evidencia |
|---|---|---|---|---|
| T1 | S7 | inline | hecho | `/health` responde `{"status":"ok","entrega":5,"regresor":true,"clasificador":true}`; checklist en «Parcial» con la prueba desde teléfono como pendiente |
| T2 | S1, S2 | inline | hecho | `src/calibracion.py` + `tests/test_calibracion.py`; clasificador entregado sin cambios (umbral 0,65, costo val 443, test 687) |
| T3 | S3, S4 | inline | hecho | 2 plantillas y 4 líneas de `presentacion/plantilla.html`; `presentacion.html`, `presentacion.pdf` e `informe_final.pdf` regenerados |
| T4 | S5 | inline | hecho | `.github/workflows/ci.yml`; `python -m pytest -q` → 53 passed |
| T5 | S6 | inline | hecho | `src/grupos.py` + `tests/test_grupos.py`; `clasificacion_por_grupo_cultivo.csv` |
| T6 | S8 | inline | hecho | 9 ramas locales borradas; bundle de la no publicada verificado antes de borrar |

## Log

- **L1** Pedido del usuario, literal: «ok solucionemos todo esto y luego te pedire otra cosa», referido a esta lista propuesta por el agente: «1. Cerrar el punto 11 (trivial). 2. Calibrar el clasificador con `CalibratedClassifierCV` — ya está listado como mejora pendiente y resuelve la incoherencia de la probabilidad mostrada. 3. Un workflow mínimo de CI con `pytest` sobre `requirements-dev.txt` — el repo ya tiene 40 tests que hoy nadie ejecuta automáticamente. 4. Recién después, modelos por grupo de cultivos (verano/invierno)», más la higiene señalada: «no hay CI (`.github` no existe, `pytest` corre solo a mano) y quedaron 15 ramas locales residuales».
- **L2** Línea de base observada antes de tocar nada: `main` limpio en `b29f645`, `python --version` 3.14.2, `scikit-learn` 1.9.0, `pandas` 3.0.2, `numpy` 2.4.2, `python -m pytest -q` → 45 passed en 7,19 s.
- **L3** Hallazgo de diseño para T2: la calibración sigmoidea o isotónica es monótona respecto de la probabilidad de la hoja del árbol, así que no puede mejorar el ordenamiento ni el recall. Su efecto es interpretativo: la probabilidad mostrada pasa a ser una probabilidad, y el umbral óptimo se acerca al teórico `COST_FP / (COST_FP + COST_FN) = 1/3`.
- **L4** Contexto para T5: el modelo ya incluye `cultivo` como variable categórica con `OneHotEncoder`, de modo que un árbol puede separar por cultivo. El experimento por grupo evalúa si imponer la separación aporta algo por encima de eso.
- **L5** Cambio de alcance en S2, decidido con evidencia: se implementó la calibración completa (`CalibratedClassifierCV`, umbral re-elegido sobre la probabilidad calibrada) y se descartó. La variante ajustada en validación (`FrozenEstimator`) da umbral 0,376 —casi el teórico 1/3—, correlación de rango 1,00 con el crudo y costo 443 idéntico, pero obliga a entrenar el árbol solo con train y rompe la comparabilidad con la regresión. La variante con validación cruzada interna sí se puede reentrenar, pero promedia cinco árboles (correlación 0,57) y empeora: costo 512, recall 0,100 contra 0,248. El clasificador entregado se restauró y se verificó que `modelos/clasificador.joblib` quedó byte a byte igual.
- **L6** S2 se reescribió como «medir antes de adoptar» en lugar de «aplicar calibración»: la spec original pedía un cambio que empeora el modelo. La decisión y los números quedan en el informe final §5.5, entrega 4 §12, README y bitácora.
- **L7** `choose_threshold` usa como candidatos los valores únicos de la probabilidad en validación, no la rejilla fija de 0,05: con la rejilla el umbral se quedaba en 0,55 y perdía recall por resolución. Vive en `src/calibracion.py`, igual que `calibrate_model`, para no cargar el módulo entregado con código que no usa.
- **L8** T5: separar por grupo empeora el costo en los dos casos (verano 400 contra 307, invierno 148 contra 136, total 548 contra 443). El árbol ya one-hotea `cultivo` y entrenar por grupo le quita filas para estimar el clima. No se adopta.
- **L9** T6: 8 ramas existían en `origin` (recuperables con `git branch <nombre> origin/<nombre>`), una no (`backup/ISS-5-clasificacion-local`). Esa se archivó primero en `C:\Users\PC Blado\AppData\Local\Temp\opencode\backup-ISS-5-clasificacion-local.bundle`, verificado con `git bundle verify` («complete history», commit `c398d3e`), y recién después se borró. Quedan `main` y la rama de trabajo.
- **L10** Verificación final: `python -m pytest -q` → 53 passed; `python -m src.clasificacion --quiet` → exit 0 con el joblib idéntico; `python -m src.calibracion` y `python -m src.grupos` regeneran sus CSV; `presentacion/construir_presentacion.py --pdf` → 14 diapositivas; `informe/construir_pdf.py` → PDF regenerado.
