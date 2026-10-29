"""Experimento de calibración de las probabilidades del clasificador.

El informe de la Entrega 4 deja pendiente "calibrar las probabilidades del
clasificador con validación". Este módulo mide las dos variantes posibles y
deja la evidencia para decidir, sin tocar el clasificador entregado.

Variantes:

1. **Crudo (entregado).** Árbol con clima, profundidad 5 y hoja mínima 50,
   con el umbral 0,65 elegido en la búsqueda. Es el modelo que usa la app.
2. **Sigmoide con validación cruzada interna.** `CalibratedClassifierCV`
   (sigmoide, 5 pliegues) sobre train, umbral re-elegido en validación.
   Puede reentrenarse con train + validación, porque el calibrador se ajusta
   con pliegues internos.
3. **Sigmoide ajustada en validación.** Árbol entrenado con train y calibrador
   ajustado con validación (`FrozenEstimator`), umbral re-elegido en
   validación sobre la probabilidad calibrada.

Solo se reporta **validación**. El test no se vuelve a medir: no hay ninguna
decisión que tomarlo justifique, y la consigna exige medirlo una sola vez.

Lectura del resultado: la sigmoide aplicada a un único árbol es monótona, así
que la variante 3 reordena exactamente igual que la variante 1 y no puede
mejorar la decisión; su valor es interpretativo, pero exige reservar la
validación como conjunto de calibración y renunciar a reentrenar con ella. La
variante 2 sí cambia el ordenamiento, porque promedia cinco árboles distintos,
y en validación empeora el costo.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .clasificacion import (
    TARGET_CLF,
    UMBRAL_TEORICO,
    _clf_metrics,
    apply_threshold,
    build_model,
    feature_columns,
    load_data,
    split_data,
)
from .config import CLASIFICACION_CALIBRACION, CLASIFICADOR_METADATA

METRICAS = ["recall", "precision", "f1", "accuracy", "costo", "tp", "fp", "fn", "tn"]

# Calibración de las probabilidades del clasificador.
# El método sigmoide ajusta una regresión logística sobre la probabilidad cruda
# del modelo; con validación cruzada interna el calibrador se ajusta con
# pliegues de train y el modelo se puede reentrenar después con train + validación.
CALIBRACION_METODO = "sigmoid"
CALIBRACION_PLIEGUES = 5


def calibrate_model(model):
    """Calibra las probabilidades del pipeline con validación cruzada interna.

    Variante que sí se puede reentrenar con train + validación, porque el
    calibrador se ajusta con pliegues internos de train.
    """
    return CalibratedClassifierCV(
        estimator=model, method=CALIBRACION_METODO, cv=CALIBRACION_PLIEGUES
    )


def choose_threshold(proba: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    """Elige el umbral de menor costo en validación.

    Los candidatos son los valores distintos de la probabilidad: en esos puntos
    cambia la decisión, así que una rejilla fija de 0,05 podría saltearse el
    óptimo y perder recall por resolución.

    Ante empates de costo se prefiere el mayor F1 y, si el empate sigue, el
    umbral más cercano al teórico COST_FP / (COST_FP + COST_FN).
    """
    candidatos = np.unique(np.round(proba, 6))
    filas = [(u, _clf_metrics(y_true, apply_threshold(proba, u))) for u in candidatos]
    filas.sort(key=lambda f: (f[1]["costo"], -f[1]["f1"], abs(f[0] - UMBRAL_TEORICO)))
    return float(filas[0][0]), filas[0][1]


def delivered_config() -> dict:
    """Configuración del clasificador entregado, leída de sus metadatos."""
    meta = json.loads(CLASIFICADOR_METADATA.read_text(encoding="utf-8"))
    return {
        "familia": "arbol",
        "variables": meta["variables"],
        "params": meta["params"],
        "umbral": float(meta["umbral_decision"]),
        "modelo": meta["modelo"],
    }


def _row(nombre: str, metodo: str, calibrador: str, umbral: float, m: dict, rho: float) -> dict:
    return {
        "variante": nombre,
        "calibracion": metodo,
        "ajuste_calibrador": calibrador,
        "umbral": round(float(umbral), 6),
        **{f"{k}_val": m[k] for k in METRICAS},
        "monotonia_spearman": rho,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def evaluar_variantes() -> pd.DataFrame:
    """Mide las tres variantes en validación. El test no se toca."""
    cfg = delivered_config()
    df = load_data()
    train, val, _ = split_data(df)
    cols = feature_columns(cfg["variables"])
    y_train = train[TARGET_CLF].to_numpy()
    y_val = val[TARGET_CLF].to_numpy()

    base = build_model(cfg["familia"], cfg["variables"], cfg["params"]).fit(
        train[cols], y_train
    )
    proba_cruda = base.predict_proba(val[cols])[:, 1]
    filas = [
        _row(
            "crudo (entregado)",
            "ninguna",
            "-",
            cfg["umbral"],
            _clf_metrics(y_val, apply_threshold(proba_cruda, cfg["umbral"])),
            1.0,
        )
    ]

    interno = calibrate_model(
        build_model(cfg["familia"], cfg["variables"], cfg["params"])
    ).fit(train[cols], y_train)
    proba_interno = interno.predict_proba(val[cols])[:, 1]
    umbral_interno, m_interno = choose_threshold(proba_interno, y_val)
    filas.append(
        _row(
            "sigmoide con validación cruzada interna",
            "sigmoid",
            "5 pliegues sobre train",
            umbral_interno,
            m_interno,
            _spearman(proba_cruda, proba_interno),
        )
    )

    en_validacion = CalibratedClassifierCV(
        estimator=FrozenEstimator(base), method="sigmoid"
    ).fit(val[cols], y_val)
    proba_validacion = en_validacion.predict_proba(val[cols])[:, 1]
    umbral_validacion, m_validacion = choose_threshold(proba_validacion, y_val)
    filas.append(
        _row(
            "sigmoide ajustada en validación",
            "sigmoid",
            "validación",
            umbral_validacion,
            m_validacion,
            _spearman(proba_cruda, proba_validacion),
        )
    )

    return pd.DataFrame(filas)


def run(verbose: bool = True) -> pd.DataFrame:
    tabla = evaluar_variantes()
    CLASIFICACION_CALIBRACION.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CLASIFICACION_CALIBRACION, index=False, encoding="utf-8")

    if verbose:
        columnas = [
            "variante",
            "calibracion",
            "ajuste_calibrador",
            "umbral",
            "recall_val",
            "precision_val",
            "costo_val",
            "monotonia_spearman",
        ]
        print("Experimento de calibración (validación, el test no se mide)")
        print(tabla[columnas].round(4).to_string(index=False))
        print(
            "\nUmbral teórico de alerta con FN = 2 × FP: "
            f"{UMBRAL_TEORICO:.4f} (COST_FP / (COST_FP + COST_FN))"
        )
        print(f"Escrito en {CLASIFICACION_CALIBRACION}")
    return tabla


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara las variantes de calibración del clasificador en validación."
    )
    parser.add_argument("--quiet", action="store_true", help="No imprime tablas.")
    args = parser.parse_args()
    run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
