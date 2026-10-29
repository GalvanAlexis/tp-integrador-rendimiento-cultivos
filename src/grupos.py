"""Experimento de modelos por grupo de cultivos.

Mejora pendiente del informe: "evaluar modelos por grupo de cultivos (verano e
invierno), donde el clima tiene un efecto distinto". El clasificador entregado
ya recibe `cultivo` como variable categórica, así que un árbol puede separar
por cultivo sin ayuda. Este módulo mide si imponer la separación aporta algo
por encima de eso.

Comparación, siempre en **validación** (el test no se mide):

1. **Global.** La configuración entregada (árbol con clima, profundidad 5,
   hoja mínima 50) entrenada con todos los cultivos y medida en las filas de
   cada grupo.
2. **Por grupo.** La misma configuración entrenada solo con las filas del
   grupo, con el umbral re-elegido en la validación de ese grupo.

Grupos:

- **Verano:** maíz, soja total y girasol.
- **Invierno:** trigo total.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .calibracion import choose_threshold, delivered_config
from .clasificacion import (
    TARGET_CLF,
    _clf_metrics,
    apply_threshold,
    build_model,
    feature_columns,
    load_data,
    split_data,
)
from .config import CLASIFICACION_GRUPOS

GRUPOS = {
    "verano": ["Maíz", "Soja total", "Girasol"],
    "invierno": ["Trigo total"],
}

METRICAS = ["recall", "precision", "f1", "accuracy", "costo", "tp", "fp", "fn", "tn"]


def _fila(grupo: str, cultivos: list[str], variante: str, umbral: float, m: dict) -> dict:
    return {
        "grupo": grupo,
        "cultivos": ", ".join(cultivos),
        "variante": variante,
        "umbral": round(float(umbral), 6),
        "positivos_val": m["tp"] + m["fn"],
        **{f"{k}_val": m[k] for k in METRICAS},
    }


def evaluar_grupos() -> pd.DataFrame:
    """Entrena y mide el modelo global y el de cada grupo. Solo validación."""
    cfg = delivered_config()
    df = load_data()
    train, val, _ = split_data(df)
    cols = feature_columns(cfg["variables"])
    umbral = cfg["umbral"]

    global_model = build_model(cfg["familia"], cfg["variables"], cfg["params"]).fit(
        train[cols], train[TARGET_CLF].to_numpy()
    )

    filas = []
    for grupo, cultivos in GRUPOS.items():
        val_grupo = val[val["cultivo"].isin(cultivos)]
        train_grupo = train[train["cultivo"].isin(cultivos)]
        if val_grupo.empty or train_grupo.empty:
            continue
        y_val = val_grupo[TARGET_CLF].to_numpy()

        proba_global = global_model.predict_proba(val_grupo[cols])[:, 1]
        filas.append(
            _fila(
                grupo,
                cultivos,
                "global (entregado)",
                umbral,
                _clf_metrics(y_val, apply_threshold(proba_global, umbral)),
            )
        )

        por_grupo = build_model(cfg["familia"], cfg["variables"], cfg["params"]).fit(
            train_grupo[cols], train_grupo[TARGET_CLF].to_numpy()
        )
        proba_grupo = por_grupo.predict_proba(val_grupo[cols])[:, 1]
        umbral_grupo, m_grupo = choose_threshold(proba_grupo, y_val)
        filas.append(_fila(grupo, cultivos, "por grupo", umbral_grupo, m_grupo))

    return pd.DataFrame(filas)


def run(verbose: bool = True) -> pd.DataFrame:
    tabla = evaluar_grupos()
    CLASIFICACION_GRUPOS.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CLASIFICACION_GRUPOS, index=False, encoding="utf-8")

    if verbose:
        columnas = [
            "grupo",
            "cultivos",
            "variante",
            "umbral",
            "positivos_val",
            "recall_val",
            "precision_val",
            "costo_val",
        ]
        print("Modelos por grupo de cultivos (validación, el test no se mide)")
        print(tabla[columnas].round(4).to_string(index=False))
        print(f"\nEscrito en {CLASIFICACION_GRUPOS}")
    return tabla


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara el modelo global con un modelo por grupo de cultivos en validación."
    )
    parser.add_argument("--quiet", action="store_true", help="No imprime tablas.")
    args = parser.parse_args()
    run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
