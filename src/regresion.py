"""Regresión del rendimiento relativo para la Entrega 3.

Protocolo:
1. Búsqueda de hiperparámetros: se entrena con train (1981/82 a 2013/14) y se
   mide en validación (2014/15 a 2018/19). El test no interviene.
2. Para cada familia de modelos se elige la configuración con menor MAE del
   ratio en validación. El regresor final es la mejor de todas.
3. Las configuraciones elegidas se reentrenan con train + validación y se
   evalúan una sola vez en test (2019/20 a 2024/25).

Modelos:
- Dummy por cultivo: media del ratio de entrenamiento de cada cultivo.
- Promedio 5 campañas: ratio = 1.
- DecisionTreeRegressor con distintas profundidades y hojas mínimas.
- KNeighborsRegressor con StandardScaler dentro del Pipeline.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .baselines import classification_metrics, regression_metrics
from .config import (
    CROPS,
    DATASET_MODELADO_CLIMA,
    MODELOS_DIR,
    RANDOM_STATE,
    REGRESION_BUSQUEDA,
    REGRESION_COMPARACION,
    REGRESION_POR_CAMPANIA,
    REGRESION_POR_CULTIVO,
    REGRESION_PREDICCIONES,
    REGRESOR_FILE,
    REGRESOR_METADATA,
    THRESHOLD,
)

TARGET = "ratio"

# Predictores disponibles en la fecha de corte. No se usa campania_inicio: el
# ratio ya no tiene tendencia y en test el año queda fuera del rango visto en
# entrenamiento. Tampoco municipio_id: su nivel productivo ya está en prev5.
NUM_AGRO = ["prev5", "prev5_ceros"]
LOG_COLS = ["superficie_sembrada"]
CAT_COLS = ["cultivo"]
CLIMA_COLS = [
    "clima_precip_total_mm",
    "clima_dias_tmax_30",
    "clima_tmax_media_c",
    "clima_tmin_media_c",
    "clima_humedad_media_pct",
    "clima_sequia_max_dias",
    "clima_precip_pre_siembra_mm",
    "clima_dias_helada",
    "clima_dias_lluvia",
]

FEATURE_SETS = {
    "sin_clima": NUM_AGRO,
    "con_clima": NUM_AGRO + CLIMA_COLS,
}

# Columnas que nunca pueden ser predictoras: objetivo, derivadas del objetivo o
# datos que solo existen después de la cosecha.
FORBIDDEN = {
    "rendimiento",
    "ratio",
    "mala80",
    "mala85",
    "mala90",
    "perdida_total",
    "produccion",
    "superficie_cosechada",
}

TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 15, None]
TREE_MIN_LEAF = [1, 10, 50, 100]
KNN_K = [1, 3, 5, 10, 15, 25, 50, 75, 100, 150, 200, 300, 500, 800]
KNN_WEIGHTS = ["uniform", "distance"]

MODEL_LABELS = {
    "dummy": "Dummy por cultivo",
    "prev5": "Promedio 5 campañas",
    "arbol": "DecisionTreeRegressor",
    "knn": "KNeighborsRegressor",
}


def feature_columns(feature_set: str) -> list[str]:
    return CAT_COLS + LOG_COLS + FEATURE_SETS[feature_set]


def load_data() -> pd.DataFrame:
    if not DATASET_MODELADO_CLIMA.exists():
        raise FileNotFoundError(
            f"No existe {DATASET_MODELADO_CLIMA}. Ejecutá primero: python -m src.clima --merge"
        )
    df = pd.read_csv(DATASET_MODELADO_CLIMA, encoding="utf-8")
    df["mala85"] = df["mala85"].astype(int)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        df[df["split"] == "train"].copy(),
        df[df["split"] == "validacion"].copy(),
        df[df["split"] == "test"].copy(),
    )


def _preprocessor(feature_set: str, scale: bool) -> ColumnTransformer:
    num = FEATURE_SETS[feature_set]
    log_steps = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]
    if scale:
        log_steps.append(("escala", StandardScaler()))
    num_transformer = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cultivo", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("superficie", Pipeline(log_steps), LOG_COLS),
            ("numericas", num_transformer, num),
        ],
        verbose_feature_names_out=False,
    )


def make_tree(feature_set: str, max_depth: int | None, min_samples_leaf: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(feature_set, scale=False)),
            (
                "modelo",
                DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_knn(feature_set: str, n_neighbors: int, weights: str) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(feature_set, scale=True)),
            ("modelo", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)),
        ]
    )


def build_model(familia: str, feature_set: str, params: dict) -> Pipeline:
    if familia == "arbol":
        return make_tree(feature_set, params["max_depth"], params["min_samples_leaf"])
    if familia == "knn":
        return make_knn(feature_set, params["n_neighbors"], params["weights"])
    raise ValueError(f"Familia desconocida: {familia}")


def candidates() -> list[tuple[str, str, dict]]:
    out: list[tuple[str, str, dict]] = []
    for feature_set in FEATURE_SETS:
        for depth in TREE_DEPTHS:
            for leaf in TREE_MIN_LEAF:
                out.append(("arbol", feature_set, {"max_depth": depth, "min_samples_leaf": leaf}))
        for k in KNN_K:
            for weights in KNN_WEIGHTS:
                out.append(("knn", feature_set, {"n_neighbors": k, "weights": weights}))
    return out


def params_label(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def predict_baseline(familia: str, fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    if familia == "prev5":
        return np.ones(len(eval_df), dtype=float)
    if familia == "dummy":
        medias = fit_df.groupby("cultivo")[TARGET].mean()
        return eval_df["cultivo"].map(medias).to_numpy(dtype=float)
    raise ValueError(f"Baseline desconocido: {familia}")


def _metrics(eval_df: pd.DataFrame, pred: np.ndarray) -> dict:
    return regression_metrics(eval_df[TARGET], pred, eval_df["prev5"])


def search(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Entrena cada candidato con train y lo mide en validación.

    Recibe solo train y validación: el test no puede influir en la búsqueda.
    """
    rows = []
    for familia, feature_set, params in candidates():
        cols = feature_columns(feature_set)
        model = build_model(familia, feature_set, params)
        model.fit(train[cols], train[TARGET])
        m_train = _metrics(train, model.predict(train[cols]))
        m_val = _metrics(val, model.predict(val[cols]))
        rows.append(
            {
                "familia": familia,
                "modelo": MODEL_LABELS[familia],
                "variables": feature_set,
                **params,
                "params": params_label(params),
                "mae_ratio_train": m_train["mae_ratio"],
                "mae_ratio_val": m_val["mae_ratio"],
                "rmse_ratio_val": m_val["rmse_ratio"],
                "r2_ratio_val": m_val["r2_ratio"],
                "mae_kg_val": m_val["mae_kg"],
            }
        )
    return pd.DataFrame(rows)


def select_best(busqueda: pd.DataFrame) -> pd.DataFrame:
    """Mejor configuración por familia y conjunto de variables, según validación."""
    idx = busqueda.groupby(["familia", "variables"])["mae_ratio_val"].idxmin()
    return busqueda.loc[idx].sort_values("mae_ratio_val").reset_index(drop=True)


def _params_from_row(row: pd.Series) -> dict:
    if row["familia"] == "arbol":
        depth = row["max_depth"]
        return {
            "max_depth": None if pd.isna(depth) else int(depth),
            "min_samples_leaf": int(row["min_samples_leaf"]),
        }
    return {"n_neighbors": int(row["n_neighbors"]), "weights": row["weights"]}


def evaluate_final(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    elegidos: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Pipeline]]:
    """Reentrena con train + validación y mide validación (con el ajuste de
    train) y test (con el ajuste final). Incluye los dos baselines."""
    train_val = pd.concat([train, val])
    rows = []
    preds_test: dict[str, np.ndarray] = {}
    modelos: dict[str, Pipeline] = {}

    for familia in ["dummy", "prev5"]:
        m_val = _metrics(val, predict_baseline(familia, train, val))
        pred_test = predict_baseline(familia, train_val, test)
        m_test = _metrics(test, pred_test)
        clase = classification_metrics(test["mala85"], (pred_test < THRESHOLD).astype(int))
        nombre = MODEL_LABELS[familia]
        preds_test[nombre] = pred_test
        rows.append(_comparison_row(nombre, "baseline", "-", "-", m_val, m_test, clase))

    for _, row in elegidos.iterrows():
        params = _params_from_row(row)
        cols = feature_columns(row["variables"])
        nombre = f"{row['modelo']} ({row['variables']})"

        model_val = build_model(row["familia"], row["variables"], params)
        model_val.fit(train[cols], train[TARGET])
        m_val = _metrics(val, model_val.predict(val[cols]))

        model = build_model(row["familia"], row["variables"], params)
        model.fit(train_val[cols], train_val[TARGET])
        pred_test = model.predict(test[cols])
        m_test = _metrics(test, pred_test)
        clase = classification_metrics(test["mala85"], (pred_test < THRESHOLD).astype(int))

        preds_test[nombre] = pred_test
        modelos[nombre] = model
        rows.append(
            _comparison_row(nombre, row["familia"], row["variables"], params_label(params), m_val, m_test, clase)
        )

    return pd.DataFrame(rows), preds_test, modelos


def _comparison_row(nombre, familia, variables, params, m_val, m_test, clase) -> dict:
    return {
        "modelo": nombre,
        "familia": familia,
        "variables": variables,
        "params": params,
        "mae_ratio_val": m_val["mae_ratio"],
        "mae_ratio_test": m_test["mae_ratio"],
        "rmse_ratio_test": m_test["rmse_ratio"],
        "r2_ratio_test": m_test["r2_ratio"],
        "mae_kg_test": m_test["mae_kg"],
        "rmse_kg_test": m_test["rmse_kg"],
        "r2_kg_test": m_test["r2_kg"],
        "recall_alerta_test": clase["recall"],
        "precision_alerta_test": clase["precision"],
    }


def metrics_by_group(test: pd.DataFrame, preds: dict[str, np.ndarray], group: str) -> pd.DataFrame:
    rows = []
    for nombre, pred in preds.items():
        tmp = test.assign(_pred=pred)
        for key, g in tmp.groupby(group):
            m = _metrics(g, g["_pred"].to_numpy())
            rows.append(
                {
                    group: key,
                    "modelo": nombre,
                    "n": m["n"],
                    "ratio_real_medio": g[TARGET].mean(),
                    "ratio_pred_medio": g["_pred"].mean(),
                    "mae_ratio": m["mae_ratio"],
                    "mae_kg": m["mae_kg"],
                    "r2_ratio": m["r2_ratio"],
                }
            )
    return pd.DataFrame(rows)


def prediction_table(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out = test[
        ["cultivo", "campania", "campania_inicio", "municipio_nombre", "superficie_sembrada",
         "prev5", "prev5_ceros", "rendimiento", TARGET, "mala85"]
    ].copy()
    out["ratio_pred"] = pred
    out["rend_pred_kg"] = pred * out["prev5"]
    out["residuo_ratio"] = out[TARGET] - out["ratio_pred"]
    out["error_kg"] = out["rendimiento"] - out["rend_pred_kg"]
    # Toneladas de diferencia en todo el partido: dimensiona el error en la escala del usuario.
    out["error_t_partido"] = out["error_kg"] * out["superficie_sembrada"] / 1000
    out["alerta_regresion"] = (out["ratio_pred"] < THRESHOLD).astype(int)
    return out.reset_index(drop=True)


def save_model(model: Pipeline, row: pd.Series, comparacion: pd.DataFrame) -> None:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, REGRESOR_FILE)
    fila = comparacion[comparacion["modelo"] == f"{row['modelo']} ({row['variables']})"].iloc[0]
    metadata = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": row["modelo"],
        "variables": row["variables"],
        "params": _params_from_row(row),
        "entrada": feature_columns(row["variables"]),
        "salida": "ratio = rendimiento / promedio de las 5 campañas anteriores",
        "conversion_kg_ha": "rendimiento_kg_ha = ratio_predicho * prev5",
        "umbral_alerta": THRESHOLD,
        "entrenado_con": "train + validacion (campañas 1981/82 a 2018/19)",
        "metricas_test": {
            k: float(fila[k])
            for k in ["mae_ratio_test", "rmse_ratio_test", "r2_ratio_test", "mae_kg_test", "r2_kg_test"]
        },
        "semilla": RANDOM_STATE,
        "versiones": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    REGRESOR_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run(verbose: bool = True) -> dict:
    df = load_data()
    train, val, test = split_data(df)

    busqueda = search(train, val)
    elegidos = select_best(busqueda)
    comparacion, preds_test, modelos = evaluate_final(train, val, test, elegidos)

    ganador = elegidos.iloc[0]
    nombre_ganador = f"{ganador['modelo']} ({ganador['variables']})"
    seleccion = {
        "dummy": MODEL_LABELS["dummy"],
        "prev5": MODEL_LABELS["prev5"],
        "seleccionado": nombre_ganador,
    }
    preds_clave = {k: preds_test[v] for k, v in seleccion.items()}

    por_cultivo = metrics_by_group(test, preds_clave, "cultivo")
    por_campania = metrics_by_group(test, preds_clave, "campania_inicio")
    predicciones = prediction_table(test, preds_test[nombre_ganador])

    busqueda.to_csv(REGRESION_BUSQUEDA, index=False, encoding="utf-8")
    comparacion.to_csv(REGRESION_COMPARACION, index=False, encoding="utf-8")
    por_cultivo.to_csv(REGRESION_POR_CULTIVO, index=False, encoding="utf-8")
    por_campania.to_csv(REGRESION_POR_CAMPANIA, index=False, encoding="utf-8")
    predicciones.to_csv(REGRESION_PREDICCIONES, index=False, encoding="utf-8")
    save_model(modelos[nombre_ganador], ganador, comparacion)

    if verbose:
        print("Mejor configuración por familia (validación)")
        print(elegidos[["modelo", "variables", "params", "mae_ratio_train", "mae_ratio_val"]].to_string(index=False))
        print("\nComparación final")
        print(comparacion.round(4).to_string(index=False))
        print(f"\nRegresor seleccionado: {nombre_ganador}")

    return {
        "train": train,
        "val": val,
        "test": test,
        "busqueda": busqueda,
        "elegidos": elegidos,
        "comparacion": comparacion,
        "preds_test": preds_test,
        "modelos": modelos,
        "seleccionado": nombre_ganador,
        "por_cultivo": por_cultivo,
        "por_campania": por_campania,
        "predicciones": predicciones,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y evalúa los regresores de la Entrega 3.")
    parser.add_argument("--quiet", action="store_true", help="No imprime tablas.")
    args = parser.parse_args()
    run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
