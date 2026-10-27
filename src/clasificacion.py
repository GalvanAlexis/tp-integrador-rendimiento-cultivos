"""Clasificaci\u00f3n directa de campa\u00f1a mala para la Entrega 4.

Protocolo (id\u00e9ntico al de regresion.py):
1. B\u00fasqueda de hiperpar\u00e1metros: se entrena con train (1981/82 a 2013/14) y se
   mide en validaci\u00f3n (2014/15 a 2018/19). El test no interviene.
2. Por cada familia se elige la configuraci\u00f3n con menor costo ponderado por
   costos en validaci\u00f3n. El clasificador final es el mejor de todos.
3. Las configuraciones elegidas se reentrenan con train + validaci\u00f3n y se
   eval\u00faan una sola vez en test (2019/20 a 2024/25).
4. Se compara el clasificador directo con la alerta derivada del regresor.

Modelos:
- Dummy (clase mayoritaria): baseline obligatorio de la consigna.
- DecisionTreeClassifier con distintas profundidades y hojas m\u00ednimas.
- KNeighborsClassifier con StandardScaler dentro del Pipeline.
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
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .baselines import classification_metrics
from .config import (
    CLASIFICACION_BUSQUEDA,
    CLASIFICACION_COMPARACION,
    CLASIFICACION_POR_CAMPANIA,
    CLASIFICACION_POR_CULTIVO,
    CLASIFICACION_PREDICCIONES,
    CLASIFICADOR_FILE,
    CLASIFICADOR_METADATA,
    COST_FN,
    COST_FP,
    DATASET_MODELADO_CLIMA,
    MODELOS_DIR,
    RANDOM_STATE,
    REGRESOR_FILE,
    THRESHOLD,
)

TARGET_CLF = "mala85"

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

TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 15, None]
TREE_MIN_LEAF = [1, 10, 30, 50, 100]
KNN_K = [1, 3, 5, 10, 15, 25, 50, 75, 100, 150, 200, 300, 500, 800]
KNN_WEIGHTS = ["uniform", "distance"]

MODEL_LABELS = {
    "dummy": "Dummy (clase mayoritaria)",
    "regresion": "Regresi\u00f3n + umbral (Entrega 3)",
    "arbol": "DecisionTreeClassifier",
    "knn": "KNeighborsClassifier",
}

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


def feature_columns(feature_set: str) -> list[str]:
    return CAT_COLS + LOG_COLS + FEATURE_SETS[feature_set]


def load_data() -> pd.DataFrame:
    if not DATASET_MODELADO_CLIMA.exists():
        raise FileNotFoundError(
            f"No existe {DATASET_MODELADO_CLIMA}. "
            "Ejecuta primero: python -m src.clima --merge"
        )
    df = pd.read_csv(DATASET_MODELADO_CLIMA, encoding="utf-8")
    df[TARGET_CLF] = df[TARGET_CLF].astype(int)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        df[df["split"] == "train"].copy(),
        df[df["split"] == "validacion"].copy(),
        df[df["split"] == "test"].copy(),
    )


def _preprocessor(feature_set: str, scale: bool) -> ColumnTransformer:
    num = FEATURE_SETS[feature_set]
    log_steps: list = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]
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
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_knn(feature_set: str, n_neighbors: int, weights: str) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(feature_set, scale=True)),
            ("modelo", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
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


def _params_from_row(row: pd.Series) -> dict:
    if row["familia"] == "arbol":
        depth = row["max_depth"]
        return {
            "max_depth": None if pd.isna(depth) else int(depth),
            "min_samples_leaf": int(row["min_samples_leaf"]),
        }
    return {"n_neighbors": int(row["n_neighbors"]), "weights": row["weights"]}


def costo_total(tp: int, fp: int, fn: int, tn: int) -> float:
    """Costo asimetrico: FN pesa COST_FN, FP pesa COST_FP."""
    return float(COST_FN * fn + COST_FP * fp)


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = classification_metrics(y_true, y_pred)
    m["costo"] = costo_total(m["tp"], m["fp"], m["fn"], m["tn"])
    positivos = m["tp"] + m["fn"]
    m["costo_por_pos"] = m["costo"] / positivos if positivos > 0 else np.nan
    return m


def search(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Entrena cada candidato con train y lo mide en validacion.

    Recibe solo train y validacion: el test no puede influir en la busqueda.
    Criterio de seleccion: menor costo total en validacion (FN pesa el doble).
    """
    rows = []
    for familia, feature_set, params in candidates():
        cols = feature_columns(feature_set)
        model = build_model(familia, feature_set, params)
        model.fit(train[cols], train[TARGET_CLF])
        for split_name, eval_df in [("train", train), ("val", val)]:
            y_pred = model.predict(eval_df[cols])
            m = _clf_metrics(eval_df[TARGET_CLF].to_numpy(), y_pred)
            rows.append(
                {
                    "familia": familia,
                    "modelo": MODEL_LABELS[familia],
                    "variables": feature_set,
                    **params,
                    "params": params_label(params),
                    "split": split_name,
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "f1": m["f1"],
                    "accuracy": m["accuracy"],
                    "costo": m["costo"],
                    "costo_por_pos": m["costo_por_pos"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "tn": m["tn"],
                }
            )
    return pd.DataFrame(rows)


def select_best(busqueda: pd.DataFrame) -> pd.DataFrame:
    """Mejor configuracion por familia y conjunto de variables, segun costo en validacion."""
    val = busqueda[busqueda["split"] == "val"].copy()
    idx = val.groupby(["familia", "variables"])["costo"].idxmin()
    return val.loc[idx].sort_values("costo").reset_index(drop=True)


def _load_regresor_predictions(test: pd.DataFrame) -> np.ndarray | None:
    """Carga el regresor de la Entrega 3 y genera predicciones de alerta en test."""
    if not REGRESOR_FILE.exists():
        return None
    try:
        regresor = joblib.load(REGRESOR_FILE)
        cols_reg = list(regresor.feature_names_in_)
        ratio_pred = regresor.predict(test[cols_reg])
        return (ratio_pred < THRESHOLD).astype(int)
    except Exception:
        return None


def evaluate_final(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    elegidos: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Pipeline]]:
    """Reentrena con train + validacion y mide en test."""
    train_val = pd.concat([train, val])
    rows = []
    preds_test: dict[str, np.ndarray] = {}
    modelos: dict[str, Pipeline] = {}

    # Baseline: Dummy clase mayoritaria.
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(train_val[["prev5"]], train_val[TARGET_CLF])
    pred_dummy = dummy.predict(test[["prev5"]])
    m_dummy = _clf_metrics(test[TARGET_CLF].to_numpy(), pred_dummy)
    preds_test[MODEL_LABELS["dummy"]] = pred_dummy
    rows.append(_comparison_row(MODEL_LABELS["dummy"], "dummy", "-", "-", m_dummy))

    # Alerta del regresor de la Entrega 3.
    pred_reg = _load_regresor_predictions(test)
    if pred_reg is not None:
        m_reg = _clf_metrics(test[TARGET_CLF].to_numpy(), pred_reg)
        preds_test[MODEL_LABELS["regresion"]] = pred_reg
        rows.append(
            _comparison_row(MODEL_LABELS["regresion"], "regresion", "con_clima", "-", m_reg)
        )

    # Clasificadores directos.
    for _, row in elegidos.iterrows():
        params = _params_from_row(row)
        cols = feature_columns(row["variables"])
        nombre = f"{row['modelo']} ({row['variables']})"
        model = build_model(row["familia"], row["variables"], params)
        model.fit(train_val[cols], train_val[TARGET_CLF])
        pred = model.predict(test[cols])
        m = _clf_metrics(test[TARGET_CLF].to_numpy(), pred)
        preds_test[nombre] = pred
        modelos[nombre] = model
        rows.append(
            _comparison_row(nombre, row["familia"], row["variables"], params_label(params), m)
        )

    return pd.DataFrame(rows), preds_test, modelos


def _comparison_row(nombre, familia, variables, params, m) -> dict:
    return {
        "modelo": nombre,
        "familia": familia,
        "variables": variables,
        "params": params,
        "recall": m["recall"],
        "precision": m["precision"],
        "f1": m["f1"],
        "accuracy": m["accuracy"],
        "costo": m["costo"],
        "costo_por_pos": m["costo_por_pos"],
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tn": m["tn"],
        "tasa_alertas": m["tasa_alertas"],
    }


def metrics_by_group(
    test: pd.DataFrame, preds: dict[str, np.ndarray], group: str
) -> pd.DataFrame:
    rows = []
    for nombre, pred in preds.items():
        tmp = test.assign(_pred=pred)
        for key, g in tmp.groupby(group):
            m = _clf_metrics(g[TARGET_CLF].to_numpy(), g["_pred"].to_numpy())
            rows.append(
                {
                    group: key,
                    "modelo": nombre,
                    "n": m["n"],
                    "positivos_reales": int(g[TARGET_CLF].sum()),
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "f1": m["f1"],
                    "costo": m["costo"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "tn": m["tn"],
                }
            )
    return pd.DataFrame(rows)


def prediction_table(
    test: pd.DataFrame,
    pred_clf: np.ndarray,
    pred_reg: np.ndarray | None,
) -> pd.DataFrame:
    out = test[
        [
            "cultivo",
            "campania",
            "campania_inicio",
            "municipio_nombre",
            "superficie_sembrada",
            "prev5",
            "prev5_ceros",
            "rendimiento",
            "ratio",
            TARGET_CLF,
        ]
    ].copy()
    out["alerta_clasificador"] = pred_clf
    if pred_reg is not None:
        out["alerta_regresion"] = pred_reg
    out["correcto"] = (out["alerta_clasificador"] == out[TARGET_CLF]).astype(int)
    return out.reset_index(drop=True)


def save_model(model: Pipeline, row: pd.Series, comparacion: pd.DataFrame) -> None:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CLASIFICADOR_FILE)
    nombre = f"{row['modelo']} ({row['variables']})"
    fila = comparacion[comparacion["modelo"] == nombre].iloc[0]
    metadata = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": row["modelo"],
        "variables": row["variables"],
        "params": _params_from_row(row),
        "entrada": feature_columns(row["variables"]),
        "salida": "mala85 = 1 si la campana se espera mala (ratio predicho < 0,85)",
        "umbral_alerta": THRESHOLD,
        "costo_fn": COST_FN,
        "costo_fp": COST_FP,
        "entrenado_con": "train + validacion (campanas 1981/82 a 2018/19)",
        "metricas_test": {
            k: float(fila[k])
            for k in ["recall", "precision", "f1", "accuracy", "costo"]
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
    CLASIFICADOR_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(verbose: bool = True) -> dict:
    df = load_data()
    train, val, test = split_data(df)

    busqueda = search(train, val)
    elegidos = select_best(busqueda)
    comparacion, preds_test, modelos = evaluate_final(train, val, test, elegidos)

    ganador = elegidos.iloc[0]
    nombre_ganador = f"{ganador['modelo']} ({ganador['variables']})"

    por_cultivo = metrics_by_group(test, preds_test, "cultivo")
    por_campania = metrics_by_group(test, preds_test, "campania_inicio")

    pred_reg = preds_test.get(MODEL_LABELS["regresion"])
    predicciones = prediction_table(test, preds_test[nombre_ganador], pred_reg)

    CLASIFICACION_BUSQUEDA.parent.mkdir(parents=True, exist_ok=True)
    busqueda.to_csv(CLASIFICACION_BUSQUEDA, index=False, encoding="utf-8")
    comparacion.to_csv(CLASIFICACION_COMPARACION, index=False, encoding="utf-8")
    por_cultivo.to_csv(CLASIFICACION_POR_CULTIVO, index=False, encoding="utf-8")
    por_campania.to_csv(CLASIFICACION_POR_CAMPANIA, index=False, encoding="utf-8")
    predicciones.to_csv(CLASIFICACION_PREDICCIONES, index=False, encoding="utf-8")
    save_model(modelos[nombre_ganador], ganador, comparacion)

    if verbose:
        print("Mejor configuracion por familia (validacion, menor costo)")
        print(
            elegidos[["modelo", "variables", "params", "recall", "precision", "costo"]]
            .to_string(index=False)
        )
        print("\nComparacion final (test)")
        print(comparacion.round(4).to_string(index=False))
        print(f"\nClasificador seleccionado: {nombre_ganador}")

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
    parser = argparse.ArgumentParser(
        description="Entrena y evalua los clasificadores de la Entrega 4."
    )
    parser.add_argument("--quiet", action="store_true", help="No imprime tablas.")
    args = parser.parse_args()
    run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
