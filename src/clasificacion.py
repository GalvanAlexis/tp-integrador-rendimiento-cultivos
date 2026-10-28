"""Clasificación directa de campaña mala para la Entrega 4.

Protocolo (idéntico al de regresion.py):
1. Búsqueda de hiperparámetros: se entrena con train (1981/82 a 2013/14) y se
   mide en validación (2014/15 a 2018/19). El test no interviene.
2. El umbral de decisión sobre la probabilidad estimada es un hiperparámetro
   más: se elige en validación junto con el resto.
3. Por cada familia se elige la configuración con menor costo total en
   validación (FN = 2 × FP). El clasificador final es el mejor de todos.
4. Las configuraciones elegidas se reentrenan con train + validación y se
   evalúan una sola vez en test (2019/20 a 2024/25).
5. Se compara el clasificador directo con la alerta derivada del regresor.

Modelos:
- DummyClassifier (clase mayoritaria): baseline obligatorio de la consigna.
- DecisionTreeClassifier con distintas profundidades, hojas mínimas y peso de clases.
- KNeighborsClassifier con StandardScaler dentro del Pipeline.
- GaussianNB con distintos valores de suavizado de la varianza.
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
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .baselines import classification_metrics
from .config import (
    CLASIFICACION_ACUERDO,
    CLASIFICACION_BUSQUEDA,
    CLASIFICACION_COMPARACION,
    CLASIFICACION_ERRORES_PERFIL,
    CLASIFICACION_ERRORES_RATIO,
    CLASIFICACION_POR_CAMPANIA,
    CLASIFICACION_POR_CULTIVO,
    CLASIFICACION_PREDICCIONES,
    CLASIFICADOR_FILE,
    CLASIFICADOR_METADATA,
    CLIMA_COLS,
    COST_FN,
    COST_FP,
    DATASET_MODELADO_CLIMA,
    MODEL_LABELS,
    MODELOS_DIR,
    RANDOM_STATE,
    REGRESOR_FILE,
    THRESHOLD,
)

TARGET_CLF = "mala85"

NUM_AGRO = ["prev5", "prev5_ceros"]
LOG_COLS = ["superficie_sembrada"]
CAT_COLS = ["cultivo"]

FEATURE_SETS = {
    "sin_clima": NUM_AGRO,
    "con_clima": NUM_AGRO + CLIMA_COLS,
}

TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 15, None]
TREE_MIN_LEAF = [1, 10, 30, 50, 100]
TREE_CLASS_WEIGHT = [None, "balanced"]
KNN_K = [1, 3, 5, 10, 15, 25, 50, 75, 100, 150, 200, 300, 500, 800]
KNN_WEIGHTS = ["uniform", "distance"]
NB_VAR_SMOOTHING = [1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 0.3]

# Umbrales de decisión sobre P(campaña mala). Con probabilidades bien
# calibradas, el umbral que minimiza el costo esperado es
# COST_FP / (COST_FP + COST_FN) = 1/3; en la práctica se elige en validación.
UMBRALES = [round(u, 2) for u in np.arange(0.05, 0.951, 0.05)]
UMBRAL_TEORICO = COST_FP / (COST_FP + COST_FN)

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

METRICAS = ["recall", "precision", "f1", "accuracy", "costo", "costo_por_pos"]
CONFUSION = ["tp", "fp", "fn", "tn"]


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


def make_tree(
    feature_set: str,
    max_depth: int | None,
    min_samples_leaf: int,
    class_weight: str | None = "balanced",
) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(feature_set, scale=False)),
            (
                "modelo",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
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


def make_nb(feature_set: str, var_smoothing: float) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(feature_set, scale=True)),
            ("modelo", GaussianNB(var_smoothing=var_smoothing)),
        ]
    )


def build_model(familia: str, feature_set: str, params: dict) -> Pipeline:
    if familia == "arbol":
        return make_tree(
            feature_set, params["max_depth"], params["min_samples_leaf"], params["class_weight"]
        )
    if familia == "knn":
        return make_knn(feature_set, params["n_neighbors"], params["weights"])
    if familia == "nb":
        return make_nb(feature_set, params["var_smoothing"])
    raise ValueError(f"Familia desconocida: {familia}")


def with_threshold(model: Pipeline, umbral: float) -> FixedThresholdClassifier:
    """Envuelve el pipeline para que predict aplique el umbral elegido en validación."""
    return FixedThresholdClassifier(model, threshold=umbral, response_method="predict_proba")


def candidates() -> list[tuple[str, str, dict]]:
    out: list[tuple[str, str, dict]] = []
    for feature_set in FEATURE_SETS:
        for depth in TREE_DEPTHS:
            for leaf in TREE_MIN_LEAF:
                for cw in TREE_CLASS_WEIGHT:
                    out.append(
                        (
                            "arbol",
                            feature_set,
                            {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": cw},
                        )
                    )
        for k in KNN_K:
            for weights in KNN_WEIGHTS:
                out.append(("knn", feature_set, {"n_neighbors": k, "weights": weights}))
        for vs in NB_VAR_SMOOTHING:
            out.append(("nb", feature_set, {"var_smoothing": vs}))
    return out


def params_label(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _params_from_row(row: pd.Series) -> dict:
    if row["familia"] == "arbol":
        depth = row["max_depth"]
        cw = row["class_weight"]
        return {
            "max_depth": None if pd.isna(depth) else int(depth),
            "min_samples_leaf": int(row["min_samples_leaf"]),
            "class_weight": None if pd.isna(cw) else cw,
        }
    if row["familia"] == "knn":
        return {"n_neighbors": int(row["n_neighbors"]), "weights": row["weights"]}
    return {"var_smoothing": float(row["var_smoothing"])}


def costo_total(tp: int, fp: int, fn: int, tn: int) -> float:
    """Costo asimétrico: FN pesa COST_FN, FP pesa COST_FP."""
    return float(COST_FN * fn + COST_FP * fp)


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = classification_metrics(y_true, y_pred)
    m["costo"] = costo_total(m["tp"], m["fp"], m["fn"], m["tn"])
    positivos = m["tp"] + m["fn"]
    m["costo_por_pos"] = m["costo"] / positivos if positivos > 0 else np.nan
    return m


def apply_threshold(proba: np.ndarray, umbral: float) -> np.ndarray:
    """Misma regla que FixedThresholdClassifier: alerta si P(mala) >= umbral."""
    return (proba >= umbral).astype(int)


def _search_row(familia, feature_set, params, umbral, m_train, m_val) -> dict:
    return {
        "familia": familia,
        "modelo": MODEL_LABELS[familia],
        "variables": feature_set,
        **params,
        "params": params_label(params),
        "umbral": umbral,
        **{f"{k}_train": m_train[k] for k in ["recall", "precision", "f1", "costo"]},
        **{f"{k}_val": m_val[k] for k in METRICAS},
        **{f"{k}_val": m_val[k] for k in CONFUSION},
    }


def search(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Entrena cada candidato con train y lo mide en validación.

    Recibe solo train y validación: el test no puede influir en la búsqueda.
    Cada configuración se entrena una vez y se mide con todos los umbrales.
    El Dummy se incluye como piso de referencia en validación.
    """
    y_train = train[TARGET_CLF].to_numpy()
    y_val = val[TARGET_CLF].to_numpy()

    dummy = DummyClassifier(strategy="most_frequent").fit(train[["prev5"]], y_train)
    rows = [
        _search_row(
            "dummy",
            "-",
            {},
            np.nan,
            _clf_metrics(y_train, dummy.predict(train[["prev5"]])),
            _clf_metrics(y_val, dummy.predict(val[["prev5"]])),
        )
    ]

    for familia, feature_set, params in candidates():
        cols = feature_columns(feature_set)
        model = build_model(familia, feature_set, params).fit(train[cols], y_train)
        proba_train = model.predict_proba(train[cols])[:, 1]
        proba_val = model.predict_proba(val[cols])[:, 1]
        for umbral in UMBRALES:
            rows.append(
                _search_row(
                    familia,
                    feature_set,
                    params,
                    umbral,
                    _clf_metrics(y_train, apply_threshold(proba_train, umbral)),
                    _clf_metrics(y_val, apply_threshold(proba_val, umbral)),
                )
            )
    return pd.DataFrame(rows)


def select_best(busqueda: pd.DataFrame) -> pd.DataFrame:
    """Mejor configuración por familia y conjunto de variables, según costo en validación.

    Ante empates de costo se prefiere el mayor F1 de validación.
    """
    val = busqueda[busqueda["familia"] != "dummy"]
    val = val.sort_values(["costo_val", "f1_val"], ascending=[True, False], kind="stable")
    return val.groupby(["familia", "variables"], sort=False).head(1).reset_index(drop=True)


def _load_regresor_predictions(test: pd.DataFrame) -> np.ndarray:
    """Carga el regresor de la Entrega 3 y genera predicciones de alerta en test."""
    if not REGRESOR_FILE.exists():
        raise FileNotFoundError(
            f"No existe {REGRESOR_FILE}. Ejecuta primero: python -m src.regresion"
        )
    regresor = joblib.load(REGRESOR_FILE)
    ratio_pred = regresor.predict(test[list(regresor.feature_names_in_)])
    return (ratio_pred < THRESHOLD).astype(int)


def _comparison_row(nombre, familia, variables, params, umbral, m, m_val=None) -> dict:
    return {
        "modelo": nombre,
        "familia": familia,
        "variables": variables,
        "params": params,
        "umbral": umbral,
        "costo_val": np.nan if m_val is None else m_val,
        **{k: m[k] for k in METRICAS + CONFUSION},
        "tasa_alertas": m["tasa_alertas"],
    }


def model_name(row: pd.Series) -> str:
    return f"{row['modelo']} ({row['variables']})"


def evaluate_final(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    elegidos: pd.DataFrame,
    costo_dummy_val: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, FixedThresholdClassifier]]:
    """Reentrena con train + validación y mide en test."""
    train_val = pd.concat([train, val])
    y_test = test[TARGET_CLF].to_numpy()
    rows = []
    preds_test: dict[str, np.ndarray] = {}
    modelos: dict[str, FixedThresholdClassifier] = {}

    # Baseline: Dummy clase mayoritaria.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_val[["prev5"]], train_val[TARGET_CLF])
    pred_dummy = dummy.predict(test[["prev5"]])
    preds_test[MODEL_LABELS["dummy"]] = pred_dummy
    rows.append(
        _comparison_row(
            MODEL_LABELS["dummy"], "dummy", "-", "-", np.nan,
            _clf_metrics(y_test, pred_dummy), costo_dummy_val,
        )
    )

    # Alerta del regresor de la Entrega 3 (ratio predicho < THRESHOLD).
    pred_reg = _load_regresor_predictions(test)
    preds_test[MODEL_LABELS["regresion"]] = pred_reg
    rows.append(
        _comparison_row(
            MODEL_LABELS["regresion"], "regresion", "con_clima", "KNN k=300, distance",
            THRESHOLD, _clf_metrics(y_test, pred_reg),
        )
    )

    # Clasificadores directos, con el umbral elegido en validación.
    for _, row in elegidos.iterrows():
        params = _params_from_row(row)
        cols = feature_columns(row["variables"])
        nombre = model_name(row)
        model = with_threshold(build_model(row["familia"], row["variables"], params), row["umbral"])
        model.fit(train_val[cols], train_val[TARGET_CLF])
        pred = model.predict(test[cols])
        preds_test[nombre] = pred
        modelos[nombre] = model
        rows.append(
            _comparison_row(
                nombre, row["familia"], row["variables"], params_label(params), row["umbral"],
                _clf_metrics(y_test, pred), row["costo_val"],
            )
        )

    return pd.DataFrame(rows), preds_test, modelos


def combined_alert(pred_clf: np.ndarray, pred_reg: np.ndarray) -> np.ndarray:
    """Alerta combinada de la app: alerta si alguno de los dos sistemas alerta."""
    return np.maximum(pred_clf, pred_reg).astype(int)


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
                    **{k: m[k] for k in CONFUSION},
                }
            )
    return pd.DataFrame(rows)


def error_type(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.select(
        [(y_true == 1) & (y_pred == 1), (y_true == 0) & (y_pred == 1), (y_true == 1) & (y_pred == 0)],
        ["VP", "FP", "FN"],
        default="VN",
    )


PERFIL_COLS = [
    "ratio",
    "prev5_ceros",
    "clima_precip_total_mm",
    "clima_sequia_max_dias",
    "clima_dias_tmax_30",
]


def error_profile(test: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Perfil medio de los casos según el tipo de resultado (VP, FP, FN, VN)."""
    rows = []
    for nombre, pred in preds.items():
        tmp = test.assign(tipo=error_type(test[TARGET_CLF].to_numpy(), pred))
        for tipo, g in tmp.groupby("tipo"):
            rows.append(
                {
                    "modelo": nombre,
                    "tipo": tipo,
                    "n": len(g),
                    "ratio_mediana": g["ratio"].median(),
                    **{f"{c}_media": g[c].mean() for c in PERFIL_COLS if c != "ratio"},
                    "share_2022": (g["campania_inicio"] == 2022).mean(),
                }
            )
    return pd.DataFrame(rows)


# Bandas de ratio real para leer qué tan lejos del umbral caen los errores.
RATIO_BANDAS = [-np.inf, 0.5, 0.75, 0.85, 0.95, 1.05, np.inf]
RATIO_BANDAS_LABELS = ["< 0,50", "0,50-0,75", "0,75-0,85", "0,85-0,95", "0,95-1,05", ">= 1,05"]


def errors_by_ratio(test: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Cantidad de FN y FP según la banda de ratio real."""
    banda = pd.cut(test["ratio"], RATIO_BANDAS, labels=RATIO_BANDAS_LABELS, right=False)
    rows = []
    for nombre, pred in preds.items():
        tipo = error_type(test[TARGET_CLF].to_numpy(), pred)
        for b in RATIO_BANDAS_LABELS:
            mask = (banda == b).to_numpy()
            if not mask.any():
                continue
            rows.append(
                {
                    "modelo": nombre,
                    "banda_ratio": b,
                    "n": int(mask.sum()),
                    "alertas": int(pred[mask].sum()),
                    "fn": int((tipo[mask] == "FN").sum()),
                    "fp": int((tipo[mask] == "FP").sum()),
                    "tasa_alerta": float(pred[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def agreement_table(test: pd.DataFrame, pred_clf: np.ndarray, pred_reg: np.ndarray) -> pd.DataFrame:
    """Cruce de las alertas de ambos sistemas contra la clase real."""
    tmp = pd.DataFrame(
        {
            "clasificador": np.where(pred_clf == 1, "alerta", "normal"),
            "regresion": np.where(pred_reg == 1, "alerta", "normal"),
            "real": test[TARGET_CLF].to_numpy(),
        }
    )
    out = (
        tmp.groupby(["clasificador", "regresion"])["real"]
        .agg(n="size", malas_reales="sum")
        .reset_index()
    )
    out["precision"] = out["malas_reales"] / out["n"]
    return out


def prediction_table(
    test: pd.DataFrame,
    pred_clf: np.ndarray,
    proba_clf: np.ndarray,
    pred_reg: np.ndarray,
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
    out["proba_clasificador"] = proba_clf
    out["alerta_clasificador"] = pred_clf
    out["alerta_regresion"] = pred_reg
    out["tipo_clasificador"] = error_type(out[TARGET_CLF].to_numpy(), pred_clf)
    out["tipo_regresion"] = error_type(out[TARGET_CLF].to_numpy(), pred_reg)
    return out.reset_index(drop=True)


def save_model(
    model: FixedThresholdClassifier, row: pd.Series, comparacion: pd.DataFrame
) -> None:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CLASIFICADOR_FILE)
    fila = comparacion[comparacion["modelo"] == model_name(row)].iloc[0]
    metadata = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": row["modelo"],
        "variables": row["variables"],
        "params": _params_from_row(row),
        "umbral_decision": float(row["umbral"]),
        "umbral_teorico": UMBRAL_TEORICO,
        "entrada": feature_columns(row["variables"]),
        "salida": (
            "mala85 = 1 (alerta) si P(campaña mala) >= umbral_decision; "
            "campaña mala significa ratio real < 0,85"
        ),
        "umbral_ratio": THRESHOLD,
        "costo_fn": COST_FN,
        "costo_fp": COST_FP,
        "entrenado_con": "train + validación (campañas 1981/82 a 2018/19)",
        "metricas_val": {
            k: float(row[f"{k}_val"]) for k in ["recall", "precision", "f1", "accuracy", "costo"]
        },
        "metricas_test": {
            k: float(fila[k]) for k in ["recall", "precision", "f1", "accuracy", "costo"]
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
    costo_dummy_val = float(busqueda.loc[busqueda["familia"] == "dummy", "costo_val"].iloc[0])
    comparacion, preds_test, modelos = evaluate_final(
        train, val, test, elegidos, costo_dummy_val
    )

    ganador = elegidos.iloc[0]
    nombre_ganador = model_name(ganador)
    modelo_ganador = modelos[nombre_ganador]
    cols_ganador = feature_columns(ganador["variables"])
    proba_ganador = modelo_ganador.estimator_.predict_proba(test[cols_ganador])[:, 1]

    pred_clf = preds_test[nombre_ganador]
    pred_reg = preds_test[MODEL_LABELS["regresion"]]
    pred_comb = combined_alert(pred_clf, pred_reg)
    m_comb = _clf_metrics(test[TARGET_CLF].to_numpy(), pred_comb)
    comparacion = pd.concat(
        [
            comparacion,
            pd.DataFrame(
                [
                    _comparison_row(
                        MODEL_LABELS["combinada"], "combinada", ganador["variables"],
                        "OR", np.nan, m_comb,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    # Análisis de errores: clasificador seleccionado contra la alerta por regresión.
    sistemas = {nombre_ganador: pred_clf, MODEL_LABELS["regresion"]: pred_reg}
    por_cultivo = metrics_by_group(test, preds_test, "cultivo")
    por_campania = metrics_by_group(test, preds_test, "campania_inicio")
    perfil = error_profile(test, sistemas)
    por_ratio = errors_by_ratio(test, sistemas)
    acuerdo = agreement_table(test, pred_clf, pred_reg)
    predicciones = prediction_table(test, pred_clf, proba_ganador, pred_reg)

    CLASIFICACION_BUSQUEDA.parent.mkdir(parents=True, exist_ok=True)
    busqueda.to_csv(CLASIFICACION_BUSQUEDA, index=False, encoding="utf-8")
    comparacion.to_csv(CLASIFICACION_COMPARACION, index=False, encoding="utf-8")
    por_cultivo.to_csv(CLASIFICACION_POR_CULTIVO, index=False, encoding="utf-8")
    por_campania.to_csv(CLASIFICACION_POR_CAMPANIA, index=False, encoding="utf-8")
    perfil.to_csv(CLASIFICACION_ERRORES_PERFIL, index=False, encoding="utf-8")
    por_ratio.to_csv(CLASIFICACION_ERRORES_RATIO, index=False, encoding="utf-8")
    acuerdo.to_csv(CLASIFICACION_ACUERDO, index=False, encoding="utf-8")
    predicciones.to_csv(CLASIFICACION_PREDICCIONES, index=False, encoding="utf-8")
    save_model(modelo_ganador, ganador, comparacion)

    if verbose:
        print(f"Costo del Dummy en validación: {costo_dummy_val:.0f}")
        print("Mejor configuración por familia (validación, menor costo)")
        print(
            elegidos[
                ["modelo", "variables", "params", "umbral", "recall_val", "precision_val", "costo_val"]
            ].to_string(index=False)
        )
        print("\nComparación final (test)")
        print(comparacion.round(4).to_string(index=False))
        print(f"\nClasificador seleccionado: {nombre_ganador}, umbral {ganador['umbral']}")

    return {
        "train": train,
        "val": val,
        "test": test,
        "busqueda": busqueda,
        "elegidos": elegidos,
        "costo_dummy_val": costo_dummy_val,
        "comparacion": comparacion,
        "preds_test": preds_test,
        "modelos": modelos,
        "seleccionado": nombre_ganador,
        "proba_seleccionado": proba_ganador,
        "por_cultivo": por_cultivo,
        "por_campania": por_campania,
        "perfil_errores": perfil,
        "errores_por_ratio": por_ratio,
        "acuerdo": acuerdo,
        "predicciones": predicciones,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa los clasificadores de la Entrega 4."
    )
    parser.add_argument("--quiet", action="store_true", help="No imprime tablas.")
    args = parser.parse_args()
    run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
