"""Baselines de regresión y clasificación para la Entrega 2.

Se evaluan sobre el mismo cohorte y con el mismo split temporal:
- train: 1981/82 a 2013/14
- validacion: 2014/15 a 2018/19
- test: 2019/20 a 2024/25

Modelos:
- DummyRegressor / DummyClassifier (los exige la consigna).
- Promedio de las 5 campañas anteriores (baseline real del problema).
- Promedio de 5 campañas ajustado por la tendencia media del train.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from .config import (
    BASELINES_CLASIFICACION,
    BASELINES_GLOBAL,
    BASELINES_REGRESION,
    CROPS,
    DATASET_MODELADO,
    FIGURAS_DIR,
    THRESHOLD,
)

MODEL_NAMES = {
    "dummy": "Dummy",
    "prev5": "Promedio 5 campañas",
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prev5: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    prev5 = np.asarray(prev5, dtype=float)

    y_true_kg = y_true * prev5
    y_pred_kg = y_pred * prev5

    return {
        "n": int(len(y_true)),
        "mae_ratio": float(mean_absolute_error(y_true, y_pred)),
        "rmse_ratio": float(root_mean_squared_error(y_true, y_pred)),
        "r2_ratio": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "mae_kg": float(mean_absolute_error(y_true_kg, y_pred_kg)),
        "rmse_kg": float(root_mean_squared_error(y_true_kg, y_pred_kg)),
        "r2_kg": float(r2_score(y_true_kg, y_pred_kg)) if len(y_true) > 1 else np.nan,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "tasa_alertas": float(y_pred.mean()) if len(y_pred) else np.nan,
    }


def _fit_predict_dummy(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    reg = DummyRegressor(strategy="mean")
    reg.fit(train[["prev5"]], train["ratio"])
    pred_reg = reg.predict(test[["prev5"]])

    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(train[["prev5"]], train["mala85"].astype(int))
    pred_clf = clf.predict(test[["prev5"]])
    return pred_reg, pred_clf


def run_baselines(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DATASET_MODELADO.exists():
        raise FileNotFoundError(
            f"No existe {DATASET_MODELADO}. Ejecutá primero: python -m src.preparar_datos"
        )

    df = pd.read_csv(DATASET_MODELADO, encoding="utf-8")
    df["mala85"] = df["mala85"].astype(int)

    reg_rows: list[dict] = []
    clf_rows: list[dict] = []
    pooled: dict[tuple[str, str], dict[str, list]] = {}

    for split in ["validacion", "test"]:
        train = df[df["split"] == "train"].copy()
        test = df[df["split"] == split].copy()
        if train.empty or test.empty:
            continue

        for crop in CROPS:
            tr = train[train["cultivo"] == crop].copy()
            ev = test[test["cultivo"] == crop].copy()
            if tr.empty or ev.empty:
                continue

            y_true = ev["ratio"].to_numpy(dtype=float)
            y_cls = ev["mala85"].to_numpy(dtype=int)
            prev5 = ev["prev5"].to_numpy(dtype=float)

            pred_dummy_reg, pred_dummy_clf = _fit_predict_dummy(tr, ev)
            pred_prev5 = np.ones(len(ev), dtype=float)
            pred_clf_prev5 = (pred_prev5 < THRESHOLD).astype(int)

            candidates = {
                "dummy": (pred_dummy_reg, pred_dummy_clf),
                "prev5": (pred_prev5, pred_clf_prev5),
            }

            for key, (pred_reg, pred_clf) in candidates.items():
                reg_m = regression_metrics(y_true, pred_reg, prev5)
                clf_m = classification_metrics(y_cls, pred_clf)
                reg_rows.append(
                    {"split": split, "cultivo": crop, "modelo": MODEL_NAMES[key], **reg_m}
                )
                clf_rows.append(
                    {"split": split, "cultivo": crop, "modelo": MODEL_NAMES[key], **clf_m}
                )

                pooled_key = (split, key)
                bucket = pooled.setdefault(
                    pooled_key,
                    {
                        "y_true": [],
                        "y_pred": [],
                        "prev5": [],
                        "y_cls": [],
                        "pred_clf": [],
                    },
                )
                bucket["y_true"].append(y_true)
                bucket["y_pred"].append(pred_reg)
                bucket["prev5"].append(prev5)
                bucket["y_cls"].append(y_cls)
                bucket["pred_clf"].append(pred_clf)

    reg = pd.DataFrame(reg_rows)
    clf = pd.DataFrame(clf_rows)

    global_rows: list[dict] = []
    for (split, key), bucket in pooled.items():
        y_true = np.concatenate(bucket["y_true"])
        y_pred = np.concatenate(bucket["y_pred"])
        prev5 = np.concatenate(bucket["prev5"])
        y_cls = np.concatenate(bucket["y_cls"])
        pred_clf = np.concatenate(bucket["pred_clf"])

        reg_m = regression_metrics(y_true, y_pred, prev5)
        clf_m = classification_metrics(y_cls, pred_clf)
        global_rows.append(
            {
                "split": split,
                "modelo": MODEL_NAMES[key],
                **reg_m,
                **{f"clf_{k}": v for k, v in clf_m.items()},
            }
        )

    glob = pd.DataFrame(global_rows)

    BASELINES_REGRESION.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(BASELINES_REGRESION, index=False, encoding="utf-8")
    clf.to_csv(BASELINES_CLASIFICACION, index=False, encoding="utf-8")
    glob.to_csv(BASELINES_GLOBAL, index=False, encoding="utf-8")

    _make_figures(reg, clf)

    if verbose:
        print("Baselines de regresión (test)")
        print(reg[reg["split"] == "test"].to_string(index=False))
        print("\nBaselines de clasificación (test)")
        print(clf[clf["split"] == "test"].to_string(index=False))
        print("\nResumen global")
        print(glob.to_string(index=False))

    return reg, clf, glob


def _make_figures(reg: pd.DataFrame, clf: pd.DataFrame) -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    test_reg = reg[reg["split"] == "test"].copy()
    if not test_reg.empty:
        pivot = test_reg.pivot(index="cultivo", columns="modelo", values="mae_ratio")
        ax = pivot.plot(kind="bar", figsize=(10, 5), rot=0)
        ax.set_title("MAE del ratio rendimiento / promedio 5 campañas (test)")
        ax.set_ylabel("MAE (ratio)")
        ax.set_xlabel("Cultivo")
        ax.legend(title="Modelo", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURAS_DIR / "baselines_mae_ratio_test.png", dpi=150)
        plt.close()

    test_clf = clf[(clf["split"] == "test") & (clf["modelo"] == MODEL_NAMES["prev5"])]
    if not test_clf.empty:
        labels = test_clf["cultivo"].tolist()
        recall = test_clf["recall"].tolist()
        precision = test_clf["precision"].tolist()
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, precision, width, label="Precision")
        ax.bar(x + width / 2, recall, width, label="Recall")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(0, 1)
        ax.set_title("Baseline Promedio 5 campañas: precision y recall (test)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURAS_DIR / "baseline_clasificacion_test.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula los baselines de la Entrega 2.")
    parser.add_argument("--quiet", action="store_true", help="No imprime tablas.")
    args = parser.parse_args()
    run_baselines(verbose=not args.quiet)


if __name__ == "__main__":
    main()
