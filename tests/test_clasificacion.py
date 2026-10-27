"""Tests de la Entrega 4: clasificacion directa de campana mala."""
import inspect

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.clasificacion import (
    CLIMA_COLS,
    FEATURE_SETS,
    FORBIDDEN,
    TARGET_CLF,
    feature_columns,
    make_knn,
    make_tree,
    search,
    select_best,
    costo_total,
)
from src.config import CLASIFICADOR_FILE, COST_FN, COST_FP


def _toy(n: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "cultivo": rng.choice(["Maiz", "Soja total"], size=n),
            "superficie_sembrada": rng.integers(100, 50_000, size=n).astype(float),
            "prev5": rng.uniform(1000, 8000, size=n),
            "prev5_ceros": rng.integers(0, 2, size=n),
        }
    )
    for col in CLIMA_COLS:
        df[col] = rng.normal(100, 20, size=n)
    ratio = 1 + (df["clima_precip_total_mm"] - 100) / 200 + rng.normal(0, 0.1, size=n)
    df[TARGET_CLF] = (ratio < 0.85).astype(int)
    return df


@pytest.mark.parametrize("feature_set", list(FEATURE_SETS))
def test_features_exclude_leakage(feature_set):
    cols = set(feature_columns(feature_set))
    assert not cols & FORBIDDEN
    assert "campania_inicio" not in cols
    assert "split" not in cols
    assert TARGET_CLF not in cols


def test_knn_scales_inside_pipeline():
    pipe = make_knn("con_clima", n_neighbors=5, weights="uniform")
    assert isinstance(pipe.steps[-1][1], KNeighborsClassifier)
    numericas = pipe.named_steps["prep"].transformers[2][1]
    assert isinstance(numericas, StandardScaler)


def test_tree_uses_balanced_class_weight():
    pipe = make_tree("con_clima", max_depth=4, min_samples_leaf=10)
    clf = pipe.named_steps["modelo"]
    assert isinstance(clf, DecisionTreeClassifier)
    assert clf.class_weight == "balanced"


@pytest.mark.parametrize("builder,params", [(make_tree, (4, 10)), (make_knn, (5, "distance"))])
def test_pipelines_fit_and_predict(builder, params):
    df = _toy()
    cols = feature_columns("con_clima")
    model = builder("con_clima", *params).fit(df[cols], df[TARGET_CLF])
    pred = model.predict(df[cols])
    assert pred.shape == (len(df),)
    assert set(pred).issubset({0, 1})


def test_search_signature_no_test():
    """La busqueda solo puede recibir train y val."""
    assert list(inspect.signature(search).parameters) == ["train", "val"]


def test_select_best_uses_costo():
    busqueda = pd.DataFrame(
        {
            "familia": ["arbol", "arbol", "knn"],
            "variables": ["con_clima"] * 3,
            "split": ["val"] * 3,
            "costo": [300.0, 200.0, 250.0],
        }
    )
    best = select_best(busqueda)
    assert best.iloc[0]["costo"] == pytest.approx(200.0)


def test_costo_asimetrico():
    """FN debe pesar COST_FN y FP debe pesar COST_FP."""
    c = costo_total(tp=0, fp=10, fn=5, tn=100)
    expected = COST_FN * 5 + COST_FP * 10
    assert c == pytest.approx(expected)


@pytest.mark.skipif(
    not CLASIFICADOR_FILE.exists(),
    reason="Falta entrenar: python -m src.clasificacion",
)
def test_saved_classifier_predicts_binary():
    model = joblib.load(CLASIFICADOR_FILE)
    df = _toy(10)
    pred = model.predict(df[list(model.feature_names_in_)])
    assert pred.shape == (10,)
    assert set(pred).issubset({0, 1})
