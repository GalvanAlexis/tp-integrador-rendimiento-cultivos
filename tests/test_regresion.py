import inspect

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.config import REGRESOR_FILE
from src.regresion import (
    CLIMA_COLS,
    FEATURE_SETS,
    FORBIDDEN,
    feature_columns,
    make_knn,
    make_tree,
    predict_baseline,
    search,
    select_best,
)


def _toy(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "cultivo": rng.choice(["Maíz", "Soja total"], size=n),
            "superficie_sembrada": rng.integers(100, 50_000, size=n).astype(float),
            "prev5": rng.uniform(1000, 8000, size=n),
            "prev5_ceros": rng.integers(0, 2, size=n),
        }
    )
    for col in CLIMA_COLS:
        df[col] = rng.normal(100, 20, size=n)
    df["ratio"] = 1 + (df["clima_precip_total_mm"] - 100) / 200 + rng.normal(0, 0.05, size=n)
    return df


@pytest.mark.parametrize("feature_set", list(FEATURE_SETS))
def test_features_exclude_leakage_and_time(feature_set):
    cols = set(feature_columns(feature_set))
    assert not cols & FORBIDDEN
    assert "campania_inicio" not in cols
    assert "split" not in cols


def test_knn_scales_inside_pipeline():
    pipe = make_knn("con_clima", n_neighbors=5, weights="uniform")
    assert isinstance(pipe.steps[-1][1], KNeighborsRegressor)
    numericas = pipe.named_steps["prep"].transformers[2][1]
    assert isinstance(numericas, StandardScaler)


@pytest.mark.parametrize("builder,params", [(make_tree, (3, 5)), (make_knn, (5, "distance"))])
def test_pipelines_fit_and_predict(builder, params):
    df = _toy()
    cols = feature_columns("con_clima")
    model = builder("con_clima", *params).fit(df[cols], df["ratio"])
    pred = model.predict(df[cols])
    assert pred.shape == (len(df),)
    assert np.isfinite(pred).all()


def test_search_only_receives_train_and_validation():
    assert list(inspect.signature(search).parameters) == ["train", "val"]


def test_select_best_uses_validation_mae():
    busqueda = pd.DataFrame(
        {
            "familia": ["knn", "knn", "arbol"],
            "variables": ["con_clima"] * 3,
            "mae_ratio_val": [0.20, 0.15, 0.18],
            "mae_ratio_train": [0.0, 0.30, 0.10],
        }
    )
    best = select_best(busqueda)
    assert best.iloc[0]["mae_ratio_val"] == pytest.approx(0.15)
    assert len(best) == 2


def test_dummy_baseline_uses_fit_means_by_crop():
    fit = pd.DataFrame({"cultivo": ["Maíz", "Maíz", "Soja total"], "ratio": [1.0, 1.2, 0.9]})
    ev = pd.DataFrame({"cultivo": ["Soja total", "Maíz"]})
    assert predict_baseline("dummy", fit, ev).tolist() == pytest.approx([0.9, 1.1])
    assert predict_baseline("prev5", fit, ev).tolist() == [1.0, 1.0]


@pytest.mark.skipif(not REGRESOR_FILE.exists(), reason="Falta entrenar: python -m src.regresion")
def test_saved_regressor_predicts_from_raw_columns():
    model = joblib.load(REGRESOR_FILE)
    df = _toy(5)
    pred = model.predict(df[list(model.feature_names_in_)])
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
