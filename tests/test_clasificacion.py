"""Tests de la Entrega 4: clasificación directa de campaña mala."""
import inspect

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.clasificacion import (
    CLIMA_COLS,
    FEATURE_SETS,
    FORBIDDEN,
    TARGET_CLF,
    UMBRAL_TEORICO,
    agreement_table,
    apply_threshold,
    candidates,
    combined_alert,
    costo_total,
    error_type,
    feature_columns,
    make_knn,
    make_nb,
    make_tree,
    search,
    select_best,
    with_threshold,
)
from src.config import CLASIFICADOR_FILE, CLASIFICADOR_METADATA, COST_FN, COST_FP


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


def test_candidates_cover_required_families():
    familias = {familia for familia, _, _ in candidates()}
    assert familias == {"arbol", "knn", "nb"}


def test_knn_scales_inside_pipeline():
    pipe = make_knn("con_clima", n_neighbors=5, weights="uniform")
    assert isinstance(pipe.steps[-1][1], KNeighborsClassifier)
    numericas = pipe.named_steps["prep"].transformers[2][1]
    assert isinstance(numericas, StandardScaler)


def test_nb_is_gaussian_nb():
    pipe = make_nb("con_clima", var_smoothing=1e-3)
    assert isinstance(pipe.named_steps["modelo"], GaussianNB)


def test_tree_class_weight_is_configurable():
    pipe = make_tree("con_clima", max_depth=4, min_samples_leaf=10, class_weight=None)
    clf = pipe.named_steps["modelo"]
    assert isinstance(clf, DecisionTreeClassifier)
    assert clf.class_weight is None


@pytest.mark.parametrize(
    "builder,params",
    [(make_tree, (4, 10)), (make_knn, (5, "distance")), (make_nb, (1e-3,))],
)
def test_pipelines_fit_and_predict(builder, params):
    df = _toy()
    cols = feature_columns("con_clima")
    model = builder("con_clima", *params).fit(df[cols], df[TARGET_CLF])
    pred = model.predict(df[cols])
    assert pred.shape == (len(df),)
    assert set(pred).issubset({0, 1})


def test_threshold_wrapper_matches_manual_rule():
    """predict del modelo guardado debe coincidir con la regla usada en la búsqueda."""
    df = _toy(200)
    cols = feature_columns("con_clima")
    base = make_tree("con_clima", max_depth=3, min_samples_leaf=5)
    for umbral in [0.2, 0.5, 0.8]:
        wrapped = with_threshold(base, umbral).fit(df[cols], df[TARGET_CLF])
        assert isinstance(wrapped, FixedThresholdClassifier)
        proba = wrapped.estimator_.predict_proba(df[cols])[:, 1]
        np.testing.assert_array_equal(wrapped.predict(df[cols]), apply_threshold(proba, umbral))


def test_search_signature_no_test():
    """La búsqueda solo puede recibir train y val."""
    assert list(inspect.signature(search).parameters) == ["train", "val"]


def test_select_best_uses_costo_and_skips_dummy():
    busqueda = pd.DataFrame(
        {
            "familia": ["dummy", "arbol", "arbol", "knn"],
            "variables": ["-", "con_clima", "con_clima", "con_clima"],
            "umbral": [np.nan, 0.5, 0.3, 0.5],
            "costo_val": [100.0, 300.0, 200.0, 250.0],
            "f1_val": [0.0, 0.2, 0.3, 0.1],
        }
    )
    best = select_best(busqueda)
    assert "dummy" not in set(best["familia"])
    assert best.iloc[0]["costo_val"] == pytest.approx(200.0)
    assert best.iloc[0]["umbral"] == pytest.approx(0.3)


def test_costo_asimetrico():
    """FN debe pesar COST_FN y FP debe pesar COST_FP."""
    c = costo_total(tp=0, fp=10, fn=5, tn=100)
    expected = COST_FN * 5 + COST_FP * 10
    assert c == pytest.approx(expected)
    assert UMBRAL_TEORICO == pytest.approx(COST_FP / (COST_FP + COST_FN))


def test_error_type_and_agreement():
    y = np.array([1, 0, 1, 0])
    clf = np.array([1, 1, 0, 0])
    reg = np.array([1, 0, 0, 1])
    assert list(error_type(y, clf)) == ["VP", "FP", "FN", "VN"]
    np.testing.assert_array_equal(combined_alert(clf, reg), [1, 1, 0, 1])
    acuerdo = agreement_table(pd.DataFrame({TARGET_CLF: y}), clf, reg)
    assert acuerdo["n"].sum() == len(y)
    assert acuerdo["malas_reales"].sum() == y.sum()


@pytest.mark.skipif(
    not CLASIFICADOR_FILE.exists(),
    reason="Falta entrenar: python -m src.clasificacion",
)
def test_saved_classifier_predicts_binary():
    model = joblib.load(CLASIFICADOR_FILE)
    assert isinstance(model, FixedThresholdClassifier)
    df = _toy(10)
    pred = model.predict(df[list(model.feature_names_in_)])
    assert pred.shape == (10,)
    assert set(pred).issubset({0, 1})


@pytest.mark.skipif(
    not CLASIFICADOR_METADATA.exists(),
    reason="Falta entrenar: python -m src.clasificacion",
)
def test_saved_classifier_beats_dummy_in_validation():
    import json

    meta = json.loads(CLASIFICADOR_METADATA.read_text(encoding="utf-8"))
    model = joblib.load(CLASIFICADOR_FILE)
    assert model.threshold == pytest.approx(meta["umbral_decision"])
    # El Dummy no alerta nunca: su costo en validación es COST_FN × positivos.
    from src.clasificacion import load_data, split_data

    _, val, _ = split_data(load_data())
    assert meta["metricas_val"]["costo"] < COST_FN * val[TARGET_CLF].sum()
