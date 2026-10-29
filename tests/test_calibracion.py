"""Tests del experimento de calibración (mejora pendiente de la Entrega 4)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.calibracion import calibrate_model, choose_threshold, delivered_config, evaluar_variantes
from src.clasificacion import (
    CLIMA_COLS,
    TARGET_CLF,
    _clf_metrics,
    apply_threshold,
    feature_columns,
    make_tree,
)
from src.config import CLASIFICADOR_METADATA, DATASET_MODELADO_CLIMA


def _toy(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "cultivo": rng.choice(["Maiz", "Trigo total"], size=n),
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


def test_choose_threshold_minimizes_cost():
    """El umbral elegido debe ser el de menor costo, sin depender de una rejilla fija."""
    proba = np.array([0.10, 0.22, 0.33, 0.47, 0.61, 0.88])
    y = np.array([0, 0, 1, 1, 0, 1])
    umbral, metricas = choose_threshold(proba, y)

    costos = {
        u: _clf_metrics(y, apply_threshold(proba, u))["costo"] for u in np.unique(proba)
    }
    mejor = min(costos.values())
    assert metricas["costo"] == pytest.approx(mejor)
    assert umbral == pytest.approx(min(costos, key=costos.get))
    assert umbral == pytest.approx(0.33)


def test_choose_threshold_considers_values_between_fixed_grid_points():
    """Un umbral fuera de la rejilla de 0,05 también tiene que ser candidato."""
    proba = np.array([0.44, 0.44, 0.51, 0.51])
    y = np.array([0, 1, 1, 1])
    umbral, _ = choose_threshold(proba, y)
    assert umbral == pytest.approx(0.44)
    assert umbral not in [round(u, 2) for u in np.arange(0.05, 0.951, 0.05)]


def test_calibrate_model_wraps_pipeline():
    pipe = make_tree("con_clima", max_depth=3, min_samples_leaf=5)
    calibrado = calibrate_model(pipe)
    assert isinstance(calibrado, CalibratedClassifierCV)


def test_sigmoid_on_frozen_tree_preserves_order():
    """La calibración sigmoidea de un árbol es monótona: no puede cambiar el ranking."""
    df = _toy(200)
    cols = feature_columns("con_clima")
    mitad = len(df) // 2
    train, val = df.iloc[:mitad], df.iloc[mitad:]
    base = make_tree("con_clima", max_depth=4, min_samples_leaf=5).fit(
        train[cols], train[TARGET_CLF]
    )
    calibrado = CalibratedClassifierCV(
        estimator=FrozenEstimator(base), method="sigmoid"
    ).fit(val[cols], val[TARGET_CLF])

    cruda = base.predict_proba(val[cols])[:, 1]
    ajustada = calibrado.predict_proba(val[cols])[:, 1]
    rho = pd.Series(cruda).corr(pd.Series(ajustada), method="spearman")
    assert rho == pytest.approx(1.0)


@pytest.mark.skipif(
    not CLASIFICADOR_METADATA.exists(),
    reason="Falta entrenar: python -m src.clasificacion",
)
def test_delivered_config_matches_saved_metadata():
    cfg = delivered_config()
    assert cfg["familia"] == "arbol"
    assert cfg["umbral"] > 0
    assert set(cfg["params"]) == {"max_depth", "min_samples_leaf", "class_weight"}


@pytest.mark.skipif(
    not (CLASIFICADOR_METADATA.exists() and DATASET_MODELADO_CLIMA.exists()),
    reason="Falta entrenar o falta el dataset de modelado",
)
def test_variants_are_measured_only_on_validation():
    tabla = evaluar_variantes()
    assert list(tabla["variante"]) == [
        "crudo (entregado)",
        "sigmoide con validación cruzada interna",
        "sigmoide ajustada en validación",
    ]
    assert "recall_test" not in tabla.columns
    assert tabla["costo_val"].notna().all()
