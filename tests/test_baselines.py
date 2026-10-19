import numpy as np
import pytest

from src.baselines import classification_metrics, regression_metrics


def test_regression_metrics_perfect():
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    prev5 = np.array([100.0, 200.0, 300.0])
    out = regression_metrics(y_true, y_pred, prev5)
    assert out["mae_ratio"] == pytest.approx(0.0)
    assert out["rmse_ratio"] == pytest.approx(0.0)
    assert out["r2_ratio"] == pytest.approx(1.0)
    assert out["mae_kg"] == pytest.approx(0.0)


def test_regression_metrics_kg_scale():
    y_true = np.array([1.0, 0.5])
    y_pred = np.array([1.0, 1.0])
    prev5 = np.array([1000.0, 2000.0])
    out = regression_metrics(y_true, y_pred, prev5)
    assert out["mae_ratio"] == pytest.approx(0.25)
    # Errores en kg: 0 y 1000 -> MAE 500
    assert out["mae_kg"] == pytest.approx(500.0)


def test_classification_metrics_confusion():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    out = classification_metrics(y_true, y_pred)
    assert out["tp"] == 1
    assert out["fn"] == 1
    assert out["tn"] == 2
    assert out["fp"] == 0
    assert out["accuracy"] == pytest.approx(0.75)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(2 / 3)
    assert out["tasa_alertas"] == pytest.approx(0.25)


def test_classification_metrics_no_alerts():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    out = classification_metrics(y_true, y_pred)
    assert out["recall"] == 0.0
    assert out["precision"] == 0.0
    assert out["f1"] == 0.0
    assert out["fn"] == 2
