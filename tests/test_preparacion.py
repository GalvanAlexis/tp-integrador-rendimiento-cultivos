import pandas as pd
import pytest

from src.clima import _window_dates, normalize_name
from src.preparar_datos import _parse_campaign, add_prev5, add_split, add_targets


def test_parse_campaign():
    assert _parse_campaign("1969/70") == (1969, 1970)
    assert _parse_campaign("1999/00") == (1999, 2000)
    assert _parse_campaign("2024/25") == (2024, 2025)
    assert _parse_campaign("2025/26") == (2025, 2026)


def test_prev5_uses_only_previous_campaigns():
    df = pd.DataFrame(
        {
            "cultivo": ["Maíz"] * 8,
            "municipio_id": [1] * 8,
            "campania_inicio": list(range(2000, 2008)),
            "rendimiento": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )
    out = add_prev5(df).sort_values("campania_inicio")
    row_2005 = out[out["campania_inicio"] == 2005].iloc[0]
    assert row_2005["prev5"] == pytest.approx(30.0)
    assert row_2005["prev5"] != row_2005["rendimiento"]

    # Cambiar el rendimiento de la campaña actual no debe cambiar su propio prev5.
    df2 = df.copy()
    df2.loc[df2["campania_inicio"] == 2005, "rendimiento"] = 999.0
    out2 = add_prev5(df2).sort_values("campania_inicio")
    row_2005_b = out2[out2["campania_inicio"] == 2005].iloc[0]
    assert row_2005_b["prev5"] == pytest.approx(30.0)

    row_2006 = out[out["campania_inicio"] == 2006].iloc[0]
    assert row_2006["prev5"] == pytest.approx(40.0)


def test_prev5_is_null_when_a_campaign_is_missing():
    df = pd.DataFrame(
        {
            "cultivo": ["Maíz"] * 6,
            "municipio_id": [1] * 6,
            "campania_inicio": [2000, 2001, 2002, 2004, 2005, 2006],
            "rendimiento": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    out = add_prev5(df).sort_values("campania_inicio")
    row_2006 = out[out["campania_inicio"] == 2006].iloc[0]
    assert pd.isna(row_2006["prev5"])


def test_targets_and_threshold():
    df = pd.DataFrame(
        {
            "rendimiento": [80.0, 85.0, 120.0],
            "prev5": [100.0, 100.0, 100.0],
        }
    )
    out = add_targets(df)
    assert out["ratio"].tolist() == [0.8, 0.85, 1.2]
    assert out["mala80"].tolist() == [0, 0, 0]
    assert out["mala85"].tolist() == [1, 0, 0]
    assert out["mala90"].tolist() == [1, 1, 0]


def test_split_boundaries():
    df = pd.DataFrame(
        {
            "campania_inicio": [1980, 1981, 2013, 2014, 2018, 2019, 2024],
        }
    )
    out = add_split(df)
    assert out["split"].tolist() == [
        "historico",
        "train",
        "train",
        "validacion",
        "validacion",
        "test",
        "test",
    ]


def test_normalize_name():
    assert normalize_name("Bahía Blanca") == "bahia blanca"
    assert normalize_name("25 De Mayo") == "25 de mayo"
    assert (
        normalize_name("Coronel De Marina L Rosales")
        == "coronel de marina leonardo rosales"
    )


def test_window_dates():
    start, end = _window_dates("Maíz", 2022)
    assert str(start.date()) == "2022-09-01"
    assert str(end.date()) == "2023-01-31"

    start, end = _window_dates("Trigo total", 2022)
    assert str(start.date()) == "2022-06-01"
    assert str(end.date()) == "2022-11-15"
