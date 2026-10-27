"""Aplicacion Flask del TP Integrador - Entrega 4.

Endpoints:
  GET  /           -> pagina principal
  GET  /predecir   -> formulario de prediccion (autocompletado)
  POST /predecir   -> ejecuta predict y devuelve resultado
  GET  /health     -> estado de la app (JSON)
  GET  /api/datos  -> JSON con partidos, cultivos y anios disponibles

La app no entrena: solo carga los pipelines guardados y ejecuta predict.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODELOS_DIR = ROOT / "modelos"
PROCESADO_DIR = ROOT / "datos" / "procesado"
DATASET_CLIMA = PROCESADO_DIR / "dataset_modelado_con_clima.csv"
REGRESOR_META = MODELOS_DIR / "regresor_metadata.json"
CLASIFICADOR_META = MODELOS_DIR / "clasificador_metadata.json"

THRESHOLD = 0.85

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Carga de modelos al inicio
# ---------------------------------------------------------------------------
_regresor = None
_clasificador = None
_df_hist: pd.DataFrame | None = None


def _load_models() -> None:
    global _regresor, _clasificador, _df_hist
    reg_path = MODELOS_DIR / "regresor.joblib"
    clf_path = MODELOS_DIR / "clasificador.joblib"
    if reg_path.exists():
        _regresor = joblib.load(reg_path)
    if clf_path.exists():
        _clasificador = joblib.load(clf_path)
    if DATASET_CLIMA.exists():
        _df_hist = pd.read_csv(DATASET_CLIMA, encoding="utf-8")


_load_models()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _partidos() -> list[str]:
    if _df_hist is not None:
        return sorted(_df_hist["municipio_nombre"].unique().tolist())
    return []


def _cultivos() -> list[str]:
    if _df_hist is not None:
        return sorted(_df_hist["cultivo"].unique().tolist())
    return ["Girasol", "Maiz", "Soja total", "Trigo total"]


def _anios() -> list[int]:
    """Ultimos 10 anos disponibles en el dataset historico."""
    if _df_hist is not None:
        return sorted(_df_hist["campania_inicio"].unique().tolist())[-10:]
    return list(range(2015, 2025))


def _autocompletar(partido: str, cultivo: str, anio: int) -> dict:
    """Devuelve prev5, prev5_ceros y variables climaticas del dataset historico."""
    if _df_hist is None:
        return {}
    fila = _df_hist[
        (_df_hist["municipio_nombre"] == partido)
        & (_df_hist["cultivo"] == cultivo)
        & (_df_hist["campania_inicio"] == anio)
    ]
    if fila.empty:
        return {}
    row = fila.iloc[0]
    clima_cols = [
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
    resultado = {
        "prev5": float(row.get("prev5", 0)),
        "prev5_ceros": int(row.get("prev5_ceros", 0)),
        "superficie_sembrada": float(row.get("superficie_sembrada", 0)),
        "rendimiento_real": float(row.get("rendimiento", 0)),
        "ratio_real": float(row.get("ratio", 0)),
        "mala85_real": int(row.get("mala85", 0)),
    }
    for col in clima_cols:
        if col in row:
            resultado[col] = float(row[col])
    return resultado


def _predecir(data: dict) -> dict:
    """Ejecuta el regresor y el clasificador sobre los datos del formulario."""
    cultivo = data["cultivo"]
    prev5 = float(data["prev5"])
    prev5_ceros = int(data["prev5_ceros"])
    superficie = float(data["superficie_sembrada"])

    clima_cols = [
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
    clima = {col: float(data.get(col, 0)) for col in clima_cols}

    fila = pd.DataFrame(
        [
            {
                "cultivo": cultivo,
                "superficie_sembrada": superficie,
                "prev5": prev5,
                "prev5_ceros": prev5_ceros,
                **clima,
            }
        ]
    )

    resultado: dict = {}

    if _regresor is not None:
        cols_reg = list(_regresor.feature_names_in_)
        ratio_pred = float(_regresor.predict(fila[cols_reg])[0])
        resultado["ratio_pred"] = round(ratio_pred, 4)
        resultado["rend_pred_kg"] = round(ratio_pred * prev5, 1)
        resultado["alerta_regresion"] = int(ratio_pred < THRESHOLD)

    if _clasificador is not None:
        cols_clf = list(_clasificador.feature_names_in_)
        alerta_clf = int(_clasificador.predict(fila[cols_clf])[0])
        resultado["alerta_clasificador"] = alerta_clf

    # Alerta final: si alguno de los dos sistemas emite alerta
    alerta_alguno = int(
        resultado.get("alerta_regresion", 0) or resultado.get("alerta_clasificador", 0)
    )
    resultado["alerta_final"] = alerta_alguno

    return resultado


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    reg_ok = (MODELOS_DIR / "regresor.joblib").exists()
    clf_ok = (MODELOS_DIR / "clasificador.joblib").exists()
    return render_template("index.html", reg_ok=reg_ok, clf_ok=clf_ok)


@app.get("/predecir")
def predecir_form():
    partido = request.args.get("partido", "")
    cultivo = request.args.get("cultivo", "")
    anio_str = request.args.get("anio", "")
    anio = int(anio_str) if anio_str.isdigit() else None
    autocomp = {}
    if partido and cultivo and anio:
        autocomp = _autocompletar(partido, cultivo, anio)
    return render_template(
        "predecir.html",
        partidos=_partidos(),
        cultivos=_cultivos(),
        anios=_anios(),
        partido=partido,
        cultivo=cultivo,
        anio=anio,
        autocomp=autocomp,
        form_data={},
        resultado=None,
    )


@app.post("/predecir")
def predecir_post():
    data = request.form.to_dict()
    try:
        resultado = _predecir(data)
    except Exception as e:
        resultado = {"error": str(e)}
    autocomp = {}
    partido = data.get("partido", "")
    cultivo = data.get("cultivo", "")
    anio_str = data.get("anio", "")
    anio = int(anio_str) if anio_str.isdigit() else None
    if partido and cultivo and anio:
        autocomp = _autocompletar(partido, cultivo, anio)
    return render_template(
        "predecir.html",
        partidos=_partidos(),
        cultivos=_cultivos(),
        anios=_anios(),
        partido=partido,
        cultivo=cultivo,
        anio=anio,
        autocomp=autocomp,
        form_data=data,
        resultado=resultado,
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "entrega": 4,
            "regresor": (MODELOS_DIR / "regresor.joblib").exists(),
            "clasificador": (MODELOS_DIR / "clasificador.joblib").exists(),
        }
    )


@app.get("/api/datos")
def api_datos():
    return jsonify(
        {
            "partidos": _partidos(),
            "cultivos": _cultivos(),
            "anios": _anios(),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
