"""Aplicacion Flask del TP Integrador - Entrega 5.

Endpoints:
  GET  /               -> pagina principal
  GET  /predecir       -> formulario; con partido, cultivo y anio autocompleta
                          y con predecir=1 ademas ejecuta la prediccion
  POST /predecir       -> ejecuta predict y devuelve resultado
  GET  /academia       -> materiales publicados (presentacion e informe)
  GET  /academia/...   -> sirve uno de esos materiales, de una lista blanca
  GET  /health         -> estado de la app (JSON)
  GET  /api/datos      -> JSON con partidos, cultivos y anios disponibles
  GET  /api/historico  -> JSON con los valores historicos de una campana

La app no entrena: solo carga los pipelines guardados y ejecuta predict.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

from src.config import CLIMA_COLS, MODEL_LABELS, TEST_START, THRESHOLD

ROOT = Path(__file__).resolve().parent
MODELOS_DIR = ROOT / "modelos"
PROCESADO_DIR = ROOT / "datos" / "procesado"
DATASET_CLIMA = PROCESADO_DIR / "dataset_modelado_con_clima.csv"
REGRESOR_META = MODELOS_DIR / "regresor_metadata.json"
CLASIFICADOR_META = MODELOS_DIR / "clasificador_metadata.json"
COMPARACION = PROCESADO_DIR / "clasificacion_comparacion.csv"
ACUERDO = PROCESADO_DIR / "clasificacion_acuerdo_sistemas_test.csv"
PRESENTACION_HTML = ROOT / "presentacion" / "presentacion.html"
PRESENTACION_PDF = ROOT / "presentacion" / "presentacion.pdf"
INFORME_PDF = ROOT / "informe" / "informe_final.pdf"

# Materiales de la seccion Academia. Es una lista blanca: la ruta solo puede
# servir uno de estos archivos, nunca una ruta escrita en el pedido.
MATERIALES = [
    {
        "slug": "presentacion",
        "titulo": "Presentación",
        "formato": "HTML",
        "detalle": (
            "Catorce diapositivas navegables, con las figuras incrustadas y las notas "
            "del orador. Se abre en el navegador y funciona sin conexión."
        ),
        "accion": "Abrir la presentación",
        "archivo": PRESENTACION_HTML,
        "mimetype": "text/html",
        "nombre_descarga": "presentacion-tp-integrador.html",
    },
    {
        "slug": "presentacion.pdf",
        "titulo": "Presentación en PDF",
        "formato": "PDF",
        "detalle": "La misma presentación, lista para proyectar o imprimir.",
        "accion": "Abrir el PDF",
        "archivo": PRESENTACION_PDF,
        "mimetype": "application/pdf",
        "nombre_descarga": "presentacion-tp-integrador.pdf",
    },
    {
        "slug": "informe.pdf",
        "titulo": "Informe final",
        "formato": "PDF",
        "detalle": (
            "El informe completo: problema, datos, protocolo, regresión, clasificación, "
            "integración y las ocho preguntas de la consigna."
        ),
        "accion": "Abrir el informe",
        "archivo": INFORME_PDF,
        "mimetype": "application/pdf",
        "nombre_descarga": "informe-final-tp-integrador.pdf",
    },
]

# Escala del ratio que se dibuja en el resultado.
RATIO_MIN, RATIO_MAX = 0.4, 1.4

# Campos del formulario: etiqueta, unidad, minimo, maximo y si es entero.
AGRO_FIELDS = {
    "superficie_sembrada": ("Superficie sembrada", "ha", 1, 1_000_000, False),
    "prev5": ("Promedio de las 5 campañas anteriores", "kg/ha", 1, 20_000, False),
    "prev5_ceros": ("Campañas sin cosecha", "", 0, 5, True),
}
CLIMA_FIELDS = {
    "clima_precip_total_mm": ("Precipitación total", "mm", 0, 3000, False),
    "clima_precip_pre_siembra_mm": ("Precipitación pre-siembra", "mm", 0, 2000, False),
    "clima_dias_lluvia": ("Días de lluvia", "días", 0, 200, True),
    "clima_tmax_media_c": ("Temperatura máxima media", "°C", -10, 50, False),
    "clima_tmin_media_c": ("Temperatura mínima media", "°C", -20, 40, False),
    "clima_humedad_media_pct": ("Humedad media", "%", 0, 100, False),
    "clima_dias_tmax_30": ("Días con máxima > 30 °C", "días", 0, 200, True),
    "clima_sequia_max_dias": ("Sequía máxima", "días", 0, 200, True),
    "clima_dias_helada": ("Días de helada", "días", 0, 200, True),
}
assert set(CLIMA_FIELDS) == set(CLIMA_COLS)
CAMPOS = {**AGRO_FIELDS, **CLIMA_FIELDS}

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Carga de modelos y datos al inicio
# ---------------------------------------------------------------------------
_regresor = None
_clasificador = None
_df_hist: pd.DataFrame | None = None
_resumen: dict = {}


def _leer_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _cargar_resumen() -> dict:
    """Metricas de test que la interfaz muestra para explicar cada modelo."""
    resumen: dict = {"umbral_ratio": THRESHOLD}
    meta_reg = _leer_json(REGRESOR_META)
    meta_clf = _leer_json(CLASIFICADOR_META)
    if meta_reg:
        resumen["regresor_desc"] = f"KNN, {meta_reg['params']['n_neighbors']} vecinos"
    if meta_clf:
        resumen["clasificador_desc"] = (
            f"Árbol de decisión, profundidad {meta_clf['params']['max_depth']}"
        )
        resumen["umbral_clasificador"] = meta_clf.get("umbral_decision")

    if COMPARACION.exists() and meta_clf:
        comp = pd.read_csv(COMPARACION, encoding="utf-8").set_index("modelo")
        nombre_clf = f"{meta_clf['modelo']} ({meta_clf['variables']})"
        for clave, nombre in [("reg", MODEL_LABELS["regresion"]), ("clf", nombre_clf)]:
            if nombre in comp.index:
                fila = comp.loc[nombre]
                resumen[clave] = {
                    "recall": float(fila["recall"]),
                    "tp": int(fila["tp"]),
                    "fp": int(fila["fp"]),
                    "positivos": int(fila["tp"] + fila["fn"]),
                }

    if ACUERDO.exists():
        acuerdo = pd.read_csv(ACUERDO, encoding="utf-8")
        resumen["acuerdo"] = {
            (r.clasificador == "alerta", r.regresion == "alerta"): {
                "precision": float(r.precision),
                "n": int(r.n),
                "malas": int(r.malas_reales),
            }
            for r in acuerdo.itertuples()
        }
    return resumen


def _load_models() -> None:
    global _regresor, _clasificador, _df_hist, _resumen
    reg_path = MODELOS_DIR / "regresor.joblib"
    clf_path = MODELOS_DIR / "clasificador.joblib"
    if reg_path.exists():
        _regresor = joblib.load(reg_path)
    if clf_path.exists():
        _clasificador = joblib.load(clf_path)
    if DATASET_CLIMA.exists():
        _df_hist = pd.read_csv(DATASET_CLIMA, encoding="utf-8")
    _resumen = _cargar_resumen()


_load_models()


@app.context_processor
def _estado_modelos() -> dict:
    return {"reg_ok": _regresor is not None, "clf_ok": _clasificador is not None}


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
    return ["Girasol", "Maíz", "Soja total", "Trigo total"]


def _anios() -> dict[str, list[int]]:
    """Campanas disponibles, separadas segun si los modelos las vieron al entrenar."""
    if _df_hist is not None:
        todos = sorted(_df_hist["campania_inicio"].unique().tolist(), reverse=True)
    else:
        todos = list(range(2024, 2014, -1))
    return {
        "prueba": [a for a in todos if a >= TEST_START],
        "entrenamiento": [a for a in todos if a < TEST_START],
    }


def _autocompletar(partido: str, cultivo: str, anio: int | None) -> dict:
    """Devuelve las variables de entrada y el resultado real de una campana historica."""
    if _df_hist is None or not (partido and cultivo and anio):
        return {}
    fila = _df_hist[
        (_df_hist["municipio_nombre"] == partido)
        & (_df_hist["cultivo"] == cultivo)
        & (_df_hist["campania_inicio"] == anio)
    ]
    if fila.empty:
        return {}
    row = fila.iloc[0]
    resultado = {col: float(row[col]) for col in CAMPOS if col in row}
    resultado["prev5_ceros"] = int(row["prev5_ceros"])
    resultado.update(
        {
            "rendimiento_real": float(row["rendimiento"]),
            "ratio_real": float(row["ratio"]),
            "mala85_real": int(row["mala85"]),
        }
    )
    return resultado


def _parse_anio(valor: str) -> int | None:
    return int(valor) if valor and valor.isdigit() else None


def _validar(data: dict) -> tuple[dict, list[str]]:
    """Convierte el formulario a numeros y devuelve los errores en lenguaje claro."""
    valores: dict = {}
    errores: list[str] = []
    cultivo = data.get("cultivo", "")
    if cultivo not in _cultivos():
        errores.append("Elegí un cultivo de la lista.")
    valores["cultivo"] = cultivo

    for col, (etiqueta, unidad, minimo, maximo, entero) in CAMPOS.items():
        texto = str(data.get(col, "")).strip().replace(",", ".")
        if texto == "":
            errores.append(f"Falta completar «{etiqueta}».")
            continue
        try:
            numero = float(texto)
        except ValueError:
            errores.append(f"«{etiqueta}» tiene que ser un número.")
            continue
        if not minimo <= numero <= maximo:
            rango = f"entre {minimo:g} y {maximo:g}" + (f" {unidad}" if unidad else "")
            errores.append(f"«{etiqueta}» tiene que estar {rango}.")
            continue
        valores[col] = int(round(numero)) if entero else numero
    return valores, errores


def _posicion(ratio: float) -> float:
    """Posicion (0 a 100) de un ratio sobre la escala dibujada."""
    r = min(max(ratio, RATIO_MIN), RATIO_MAX)
    return round((r - RATIO_MIN) / (RATIO_MAX - RATIO_MIN) * 100, 1)


def _predecir(valores: dict) -> dict:
    """Ejecuta el regresor y el clasificador sobre valores ya validados."""
    fila = pd.DataFrame([valores])
    resultado: dict = {"pos_umbral": _posicion(THRESHOLD)}

    if _regresor is not None:
        cols_reg = list(_regresor.feature_names_in_)
        ratio_pred = float(_regresor.predict(fila[cols_reg])[0])
        resultado["ratio_pred"] = ratio_pred
        resultado["rend_pred_kg"] = ratio_pred * valores["prev5"]
        resultado["alerta_regresion"] = int(ratio_pred < THRESHOLD)
        resultado["pos_ratio"] = _posicion(ratio_pred)

    if _clasificador is not None:
        cols_clf = list(_clasificador.feature_names_in_)
        resultado["alerta_clasificador"] = int(_clasificador.predict(fila[cols_clf])[0])
        proba = _clasificador.estimator_.predict_proba(fila[cols_clf])[0, 1]
        resultado["proba_clasificador"] = float(proba)
        resultado["umbral_clasificador"] = float(_clasificador.threshold)

    # La alerta principal es la del clasificador, elegido por costo en validación.
    # La regresión se muestra como segunda opinión: en test, la alerta combinada
    # (alerta si alguno alerta) costó más que el clasificador solo.
    alerta_clf = resultado.get("alerta_clasificador")
    alerta_reg = resultado.get("alerta_regresion", 0)
    if alerta_clf is None:
        alerta_clf = alerta_reg
    resultado["alerta_final"] = alerta_clf
    if alerta_clf and alerta_reg:
        resultado["nivel"] = "alta"
    elif alerta_clf:
        resultado["nivel"] = "clasificador"
    elif alerta_reg:
        resultado["nivel"] = "solo_regresion"
    else:
        resultado["nivel"] = "normal"
    resultado["acuerdo"] = _resumen.get("acuerdo", {}).get((bool(alerta_clf), bool(alerta_reg)))
    return resultado


def _cambios_escenario(valores: dict, autocomp: dict) -> list[dict]:
    """Variables que el usuario modifico respecto del dato historico."""
    if not autocomp:
        return []
    cambios = []
    for col, (etiqueta, unidad, *_rest) in CAMPOS.items():
        # El formulario muestra los valores redondeados a 2 decimales.
        if col in valores and col in autocomp and abs(valores[col] - round(autocomp[col], 2)) > 0.006:
            cambios.append(
                {"etiqueta": etiqueta, "unidad": unidad, "real": autocomp[col], "nuevo": valores[col]}
            )
    return cambios


def _real(autocomp: dict, resultado: dict) -> dict | None:
    """Resultado real de la campana y si cada modelo acerto."""
    if not autocomp or "ratio_real" not in autocomp:
        return None
    mala = bool(autocomp["mala85_real"])
    real = {
        "ratio": autocomp["ratio_real"],
        "rendimiento": autocomp["rendimiento_real"],
        "mala": mala,
        "pos": _posicion(autocomp["ratio_real"]),
    }
    if "alerta_regresion" in resultado:
        real["acierto_regresion"] = bool(resultado["alerta_regresion"]) == mala
    if "alerta_clasificador" in resultado:
        real["acierto_clasificador"] = bool(resultado["alerta_clasificador"]) == mala
    return real


def _render_predecir(data: dict, ejecutar: bool):
    partido = data.get("partido", "")
    cultivo = data.get("cultivo", "")
    anio = _parse_anio(str(data.get("anio", "")))
    autocomp = _autocompletar(partido, cultivo, anio)

    resultado = None
    errores: list[str] = []
    cambios: list[dict] = []
    real = None
    if ejecutar:
        valores, errores = _validar(data)
        if not errores:
            try:
                resultado = _predecir(valores)
            except Exception as e:  # el modelo guardado no acepta la entrada
                errores = [f"No se pudo calcular la predicción: {e}"]
        if resultado:
            cambios = _cambios_escenario(valores, autocomp)
            real = _real(autocomp, resultado)

    form_data = {k: data[k] for k in CAMPOS if k in data} if data else {}
    return render_template(
        "predecir.html",
        partidos=_partidos(),
        cultivos=_cultivos(),
        anios=_anios(),
        partido=partido,
        cultivo=cultivo,
        anio=anio,
        anio_prueba=anio is not None and anio >= TEST_START,
        autocomp=autocomp,
        form_data=form_data,
        agro_fields=AGRO_FIELDS,
        clima_fields=CLIMA_FIELDS,
        resultado=resultado,
        errores=errores,
        cambios=cambios,
        real=real,
        resumen=_resumen,
        ratio_min=RATIO_MIN,
        ratio_max=RATIO_MAX,
    )


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html", resumen=_resumen)


@app.get("/predecir")
def predecir_form():
    data = request.args.to_dict()
    ejecutar = data.pop("predecir", "") == "1"
    if ejecutar:
        # Desde un enlace de ejemplo: se predice con los valores historicos.
        autocomp = _autocompletar(
            data.get("partido", ""), data.get("cultivo", ""), _parse_anio(data.get("anio", ""))
        )
        data = {**{k: autocomp[k] for k in CAMPOS if k in autocomp}, **data}
    return _render_predecir(data, ejecutar=ejecutar)


@app.post("/predecir")
def predecir_post():
    return _render_predecir(request.form.to_dict(), ejecutar=True)


@app.get("/academia")
def academia():
    """Pagina con los materiales publicados: presentacion e informe."""
    return render_template(
        "academia.html",
        materiales=[{**m, "disponible": m["archivo"].exists()} for m in MATERIALES],
    )


@app.get("/academia/<slug>")
def academia_archivo(slug: str):
    """Sirve uno de los materiales de la lista blanca, sin aceptar rutas."""
    material = next((m for m in MATERIALES if m["slug"] == slug), None)
    if material is None or not material["archivo"].exists():
        abort(404)
    return send_file(
        material["archivo"],
        mimetype=material["mimetype"],
        download_name=material["nombre_descarga"],
        conditional=True,
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "entrega": 5,
            "regresor": _regresor is not None,
            "clasificador": _clasificador is not None,
        }
    )


@app.get("/api/datos")
def api_datos():
    anios = _anios()
    return jsonify(
        {
            "partidos": _partidos(),
            "cultivos": _cultivos(),
            "anios": sorted(anios["prueba"] + anios["entrenamiento"]),
        }
    )


@app.get("/api/historico")
def api_historico():
    autocomp = _autocompletar(
        request.args.get("partido", ""),
        request.args.get("cultivo", ""),
        _parse_anio(request.args.get("anio", "")),
    )
    if not autocomp:
        return jsonify({"encontrado": False}), 404
    return jsonify({"encontrado": True, "valores": {k: autocomp[k] for k in CAMPOS}})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
