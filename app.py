"""Aplicación Flask del TP Integrador.

En la Entrega 2 la app queda como esqueleto: muestra el estado del proyecto y
un endpoint de salud. La carga de pipelines y el formulario de predicción se
completan en la Clase 4.

La app nunca entrena modelos ni acepta archivos joblib o pickle externos.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "entrega": 2})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
