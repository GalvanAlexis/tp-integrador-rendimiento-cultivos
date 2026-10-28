"""Tests de la aplicacion Flask: rutas, validacion del formulario y resultado."""
import pytest

from src.config import CLASIFICADOR_FILE, DATASET_MODELADO_CLIMA, REGRESOR_FILE

pytestmark = pytest.mark.skipif(
    not (CLASIFICADOR_FILE.exists() and REGRESOR_FILE.exists() and DATASET_MODELADO_CLIMA.exists()),
    reason="Faltan los modelos o el dataset: python -m src.regresion y python -m src.clasificacion",
)

EJEMPLO = {"partido": "Pergamino", "cultivo": "Soja total", "anio": "2022"}


@pytest.fixture()
def client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


def _form_historico(client, **cambios):
    datos = client.get("/api/historico", query_string=EJEMPLO).get_json()["valores"]
    form = {**EJEMPLO, **{k: str(round(v, 2)) for k, v in datos.items()}}
    form.update(cambios)
    return form


def test_paginas_responden(client):
    assert client.get("/").status_code == 200
    assert client.get("/predecir").status_code == 200
    assert client.get("/health").get_json()["clasificador"] is True


def test_api_historico(client):
    r = client.get("/api/historico", query_string=EJEMPLO)
    assert r.status_code == 200
    valores = r.get_json()["valores"]
    assert valores["prev5"] > 0
    assert "clima_precip_total_mm" in valores

    r = client.get("/api/historico", query_string={**EJEMPLO, "anio": "1900"})
    assert r.status_code == 404


def test_post_muestra_los_dos_modelos(client):
    html = client.post("/predecir", data=_form_historico(client)).get_data(as_text=True)
    assert "¿Cuánto va a rendir?" in html
    assert "¿Será una campaña mala?" in html
    assert "Lo que pasó realmente" in html
    assert "Escenario modificado" not in html


def test_campo_vacio_da_mensaje_claro(client):
    html = client.post(
        "/predecir", data=_form_historico(client, clima_precip_total_mm="")
    ).get_data(as_text=True)
    assert "Falta completar «Precipitación total»." in html
    assert "could not convert" not in html


def test_valor_fuera_de_rango(client):
    html = client.post(
        "/predecir", data=_form_historico(client, clima_humedad_media_pct="150")
    ).get_data(as_text=True)
    assert "«Humedad media» tiene que estar entre 0 y 100 %." in html


def test_escenario_modificado(client):
    html = client.post(
        "/predecir", data=_form_historico(client, clima_precip_total_mm="900")
    ).get_data(as_text=True)
    assert "Escenario modificado" in html
    assert "no se compara con lo que pasó" in html


def test_enlace_de_ejemplo_predice(client):
    html = client.get("/predecir", query_string={**EJEMPLO, "predecir": "1"}).get_data(as_text=True)
    assert "Alerta de alta confianza" in html
    assert "Campaña de prueba" in html


def test_advertencia_de_limitaciones(client):
    """La app debe advertir sus límites, como pide la consigna."""
    inicio = client.get("/").get_data(as_text=True)
    assert "Cuándo no conviene usarla" in inicio
    assert "nunca como la única fuente" in inicio

    formulario = client.get("/predecir").get_data(as_text=True)
    assert "Limitaciones de esta herramienta" in formulario
    assert "No es una garantía" in formulario
    assert "no como la única fuente" in formulario
