"""Tests de la seccion Academia: materiales publicados desde la app."""
import pytest

from app import MATERIALES, app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


def _slugs() -> list[str]:
    return [m["slug"] for m in MATERIALES]


def test_academia_lista_los_materiales(client):
    html = client.get("/academia").get_data(as_text=True)
    assert "Academia" in html
    for slug in _slugs():
        assert f'href="/academia/{slug}"' in html, f"falta el enlace a {slug}"


def test_la_navbar_lleva_a_academia(client):
    for ruta in ("/", "/predecir", "/academia"):
        html = client.get(ruta).get_data(as_text=True)
        assert 'href="/academia"' in html, f"falta Academia en la navbar de {ruta}"


def test_academia_marca_la_pagina_actual(client):
    html = client.get("/academia").get_data(as_text=True)
    assert 'href="/academia" aria-current="page"' in html


@pytest.mark.skipif(
    not all(m["archivo"].exists() for m in MATERIALES),
    reason="Faltan los materiales: construir presentacion.html, presentacion.pdf e informe_final.pdf",
)
def test_los_materiales_se_sirven(client):
    for material in MATERIALES:
        r = client.get(f"/academia/{material['slug']}")
        assert r.status_code == 200, material["slug"]
        assert r.headers["Content-Type"].startswith(material["mimetype"])
        assert "public" in r.headers.get("Cache-Control", ""), material["slug"]
        assert "s-maxage=604800" in r.headers.get("Cache-Control", ""), material["slug"]
        datos = r.get_data()
        assert len(datos) > 1000, material["slug"]
        if material["mimetype"] == "application/pdf":
            assert datos[:4] == b"%PDF", material["slug"]
        else:
            assert b'<section class="slide"' in datos, material["slug"]


def test_un_slug_desconocido_da_404(client):
    assert client.get("/academia/notas").status_code == 404
    assert client.get("/academia/apuntes.pdf").status_code == 404
    assert client.get("/academia/presentacion.pdf").status_code == 404


def test_la_ruta_no_sirve_archivos_fuera_de_la_lista(client):
    """La lista blanca no puede convertirse en lectura arbitraria del repositorio."""
    assert client.get("/academia/app.py").status_code == 404
    assert client.get("/academia/..%2Fapp.py").status_code == 404
    assert client.get("/academia/%2Fetc%2Fpasswd").status_code == 404


def test_landing_page_incluye_acceso_a_presentacion(client):
    """La pagina de inicio ofrece accesos directos a la presentacion e informe final."""
    html = client.get("/").get_data(as_text=True)
    assert 'href="/academia/presentacion"' in html
    assert 'href="/academia/informe.pdf"' in html
    assert 'href="/academia/presentacion.pdf"' not in html


def test_navbar_contiene_dropdown_con_materiales(client):
    """El navbar incluye el menu desplegable con accesos a la presentacion en cada pagina."""
    for ruta in ("/", "/predecir", "/academia"):
        html = client.get(ruta).get_data(as_text=True)
        assert 'nav-dropdown' in html, f"falta nav-dropdown en {ruta}"
        assert 'href="/academia/presentacion"' in html, f"falta link a presentacion en navbar de {ruta}"
        assert 'href="/academia/informe.pdf"' in html, f"falta link a informe.pdf en navbar de {ruta}"
        assert 'href="/academia/presentacion.pdf"' not in html


def test_presentacion_html_contiene_enlace_volver(client):
    """La presentacion en HTML tiene un boton para volver a la aplicacion web."""
    r = client.get("/academia/presentacion")
    if r.status_code == 200:
        html = r.get_data(as_text=True)
        assert 'id="volver"' in html
        assert "Volver" in html

