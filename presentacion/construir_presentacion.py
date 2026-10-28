"""Construye la presentacion a partir de presentacion/plantilla.html.

Reemplaza cada marca {{fig:archivo.png}} por la figura de informe/figuras/
incrustada como data URI, asi el HTML final es un unico archivo que se abre
con doble clic y funciona sin servidor ni conexion.

Uso:
    python presentacion/construir_presentacion.py
    python presentacion/construir_presentacion.py --pdf

Con --pdf tambien genera presentacion/presentacion.pdf con Chrome en modo
headless. Las diapositivas estan pensadas en 1280 x 720.
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PRESENTACION_DIR = RAIZ / "presentacion"
FIGURAS_DIRS = [PRESENTACION_DIR / "figuras", RAIZ / "informe" / "figuras"]
PLANTILLA = PRESENTACION_DIR / "plantilla.html"
SALIDA = PRESENTACION_DIR / "presentacion.html"
PDF = PRESENTACION_DIR / "presentacion.pdf"

PATRON_FIGURA = re.compile(r"\{\{fig:([^}]+)\}\}")


def incrustar_figuras(html: str) -> tuple[str, list[str]]:
    """Devuelve el HTML con las figuras embebidas y la lista de figuras usadas."""
    usadas: list[str] = []

    def reemplazar(match: re.Match[str]) -> str:
        nombre = match.group(1).strip()
        for carpeta in FIGURAS_DIRS:
            ruta = carpeta / nombre
            if ruta.exists():
                break
        else:
            buscadas = ", ".join(str(c / nombre) for c in FIGURAS_DIRS)
            raise SystemExit(f"Falta la figura {nombre}. Se busco en: {buscadas}")
        if nombre not in usadas:
            usadas.append(nombre)
        datos = base64.b64encode(ruta.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{datos}"

    return PATRON_FIGURA.sub(reemplazar, html), usadas


def buscar_navegador() -> Path:
    """Devuelve la ruta de Chrome o Edge."""
    candidatos = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    for nombre in ("chrome", "google-chrome", "msedge", "chromium"):
        encontrado = shutil.which(nombre)
        if encontrado:
            return Path(encontrado)
    raise SystemExit(
        "No se encontro Chrome ni Edge. Instala uno, o abri el HTML y exportalo a PDF."
    )


def generar_pdf() -> None:
    navegador = buscar_navegador()
    comando = [
        str(navegador),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={PDF}",
        SALIDA.resolve().as_uri(),
    ]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    if resultado.returncode != 0 or not PDF.exists():
        print(resultado.stdout)
        print(resultado.stderr, file=sys.stderr)
        raise SystemExit("Chrome no pudo generar el PDF")


def main() -> int:
    if not PLANTILLA.exists():
        raise SystemExit(f"No existe {PLANTILLA}")

    html, usadas = incrustar_figuras(PLANTILLA.read_text(encoding="utf-8"))
    SALIDA.write_text(html, encoding="utf-8")

    diapositivas = html.count('class="slide')
    print(f"HTML generado: {SALIDA.relative_to(RAIZ)}")
    print(f"Diapositivas: {diapositivas}")
    print(f"Figuras incrustadas: {len(usadas)} ({', '.join(usadas)})")
    print(f"Tamano: {SALIDA.stat().st_size / 1024:.0f} KB")

    if "--pdf" in sys.argv[1:]:
        generar_pdf()
        print(f"PDF generado: {PDF.relative_to(RAIZ)}")
        print(f"Tamano: {PDF.stat().st_size / 1024:.0f} KB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
