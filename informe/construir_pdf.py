"""Convierte informe/informe_final.md en informe/informe_final.pdf.

El PDF se arma con Chrome en modo headless, que ya viene instalado en cualquier
equipo con navegador. No hace falta LaTeX ni LibreOffice.

Uso:
    python informe/construir_pdf.py

Requiere `markdown-it-py` (esta en requirements-dev.txt) y Chrome o Edge.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

INFORME_DIR = Path(__file__).resolve().parent
ORIGEN = INFORME_DIR / "informe_final.md"
PDF = INFORME_DIR / "informe_final.pdf"

CSS = """
@page {
  size: A4;
  margin: 17mm 15mm 16mm 15mm;
}
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Calibri", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.4pt;
  line-height: 1.55;
  color: #1c1c1c;
  margin: 0;
}
h1 {
  font-size: 19pt;
  line-height: 1.25;
  margin: 0 0 4pt;
  color: #14342b;
}
h2 {
  font-size: 14pt;
  margin: 22pt 0 8pt;
  padding-bottom: 4pt;
  border-bottom: 1pt solid #d6d5cf;
  color: #14342b;
  page-break-after: avoid;
}
h3 {
  font-size: 11.4pt;
  margin: 16pt 0 6pt;
  color: #2a4d3f;
  page-break-after: avoid;
}
p { margin: 6pt 0; }
strong { color: #101c17; }
em { color: #3c3b37; }
a { color: #1c5f8a; text-decoration: none; }
ul, ol { margin: 6pt 0 8pt; padding-left: 18pt; }
li { margin: 3pt 0; }
hr {
  border: 0;
  border-top: 1pt solid #e2e1dc;
  margin: 16pt 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 8.5pt;
  line-height: 1.35;
  margin: 9pt 0 14pt;
  page-break-inside: avoid;
}
th, td {
  border: 0.6pt solid #cdccc6;
  padding: 3pt 5pt;
  vertical-align: top;
}
th {
  background: #f1f0ec;
  font-weight: 600;
  color: #14342b;
}
tbody tr:nth-child(even) td { background: #faf9f7; }
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 10pt auto 4pt;
  page-break-inside: avoid;
}
code {
  font-family: "Cascadia Mono", "Consolas", monospace;
  font-size: 8.8pt;
  background: #f3f2ee;
  border: 0.5pt solid #e0dfd9;
  border-radius: 2pt;
  padding: 0.5pt 2.5pt;
}
pre {
  background: #f7f6f3;
  border: 0.5pt solid #e0dfd9;
  border-left: 2.5pt solid #14342b;
  border-radius: 3pt;
  padding: 8pt 10pt;
  font-size: 8.8pt;
  line-height: 1.4;
  overflow-x: auto;
  page-break-inside: avoid;
}
pre code {
  background: none;
  border: 0;
  padding: 0;
}
blockquote {
  margin: 8pt 0;
  padding: 4pt 10pt;
  border-left: 3pt solid #b9c9c1;
  color: #3c3b37;
  background: #f8f8f6;
}
.cover {
  page-break-after: always;
  padding-top: 24mm;
}
.cover h1 {
  font-size: 26pt;
  line-height: 1.2;
  margin-bottom: 8pt;
}
.cover .subtitulo {
  font-size: 12.5pt;
  color: #2a4d3f;
  margin: 0 0 26mm;
}
.cover table {
  font-size: 9.6pt;
}
.cover th {
  width: 34%;
}
"""

PLANTILLA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Informe final - TP Integrador de Aprendizaje Automatico</title>
<style>{css}</style>
</head>
<body>
{contenido}
</body>
</html>
"""


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
        "No se encontró Chrome ni Edge. Instalá uno, o abrí el .md y exportalo a PDF."
    )


def renderizar(md: MarkdownIt, texto: str) -> str:
    return md.render(texto)


def main() -> int:
    if not ORIGEN.exists():
        raise SystemExit(f"No existe {ORIGEN}")

    texto = ORIGEN.read_text(encoding="utf-8")

    # La portada es todo lo que está antes del primer separador horizontal.
    partes = texto.split("\n---\n", 1)
    if len(partes) != 2:
        raise SystemExit("El informe debe tener un separador --- después de la portada")
    portada, cuerpo = partes

    # El subtítulo de la portada se separa del bloque de datos.
    lineas = portada.splitlines()
    titulo = lineas[0].lstrip("# ").strip()
    resto = [
        linea
        for linea in lineas[1:]
        if linea.strip() and not linea.startswith("**Informe final")
    ]
    subtitulo = next(
        (linea.strip("* ") for linea in lineas if linea.startswith("**Informe final")),
        "Informe final",
    )

    md = MarkdownIt("js-default")
    cuerpo_html = renderizar(md, cuerpo)
    datos_html = renderizar(md, "\n".join(resto))
    # En la portada la tabla va sin encabezado: es una ficha, no una tabla de datos.
    datos_html = re.sub(r"<thead>.*?</thead>", "", datos_html, flags=re.S)

    html = PLANTILLA.format(
        css=CSS,
        contenido=(
            '<div class="cover">'
            f"<h1>{titulo}</h1>"
            f'<p class="subtitulo">{subtitulo}</p>'
            f"{datos_html}"
            "</div>"
            f"{cuerpo_html}"
        ),
    )

    # Chrome resuelve las rutas relativas de las figuras desde la ubicación del
    # HTML, así que el archivo temporal se crea dentro de informe/.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", dir=INFORME_DIR, delete=False, encoding="utf-8"
    ) as fh:
        fh.write(html)
        html_path = Path(fh.name)

    try:
        navegador = buscar_navegador()
        comando = [
            str(navegador),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",
            f"--print-to-pdf={PDF}",
            html_path.resolve().as_uri(),
        ]
        resultado = subprocess.run(comando, capture_output=True, text=True)
        if resultado.returncode != 0 or not PDF.exists():
            print(resultado.stdout)
            print(resultado.stderr, file=sys.stderr)
            raise SystemExit("Chrome no pudo generar el PDF")
    finally:
        html_path.unlink(missing_ok=True)

    print(f"PDF generado: {PDF.relative_to(INFORME_DIR.parent)}")
    print(f"Tamaño: {PDF.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
