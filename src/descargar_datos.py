"""Descarga y verificación del CSV oficial de Estimaciones Agrícolas."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import RAW_DIR, RAW_FILE, SOURCE_LICENSE, SOURCE_PAGE, SOURCE_URL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False, verbose: bool = True) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        if verbose:
            print(f"Archivo existente: {RAW_FILE}")
    else:
        if verbose:
            print(f"Descargando: {SOURCE_URL}")
        response = requests.get(SOURCE_URL, timeout=180)
        response.raise_for_status()
        RAW_FILE.write_bytes(response.content)
        if verbose:
            print(f"Guardado en: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE, encoding="utf-8")
    metadata = {
        "fuente": SOURCE_PAGE,
        "url_descarga": SOURCE_URL,
        "licencia": SOURCE_LICENSE,
        "archivo": RAW_FILE.name,
        "bytes": RAW_FILE.stat().st_size,
        "sha256": sha256(RAW_FILE),
        "filas": int(len(df)),
        "columnas": list(df.columns),
        "descargado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = RAW_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return RAW_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el CSV de Estimaciones Agrícolas.")
    parser.add_argument("--force", action="store_true", help="Vuelve a descargar el archivo.")
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
