"""Preparación del dataset de rendimiento de cultivos.

Flujo:
1. Lee el CSV oficial.
2. Limpia tipos, filtra cultivos y campañas, elimina duplicados y registra
   cada exclusión.
3. Calcula el promedio de las cinco campañas anteriores por cultivo y
   municipio, usando una grilla calendario para no mezclar campañas salteadas.
4. Construye el rendimiento relativo (ratio) y las clases binarias.
5. Define el split temporal y guarda los datasets de trabajo.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CROPS,
    DATASET_LIMPIO,
    DATASET_MODELADO,
    END_YEAR,
    GRID_END_YEAR,
    GRID_START_YEAR,
    PROCESADO_DIR,
    RAW_FILE,
    REGISTRO_LIMPIEZA,
    RESUMEN_CLASES_CAMPANIA,
    RESUMEN_CLASES_CULTIVO,
    START_YEAR,
    TEST_START,
    THRESHOLDS,
    TRAIN_END,
    VALIDATION_END,
)

logger = logging.getLogger(__name__)

NUMERIC_COLS = [
    "superficie_sembrada",
    "superficie_cosechada",
    "produccion",
    "rendimiento",
]

MODEL_COLS = [
    "cultivo",
    "campania",
    "campania_inicio",
    "municipio_id",
    "municipio_nombre",
    "superficie_sembrada",
    "prev5",
    "rendimiento",
    "ratio",
    "mala80",
    "mala85",
    "mala90",
    "perdida_total",
    "peso_superficie",
    "split",
]


def _parse_campaign(value: str) -> tuple[int, int]:
    """Convierte '2015/16' en (2015, 2016) y '1999/00' en (1999, 2000)."""
    start_s, end_s = str(value).split("/")
    start = int(start_s)
    end = int(end_s)
    if end < 100:
        century = (start // 100) * 100
        end = century + end
        if end < start:
            end += 100
    return start, end


def read_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecutá primero: python -m src.descargar_datos"
        )
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df


def _add_step(
    log: list[dict],
    paso: str,
    antes: int,
    despues: int,
    motivo: str,
) -> None:
    log.append(
        {
            "paso": paso,
            "filas_antes": int(antes),
            "filas_despues": int(despues),
            "filas_removidas": int(antes - despues),
            "motivo": motivo,
        }
    )


def clean_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia el CSV y devuelve (datos, registro de limpieza)."""
    log: list[dict] = []
    _add_step(log, "lectura_csv", len(df), len(df), "Archivo original")

    out = df.copy()
    out["cultivo"] = out["cultivo"].astype(str).str.strip()
    out["municipio_nombre"] = out["municipio_nombre"].astype(str).str.strip()
    out["campania"] = out["campania"].astype(str).str.strip()

    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    parsed = out["campania"].map(_parse_campaign)
    out["campania_inicio"] = parsed.map(lambda x: x[0]).astype("Int64")
    out["campania_fin"] = parsed.map(lambda x: x[1]).astype("Int64")

    before = len(out)
    out = out[out["cultivo"].isin(CROPS)].copy()
    _add_step(log, "filtrar_cultivos", before, len(out), f"Cultivos del alcance: {', '.join(CROPS)}")

    before = len(out)
    out = out[out["campania_inicio"] <= END_YEAR].copy()
    _add_step(
        log,
        "filtrar_campanias",
        before,
        len(out),
        f"Se conservan campañas {GRID_START_YEAR} a {END_YEAR}; se excluye 2025/26 por estar incompleta",
    )

    before = len(out)
    out = out[out["municipio_nombre"].str.lower() != "ignorado"].copy()
    _add_step(log, "quitar_municipio_ignorado", before, len(out), "Municipio 'Ignorado' sin partido asignado")

    before = len(out)
    out = out[out["rendimiento"].notna() & (out["rendimiento"] >= 0)].copy()
    _add_step(log, "quitar_rendimiento_invalido", before, len(out), "Rendimiento nulo o negativo")

    before = len(out)
    out = out[out["superficie_sembrada"].notna() & (out["superficie_sembrada"] > 0)].copy()
    _add_step(log, "quitar_superficie_invalida", before, len(out), "Superficie sembrada nula o no positiva")

    before = len(out)
    out = out.drop_duplicates(subset=["cultivo", "campania", "municipio_id"]).copy()
    _add_step(log, "quitar_duplicados", before, len(out), "Duplicados por cultivo, campaña y municipio")

    out["perdida_total"] = (out["rendimiento"] == 0) & (out["superficie_cosechada"] == 0)
    out["peso_superficie"] = out["superficie_sembrada"].astype(float)

    out = out.sort_values(["cultivo", "municipio_id", "campania_inicio"]).reset_index(drop=True)
    _add_step(log, "ordenar", len(out), len(out), "Orden por cultivo, municipio y campaña")

    return out, pd.DataFrame(log)


def add_prev5(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el promedio de las cinco campañas anteriores.

    Se construye una grilla completa cultivo x municipio x campaña para que el
    promedio use cinco campañas calendario consecutivas. Si falta un año, el
    promedio queda nulo en lugar de saltear el hueco.
    """
    pairs = df[["cultivo", "municipio_id"]].drop_duplicates()
    years = pd.DataFrame({"campania_inicio": range(GRID_START_YEAR, GRID_END_YEAR + 1)})
    pairs["_key"] = 1
    years["_key"] = 1
    grid = pairs.merge(years, on="_key").drop(columns="_key")

    yields = df[["cultivo", "municipio_id", "campania_inicio", "rendimiento"]].copy()
    grid = grid.merge(
        yields,
        on=["cultivo", "municipio_id", "campania_inicio"],
        how="left",
        validate="one_to_one",
    )
    grid = grid.sort_values(["cultivo", "municipio_id", "campania_inicio"])
    grid["prev5"] = (
        grid.groupby(["cultivo", "municipio_id"], group_keys=False)["rendimiento"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
    )

    return df.merge(
        grid[["cultivo", "municipio_id", "campania_inicio", "prev5"]],
        on=["cultivo", "municipio_id", "campania_inicio"],
        how="left",
        validate="one_to_one",
    )


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid = out["prev5"].notna() & (out["prev5"] > 0)
    out["ratio"] = np.where(valid, out["rendimiento"] / out["prev5"], np.nan)

    for threshold in THRESHOLDS:
        name = f"mala{int(round(threshold * 100))}"
        out[name] = np.where(valid, (out["ratio"] < threshold).astype(float), np.nan)
        out[name] = out[name].astype("Int64")

    return out


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    year = out["campania_inicio"]

    def label(y: int) -> str:
        if y < START_YEAR:
            return "historico"
        if y <= TRAIN_END:
            return "train"
        if y <= VALIDATION_END:
            return "validacion"
        if y <= END_YEAR:
            return "test"
        return "fuera"

    out["split"] = year.map(lambda y: label(int(y)) if pd.notna(y) else "fuera")
    return out


def build_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    limpio = add_split(add_targets(add_prev5(df)))
    modelado = limpio[
        (limpio["campania_inicio"] >= START_YEAR)
        & (limpio["campania_inicio"] <= END_YEAR)
        & limpio["prev5"].notna()
        & (limpio["prev5"] > 0)
    ].copy()
    modelado = modelado[modelado["split"].isin(["train", "validacion", "test"])].copy()
    modelado = modelado.reset_index(drop=True)
    return limpio, modelado


def class_summaries(modelado: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_crop = (
        modelado.groupby(["cultivo", "split"], observed=True)["mala85"]
        .agg(n="size", positivos="sum")
        .reset_index()
    )
    by_crop["tasa_positivos"] = by_crop["positivos"] / by_crop["n"]

    by_campaign = (
        modelado.groupby(["campania_inicio", "split"], observed=True)["mala85"]
        .agg(n="size", positivos="sum")
        .reset_index()
    )
    by_campaign["tasa_positivos"] = by_campaign["positivos"] / by_campaign["n"]

    return by_crop, by_campaign


def save_outputs(
    limpio: pd.DataFrame,
    modelado: pd.DataFrame,
    registro: pd.DataFrame,
    by_crop: pd.DataFrame,
    by_campaign: pd.DataFrame,
) -> None:
    PROCESADO_DIR.mkdir(parents=True, exist_ok=True)

    limpio.to_csv(DATASET_LIMPIO, index=False, encoding="utf-8")
    modelado[MODEL_COLS].to_csv(DATASET_MODELADO, index=False, encoding="utf-8")
    registro.to_csv(REGISTRO_LIMPIEZA, index=False, encoding="utf-8")
    by_crop.to_csv(RESUMEN_CLASES_CULTIVO, index=False, encoding="utf-8")
    by_campaign.to_csv(RESUMEN_CLASES_CAMPANIA, index=False, encoding="utf-8")

    metadata = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cultivos": CROPS,
        "campanias_modelado": [START_YEAR, END_YEAR],
        "umbral_principal": 0.85,
        "umbrales_analizados": list(THRESHOLDS),
        "split": {
            "train": [START_YEAR, TRAIN_END],
            "validacion": [TRAIN_END + 1, VALIDATION_END],
            "test": [TEST_START, END_YEAR],
        },
        "filas_limpias": int(len(limpio)),
        "filas_modelado": int(len(modelado)),
        "filas_por_split": modelado["split"].value_counts().to_dict(),
        "filas_por_cultivo": modelado["cultivo"].value_counts().to_dict(),
        "perdidas_totales": int(modelado["perdida_total"].sum()),
    }
    (PROCESADO_DIR / "metadata_preparacion.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = read_raw()
    limpio_base, registro = clean_raw(raw)
    limpio, modelado = build_datasets(limpio_base)
    by_crop, by_campaign = class_summaries(modelado)
    save_outputs(limpio, modelado, registro, by_crop, by_campaign)

    if verbose:
        print("Registro de limpieza")
        print(registro.to_string(index=False))
        print("\nFilas por split")
        print(modelado["split"].value_counts().to_string())
        print("\nBalance de clases por cultivo")
        print(by_crop.to_string(index=False))
        print("\nBalance por campaña")
        print(by_campaign.to_string(index=False))

    return limpio, modelado, registro


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara el dataset de rendimiento.")
    parser.add_argument("--quiet", action="store_true", help="No imprime tablas.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare(verbose=not args.quiet)


if __name__ == "__main__":
    main()
