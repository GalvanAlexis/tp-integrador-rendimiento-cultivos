"""Descarga y agregación de datos climáticos de NASA POWER.

La app no consulta la API en cada predicción: el clima se descarga una vez,
se cachea por municipio y se agrega a ventanas por cultivo. El archivo
resultante se une al dataset de modelado.

Uso:
    python -m src.clima --sample
    python -m src.clima --full
    python -m src.clima --full --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import (
    CACHE_DIR,
    CLIMA_CACHE_DIR,
    CLIMA_FEATURES,
    CROP_WINDOWS,
    CROPS,
    DATASET_MODELADO,
    DATASET_MODELADO_CLIMA,
    GEOREF_URL,
    NASA_END,
    NASA_PARAMETERS,
    NASA_START,
    PROCESADO_DIR,
)

logger = logging.getLogger(__name__)

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
GEOREF_CACHE = CACHE_DIR / "georef_ba.json"

MANUAL_NAME_MAP = {
    "coronel de marina l rosales": "coronel de marina leonardo rosales",
}

CLIMA_COLS = [
    "municipio_id",
    "cultivo",
    "campania_inicio",
    "clima_precip_total_mm",
    "clima_dias_tmax_30",
    "clima_tmax_media_c",
    "clima_tmin_media_c",
    "clima_radiacion_kwh_m2",
    "clima_humedad_media_pct",
    "clima_sequia_max_dias",
    "clima_precip_pre_siembra_mm",
    "clima_dias_helada",
    "clima_dias_lluvia",
]


def normalize_name(value: str) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.replace("partido de", "").replace("partido", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return MANUAL_NAME_MAP.get(text, text)


def load_georef(force: bool = False) -> dict[str, dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if GEOREF_CACHE.exists() and not force:
        data = json.loads(GEOREF_CACHE.read_text(encoding="utf-8"))
    else:
        response = requests.get(GEOREF_URL, timeout=120)
        response.raise_for_status()
        data = response.json()
        GEOREF_CACHE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    out: dict[str, dict] = {}
    for item in data["municipios"]:
        key = normalize_name(item["nombre"])
        out[key] = {
            "nombre": item["nombre"],
            "lat": float(item["centroide"]["lat"]),
            "lon": float(item["centroide"]["lon"]),
        }
    return out


def match_municipios(df: pd.DataFrame, georef: dict[str, dict]) -> pd.DataFrame:
    """Devuelve municipio_id, nombre canónico y centroide."""
    names = (
        df.groupby("municipio_id", observed=True)["municipio_nombre"]
        .agg(lambda s: max(s.astype(str), key=len))
        .reset_index()
    )
    names["key"] = names["municipio_nombre"].map(normalize_name)
    names["geo_nombre"] = names["key"].map(lambda k: georef.get(k, {}).get("nombre"))
    names["lat"] = names["key"].map(lambda k: georef.get(k, {}).get("lat"))
    names["lon"] = names["key"].map(lambda k: georef.get(k, {}).get("lon"))
    return names


def _max_dry_spell(precip: pd.Series) -> int:
    dry = (precip.fillna(0).to_numpy() < 1.0).astype(int)
    best = run = 0
    for value in dry:
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def fetch_daily(
    lat: float,
    lon: float,
    cache_path: Path,
    start: str = NASA_START,
    end: str = NASA_END,
    retries: int = 3,
) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["fecha"])

    params = {
        "parameters": ",".join(NASA_PARAMETERS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(NASA_URL, params=params, timeout=180)
            response.raise_for_status()
            payload = response.json()
            parameter = payload["properties"]["parameter"]
            dates = sorted(parameter["PRECTOTCORR"].keys())
            daily = pd.DataFrame(
                {
                    "fecha": pd.to_datetime(dates, format="%Y%m%d"),
                    "precip_mm": [parameter["PRECTOTCORR"][d] for d in dates],
                    "tmax_c": [parameter["T2M_MAX"][d] for d in dates],
                    "tmin_c": [parameter["T2M_MIN"][d] for d in dates],
                    "radiacion_kwh_m2": [parameter["ALLSKY_SFC_SW_DWN"][d] for d in dates],
                    "humedad_pct": [parameter["RH2M"][d] for d in dates],
                }
            )
            daily = daily.replace(-999, np.nan).replace(-999.0, np.nan)
            daily.to_csv(cache_path, index=False)
            return daily
        except Exception as exc:  # noqa: BLE001 - se registra y se reintenta
            last_error = exc
            logger.warning("Intento %s/%s falló para %s: %s", attempt, retries, cache_path.name, exc)
            time.sleep(2 * attempt)

    raise RuntimeError(f"No se pudo descargar {cache_path.name}: {last_error}")


def _window_dates(crop: str, campaign_start: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg = CROP_WINDOWS[crop]
    start_month, start_day = cfg["start"]
    end_month, end_day = cfg["end"]
    start = pd.Timestamp(year=campaign_start, month=start_month, day=start_day)
    end_year = campaign_start + 1 if cfg["next_year"] else campaign_start
    end = pd.Timestamp(year=end_year, month=end_month, day=end_day)
    return start, end


def aggregate_window(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    window = daily[(daily["fecha"] >= start) & (daily["fecha"] <= end)]
    if window.empty:
        return {c: np.nan for c in CLIMA_COLS[3:]}

    pre_start = start - pd.Timedelta(days=60)
    pre = daily[(daily["fecha"] >= pre_start) & (daily["fecha"] < start)]

    return {
        "clima_precip_total_mm": float(window["precip_mm"].sum(min_count=1)),
        "clima_dias_tmax_30": int((window["tmax_c"] > 30).sum()),
        "clima_tmax_media_c": float(window["tmax_c"].mean()),
        "clima_tmin_media_c": float(window["tmin_c"].mean()),
        "clima_radiacion_kwh_m2": float(window["radiacion_kwh_m2"].sum(min_count=1)),
        "clima_humedad_media_pct": float(window["humedad_pct"].mean()),
        "clima_sequia_max_dias": _max_dry_spell(window["precip_mm"]),
        "clima_precip_pre_siembra_mm": float(pre["precip_mm"].sum(min_count=1)),
        "clima_dias_helada": int((window["tmin_c"] < 0).sum()),
        "clima_dias_lluvia": int((window["precip_mm"] > 1).sum()),
    }


def build_climate_features(limit: int | None = None, verbose: bool = True) -> pd.DataFrame:
    if not DATASET_MODELADO.exists():
        raise FileNotFoundError(
            f"No existe {DATASET_MODELADO}. Ejecutá primero: python -m src.preparar_datos"
        )

    CLIMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESADO_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_MODELADO, encoding="utf-8")
    georef = load_georef()
    matches = match_municipios(df, georef)

    missing = matches[matches["lat"].isna()]
    if not missing.empty:
        logger.warning("Municipios sin centroide: %s", missing["municipio_nombre"].tolist())

    matches = matches.dropna(subset=["lat", "lon"]).copy()
    if limit:
        matches = matches.head(limit)

    rows: list[dict] = []
    log_rows: list[dict] = []
    total = len(matches)

    for i, row in enumerate(matches.itertuples(index=False), start=1):
        cache_path = CLIMA_CACHE_DIR / f"{int(row.municipio_id)}.csv"
        try:
            daily = fetch_daily(row.lat, row.lon, cache_path)
            for crop in CROPS:
                for campaign_start in range(1981, 2025):
                    start, end = _window_dates(crop, campaign_start)
                    features = aggregate_window(daily, start, end)
                    rows.append(
                        {
                            "municipio_id": int(row.municipio_id),
                            "cultivo": crop,
                            "campania_inicio": campaign_start,
                            **features,
                        }
                    )
            log_rows.append(
                {
                    "municipio_id": int(row.municipio_id),
                    "municipio_nombre": row.municipio_nombre,
                    "lat": row.lat,
                    "lon": row.lon,
                    "estado": "ok",
                    "dias": int(len(daily)),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Error con %s: %s", row.municipio_nombre, exc)
            log_rows.append(
                {
                    "municipio_id": int(row.municipio_id),
                    "municipio_nombre": row.municipio_nombre,
                    "lat": row.lat,
                    "lon": row.lon,
                    "estado": "error",
                    "dias": 0,
                    "error": str(exc),
                }
            )

        if verbose and (i % 10 == 0 or i == total):
            print(f"Clima: {i}/{total} municipios procesados")

    features_df = pd.DataFrame(rows, columns=CLIMA_COLS)
    features_df.to_csv(CLIMA_FEATURES, index=False, encoding="utf-8")
    pd.DataFrame(log_rows).to_csv(PROCESADO_DIR / "clima_fetch_log.csv", index=False, encoding="utf-8")

    if verbose:
        print(f"Guardado: {CLIMA_FEATURES} ({len(features_df)} filas)")
    return features_df


def merge_climate(drop_incomplete: bool = True) -> pd.DataFrame:
    """Une el clima al dataset de modelado.

    La radiación de NASA POWER no está disponible para las ventanas de
    1981-1983 (el satélite/reanálisis arranca después). Se conserva en la
    tabla cruda de clima, pero se excluye del dataset de modelado para no
    dejar faltantes en un predictor. Las restantes variables climáticas
    tienen cobertura completa.
    """
    if not CLIMA_FEATURES.exists():
        raise FileNotFoundError(
            f"No existe {CLIMA_FEATURES}. Ejecutá primero: python -m src.clima --full"
        )
    df = pd.read_csv(DATASET_MODELADO, encoding="utf-8")
    clima = pd.read_csv(CLIMA_FEATURES, encoding="utf-8")
    merged = df.merge(
        clima,
        on=["municipio_id", "cultivo", "campania_inicio"],
        how="left",
        validate="one_to_one",
    )
    if drop_incomplete:
        merged = merged.drop(columns=["clima_radiacion_kwh_m2"], errors="ignore")
    out = DATASET_MODELADO_CLIMA
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"Guardado: {out} ({len(merged)} filas, {merged['clima_precip_total_mm'].notna().sum()} con clima)")
    return merged


def sample() -> None:
    georef = load_georef()
    key = normalize_name("Chascomús")
    info = georef.get(key)
    if info is None:
        raise SystemExit("No se encontró Chascomús en Georef.")
    cache_path = CLIMA_CACHE_DIR / "sample_chascomus.csv"
    CLIMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    daily = fetch_daily(info["lat"], info["lon"], cache_path)
    print(f"Chascomús: {info}")
    print(f"Días descargados: {len(daily)}")
    print(daily.tail(3).to_string(index=False))
    for crop in CROPS:
        start, end = _window_dates(crop, 2022)
        print(crop, start.date(), end.date(), aggregate_window(daily, start, end))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clima NASA POWER para el TP.")
    parser.add_argument("--sample", action="store_true", help="Prueba con Chascomús.")
    parser.add_argument("--full", action="store_true", help="Descarga todos los municipios.")
    parser.add_argument("--merge", action="store_true", help="Une clima al dataset de modelado.")
    parser.add_argument("--limit", type=int, default=None, help="Limita la cantidad de municipios.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.sample:
        sample()
    if args.full:
        build_climate_features(limit=args.limit)
    if args.merge:
        merge_climate()
    if not (args.sample or args.full or args.merge):
        parser.print_help()


if __name__ == "__main__":
    main()
