"""Configuración central del proyecto.

Todas las rutas se resuelven a partir de la raíz del repositorio para que los
scripts y el notebook funcionen sin importar el directorio desde el que se
ejecuten.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATOS = ROOT / "datos"
RAW_DIR = DATOS / "raw"
PROCESADO_DIR = DATOS / "procesado"
CACHE_DIR = DATOS / "cache"
CLIMA_CACHE_DIR = CACHE_DIR / "clima"
FIGURAS_DIR = ROOT / "informe" / "figuras"

RAW_FILE = RAW_DIR / "estimaciones-agricolas-1969_2026.csv"
DATASET_LIMPIO = PROCESADO_DIR / "dataset_limpio.csv"
DATASET_MODELADO = PROCESADO_DIR / "dataset_modelado.csv"
REGISTRO_LIMPIEZA = PROCESADO_DIR / "registro_limpieza.csv"
RESUMEN_CLASES_CULTIVO = PROCESADO_DIR / "resumen_clases_por_cultivo.csv"
RESUMEN_CLASES_CAMPANIA = PROCESADO_DIR / "resumen_clases_por_campania.csv"
CLIMA_FEATURES = PROCESADO_DIR / "clima_features.csv"
BASELINES_REGRESION = PROCESADO_DIR / "baselines_regresion.csv"
BASELINES_CLASIFICACION = PROCESADO_DIR / "baselines_clasificacion.csv"
BASELINES_GLOBAL = PROCESADO_DIR / "baselines_global.csv"

SOURCE_URL = (
    "https://catalogo.datos.gba.gob.ar/dataset/4ee50a8f-6866-4191-9bf8-aebb30b0d718/"
    "resource/d3e23031-164a-43e0-b104-01ced63e4abd/download/"
    "estimaciones-agricolas-1969_2026.csv"
)
SOURCE_PAGE = "https://catalogo.datos.gba.gob.ar/dataset/estimaciones-agro"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_ENTITY = (
    "Ministerio de Economía de la Provincia de Buenos Aires. "
    "Secretaría de Agricultura, Ganadería y Pesca. "
    "Dirección de Estimaciones Agrícolas."
)

# Cultivos del alcance principal. Se excluyen Sorgo (alta proporción de
# rendimiento cero) y Cebada cervecera (serie corta, termina en 2015/16).
CROPS = ["Maíz", "Trigo total", "Girasol", "Soja total"]

START_YEAR = 1981
END_YEAR = 2024
GRID_START_YEAR = 1969
GRID_END_YEAR = 2024

THRESHOLDS = (0.80, 0.85, 0.90)
THRESHOLD = 0.85

TRAIN_END = 2013
VALIDATION_END = 2018
TEST_START = 2019

RANDOM_STATE = 42

# Ventanas climáticas por cultivo. El año de campaña es el año de inicio.
# (mes, día) de inicio y de corte. El corte del año siguiente se expresa con
# next_year=True.
CROP_WINDOWS = {
    "Trigo total": {"start": (6, 1), "end": (11, 15), "next_year": False},
    "Maíz": {"start": (9, 1), "end": (1, 31), "next_year": True},
    "Girasol": {"start": (9, 1), "end": (1, 31), "next_year": True},
    "Soja total": {"start": (11, 1), "end": (2, 28), "next_year": True},
}

NASA_PARAMETERS = [
    "PRECTOTCORR",
    "T2M_MAX",
    "T2M_MIN",
    "ALLSKY_SFC_SW_DWN",
    "RH2M",
]

NASA_START = "19810101"
NASA_END = "20250331"

GEOREF_URL = (
    "https://apis.datos.gob.ar/georef/api/municipios"
    "?provincia=06&max=200&campos=id,nombre,centroide"
)
