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
CASOS_RATIO_EXTREMO = PROCESADO_DIR / "casos_ratio_extremo.csv"
CLIMA_FEATURES = PROCESADO_DIR / "clima_features.csv"
DATASET_MODELADO_CLIMA = PROCESADO_DIR / "dataset_modelado_con_clima.csv"
BASELINES_REGRESION = PROCESADO_DIR / "baselines_regresion.csv"
BASELINES_CLASIFICACION = PROCESADO_DIR / "baselines_clasificacion.csv"
BASELINES_GLOBAL = PROCESADO_DIR / "baselines_global.csv"

MODELOS_DIR = ROOT / "modelos"
REGRESOR_FILE = MODELOS_DIR / "regresor.joblib"
REGRESOR_METADATA = MODELOS_DIR / "regresor_metadata.json"
REGRESION_BUSQUEDA = PROCESADO_DIR / "regresion_busqueda_validacion.csv"
REGRESION_COMPARACION = PROCESADO_DIR / "regresion_comparacion.csv"
REGRESION_POR_CULTIVO = PROCESADO_DIR / "regresion_por_cultivo_test.csv"
REGRESION_POR_CAMPANIA = PROCESADO_DIR / "regresion_por_campania_test.csv"
REGRESION_PREDICCIONES = PROCESADO_DIR / "regresion_predicciones_test.csv"

CLASIFICADOR_FILE = MODELOS_DIR / "clasificador.joblib"
CLASIFICADOR_METADATA = MODELOS_DIR / "clasificador_metadata.json"
CLASIFICACION_BUSQUEDA = PROCESADO_DIR / "clasificacion_busqueda_validacion.csv"
CLASIFICACION_COMPARACION = PROCESADO_DIR / "clasificacion_comparacion.csv"
CLASIFICACION_POR_CULTIVO = PROCESADO_DIR / "clasificacion_por_cultivo_test.csv"
CLASIFICACION_POR_CAMPANIA = PROCESADO_DIR / "clasificacion_por_campania_test.csv"
CLASIFICACION_PREDICCIONES = PROCESADO_DIR / "clasificacion_predicciones_test.csv"
CLASIFICACION_ERRORES_PERFIL = PROCESADO_DIR / "clasificacion_errores_perfil_test.csv"
CLASIFICACION_ERRORES_RATIO = PROCESADO_DIR / "clasificacion_errores_por_ratio_test.csv"
CLASIFICACION_ACUERDO = PROCESADO_DIR / "clasificacion_acuerdo_sistemas_test.csv"
CLASIFICACION_CALIBRACION = PROCESADO_DIR / "clasificacion_calibracion_comparacion.csv"
CLASIFICACION_GRUPOS = PROCESADO_DIR / "clasificacion_por_grupo_cultivo.csv"

# Esquema de costos de la alerta de campaña mala.
# Un falso negativo (campaña mala no anticipada) se penaliza el doble que un
# falso positivo (alerta en una campaña normal).
COST_FN = 2.0
COST_FP = 1.0

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

# Estas dos definiciones viven en config para que la app web no tenga que
# importar src.clasificacion (y, a traves de el, matplotlib) en produccion.
CLIMA_COLS = [
    "clima_precip_total_mm",
    "clima_dias_tmax_30",
    "clima_tmax_media_c",
    "clima_tmin_media_c",
    "clima_humedad_media_pct",
    "clima_sequia_max_dias",
    "clima_precip_pre_siembra_mm",
    "clima_dias_helada",
    "clima_dias_lluvia",
]

MODEL_LABELS = {
    "dummy": "Dummy (clase mayoritaria)",
    "regresion": "Regresión + umbral (Entrega 3)",
    "combinada": "Clasificador o regresión (alerta combinada)",
    "arbol": "DecisionTreeClassifier",
    "knn": "KNeighborsClassifier",
    "nb": "GaussianNB",
}

# Un ratio mayor a 3 (rendimiento más de tres veces el promedio de las cinco
# campañas anteriores) se marca para revisión. No se elimina: el dato actual es
# consistente, el promedio queda bajo por campañas previas sin cosecha.
RATIO_EXTREMO = 3.0

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
