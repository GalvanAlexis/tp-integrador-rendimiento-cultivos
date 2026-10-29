"""Tests del experimento de modelos por grupo de cultivos."""
import pytest

from src.config import CLASIFICADOR_METADATA, DATASET_MODELADO_CLIMA
from src.grupos import GRUPOS, evaluar_grupos


def test_grupos_cubren_los_cultivos_del_alcance():
    cultivos = [c for lista in GRUPOS.values() for c in lista]
    assert set(cultivos) == {"Maíz", "Soja total", "Girasol", "Trigo total"}
    assert len(cultivos) == len(set(cultivos))
    assert set(GRUPOS) == {"verano", "invierno"}


@pytest.mark.skipif(
    not (CLASIFICADOR_METADATA.exists() and DATASET_MODELADO_CLIMA.exists()),
    reason="Falta entrenar o falta el dataset de modelado",
)
def test_experimento_compara_global_y_por_grupo_solo_en_validacion():
    tabla = evaluar_grupos()
    assert len(tabla) == 2 * len(GRUPOS)
    assert set(tabla["variante"]) == {"global (entregado)", "por grupo"}
    assert not [c for c in tabla.columns if c.endswith("_test")]
    assert tabla["costo_val"].notna().all()
