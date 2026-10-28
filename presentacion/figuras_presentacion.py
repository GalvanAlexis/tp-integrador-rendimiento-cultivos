"""Genera las figuras que usa la presentacion.

Las figuras de informe/figuras/ salen de los notebooks y estan pensadas para
leerse en pantalla o impresas. En una diapositiva, a 1280 x 720, sus etiquetas
quedan demasiado chicas. Este script arma cuatro versiones con tipografia
grande y menos elementos, a partir de los mismos CSV de datos/procesado/.

Uso:
    python presentacion/figuras_presentacion.py

Salida: presentacion/figuras/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent
PROCESADO = RAIZ / "datos" / "procesado"
SALIDA = Path(__file__).resolve().parent / "figuras"

VERDE = "#14342b"
VERDE2 = "#2a4d3f"
NARANJA = "#eb6834"
AZUL = "#2f6f9f"
OCRE = "#c9a227"
GRIS = "#6b6a65"

# Misma paleta que las figuras de los notebooks, para que el informe y la
# presentacion se lean como el mismo trabajo.
COLORES_CULTIVO = {
    "Maíz": AZUL,
    "Soja total": NARANJA,
    "Trigo total": VERDE2,
    "Girasol": OCRE,
}

plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "axes.edgecolor": "#cdccc6",
        "axes.labelcolor": VERDE,
        "xtick.color": GRIS,
        "ytick.color": GRIS,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

MILES = FuncFormatter(lambda valor, _: f"{valor:,.0f}".replace(",", "."))


def guardar(fig: plt.Figure, nombre: str) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / nombre
    fig.savefig(destino, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {destino.name}")


def rendimiento_por_campania() -> None:
    """El rendimiento mediano se duplico: por eso se modela el ratio."""
    df = pd.read_csv(PROCESADO / "dataset_modelado_con_clima.csv")
    tabla = (
        df.groupby(["campania_inicio", "cultivo"])["rendimiento"]
        .median()
        .unstack("cultivo")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for cultivo in ["Maíz", "Soja total", "Trigo total", "Girasol"]:
        if cultivo not in tabla:
            continue
        serie = tabla[cultivo].dropna()
        ax.plot(
            serie.index,
            serie.values,
            color=COLORES_CULTIVO[cultivo],
            linewidth=2.4,
            label=cultivo,
        )

    ax.set_xlim(1981, 2025)
    ax.set_ylim(0, 9800)
    ax.yaxis.set_major_formatter(MILES)
    ax.set_ylabel("kg/ha")
    ax.set_title("Rendimiento mediano por campaña", loc="left", color=VERDE, pad=12)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        columnspacing=1.6,
        handlelength=1.6,
        fontsize=12,
    )
    guardar(fig, "p_rendimiento.png")


def predicho_contra_real() -> None:
    """La regresion ordena pero comprime: la nube es mas horizontal que la diagonal."""
    df = pd.read_csv(PROCESADO / "regresion_predicciones_test.csv")

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.scatter(
        df["ratio"],
        df["ratio_pred"],
        s=9,
        color=VERDE2,
        alpha=0.22,
        linewidths=0,
        label="Campaña de test",
    )
    limite = [0, 2.1]
    ax.plot(limite, limite, color=GRIS, linewidth=1.4, linestyle="--", label="Predicción perfecta")
    ax.axhline(0.85, color=NARANJA, linewidth=1.4, linestyle=":")
    ax.axvline(0.85, color=NARANJA, linewidth=1.4, linestyle=":")
    ax.text(0.87, 1.93, "umbral 0,85", color=NARANJA, fontsize=12, fontweight="bold", ha="left")

    medios = df[df["ratio"] < 0.60]["ratio_pred"].mean()
    ax.annotate(
        f"ratio real < 0,60\nse predice {medios:.2f}".replace(".", ","),
        xy=(0.45, medios),
        xytext=(0.10, 1.55),
        fontsize=12,
        color=NARANJA,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": NARANJA, "linewidth": 1.4},
    )

    ax.set_xlim(limite)
    ax.set_ylim(limite)
    ax.set_xlabel("Ratio real")
    ax.set_ylabel("Ratio predicho")
    ax.set_title("Predicción contra realidad en test", loc="left", color=VERDE, pad=12)
    ax.grid(color="#eeede9", linewidth=0.8)
    ax.set_axisbelow(True)
    guardar(fig, "p_predicho_vs_real.png")


def costo_por_umbral() -> None:
    """El minimo de validacion esta en 0,65: las probabilidades no estan calibradas."""
    df = pd.read_csv(PROCESADO / "clasificacion_busqueda_validacion.csv")
    arbol = df[
        (df["familia"] == "arbol")
        & (df["variables"] == "con_clima")
        & (df["params"] == "max_depth=5, min_samples_leaf=50, class_weight=None")
    ].sort_values("umbral")

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(
        arbol["umbral"],
        arbol["costo_val"],
        color=NARANJA,
        linewidth=2.6,
        marker="o",
        markersize=5,
        label="Árbol con clima",
    )
    ax.axhline(540, color=GRIS, linewidth=1.4, linestyle="--", label="No alertar nunca")

    elegido = arbol.loc[arbol["costo_val"].idxmin()]
    ax.scatter([elegido["umbral"]], [elegido["costo_val"]], s=150, color=VERDE, zorder=5)
    ax.annotate(
        f"elegido 0,65\ncosto {elegido['costo_val']:.0f}".replace(".", ","),
        xy=(elegido["umbral"], elegido["costo_val"]),
        xytext=(0.26, 412),
        textcoords="data",
        fontsize=12,
        color=VERDE,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": VERDE, "linewidth": 1.2},
    )

    ax.set_xlabel("Umbral de probabilidad")
    ax.set_ylabel("Costo en validación")
    ax.set_ylim(400, 745)
    ax.set_title("El umbral se elige en validación", loc="left", color=VERDE, pad=12)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    guardar(fig, "p_costo_umbral.png")


def alertas_por_banda() -> None:
    """El arbol detecta mas en los extremos; los dos fallan en la zona gris."""
    df = pd.read_csv(PROCESADO / "clasificacion_errores_por_ratio_test.csv")
    orden = ["< 0,50", "0,50-0,75", "0,75-0,85", "0,85-0,95", "0,95-1,05", ">= 1,05"]
    arbol = (
        df[df["modelo"] == "DecisionTreeClassifier (con_clima)"]
        .set_index("banda_ratio")
        .loc[orden]
    )
    regresion = (
        df[df["modelo"] == "Regresión + umbral (Entrega 3)"]
        .set_index("banda_ratio")
        .loc[orden]
    )

    posiciones = range(len(orden))
    ancho = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    barras_arbol = ax.bar(
        [p - ancho / 2 for p in posiciones],
        arbol["tasa_alerta"] * 100,
        width=ancho,
        color=NARANJA,
        label="Clasificador",
    )
    barras_reg = ax.bar(
        [p + ancho / 2 for p in posiciones],
        regresion["tasa_alerta"] * 100,
        width=ancho,
        color=VERDE2,
        label="Regresión + umbral",
    )

    for barras in (barras_arbol, barras_reg):
        for barra in barras:
            alto = barra.get_height()
            ax.annotate(
                f"{alto:.0f}",
                xy=(barra.get_x() + barra.get_width() / 2, alto),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                color=GRIS,
            )

    ax.axvline(2.5, color="#cdccc6", linewidth=1.2, linestyle=":")
    ax.axvspan(-0.5, 2.5, color="#fbf1ea", zorder=0)
    ax.text(1, 62, "campañas malas", ha="center", fontsize=12, color=GRIS, fontweight="bold")
    ax.text(4, 62, "campañas normales", ha="center", fontsize=12, color=GRIS, fontweight="bold")

    ax.set_xticks(list(posiciones))
    ax.set_xticklabels(orden, fontsize=11)
    ax.set_ylim(0, 96)
    ax.set_xlabel("Ratio real")
    ax.set_ylabel("% de casos con alerta")
    ax.set_title("Tasa de alerta por banda de ratio en test", loc="left", color=VERDE, pad=12)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=2, fontsize=11)
    guardar(fig, "p_alertas_por_banda.png")


def main() -> int:
    print("Figuras de la presentacion:")
    rendimiento_por_campania()
    predicho_contra_real()
    costo_por_umbral()
    alertas_por_banda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
