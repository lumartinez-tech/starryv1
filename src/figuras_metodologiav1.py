# =============================================================================
# GENERADOR DE FIGURAS PARA LA METODOLOGÍA DE TESIS
# Caracterización y modelaje de estrellas binarias eclipsantes
# usando datos de TESS y Gaia DR3 en las regiones de Pléyades y Orión
#
# Figuras:
#   3.1 Diagrama de flujo metodológico
#   3.2 Preprocesamiento de curvas de luz
#   3.3 Medición de profundidad de eclipse
#   3.4 Diagrama HR e isocronas esquemáticas
#   3.5 Ajuste de eclipse y residuos
#
# Dependencias: numpy, matplotlib
#
# IMPORTANTE:
# Las figuras 3.2-3.5 usan datos sintéticos con fines ilustrativos.
# Las isocronas de la figura 3.4 son esquemáticas, no modelos físicos.
# Sustituir estos ejemplos por los resultados reales antes de publicarlos.
# =============================================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch


# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================

OUTPUT_DIR = Path("figuras_metodologia_tesis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta oficial de la tesis
TESIS_VERDE_OSCURO = "#273A2F"
TESIS_VERDE = "#4C6A4A"
TESIS_TERRACOTA = "#B07D4E"
TESIS_BEIGE = "#E9D8BF"
TESIS_ROJO_TIERRA = "#8A3F2B"

TESIS_TEXTO = TESIS_VERDE_OSCURO
TESIS_ENFASIS = TESIS_ROJO_TIERRA
TESIS_SUBTITULO = TESIS_VERDE
TESIS_ACENTO = TESIS_TERRACOTA

BLANCO = "#FFFFFF"
GRIS = "#6D6B63"
VERDE_CLARO = "#E9EEE7"
BEIGE_CLARO = "#F3E8D7"

# Reproducibilidad de los ejemplos sintéticos
rng = np.random.default_rng(42)

# Tipografía
# Se usa serif como configuración provisional.
# Para igualar exactamente la fuente de LaTeX hay que comprobar
# la familia tipográfica declarada en el preámbulo de la tesis.
FONT_FAMILY = "serif"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 10,
    "font.weight": "normal",

    "text.color": TESIS_TEXTO,

    "axes.labelsize": 10,
    "axes.labelcolor": TESIS_TEXTO,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.edgecolor": TESIS_VERDE_OSCURO,
    "axes.linewidth": 0.8,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": TESIS_TEXTO,
    "ytick.color": TESIS_TEXTO,
    "xtick.direction": "in",
    "ytick.direction": "in",

    "legend.fontsize": 8.5,
    "legend.frameon": False,

    "figure.titlesize": 12,
    "figure.facecolor": BLANCO,
    "axes.facecolor": BLANCO,

    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": BLANCO,

    # Fuentes incrustadas en los PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================

def guardar_figura(fig, nombre):
    """
    Guarda cada figura en PDF y PNG.
    PDF: formato vectorial recomendado para la tesis.
    PNG: resolución de 300 dpi.
    """
    ruta_pdf = OUTPUT_DIR / f"{nombre}.pdf"
    ruta_png = OUTPUT_DIR / f"{nombre}.png"

    fig.savefig(ruta_pdf)
    fig.savefig(ruta_png, dpi=300)

    plt.close(fig)

    print(f"Generada: {ruta_pdf}")
    print(f"Generada: {ruta_png}")


def agregar_caja(
    ax,
    x,
    y,
    ancho,
    alto,
    titulo,
    contenido="",
    fondo=BEIGE_CLARO,
    borde=TESIS_TERRACOTA,
    color_titulo=TESIS_VERDE_OSCURO,
    color_contenido=TESIS_TEXTO,
    tam_titulo=11,
    tam_contenido=9,
    alineacion="left",
):
    """Añade una caja de texto con esquinas redondeadas."""

    caja = FancyBboxPatch(
        (x, y),
        ancho,
        alto,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=borde,
        facecolor=fondo,
    )

    ax.add_patch(caja)

    if alineacion == "left":
        tx = x + 0.035 * ancho
        ha = "left"
    else:
        tx = x + ancho / 2
        ha = "center"

    ax.text(
        tx,
        y + alto * 0.68,
        titulo,
        ha=ha,
        va="center",
        fontsize=tam_titulo,
        fontweight="bold",
        color=color_titulo,
    )

    if contenido:
        ax.text(
            tx,
            y + alto * 0.34,
            contenido,
            ha=ha,
            va="center",
            fontsize=tam_contenido,
            color=color_contenido,
            linespacing=1.4,
        )

    return caja


def agregar_flecha(
    ax,
    inicio,
    fin,
    color=TESIS_VERDE,
    grosor=1.6,
    estilo="-",
    escala=13,
):
    """Añade una flecha de flujo."""

    flecha = FancyArrowPatch(
        inicio,
        fin,
        arrowstyle="-|>",
        mutation_scale=escala,
        linewidth=grosor,
        color=color,
        linestyle=estilo,
        connectionstyle="arc3,rad=0",
    )

    ax.add_patch(flecha)
    return flecha


def estilizar_ejes(ax):
    """Formato homogéneo para las gráficas científicas."""

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        length=4,
        width=0.8,
    )

    ax.grid(
        True,
        color=GRIS,
        alpha=0.18,
        linestyle=":",
        linewidth=0.7,
    )


# =============================================================================
# FIGURA 3.1
# DIAGRAMA DE FLUJO METODOLÓGICO
# =============================================================================

fig, ax = plt.subplots(figsize=(15, 6))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Encabezado conceptual
ax.add_patch(
    Circle(
        (0.035, 0.92),
        0.014,
        facecolor=TESIS_TERRACOTA,
        edgecolor="none",
    )
)

ax.text(
    0.06,
    0.92,
    "Los observables sugieren una estrella",
    va="center",
    fontsize=11,
)

agregar_flecha(
    ax,
    (0.34, 0.92),
    (0.42, 0.92),
    color=TESIS_ROJO_TIERRA,
)

ax.add_patch(
    Circle(
        (0.45, 0.92),
        0.014,
        facecolor=TESIS_TERRACOTA,
        edgecolor="none",
    )
)

ax.add_patch(
    Circle(
        (0.475, 0.92),
        0.008,
        facecolor=TESIS_ROJO_TIERRA,
        edgecolor="none",
    )
)

ax.text(
    0.50,
    0.92,
    "Los eclipses revelan dos componentes",
    va="center",
    fontsize=11,
)

ax.text(
    0.99,
    0.92,
    "S1–S3",
    ha="right",
    va="center",
    fontsize=10,
    color=TESIS_ROJO_TIERRA,
    fontweight="bold",
)

# Entradas observacionales
agregar_caja(
    ax,
    0.015, 0.58, 0.18, 0.21,
    "Gaia DR3 + espectro",
    r"$T_{\mathrm{eff},1}$ o clase espectral",
    fondo=BLANCO,
    borde=TESIS_VERDE,
    tam_titulo=11,
    tam_contenido=9,
)

agregar_caja(
    ax,
    0.015, 0.13, 0.18, 0.21,
    "Curva de luz TESS",
    "Detrending\nWōtan / hspline",
    fondo=BLANCO,
    borde=TESIS_VERDE,
    tam_titulo=11,
    tam_contenido=9,
)

# Ajuste de la primaria
agregar_caja(
    ax,
    0.25, 0.58, 0.205, 0.21,
    "Ajuste MassAge",
    r"$M_1,\ T_{\mathrm{eff},1},\ L_1$" "\n"
    r"$R_1,\ t_1$",
    fondo=TESIS_BEIGE,
    borde=TESIS_TERRACOTA,
    tam_titulo=12,
    tam_contenido=9,
)

# Búsqueda de periodos y profundidad
agregar_caja(
    ax,
    0.25, 0.13, 0.205, 0.21,
    "BLS + Lomb–Scargle",
    r"$P_{\mathrm{orb}},\ P_{\mathrm{rot}},\ \delta$",
    fondo=TESIS_BEIGE,
    borde=TESIS_TERRACOTA,
    tam_titulo=11,
    tam_contenido=9,
)

# Hipótesis S1
ax.add_patch(
    Circle(
        (0.225, 0.685),
        0.025,
        facecolor=TESIS_ROJO_TIERRA,
        edgecolor="none",
        zorder=5,
    )
)

ax.text(
    0.225, 0.685, "S1",
    color=BLANCO,
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    zorder=6,
)

# Hipótesis S2
ax.add_patch(
    Circle(
        (0.48, 0.685),
        0.025,
        facecolor=TESIS_ROJO_TIERRA,
        edgecolor="none",
        zorder=5,
    )
)

ax.text(
    0.48, 0.685, "S2",
    color=BLANCO,
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    zorder=6,
)

# Coetaneidad
agregar_caja(
    ax,
    0.53, 0.58, 0.14, 0.21,
    r"$t_2=t_1$",
    "Sistema coetáneo",
    fondo=TESIS_BEIGE,
    borde=TESIS_TERRACOTA,
    tam_titulo=13,
    tam_contenido=8,
    alineacion="center",
)

# Hipótesis S3
ax.add_patch(
    Circle(
        (0.48, 0.235),
        0.025,
        facecolor=TESIS_ROJO_TIERRA,
        edgecolor="none",
        zorder=5,
    )
)

ax.text(
    0.48, 0.235, "S3",
    color=BLANCO,
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    zorder=6,
)

# Radio secundario
agregar_caja(
    ax,
    0.53, 0.13, 0.14, 0.21,
    r"$R_2=R_1\sqrt{\delta}$",
    "Estimación aproximada",
    fondo=TESIS_BEIGE,
    borde=TESIS_TERRACOTA,
    tam_titulo=10,
    tam_contenido=8,
    alineacion="center",
)

# Interpolador
agregar_caja(
    ax,
    0.705, 0.34, 0.16, 0.29,
    "Interpolador",
    r"$T_{\mathrm{eff},2}$" "\n"
    r"$=T_{\mathrm{eff}}(R_2,t_1)$" "\n\n"
    "Isocronas estelares",
    fondo=BLANCO,
    borde=TESIS_ROJO_TIERRA,
    tam_titulo=11,
    tam_contenido=8.5,
)

# Borde discontinuo alrededor del interpolador
ax.add_patch(
    FancyBboxPatch(
        (0.697, 0.33),
        0.176,
        0.31,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=TESIS_ROJO_TIERRA,
        facecolor="none",
        linestyle=(0, (4, 3)),
    )
)

# Modelo final
ax.add_patch(
    FancyBboxPatch(
        (0.89, 0.33),
        0.10,
        0.29,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=TESIS_VERDE_OSCURO,
        facecolor=TESIS_VERDE_OSCURO,
    )
)

ax.text(
    0.94, 0.55,
    "Modelo del\nsistema",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color=BLANCO,
)

ax.text(
    0.94, 0.415,
    r"$R_1,\ M_1$" "\n"
    r"$R_2,\ M_2$" "\n"
    r"$P_{\mathrm{orb}}$" "\n"
    "MCMC",
    ha="center",
    va="center",
    fontsize=8.5,
    color=BLANCO,
    linespacing=1.4,
)

# Flechas de flujo
agregar_flecha(ax, (0.195, 0.685), (0.25, 0.685))
agregar_flecha(ax, (0.455, 0.685), (0.53, 0.685))
agregar_flecha(ax, (0.67, 0.685), (0.72, 0.55))

agregar_flecha(ax, (0.195, 0.235), (0.25, 0.235))
agregar_flecha(ax, (0.455, 0.235), (0.53, 0.235))
agregar_flecha(ax, (0.67, 0.235), (0.72, 0.41))

agregar_flecha(ax, (0.865, 0.475), (0.89, 0.475))

# Dependencia del radio primario
agregar_flecha(
    ax,
    (0.40, 0.58),
    (0.56, 0.35),
    color=TESIS_TERRACOTA,
    grosor=1.3,
    estilo="--",
    escala=11,
)

ax.text(
    0.465, 0.435,
    r"$R_1$",
    color=TESIS_TERRACOTA,
    fontsize=10,
    rotation=-30,
)

ax.text(
    0.5, 0.035,
    "Esquema conceptual de las dependencias usadas en la inferencia.",
    ha="center",
    fontsize=8,
    color=GRIS,
    style="italic",
)

fig.tight_layout()

guardar_figura(
    fig,
    "Figura_3_1_Pipeline_Flujo",
)


# =============================================================================
# FIGURA 3.2
# PREPROCESAMIENTO DE CURVAS DE LUZ
# =============================================================================

tiempo = np.linspace(0, 27, 2000)

P_orb = 3.42
P_rot = 1.85
t0 = 0.85

tendencia = (
    1.0
    + 0.025 * np.sin(2 * np.pi * tiempo / 15)
    + 0.012 * (tiempo / 27) ** 2
)

manchas = 0.012 * np.cos(2 * np.pi * tiempo / P_rot)

fase_orbital = ((tiempo - t0) / P_orb) % 1

dist_primario = np.minimum(
    fase_orbital,
    1 - fase_orbital,
)

dist_secundario = np.abs(
    fase_orbital - 0.5
)

eclipses = np.ones_like(tiempo)

eclipses -= 0.12 * np.exp(
    -0.5 * (dist_primario / 0.018) ** 2
)

eclipses -= 0.035 * np.exp(
    -0.5 * (dist_secundario / 0.025) ** 2
)

flujo_raw = (
    tendencia
    * (1 + manchas)
    * eclipses
    + rng.normal(0, 0.0025, len(tiempo))
)

flujo_aplanado = (
    flujo_raw / (tendencia * (1 + manchas))
    + rng.normal(0, 0.001, len(tiempo))
)

fig, axs = plt.subplots(
    2, 2,
    figsize=(10.5, 6.7),
)

# Panel A
axs[0, 0].plot(
    tiempo,
    flujo_raw,
    ".",
    ms=1.5,
    color=TESIS_VERDE_OSCURO,
    alpha=0.45,
    label="Flujo sintético",
)

axs[0, 0].plot(
    tiempo,
    tendencia * (1 + manchas),
    color=TESIS_TERRACOTA,
    lw=1.3,
    label="Tendencia ilustrativa",
)

axs[0, 0].set_title("(a) Serie temporal y tendencia")
axs[0, 0].set_xlabel("Tiempo (d)")
axs[0, 0].set_ylabel("Flujo relativo")
axs[0, 0].legend()

# Panel B
axs[0, 1].plot(
    tiempo,
    flujo_aplanado,
    ".",
    ms=1.5,
    color=TESIS_VERDE_OSCURO,
    alpha=0.5,
)

axs[0, 1].axhline(
    1,
    color=TESIS_ROJO_TIERRA,
    ls="--",
    lw=1,
)

axs[0, 1].set_title("(b) Curva aplanada")
axs[0, 1].set_xlabel("Tiempo (d)")
axs[0, 1].set_ylabel("Flujo relativo")

# Panel C
axs[1, 0].plot(
    fase_orbital,
    flujo_aplanado,
    ".",
    ms=1.6,
    color=TESIS_VERDE_OSCURO,
    alpha=0.45,
)

axs[1, 0].set_xlim(0, 1)
axs[1, 0].set_title(
    rf"(c) Plegado orbital ($P_{{\mathrm{{orb}}}}={P_orb:.2f}$ d)"
)
axs[1, 0].set_xlabel("Fase orbital")
axs[1, 0].set_ylabel("Flujo relativo")

# Panel D
sin_eclipse = (
    (dist_primario > 0.035)
    & (dist_secundario > 0.04)
)

fase_rotacional = (
    (tiempo[sin_eclipse] % P_rot) / P_rot
)

flujo_rotacional = (
    flujo_raw[sin_eclipse]
    / tendencia[sin_eclipse]
)

axs[1, 1].plot(
    fase_rotacional,
    flujo_rotacional,
    ".",
    ms=1.6,
    color=TESIS_TERRACOTA,
    alpha=0.5,
)

axs[1, 1].set_xlim(0, 1)
axs[1, 1].set_title(
    rf"(d) Modulación rotacional ($P_{{\mathrm{{rot}}}}={P_rot:.2f}$ d)"
)
axs[1, 1].set_xlabel("Fase rotacional")
axs[1, 1].set_ylabel("Flujo relativo")

for ax_ in axs.flat:
    estilizar_ejes(ax_)

fig.suptitle(
    "Figura 3.2. Etapas de procesamiento de la curva de luz",
    fontweight="bold",
)

fig.text(
    0.5, 0.005,
    "Datos sintéticos para ilustración; no corresponden a una estrella de la muestra.",
    ha="center",
    fontsize=8,
    color=GRIS,
    style="italic",
)

fig.tight_layout(rect=[0, 0.035, 1, 0.96])

guardar_figura(
    fig,
    "Figura_3_2_Preprocesamiento_Diagnostico",
)


# =============================================================================
# FIGURA 3.3
# PROFUNDIDAD DE ECLIPSE Y SELECCIÓN DE VENTANA
# =============================================================================

fase_zoom = np.linspace(-0.05, 0.05, 500)

profundidad = 0.29
ancho_eclipse = 0.018

perfil_eclipse = (
    1
    - profundidad
    * np.exp(
        -0.5
        * (fase_zoom / (ancho_eclipse / 2.6)) ** 2
    )
)

flujo_observado = (
    perfil_eclipse
    + rng.normal(0, 0.008, len(fase_zoom))
)

fig, ax = plt.subplots(figsize=(7.5, 4.8))

ax.plot(
    fase_zoom,
    flujo_observado,
    ".",
    ms=3,
    color=TESIS_VERDE_OSCURO,
    alpha=0.55,
    label="Datos sintéticos",
)

ax.plot(
    fase_zoom,
    perfil_eclipse,
    color=TESIS_TERRACOTA,
    lw=1.6,
    label="Perfil ilustrativo",
)

ax.axvspan(
    -0.03, 0.03,
    color=TESIS_ROJO_TIERRA,
    alpha=0.12,
    label="Ventana fija amplia",
)

ax.axvspan(
    -ancho_eclipse / 2,
    ancho_eclipse / 2,
    color=TESIS_VERDE,
    alpha=0.20,
    label="Ventana centrada en el eclipse",
)

ax.axhline(
    1 - profundidad,
    color=TESIS_VERDE,
    ls="--",
    lw=1.2,
    label=rf"Profundidad ilustrativa $\delta={profundidad:.2f}$",
)

ax.set_xlabel("Fase orbital respecto al centro del eclipse")
ax.set_ylabel("Flujo relativo")

ax.set_title(
    "Figura 3.3. Selección de ventana para medir la profundidad"
)

estilizar_ejes(ax)

ax.legend(
    frameon=False,
    loc="lower right",
)

fig.text(
    0.5, 0.005,
    "Esquema ilustrativo: la profundidad depende de la geometría del eclipse "
    "y de la definición de la ventana.",
    ha="center",
    fontsize=8,
    color=GRIS,
    style="italic",
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

guardar_figura(
    fig,
    "Figura_3_3_Profundidad_Adaptativa",
)


# =============================================================================
# FIGURA 3.4
# DIAGRAMA HR E INFERENCIA EVOLUTIVA
# =============================================================================

Teff = np.linspace(3000, 7500, 240)

# Curvas esquemáticas, no isocronas físicas de MIST o BHAC15.
logL_10Myr = (
    -0.65
    + 3.6 * np.log10(Teff / 4000)
)

logL_100Myr = (
    -1.00
    + 3.15 * np.log10(Teff / 4000)
)

logL_zams = (
    -1.25
    + 2.75 * np.log10(Teff / 4000)
)

fig, ax = plt.subplots(figsize=(7.5, 5.2))

ax.plot(
    Teff,
    logL_10Myr,
    color=TESIS_TERRACOTA,
    lw=1.8,
    label="Isocrona joven (esquemática)",
)

ax.plot(
    Teff,
    logL_100Myr,
    color=TESIS_VERDE,
    lw=1.6,
    ls="--",
    label="Isocrona evolucionada (esquemática)",
)

ax.plot(
    Teff,
    logL_zams,
    color=TESIS_ROJO_TIERRA,
    lw=1.5,
    ls=":",
    label="Secuencia principal (esquemática)",
)

# Punto de ejemplo
ax.errorbar(
    6200,
    0.35,
    xerr=150,
    yerr=0.08,
    fmt="o",
    ms=6,
    color=TESIS_VERDE_OSCURO,
    ecolor=TESIS_VERDE_OSCURO,
    capsize=3,
    label="Primaria (ejemplo)",
)

ax.invert_xaxis()

ax.set_xlabel(
    r"Temperatura efectiva $T_{\mathrm{eff}}$ (K)"
)

ax.set_ylabel(
    r"$\log(L/L_\odot)$"
)

ax.set_title(
    "Figura 3.4. Diagrama HR e inferencia de parámetros estelares"
)

estilizar_ejes(ax)

ax.legend(
    frameon=False,
    loc="lower left",
)

fig.text(
    0.5, 0.005,
    "Curvas esquemáticas. Sustituir por las isocronas de los modelos "
    "evolutivos utilizados en el análisis.",
    ha="center",
    fontsize=8,
    color=GRIS,
    style="italic",
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

guardar_figura(
    fig,
    "Figura_3_4_HR_Isocronas",
)


# =============================================================================
# FIGURA 3.5
# AJUSTE DE ECLIPSE Y RESIDUOS
# =============================================================================

tiempo_eclipse = np.linspace(-0.04, 0.04, 300)

modelo = (
    1
    - 0.12
    * np.exp(
        -0.5
        * (tiempo_eclipse / 0.012) ** 2
    )
)

sigma = 0.002

flujo_observado = (
    modelo
    + rng.normal(0, sigma, len(tiempo_eclipse))
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 5.8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

# Ajuste
ax1.plot(
    tiempo_eclipse,
    flujo_observado,
    ".",
    ms=2.5,
    color=TESIS_VERDE_OSCURO,
    alpha=0.55,
    label="Datos sintéticos",
)

ax1.plot(
    tiempo_eclipse,
    modelo,
    color=TESIS_ROJO_TIERRA,
    lw=1.7,
    label="Modelo ilustrativo",
)

ax1.fill_between(
    tiempo_eclipse,
    modelo - 2 * sigma,
    modelo + 2 * sigma,
    color=TESIS_TERRACOTA,
    alpha=0.22,
    label=r"Banda ilustrativa de $\pm 2\sigma$",
)

ax1.set_ylabel("Flujo relativo")

ax1.set_title(
    "Figura 3.5. Ajuste de eclipse y análisis de residuos"
)

ax1.legend(
    frameon=False,
    loc="lower right",
)

# Residuos normalizados
residuos = (
    flujo_observado - modelo
) / sigma

ax2.plot(
    tiempo_eclipse,
    residuos,
    ".",
    ms=2.2,
    color=TESIS_VERDE_OSCURO,
    alpha=0.6,
)

ax2.axhline(
    0,
    color=TESIS_ROJO_TIERRA,
    ls="--",
    lw=1,
)

ax2.axhline(
    3,
    color=GRIS,
    ls=":",
    lw=0.8,
)

ax2.axhline(
    -3,
    color=GRIS,
    ls=":",
    lw=0.8,
)

ax2.set_ylabel(r"Residuo ($\sigma$)")
ax2.set_xlabel("Tiempo relativo al centro del eclipse (d)")
ax2.set_ylim(-4.5, 4.5)

estilizar_ejes(ax1)
estilizar_ejes(ax2)

fig.text(
    0.5, 0.005,
    "Demostración sintética. Sustituir por la curva observada y el ajuste "
    "posterior correspondiente.",
    ha="center",
    fontsize=8,
    color=GRIS,
    style="italic",
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

guardar_figura(
    fig,
    "Figura_3_5_Ajuste_Starry_Residuos",
)


# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "=" * 65)
print("GENERACIÓN DE FIGURAS COMPLETADA")
print("=" * 65)

print(f"\nDirectorio de salida: {OUTPUT_DIR.resolve()}")

print("\nArchivos generados:")
for archivo in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {archivo.name}")

print("\nFormato:")
print("  PDF: vectorial, recomendado para la tesis.")
print("  PNG: 300 dpi.")
print("\nRecuerda sustituir los datos sintéticos por los resultados reales.")