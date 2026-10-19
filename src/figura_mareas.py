"""Circularización y sincronización frente al periodo orbital, coloreado por edad.

Lee diagnostico_resultados/tabla_maestra.csv (una fila por corrida TIC-sector) y toma la mediana
por sistema. Usa |e cos w| (lo que la fotometría mide bien), no e.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
TABLA = AQUI.parents[1] / "diagnostico_resultados" / "tabla_maestra.csv"
SALIDA = AQUI / "figuras"
SALIDA.mkdir(exist_ok=True)

T = pd.read_csv(TABLA)
S = T.groupby("sistema").agg(
    P=("P", "median"), ecosw=("ecosw", "median"), ecosw_sd=("ecosw_sd", "median"),
    abs95=("abs_ecosw_95", "median"), e=("e", "median"), edad=("edad_Myr", "first"),
    fuente=("edad_fuente", "first"), Prot=("P_rot_consenso", "first"), TeffA=("Teff_A", "first"),
).reset_index()
S["sinc"] = S.Prot / S.P
S["caliente"] = S.TeffA > 10_000          # envolvente radiativa: P_LS es elipsoidal, no rotación
S["signif"] = np.abs(S.ecosw) > 3 * S.ecosw_sd


def pseudo_sinc(e):
    """Omega_ps / n de Hut (1981)."""
    e2 = e**2
    f2 = 1 + 7.5 * e2 + 45 / 8 * e2**2 + 5 / 16 * e2**3
    f5 = 1 + 3 * e2 + 3 / 8 * e2**2
    return f2 / (f5 * (1 - e2) ** 1.5)


# rampa secuencial azul (claro = joven, oscuro = viejo), escala logarítmica en edad
rampa = mpl.colors.LinearSegmentedColormap.from_list(
    "azul", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
norma = mpl.colors.LogNorm(vmin=0.15, vmax=40)
TINTA, TINTA2, REJILLA = "#1f1f1f", "#5f5f5f", "#e6e6e6"
# desplazamientos de etiqueta (puntos) donde los sistemas se enciman
DESP_A = {3: (-11, 2), 5: (-11, 2), 10: (-12, 4), 4: (5, -8)}
DESP_B = {3: (-12, 2), 5: (-6, 6), 6: (2, 6), 7: (6, -10), 9: (-10, 4), 10: (5, -10)}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True,
                               gridspec_kw={"hspace": 0.08})

# --- (a) |e cos w| frente a P --------------------------------------------------------------
PISO = 3e-4
for _, r in S.iterrows():
    c = rampa(norma(r.edad))
    mk = "s" if r.fuente == "vecindad" else "o"
    if r.signif and abs(r.ecosw) > 0.005:
        ax1.errorbar(r.P, abs(r.ecosw), yerr=r.ecosw_sd, fmt=mk, ms=7, color=c, mec="white",
                     mew=1, ecolor=c, zorder=3)
    else:  # límite superior al 95 %
        y = max(r.abs95, PISO)
        ax1.errorbar(r.P, y, yerr=0.35 * y, uplims=True, fmt=mk, ms=7, color=c, mec="white",
                     mew=1, ecolor=c, zorder=3, capsize=0)
    ax1.annotate(str(r.sistema), (r.P, max(r.abs95, PISO, abs(r.ecosw))),
                 xytext=DESP_A.get(r.sistema, (5, 2)),
                 textcoords="offset points", fontsize=7, color=TINTA2)
ax1.axvspan(6.3, 7.46, color="#f0efec", zorder=0)
ax1.text(6.88, 0.3, "último circular (20)\nprimer excéntrico (21)", ha="center", va="top",
         fontsize=6.5, color=TINTA2)
ax1.set_yscale("log")
ax1.set_ylim(1.5e-4, 0.4)
ax1.set_ylabel(r"$|e\cos\omega|$")
ax1.grid(axis="y", color=REJILLA, lw=0.6)
ax1.text(0.01, 0.98, "(a) circularización", transform=ax1.transAxes, va="top", color=TINTA)

# --- (b) P_LS / P_orb frente a P -----------------------------------------------------------
for y0, txt in [(1, "sincronizada"), (0.5, r"$P_{\rm orb}/2$ (elipsoidal)")]:
    ax2.axhline(y0, color=TINTA2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax2.text(8.45, y0, txt, fontsize=7, color=TINTA2, va="bottom", ha="right")
for _, r in S.iterrows():
    c = rampa(norma(r.edad))
    mk = "s" if r.fuente == "vecindad" else "o"
    kw = dict(mfc="white", mec=c, mew=1.6) if r.caliente else dict(mfc=c, mec="white", mew=1)
    ax2.plot(r.P, r.sinc, mk, ms=7, zorder=3, **kw)
    ax2.annotate(str(r.sistema), (r.P, r.sinc), xytext=DESP_B.get(r.sistema, (5, 2)),
                 textcoords="offset points",
                 fontsize=7, color=TINTA2)
s21 = S.set_index("sistema").loc[21]
yps = 1 / pseudo_sinc(s21.e)
ax2.plot([s21.P - 0.25, s21.P + 0.25], [yps, yps], color=TINTA, lw=2)
ax2.annotate(f"pseudo-sinc. Hut (1981)\ncon e={s21.e:.2f}: {yps:.2f}", (s21.P, yps),
             xytext=(-150, 18), textcoords="offset points", fontsize=7, color=TINTA,
             arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.6))
ax2.set_ylim(0, 1.2)
ax2.set_xlim(0.5, 8.5)
ax2.set_xlabel(r"$P_{\rm orb}$ (d)")
ax2.set_ylabel(r"$P_{\rm LS}/P_{\rm orb}$")
ax2.grid(axis="y", color=REJILLA, lw=0.6)
ax2.text(0.01, 0.98, "(b) sincronización", transform=ax2.transAxes, va="top", color=TINTA)

# leyenda de símbolos + barra de edad
h = [
    plt.Line2D([], [], marker="o", ls="", color=TINTA2, mec="white", label="edad: massage"),
    plt.Line2D([], [], marker="s", ls="", color=TINTA2, mec="white", label="edad: vecindad"),
    plt.Line2D([], [], marker="o", ls="", mfc="white", mec=TINTA2, mew=1.6,
               label=r"primaria caliente ($T_{\rm eff}>10^4$ K)"),
    plt.Line2D([], [], marker="v", ls="", color=TINTA2, label="límite sup. 95 %"),
]
ax2.legend(handles=h, loc="upper center", fontsize=7, frameon=False, ncol=2,
           bbox_to_anchor=(0.5, -0.16))
cb = fig.colorbar(mpl.cm.ScalarMappable(norma, rampa), ax=[ax1, ax2], pad=0.02, aspect=35)
cb.set_label("edad (Myr)")
cb.set_ticks([0.2, 0.5, 1, 2, 5, 10, 20, 40])
cb.set_ticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "40"])
cb.minorticks_off()
cb.outline.set_visible(False)

for ext in ("png", "pdf"):
    fig.savefig(SALIDA / f"fig_mareas_P.{ext}", dpi=200, bbox_inches="tight")
S.round(4).to_csv(AQUI / "numeros_mareas.csv", index=False)
print(S[["sistema", "P", "ecosw", "ecosw_sd", "abs95", "e", "sinc", "edad", "fuente",
         "caliente"]].round(3).to_string(index=False))
print(f"\nSistema 21: pseudo-sincronización predicha {yps:.3f}, observada {s21.sinc:.3f}")
