"""Construye secundarias_con_edad_de_grupo.ipynb (kernel starry)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
celdas = [
md(r"""# Propiedades de la secundaria con la edad de la vecindad

Para los sistemas 7, 11 y 20 se usa la edad asignada por la vecindad (`edades_asignadas.csv`) en lugar de la
edad individual de la primaria, y el interpolador de isócronas `isocronas_interp.py`
(`datos_estrellas_massage_orion/`) para pasar de $(R_2, t)$ a $M_2$, $T_{{\rm eff},2}$ y $\log L_2$.
El interpolador invierte $R(M)$ a edad fija y evalúa las propiedades en esa masa; no extrapola (fuera de la
malla devuelve NaN).

**Radio de la secundaria**, dos estimaciones independientes por sector:
1. **Profundidad:** $R_2 = R_1\sqrt{D_1}$, con $R_1$ del catálogo (Stefan–Boltzmann, rama MIST y Somers17) y
   $D_1$ de `priors_v2.csv` (profundidad medida en el fondo del eclipse; la versión anterior de este cálculo
   usaba profundidades v1, que subestimaban $D_1$).
2. **starry:** el radio $R_B$ ajustado (`B_r`) en `starry_por_tic/`. Es geométrico, pero hereda la dilución
   de las curvas de tess_extractor (los eclipses son 15–50 % más someros que en TGLC), así que tiende a
   subestimar $R_2$.

**Edad:** la mediana del grupo y, para ver la sensibilidad, los percentiles 16 y 84 de las edades de sus
miembros.

**Modelos:** MIST, Somers et al. (2017) con 17 % de manchas (`f017`) y BHAC15 (Baraffe et al. 2015)."""),
code(r"""import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUTA = Path("/home/luciam/Documents/starry/Resultados_Orion")
RUTA_MASSAGE = RUTA / "datos_estrellas_massage_orion"
sys.path.insert(0, str(RUTA_MASSAGE))
from isocronas_interp import load_isochrone_model, IsochroneInterpolator, invert_radius_to_mass

modelos_dir = RUTA_MASSAGE / "modelos"
INTERP = {
    "MIST": IsochroneInterpolator(load_isochrone_model(
        "MIST", modelos_dir / "iso_mist" / "MIST_iso_6a9a687177c08_todas_edades.csv"), "MIST"),
    "Somers17": IsochroneInterpolator(load_isochrone_model(
        "Somers17", modelos_dir / "iso_somers17" / "f017track_ms.csv"), "Somers17"),
    "BHAC15": IsochroneInterpolator(load_isochrone_model(
        "BHAC15", modelos_dir / "iso_barafe" / "BHAC15_iso.txt"), "BHAC15"),
}

SISTEMAS = {264483228: "sistema 7", 138829403: "sistema 11", 264459857: "sistema 20"}
edades = pd.read_csv(RUTA / "edades_vecindad" / "edades_asignadas.csv").set_index("TIC")
priors = pd.read_csv(RUTA / "priors_v2.csv")
cat_prev = pd.read_csv(RUTA_MASSAGE / "teff2_desde_profundidad_transito.csv")
R1 = cat_prev.groupby("TIC")[["R1_MIST", "R1_Somers17"]].first()
edades.loc[list(SISTEMAS), ["metodo", "edad_grupo_Myr", "err_edad_grupo_Myr", "p16_Myr", "p84_Myr"]]"""),
md("## 1) Radio de la secundaria por sector"),
code(r"""def starry_r(tic, sector):
    # radio ajustado con starry (Rsun) y su error; carpeta S<s> o S<s>_ecc más reciente
    cands = sorted((RUTA / "starry_por_tic" / f"TIC_{tic}").glob(f"S{sector}*/*_Summary.csv"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        return np.nan, np.nan
    s = pd.read_csv(cands[-1], index_col=0)
    return s.loc["B_r", "mean"], s.loc["B_r", "sd"]


filas = []
for tic in SISTEMAS:
    for r in priors.loc[priors["TIC"] == tic].itertuples():
        rs, rs_sd = starry_r(tic, int(r.Sector))
        for rama in ("MIST", "Somers17"):
            filas.append(dict(TIC=tic, sistema=SISTEMAS[tic], Sector=int(r.Sector), fuente_R2=f"profundidad ({rama})",
                              R2=R1.loc[tic, f"R1_{rama}"] * np.sqrt(r.Depth_primary), D1=r.Depth_primary))
        filas.append(dict(TIC=tic, sistema=SISTEMAS[tic], Sector=int(r.Sector), fuente_R2="starry", R2=rs, R2_sd=rs_sd))
radios = pd.DataFrame(filas)
radios.pivot_table(index=["sistema", "Sector"], columns="fuente_R2", values="R2").round(3)"""),
md(r"""## 2) $M_2$, $T_{{\rm eff},2}$ y $\log L_2$ con cada modelo

Para cada radio se evalúa a la edad del grupo (mediana) y a sus percentiles 16 y 84."""),
code(r"""def propiedades(R2, edad_Myr, interp):
    lg = np.log10(edad_Myr * 1e6)
    m = invert_radius_to_mass(lg, R2, interp)
    if not np.isfinite(m):
        return np.nan, np.nan, np.nan
    return m, float(interp.teff(lg, m)), float(interp.logL(lg, m))


res = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for r in radios.itertuples():
        e = edades.loc[r.TIC]
        for modelo, interp in INTERP.items():
            fila = dict(TIC=r.TIC, sistema=r.sistema, Sector=r.Sector, fuente_R2=r.fuente_R2, R2=r.R2, modelo=modelo,
                        edad_Myr=e["edad_grupo_Myr"])
            for etiqueta, t in (("", e["edad_grupo_Myr"]), ("_p16", e["p16_Myr"]), ("_p84", e["p84_Myr"])):
                m, T, L = propiedades(r.R2, t, interp)
                fila.update({f"M2{etiqueta}": m, f"Teff2{etiqueta}": T, f"logL2{etiqueta}": L})
            res.append(fila)
res = pd.DataFrame(res)
res.to_csv(RUTA / "edades_vecindad" / "secundarias_con_edad_de_grupo.csv", index=False)
pd.set_option("display.width", 220)
res.pivot_table(index=["sistema", "fuente_R2", "Sector"], columns="modelo", values=["M2", "Teff2"]).round(2)"""),
md(r"""## 3) Resumen por sistema

Mediana sobre sectores de cada combinación (fuente de $R_2$, modelo). El rango entre paréntesis es el que
resulta de usar los percentiles 16 y 84 de la edad del grupo."""),
code(r"""def resumen(g):
    return pd.Series(dict(
        R2=g["R2"].median(),
        M2=g["M2"].median(), M2_rango=f"{min(g['M2_p16'].median(), g['M2_p84'].median()):.2f}–{max(g['M2_p16'].median(), g['M2_p84'].median()):.2f}",
        Teff2=g["Teff2"].median(), Teff2_rango=f"{min(g['Teff2_p16'].median(), g['Teff2_p84'].median()):.0f}–{max(g['Teff2_p16'].median(), g['Teff2_p84'].median()):.0f}",
        logL2=g["logL2"].median(), n_sectores=int(g["M2"].notna().sum())))


tabla = res.groupby(["sistema", "fuente_R2", "modelo"]).apply(resumen).reset_index()
tabla.to_csv(RUTA / "edades_vecindad" / "secundarias_resumen.csv", index=False)
tabla.round(3)"""),
code(r"""fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
marcas = {"MIST": "o", "Somers17": "s", "BHAC15": "^"}
colores = {"profundidad (MIST)": "#2a78d6", "profundidad (Somers17)": "#1baf7a", "starry": "#c0392b"}
for ax, (tic, nombre) in zip(axs, SISTEMAS.items()):
    g = res.loc[res["TIC"] == tic]
    for (fuente, modelo), h in g.groupby(["fuente_R2", "modelo"]):
        h = h.dropna(subset=["Teff2"])
        if h.empty:
            continue
        lo = h[["Teff2", "Teff2_p16", "Teff2_p84"]].min(axis=1)
        hi = h[["Teff2", "Teff2_p16", "Teff2_p84"]].max(axis=1)
        ax.errorbar(h["Teff2"], h["M2"], xerr=[h["Teff2"] - lo, hi - h["Teff2"]],
                    fmt=marcas[modelo], color=colores[fuente], alpha=0.8, ms=7, capsize=2,
                    label=f"{fuente}, {modelo}")
    e = edades.loc[tic]
    ax.set_title(f"{nombre} (TIC {tic}): edad {e['edad_grupo_Myr']:.1f} Myr")
    ax.set_xlabel("Teff2 [K]"); ax.set_ylabel("M2 [M☉]"); ax.grid(alpha=0.25)
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(RUTA / "edades_vecindad" / "secundarias_con_edad_de_grupo.png", dpi=120)
plt.show()"""),
]
nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, "/home/luciam/Documents/starry/Resultados_Orion/edades_vecindad/secundarias_con_edad_de_grupo.ipynb")
