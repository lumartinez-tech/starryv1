"""Construye 2_candidatas_binarias.ipynb (kernel: Python (cumulos))."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
celdas = [
md(r"""# Pléyades 2: preparación de la inspección visual de binarias eclipsantes

Para los miembros del grupo elegido (`pleyades_miembros.csv`):

1. **Cruce con el TIC 8.2** (1″, el más cercano) para el TIC, Tmag y la razón de contaminación.
2. **Sectores de TESS** que observaron a cada miembro, con `tess-point` (cálculo local, sin TESScut).
3. **Lista para TESSExtractor** en el formato de `pleyades_tessextractor_sample2` (`# TIC RA Dec Sector Tmag`).
4. **Binarias eclipsantes ya catalogadas** entre los miembros, para priorizar la inspección y como control:
   VSX (tipos con `E`: EA, EB, EW, E), binarias eclipsantes de TESS (Prša et al. 2022) y binarias eclipsantes
   de Gaia DR3 (`vari_eclipsing_binary`, Mowlavi et al. 2023)."""),
code(r"""import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import Table
from astroquery.xmatch import XMatch
from tess_stars2px import tess_stars2px_function_entry

warnings.simplefilter("ignore")
AQUI = Path("/home/luciam/Documents/starry/Resultados_Pleyades/membresia")
m = pd.read_csv(AQUI / "pleyades_miembros.csv")
print(f"{len(m)} miembros")


def xmatch(df, cat, radio_arcsec):
    t = Table.from_pandas(df[["source_id", "ra", "dec"]])
    r = XMatch.query(cat1=t, cat2=cat, max_distance=radio_arcsec * u.arcsec, colRA1="ra", colDec1="dec").to_pandas()
    return r.sort_values("angDist").drop_duplicates("source_id")"""),
md("## 1) TIC 8.2"),
code(r"""tic = xmatch(m, "vizier:IV/39/tic82", 1.0)
m = m.merge(tic[["source_id", "TIC", "Tmag", "Teff", "Rcont", "Jmag", "Hmag", "Kmag"]], on="source_id", how="left")
print(f"con TIC: {m['TIC'].notna().sum()} de {len(m)}; Tmag: {m['Tmag'].min():.1f}–{m['Tmag'].max():.1f}")
m.to_csv(AQUI / "pleyades_miembros_TIC82.csv", index=False)"""),
md("## 2) Sectores de TESS y lista para TESSExtractor"),
code(r"""ok = m["TIC"].notna()
tics = m.loc[ok, "TIC"].astype(int).values
res = tess_stars2px_function_entry(tics, m.loc[ok, "ra"].values, m.loc[ok, "dec"].values)
sect = pd.DataFrame({"TIC": res[0], "Sector": res[3]})
SECTOR_MAX = 107   # tess-point devuelve también sectores fuera de la misión (p. ej. 1751); se descartan
sect = sect.loc[(sect["Sector"] > 0) & (sect["Sector"] <= SECTOR_MAX)].drop_duplicates()
lista = sect.merge(m.loc[ok, ["TIC", "ra", "dec", "Tmag"]].astype({"TIC": int}), on="TIC").sort_values(["TIC", "Sector"])
with open(AQUI / "pleyades_tessextractor_miembros.txt", "w") as f:
    f.write("# TIC       RA         Dec        Sector Tmag  \n")
    for r in lista.itertuples():
        f.write(f"  {r.TIC:<9d} {r.ra:<10.6f} {r.dec:<10.6f} {r.Sector:<6d} {r.Tmag:<6.3f}\n")
print(f"{lista['TIC'].nunique()} estrellas, {len(lista)} pares (TIC, sector); sectores: {sorted(lista['Sector'].unique())}")
lista.groupby("Sector").size().rename("estrellas").to_frame().T"""),
md("## 3) Binarias eclipsantes ya catalogadas"),
code(r"""vsx = xmatch(m, "vizier:B/vsx/vsx", 2.0)
# tipos eclipsantes de VSX: E, EA, EB, EW (con ":" de incierto), en cualquier componente del tipo compuesto
TIPOS_E = {"E", "EA", "EB", "EW", "E:", "EA:", "EB:", "EW:"}
vsx = vsx.loc[vsx["Type"].astype(str).apply(lambda t: any(x in TIPOS_E for x in __import__("re").split(r"[+|/]", t)))]
teb = xmatch(m, "vizier:J/ApJS/258/16/tess-ebs", 2.0)
geb = xmatch(m, "vizier:I/358/veb", 1.0)
cand = m[["source_id", "TIC", "ra", "dec", "Gmag", "Tmag", "BPmag", "RPmag", "PLXcorr", "ruwe", "Rcont"]].copy()
cand["VSX"] = cand["source_id"].map(dict(zip(vsx["source_id"], vsx["Name"].astype(str) + " (" + vsx["Type"].astype(str) + ")")))
cand["P_VSX"] = cand["source_id"].map(dict(zip(vsx["source_id"], vsx["Period"])))
cols_teb = [c for c in teb.columns if c.lower() in ("period", "per", "p")]
cand["TESS_EB"] = cand["source_id"].isin(teb["source_id"])
if cols_teb:
    cand["P_TESS_EB"] = cand["source_id"].map(dict(zip(teb["source_id"], teb[cols_teb[0]])))
cols_geb = [c for c in geb.columns if c.lower() in ("freq", "frequency")]
cand["Gaia_EB"] = cand["source_id"].isin(geb["source_id"])
if cols_geb:
    cand["P_Gaia_EB"] = cand["source_id"].map(dict(zip(geb["source_id"], 1 / geb[cols_geb[0]])))
cand["n_catalogos"] = cand["VSX"].notna().astype(int) + cand["TESS_EB"].astype(int) + cand["Gaia_EB"].astype(int)
conocidas = cand.loc[cand["n_catalogos"] > 0].sort_values(["n_catalogos", "Tmag"], ascending=[False, True])
conocidas.to_csv(AQUI / "binarias_eclipsantes_conocidas.csv", index=False)
# nota: ninguna binaria de Gaia DR3 (287 en 3°) es miembro, y el catálogo de TESS (S1-26) casi no cubre las Pléyades
print(f"miembros en algún catálogo de binarias eclipsantes: {len(conocidas)}  "
      f"(VSX: {cand['VSX'].notna().sum()}, TESS EB: {cand['TESS_EB'].sum()}, Gaia DR3 EB: {cand['Gaia_EB'].sum()})")
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 40)
conocidas.drop(columns=["source_id"]).round(3)"""),
]
nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata = {"kernelspec": {"display_name": "Python (cumulos)", "language": "python", "name": "cumulos"},
               "language_info": {"name": "python"}}
nbf.write(nb, "/home/luciam/Documents/starry/Resultados_Pleyades/membresia/2_candidatas_binarias.ipynb")
