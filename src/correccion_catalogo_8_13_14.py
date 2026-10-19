"""Corrige el intercambio MIST/Somers17 de catalogo_completo_mist_sommers.csv en los sistemas 8, 13 y 14 (las tres
primarias con MassAge por T_eff) y mide cuánto cambian los priors de starry.

Evidencia del intercambio: las columnas _1 ("MIST") de esos tres TIC coinciden en T, L, masa y edad con
tess_somers_teff_massage1 y yceb_Teff_somers17_results.csv, y las _2 con tess_mist_teff_massage1 y
yceb_Teff_mist_results.csv (dos series de archivos independientes).

Qué hace:
1. Guarda una copia del catálogo original y corrige el intercambio (todas las columnas _1 <-> _2 de esos TIC).
2. Recalcula, como en 2_teff_desde_isocronas.ipynb / 3_diagramas_hr_isocronas.ipynb / genera_starry.py, la rama
   que usa el pipeline (MIST primero): R_A por Stefan-Boltzmann, M_A = mass_median, edad, R_B = R_A sqrt(D) por sector
   y M_B invirtiendo R_B en BHAC15 a esa edad.
3. Compara con los priors de las corridas vigentes (tabla_maestra.csv), en unidades de la sd del prior.
No toca resumen_por_tic_sector.csv ni las corridas.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion")
MA = RAIZ / "datos_estrellas_massage_orion"
sys.path.insert(0, str(MA))
from isocronas_interp import IsochroneInterpolator, invert_radius_to_mass, load_isochrone_model  # noqa: E402

CAT = RAIZ / "catalogo_completo_mist_sommers.csv"
COPIA = RAIZ / "catalogo_completo_mist_sommers_antes_correccion_20261013.csv"
TICS = {11300044: "8", 249032831: "13", 199780530: "14"}
T_SUN = 5772.0


def corregir():
    if not COPIA.exists():
        shutil.copy2(CAT, COPIA)
    c = pd.read_csv(COPIA)
    base = sorted({k[:-2] for k in c.columns if k.endswith("_1") and k[:-2] + "_2" in c.columns})
    m = c.TIC.isin(TICS)
    for b in base:
        c.loc[m, [f"{b}_1", f"{b}_2"]] = c.loc[m, [f"{b}_2", f"{b}_1"]].to_numpy()
    # verificación contra los archivos por modelo
    for f, suf in (("tess_mist_teff_massage1", "1"), ("tess_somers_teff_massage1", "2")):
        ref = pd.read_csv(MA / f).set_index("TIC")
        for t in TICS:
            for q in ("T_mean", "mass_mean", "age_mean"):
                assert np.isclose(c.loc[c.TIC == t, f"{q}_{suf}"].iloc[0], ref.loc[t, q], rtol=1e-3), (t, q, f)
    c.to_csv(CAT, index=False)
    return c, pd.read_csv(COPIA)


def priors_rama(fila, suf, D, bhac):
    R_A = 10 ** (fila[f"lgL_median_{suf}"] / 2) * (T_SUN / fila[f"T_median_{suf}"]) ** 2
    edad = fila[f"age_median_{suf}"]
    R_B = R_A * np.sqrt(D)
    try:
        M_B = invert_radius_to_mass(np.log10(edad * 1e6), R_B, bhac)
    except Exception:
        M_B = np.nan
    return dict(edad=edad, M_A=fila[f"mass_median_{suf}"], sd_M_A=max(fila[f"mass_std_{suf}"], 0.1 * fila[f"mass_median_{suf}"]),
                R_A=R_A, R_B=R_B, M_B=M_B)


if __name__ == "__main__":
    nuevo, viejo = corregir()
    bhac = IsochroneInterpolator(load_isochrone_model("BHAC15", MA / "modelos/iso_barafe/BHAC15_iso.txt"), "BHAC15")
    pri = pd.read_csv(RAIZ / "priors_v2.csv").drop_duplicates(["TIC", "Sector"]).set_index(["TIC", "Sector"])
    T = pd.read_csv(RAIZ / "diagnostico_resultados/tabla_maestra.csv")
    filas = []
    for _, r in T[T.TIC.isin(TICS)].iterrows():
        D = pri.loc[(r.TIC, r.Sector), "Depth_primary"]
        a = priors_rama(viejo[viejo.TIC == r.TIC].iloc[0], "1", D, bhac)
        b = priors_rama(nuevo[nuevo.TIC == r.TIC].iloc[0], "1", D, bhac)
        filas.append(dict(sistema=TICS[r.TIC], TIC=r.TIC, Sector=r.Sector,
                          edad_antes=a["edad"], edad_nueva=b["edad"],
                          M_A_corrida=r.prior_A_m, M_A_nueva=b["M_A"], dM_A_en_sd=(b["M_A"] - r.prior_A_m) / r.prior_A_m_sd,
                          R_A_antes=a["R_A"], R_A_nueva=b["R_A"],
                          M_B_corrida=r.prior_B_m, M_B_recalc_antes=a["M_B"], M_B_nueva=b["M_B"],
                          dM_B_en_sd=(b["M_B"] - r.prior_B_m) / r.prior_B_m_sd,
                          M_A_post=r.A_m, M_B_post=r.B_m, q_post=r.B_m / r.A_m,
                          q_nuevo_esperado=b["M_B"] / b["M_A"] if np.isfinite(b["M_B"]) else np.nan))
    R = pd.DataFrame(filas)
    R.round(4).to_csv(RAIZ / "diagnostico_resultados/impacto_correccion_8_13_14.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
