"""Notebooks de starry para los sistemas 3, 7, 11 y 20 con la edad de su vecindad y los parámetros de la
secundaria recalculados a esa edad (interpolador de isócronas, BHAC15).

Usa la plantilla y los priors de ../starry_por_tic/genera_starry.py (sin ejecutar su bucle principal) y
cambia solo: la edad del sistema, la masa de la secundaria (prior m_B) y su temperatura. Además acota las
amplitudes a valores positivos (corrección de starry_correcciones/). No ejecuta los notebooks.
Uso: python genera_starry_edad_vecindad.py
"""
import sys
import warnings
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RUTA = AQUI.parent
GEN = RUTA / "starry_por_tic" / "genera_starry.py"
fuente_gen = GEN.read_text()
exec(compile(fuente_gen[:fuente_gen.index("filas_indice = []")], str(GEN), "exec"))

sys.path.insert(0, str(RUTA / "datos_estrellas_massage_orion"))
from isocronas_interp import load_isochrone_model, IsochroneInterpolator, invert_radius_to_mass

BHAC15 = IsochroneInterpolator(load_isochrone_model(
    "BHAC15", RUTA / "datos_estrellas_massage_orion" / "modelos" / "iso_barafe" / "BHAC15_iso.txt"), "BHAC15")

edades = pd.read_csv(RUTA / "edades_vecindad" / "edades_asignadas.csv").set_index("TIC")
R1 = pd.read_csv(RUTA / "datos_estrellas_massage_orion" / "teff2_desde_profundidad_transito.csv") \
    .groupby("TIC")["R1_MIST"].first()

# edad (Myr) y su rango (p16, p84) por sistema; el 3 no tiene grupo: se usa su propia edad (SpT de la
# literatura, MassAge PARSEC pms, con el rango de su Monte Carlo)
EDADES = {
    11412481: dict(sistema="sistema 3", edad=15.93, lo=14.68, hi=17.10,
                   fuente="edad propia de la primaria (SpT de la literatura, MassAge PARSEC pms); sin grupo co-móvil"),
    264483228: dict(sistema="sistema 7"),
    138829403: dict(sistema="sistema 11"),
    264459857: dict(sistema="sistema 20"),
}
for tic, d in EDADES.items():
    if "edad" not in d:
        e = edades.loc[tic]
        d.update(edad=float(e["edad_grupo_Myr"]), lo=float(e["p16_Myr"]), hi=float(e["p84_Myr"]),
                 fuente=f"edad de la vecindad ({e['metodo']})")

GENERALES = [
    ('A_amp = pm.Normal("A_amp", mu=amp_A, sd=sd_amp_A, testval=amp_A)',
     'A_amp = PositiveNormal("A_amp", mu=amp_A, sd=sd_amp_A, testval=amp_A)'),
    ('B_amp = pm.Normal("B_amp", mu=amp_B, sd=sd_amp_B, testval=amp_B)',
     'B_amp = PositiveNormal("B_amp", mu=amp_B, sd=sd_amp_B, testval=amp_B)'),
]


def secundaria(R2, edad):
    # masa y Teff de la secundaria con BHAC15 a una edad dada (NaN fuera de la malla)
    lg = np.log10(edad * 1e6)
    m = invert_radius_to_mass(lg, R2, BHAC15)
    return (m, float(BHAC15.teff(lg, m))) if np.isfinite(m) else (np.nan, np.nan)


filas = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for tic, d in EDADES.items():
        for _, p in priors.loc[priors["TIC"] == tic].sort_values("Sector").iterrows():
            sector = int(p["Sector"])
            v, avisos, fuente = priors_de(tic, sector)
            v["prot"] = p["P_rot"]
            R2 = R1.loc[tic] * np.sqrt(p["Depth_primary"])
            m, T = secundaria(R2, d["edad"])
            m_lo, _ = secundaria(R2, d["hi"])    # más vieja -> menos masa para el mismo radio
            m_hi, _ = secundaria(R2, d["lo"])
            masas = [x for x in (m_lo, m, m_hi) if np.isfinite(x)]
            sd_m = max((max(masas) - min(masas)) / 2, 0.1 * m) if np.isfinite(m) else np.nan
            antes = (v["m_B"], v["sd_m_B"], v["T2"], v["edad"])
            if np.isfinite(m):
                v["m_B"], v["sd_m_B"], v["T2"] = m, sd_m, T
            v["edad"] = d["edad"]
            fuente["m_B"] = (f"BHAC15 a la edad de la vecindad ({d['edad']:.1f} Myr; rango {d['lo']:.1f}–{d['hi']:.1f} Myr) "
                             f"con R_2 = R_1 √D_1 = {R2:.3f} R☉ (R_1 = {R1.loc[tic]:.3f}, D_1 v2 = {p['Depth_primary']:.4f}); "
                             f"sd = mitad del rango de masas en ese rango de edades (mínimo 10 %). "
                             f"Antes: m_B = {antes[0]:.3f} ± {antes[1]:.3f} M☉ con la edad individual de {antes[3]:.3g} Myr")
            avisos = [a for a in avisos if not a.startswith("M_B no sale de BHAC15")]
            avisos.insert(0, f"**Edad de la vecindad:** {d['fuente']}: {d['edad']:.1f} Myr (antes {antes[3]:.3g} Myr). "
                             f"Secundaria con BHAC15: M_B = {v['m_B']:.3f} ± {v['sd_m_B']:.3f} M☉ (antes "
                             f"{antes[0]:.3f} ± {antes[1]:.3f}), Teff_B = {v['T2']:.0f} K.")
            if tic == 138829403:
                avisos.insert(1, "R_1 del catálogo (1.63 R☉) es pequeño para una B3.5 y la curva está diluida: R_2 y M_B "
                                 "probablemente están subestimados (ver edades_vecindad/secundarias_con_edad_de_grupo.ipynb).")
            file_lc = RUTA_ORION / "Light_Curve_Files" / f"tic {tic}_S_{sector}_LC.txt"
            t = pd.read_csv(file_lc, sep=r"\s+", usecols=["#Time"], nrows=50)["#Time"].to_numpy()
            cadencia = np.median(np.diff(t)) * 1440
            out_dir = AQUI / f"TIC_{tic}" / f"S{sector}"
            nb = crear(tic, sector, v, avisos, fuente, file_lc, n_puntos(file_lc), cadencia, out_dir)
            hechos = 0
            for c in nb.cells:
                for a, b in GENERALES:
                    if a in c.source:
                        c.source = c.source.replace(a, b); hechos += 1
            assert hechos == len(GENERALES)
            nb.cells.insert(1, nbf.v4.new_markdown_cell(
                f"## Corrida con la edad de la vecindad ({d['sistema']})\n\n"
                f"- **Edad:** {d['edad']:.1f} Myr ({d['fuente']}); antes {antes[3]:.3g} Myr.\n"
                f"- **Secundaria** (prior de `m_B`): {v['m_B']:.3f} ± {v['sd_m_B']:.3f} M☉ con BHAC15 "
                f"(antes {antes[0]:.3f} ± {antes[1]:.3f} M☉); $T_{{\\rm eff,B}}$ = {v['T2']:.0f} K.\n"
                f"- El resto de los priors y del modelo es el del generador original; además las amplitudes `A_amp` y "
                f"`B_amp` están acotadas a valores positivos.\n"
                f"- Salidas en `starry_edad_vecindad/TIC_{tic}/S{sector}/`; las corridas anteriores no se tocan."))
            ruta = AQUI / f"TIC_{tic}" / f"full_{tic}_s{sector}_edadvec.ipynb"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            nbf.write(nb, ruta)
            filas.append(dict(TIC=tic, sistema=d["sistema"], Sector=sector, edad_Myr=round(d["edad"], 2),
                              edad_antes_Myr=antes[3], R2_Rsun=round(R2, 3),
                              m_B=round(v["m_B"], 3), sd_m_B=round(v["sd_m_B"], 3), Teff_B=round(v["T2"]),
                              m_B_antes=round(antes[0], 3), sd_m_B_antes=round(antes[1], 3),
                              notebook=str(ruta.relative_to(AQUI))))
ind = pd.DataFrame(filas)
ind.to_csv(AQUI / "indice_edad_vecindad.csv", index=False)
print(ind.drop(columns="notebook").to_string(index=False))
