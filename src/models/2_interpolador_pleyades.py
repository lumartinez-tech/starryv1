"""Pasos 2-4 de la metodología de Orión (4_pipeline_completo) aplicados a las EB de las Pléyades con la edad del cúmulo fija.

- Primaria: Teff y L de massage (MIST; clase espectral donde existe, Teff de TIC en 26017840). R1 = Stefan-Boltzmann, como en Orión.
- Profundidad: estimador v2 de 1b_profundidades_eclipse_v2 (se ejecuta su código tal cual) sobre las curvas de TESSExtractor.
- Secundaria: R2 = R1*sqrt(D); M2 y Teff2 con la clase Malla de 4_pipeline_completo (mismo interpolador y errores por diferencias finitas).
- Edad fija 125 +- 8 Myr (Stauffer et al. 1998, límite de litio). Las isocronas MIST de Orión llegan solo a 100 Myr, así que las mallas son
  PARSEC (interpolada entre 100 y 150 Myr), Somers17 y BHAC15.
Salida: interpolador/*.csv y figuras.
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ORION = Path("/home/luciam/Documents/starry/Resultados_Orion")
PLEY = Path("/home/luciam/Documents/starry/Resultados_Pleyades")
AQUI = PLEY / "pipeline"
EDAD, EDAD_ERR = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (125.0, 8.0)   # Myr
SAL = AQUI / ("interpolador" if len(sys.argv) <= 2 else f"interpolador_edad{EDAD:.0f}"); SAL.mkdir(exist_ok=True)
LCDIR = PLEY / "tessextractor_local/tess_extractor_corr/Light_Curve_Files"
LCDIR_346 = PLEY / "tessextractor_local/tess_extractor_ 346541355/Light_Curve_Files"
MODELOS = ORION / "datos_estrellas_massage_orion/modelos"
PARSEC_DIR = Path("/home/luciam/Documents/catalogos_tesis/isocronas-parsec")
MIST_NUEVA = Path("/home/luciam/Documents/catalogos_tesis/iso_mist/MIST_iso_6ac961c58ea03.iso")   # MIST v2.5, log edad 8.07-8.30

T_SUN = 5772.0
PROPAGAR_ERROR_R1 = False            # igual que el default de Orión
P_SISTEMA = {26017840: 1.3854, 35159657: 7.0513, 125754991: 2.4609, 346541355: 8.1634}   # d (priors_pleyades: P/2P resuelto en el paso 1)
TICS = list(P_SISTEMA)
W_MAX = 0.25

# ---------------------------------------------------------------- código de Orión, ejecutado sin cambios
nb1b = json.load(open(ORION / "1b_profundidades_eclipse_v2.ipynb"))["cells"]
nb4 = json.load(open(ORION / "4_pipeline_completo.ipynb"))["cells"]
src = lambda nb, i: "".join(nb[i]["source"])
g = {}
exec(src(nb1b, 4).replace("MOSTRAR_FIGURAS = True", "MOSTRAR_FIGURAS = False"), g)      # imports y constantes de 1b
g["RUTA_CURVAS"] = LCDIR
g["DIR_FIG"] = SAL
exec(src(nb1b, 5), g)                                                                  # medir() y auxiliares
exec(src(nb4, 1), g)                                                                   # imports de 4_
g.update(T_SUN=T_SUN, PROPAGAR_ERROR_R1=PROPAGAR_ERROR_R1)
exec(src(nb4, 8).split("mallas = {")[0], g)                                            # COLS, Malla, cargadores
exec(src(nb4, 10).split("cat = pd.read_csv")[0], g)                                    # radio_sb, lum_sb
exec(src(nb4, 12).split("def calcular_secundarias")[0], g)                             # propiedades_secundaria
Malla, medir, cargar_curva = g["Malla"], g["medir"], None
cargar_bhac15, cargar_somers17 = g["cargar_bhac15"], g["cargar_somers17"]
radio_sb, lum_sb, propiedades_secundaria = g["radio_sb"], g["lum_sb"], g["propiedades_secundaria"]


def cargar_curva_pl(tic, sector):
    d = pd.read_csv((LCDIR_346 if tic == 346541355 else LCDIR) / f"tic {tic}_S_{sector}_LC.txt", sep=r"\s+")
    d = d.dropna(subset=["#Time", "Flux_CBV_SS"]).sort_values("#Time").drop_duplicates("#Time")
    return d["#Time"].to_numpy(float), d["Flux_CBV_SS"].to_numpy(float)


def cargar_parsec(archivos):
    filas = []
    for f in archivos:
        t = pd.read_csv(f, sep=r"\s+", comment="#", header=None)
        t = t[t[9].isin([0, 1])]                  # solo pre-MS y secuencia principal (label PARSEC 0 y 1)
        d = pd.DataFrame({"logAge": t[2], "m_ini": t[3], "mass": t[5], "logL": t[6], "logTeff": t[7]})
        d["logR"] = 0.5 * d["logL"] - 2 * (d["logTeff"] - np.log10(T_SUN))
        filas.append(d)
    return pd.concat(filas, ignore_index=True)


def cargar_mist_iso(path):
    # archivo .iso de la web de MIST (formato de Orión: logAge, m_ini, mass, logTeff, logR, logL); solo pre-MS y MS (phase -1, 0)
    filas, cols = [], None
    for l in open(path):
        if l.startswith("# EEP"):
            cols = l.lstrip("# ").split()
        elif not l.startswith("#") and l.strip():
            filas.append(l.split())
    t = pd.DataFrame(filas, columns=cols).astype(float)
    t = t[t["phase"].isin([-1, 0])]
    return pd.DataFrame({"logAge": t["log10_isochrone_age_yr"], "m_ini": t["initial_mass"], "mass": t["star_mass"],
                         "logTeff": t["log_Teff"], "logR": t["log_R"], "logL": t["log_L"]})


mallas = {
    "MIST": Malla("MIST", cargar_mist_iso(MIST_NUEVA), "isocronas"),
    "PARSEC": Malla("PARSEC", cargar_parsec([PARSEC_DIR / "100Myr.dat", PARSEC_DIR / "150Myr.dat"]), "isocronas"),
    "Somers17": Malla("Somers17", cargar_somers17(MODELOS / "iso_somers17/f017track_ms.csv"), "tracks"),
    "BHAC15": Malla("BHAC15", cargar_bhac15(MODELOS / "iso_barafe/BHAC15_iso.txt"), "isocronas"),
}

# ---------------------------------------------------------------- 1) profundidades v2 por sector
pri = pd.concat([pd.read_csv(AQUI / "priors_pleyades.csv"), pd.read_csv(AQUI / "priors_346541355.csv")])
pri = pri[pri.TIC.isin(TICS) & (pri.Sector < 200)]
dur_med = pri.groupby("TIC")["Duration"].median()
filas = []
for _, p in pri.iterrows():
    tic, s = int(p.TIC), int(p.Sector)
    t, f = cargar_curva_pl(tic, s)
    P = P_SISTEMA[tic]
    dur = p.Duration
    if abs(dur / dur_med[tic] - 1) > 0.4 and dur / P <= W_MAX:
        dur = dur_med[tic]
    t0 = p.t0
    m, F, ph = medir(t, f, P, t0, dur)
    # §2.9 de 1b: el eclipse primario es el más profundo; si el otro lo es por > 3 sigma, t0 pasa a ese eclipse
    d1, d2, e12 = m["D_primario"], m["D_secundario"], np.hypot(m["D_primario_err"], m["D_secundario_err"])
    intercambiado = bool(np.isfinite(d1) and np.isfinite(d2) and d2 - d1 > 3 * e12 and not m["eclipse_ancho"])
    if intercambiado:
        phi2 = m["fase_secundario"] if np.isfinite(m["fase_secundario"]) else 0.5
        t0 = t0 + phi2 * P
        m, F, ph = medir(t, f, P, t0, dur)
    filas.append(dict(TIC=tic, Sector=s, P=P, t0=t0, intercambiado=intercambiado, Duration=dur, texp_sobre_duracion=np.median(np.diff(t)) / dur, **m))
med = pd.DataFrame(filas)
med["Calidad_depth"] = ["ok" if (r.D_primario > 3 * r.D_primario_err and not r.eclipse_ancho) else
                        ("eclipse_ancho" if r.eclipse_ancho else "no_significativo") for r in med.itertuples()]
v1 = pri.set_index(["TIC", "Sector"])["Depth_primary"]
med["Depth_v1"] = [v1[(r.TIC, r.Sector)] for r in med.itertuples()]
med.to_csv(SAL / "profundidades_v2_pleyades.csv", index=False)
print(med[["TIC", "Sector", "P", "Duration", "W", "D_primario", "D_primario_err", "Depth_v1", "D_secundario", "secundario_detectado", "intercambiado", "texp_sobre_duracion", "Calidad_depth"]].round(4).to_string())


def combinar(gr):                       # promedio por inverso de varianza (sección 5 de 4_pipeline_completo)
    gr = gr[(gr.D_primario_err > 0) & gr.D_primario.notna()]
    w = 1 / gr.D_primario_err ** 2
    D = (w * gr.D_primario).sum() / w.sum(); sD = w.sum() ** -0.5
    chi = (w * (gr.D_primario - D) ** 2).sum() / max(len(gr) - 1, 1)
    return pd.Series(dict(Depth=D, Depth_err=sD * np.sqrt(max(chi, 1)), n_sectores=len(gr), chi2_red=chi))


comb = med[med.Calidad_depth == "ok"].groupby("TIC").apply(combinar).reset_index()

# ---------------------------------------------------------------- 2) primaria desde massage
ms = pd.concat([pd.read_csv(AQUI / "massage/pleyades3eb_SpT_mist_results.csv").assign(entrada="SpT"),
                pd.read_csv(AQUI / "massage/pleyades4eb_mist_results.csv").assign(entrada="Teff TIC")])
ms.columns = ms.columns.str.strip(); ms["TIC"] = ms.OBJECT.astype(int)
ms = ms[(ms.entrada == "SpT") | (ms.TIC == 26017840)]
ms = ms[ms.TIC.isin(TICS)].drop_duplicates("TIC").set_index("TIC")
prim = pd.DataFrame({"Teff1": ms.T_median, "Teff1_err": ms.T_std, "logL1": ms.lgL_median, "logL1_err": ms.lgL_std,
                     "M1_massage": ms.mass_median, "M1_massage_err": ms.mass_std, "entrada": ms.entrada})
prim["R1"] = radio_sb(prim.Teff1, prim.logL1)
prim["R1_err"] = prim.R1 * np.hypot(np.log(10) / 2 * prim.logL1_err, 2 * prim.Teff1_err / prim.Teff1)
for nom, mall in mallas.items():                 # masa de la primaria a la edad del cúmulo, por radio
    prim[f"M1_{nom}"] = [mall.en_radio(r, EDAD)["M"] for r in prim.R1]
prim.to_csv(SAL / "primarias_pleyades.csv")
print("\nPRIMARIAS\n", prim.round(3).to_string())

# ---------------------------------------------------------------- 3) secundaria
def fila_sec(tic, sector, D, sD, nombre_malla):
    a = prim.loc[tic]
    R2 = a.R1 * np.sqrt(D)
    rel2 = (0.5 * sD / D) ** 2 + (((a.R1_err / a.R1) ** 2) if PROPAGAR_ERROR_R1 else 0)
    R2_err = R2 * np.sqrt(rel2)
    props = propiedades_secundaria(mallas[nombre_malla], R2, R2_err, EDAD, EDAD_ERR)
    return dict(TIC=tic, Sector=sector, malla=nombre_malla, Depth=D, Depth_err=sD, R1=a.R1, R1_err=a.R1_err,
                R2=R2, R2_err=R2_err, k=np.sqrt(D), **props)


sec_s = [fila_sec(r.TIC, r.Sector, r.D_primario, r.D_primario_err, m) for r in med[med.Calidad_depth == "ok"].itertuples() for m in mallas]
sec_c = [fila_sec(r.TIC, "combinado", r.Depth, r.Depth_err, m) for r in comb.itertuples() for m in mallas]
sec_s, sec_c = pd.DataFrame(sec_s), pd.DataFrame(sec_c)
sec_c = sec_c.merge(comb[["TIC", "n_sectores", "chi2_red"]], on="TIC")
H, C, KB, LAM = 6.62607e-34, 2.99792e8, 1.380649e-23, 786.5e-9
planck = lambda T: 1.0 / np.expm1(H * C / (LAM * KB * T))
# diagnóstico (no entra al modelo): razón de brillo superficial observada J = D_sec / D_prim frente a la que da Teff2 de la isocrona
dd = med.groupby("TIC")[["D_primario", "D_secundario"]].median()
sec_c["J_obs"] = sec_c.TIC.map(dd.D_secundario / dd.D_primario)
sec_c["J_isocrona"] = [planck(r.Teff2) / planck(prim.loc[r.TIC, "Teff1"]) if np.isfinite(r.Teff2) else np.nan for r in sec_c.itertuples()]
sec_s.to_csv(SAL / "secundarias_por_sector_pleyades.csv", index=False)
sec_c.to_csv(SAL / "secundarias_combinadas_pleyades.csv", index=False)
print(f"\nSECUNDARIAS COMBINADAS (edad {EDAD:.0f} Myr)\n", sec_c.drop(columns=["Sector"]).round(4).to_string())

# ---------------------------------------------------------------- figura: radio-masa a 125 Myr
col = {"MIST": "tab:blue", "PARSEC": "tab:green", "Somers17": "tab:orange", "BHAC15": "tab:purple"}
fig, ax = plt.subplots(1, len(TICS), figsize=(5 * len(TICS), 4.3))
for a_, tic in zip(ax, TICS):
    for nom, mall in mallas.items():
        iso = mall.isocrona(EDAD)
        if iso is not None:
            a_.plot(iso["mass"], 10 ** iso["logR"], color=col[nom], lw=1.4, label=f"{nom} a {EDAD:.0f} Myr")
    r1 = prim.loc[tic]
    a_.errorbar(prim.loc[tic, "M1_massage"], r1.R1, xerr=prim.loc[tic, "M1_massage_err"], yerr=r1.R1_err, fmt="*", ms=13, color="k", label="primaria (massage)")
    for _, r in sec_c[sec_c.TIC == tic].iterrows():
        a_.axhspan(r.R2 - r.R2_err, r.R2 + r.R2_err, color=col[r.malla], alpha=0.12)
        if np.isfinite(r.M2):
            a_.errorbar(r.M2, r.R2, xerr=r.M2_err, yerr=r.R2_err, fmt="o", color=col[r.malla], mec="k")
    a_.set(xscale="log", yscale="log", xlabel="masa [Msun]", ylabel="radio [Rsun]", title=f"TIC {tic}")
    a_.set_xlim(0.08, max(3.5, prim.loc[tic, "M1_massage"] * 1.4))
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(SAL / "radio_masa_pleyades.png", dpi=140)
