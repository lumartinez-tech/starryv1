"""Construye edades_por_vecindad.ipynb (kernel: Python (cumulos))."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
celdas = [
md(r"""# Edad de la primaria a partir de las estrellas vecinas

Para los sistemas 3, 7, 11 y 20, la edad individual de la primaria (MIST) sale de 200 a 370 Myr, muy por
encima de la del complejo de Orión. Son primarias de 1.4–2.8 M☉ que ya están en (o cerca de) la secuencia
principal, donde la edad casi no cambia la posición en el diagrama HR. En lugar de la edad individual se
usa la **edad del grupo co-móvil** al que pertenece cada sistema, medida con sus miembros de baja masa,
que todavía están en la pre-secuencia principal y tienen edades bien acotadas.

Se siguen los pasos del *Taller: caracterización global de las estrellas en la región de ε Ori*
(`~/Downloads/Taller_cluster2.pdf`), con el campo centrado en cada sistema en lugar de en ε Ori:

1. **Catálogo:** Gaia DR3 en un radio de 3°, con $\sigma_\varpi/\varpi<0.2$ y $\varpi\ge1$ mas; `fidelity_v2`
   (Rybizki et al. 2022) y paralaje corregida (`parallax_corr`) del servicio TAP de ARI (scripts del taller
   `download_gaiadr3_fidelity_plxcorr.py`).
2. **Cortes:** calidad `(ruwe<1.4 || fidelity_v2>0.2)`; pre-secuencia principal
   $M_{RP}<2.1\,(BP-RP)+2.6$; cinemático $2<\varpi_{\rm corr}<3.6$ mas y $|\mu|<5.425$ mas/año.
3. **HDBSCAN** en $(\alpha,\delta,\varpi_{\rm corr},\mu_\alpha,\mu_\delta)$ con `RobustScaler`,
   `min_samples = 1`, `cluster_selection_method = "leaf"`. Primero se barre `min_cluster_size` de 1000 a 10
   (`hdbscan_test_minsample.py`) y luego se corre con el valor elegido (`clusters_finder.py`); se conservan
   los miembros con probabilidad > 0.37 (el corte del taller).
4. **Diagrama HR:** cruce con el TESS Input Catalog v8.2 (1″) para $T_{\rm eff}$, $L$ y 2MASS.
5. **Discos** (W3, W4 de ALLWISE) y 6. **rayos X** (eRASS1, Freund et al. 2024): indicadores
   independientes de juventud del grupo.
7. **MassAge** (PARSEC, fase `pms`) para masa y edad de cada miembro. La edad del grupo es la mediana de las
   edades de sus miembros; su incertidumbre, la del bootstrap de la mediana.

Diferencias con el taller (todas para hacerlo reproducible sin TOPCAT):
- los cortes y el cruce con el TIC se hacen en Python (`astroquery.xmatch`, CDS) con los mismos criterios;
- el error de las magnitudes de Gaia se calcula como $1.0857/(\text{flux\_over\_error})$ (el script del
  taller multiplica en lugar de dividir; en el taller los errores finales venían del TIC);
- el `min_cluster_size` se elige en el plateau más largo de la curva (número de cúmulos frente a
  `min_cluster_size`), como se hace a ojo en el taller, y se puede fijar a mano en `MCS_MANUAL`."""),
code(r"""import os
import json
import time
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.utils.tap.core import TapPlus
from astroquery.xmatch import XMatch

warnings.simplefilter("ignore")
RUTA_ORION = Path("/home/luciam/Documents/starry/Resultados_Orion")
AQUI = RUTA_ORION / "edades_vecindad"
CATALOGO = pd.read_csv(RUTA_ORION / "catalogo_completo_mist_sommers.csv").set_index("TIC")   # resultados MassAge previos (MIST)
# datos de la literatura de cada sistema, antes de MassAge: tipo espectral (escala PM13: 20 = B0, 30 = A0,
# 40 = F0, 50 = G0, 60 = K0, 70 = M0), astrometría y fotometría Gaia DR3, 2MASS y ALLWISE
MACRO = pd.read_csv(RUTA_ORION / "datos_estrellas_massage_orion" / "Macro_21aug2026.csv").set_index("TIC")

OBJETIVOS = {11412481: "sistema 3", 264483228: "sistema 7", 138829403: "sistema 11", 264459857: "sistema 20"}
RADIO_DEG = 3.0
P_CORTE = 0.37                       # probabilidad mínima de pertenencia (taller)
MCS_MANUAL = {}                      # {TIC: min_cluster_size} para fijarlo a mano tras ver la curva
MCS_RANGO = (30, 400)                # rango donde se busca el plateau

# MassAge (instalación del taller)
MASSAGE_DIR = Path("/home/luciam/Documents/massage_workshop/massage.py")
MASSAGE_MODELOS = "/home/luciam/Documents/massage_workshop/MODELS/MASSAGE_models"
MASSAGE_PY = "/home/luciam/anaconda3/envs/massage_env/bin/python"
MASSAGE_NPROC = 3                    # núcleos para MassAge (hay corridas de starry en paralelo)
# iteraciones Monte Carlo de MassAge en los grupos: 50 (el taller usa 100) para acelerar; con 250-900
# miembros por grupo la mediana de la edad queda igual de bien determinada
MASSAGE_NMC = {}   # el grupo del sistema 3 se corrió primero con 100 y falló por una estrella sin e_Jmag
NMC_DEFECTO = 50
REHACER = False                      # True: volver a bajar y recalcular todo"""),
md("### Funciones de MassAge (se usan en las secciones 4 y 7)"),
code(r"""CONFIG = '''parameter|option|comments
modelo|PARSEC|Teoretical Models
phase_sel|{fase}|evolutionary phase:pms,ms,all
inter_mode|cubic|interpolation option
min_mass|0.01|minimal mass in Msun
max_mass|10|maximal mass in Msun
min_age|0.01|minimal age in Myr
max_age|1000|maximal age in Myr
num_nmc1|{nmc}|montecarlo iteration
extlaw|c89|extinction law
Rv|3.1|extinction selection
anch1|J|anchor filter
Xopt1|{xopt}|temperature values
Yopt1|PARSEC|Bolometric correction and intrinsic color option
massagedir|{dir}|Models folder
output|{out}|Base name for outputfiles
mcoutput|N|monte carlo files
nprocess|{np}|Number of processes
'''


def entrada_primaria(tic):
    # entrada de MassAge para la primaria con los datos de la literatura (Macro_21aug2026.csv)
    m = MACRO.loc[[tic]]
    return pd.DataFrame({
        "Source": [str(tic)], "RA": m["RA"].values, "DEC": m["DEC"].values,
        "Gmag": m["Gmag"].values, "BPmag": m["BPmag"].values, "RPmag": m["RPmag"].values,
        "e_Gmag": m["e_Gmag"].values, "e_BPmag": m["e_BPmag"].values, "e_RPmag": m["e_RPmag"].values,
        "Jmag": m["Jmag"].values, "e_Jmag": m["e_Jmag"].values, "Hmag": m["Hmag"].values, "e_Hmag": m["e_Hmag"].values,
        "SpT": m["SpT"].values, "SpTerr": m["SpTerr"].values, "PLXcorr": m["PLXcorr"].values, "e_Plx": m["e_Plx"].values})


def entrada_massage(df):
    e = pd.DataFrame({
        "Source": df["source_id"].astype(str), "RA": df["ra"], "DEC": df["dec"],
        "Gmag": df["Gmag"], "BPmag": df["BPmag"], "RPmag": df["RPmag"],
        "e_Gmag": df["e_Gmag"], "e_BPmag": df["e_BPmag"], "e_RPmag": df["e_RPmag"],
        "Jmag": df["Jmag"], "e_Jmag": df["e_Jmag"], "Hmag": df["Hmag"], "e_Hmag": df["e_Hmag"],
        "Teff": df["Teff"], "eTeff": df["e_Teff"].fillna(0.05 * df["Teff"]),
        "PLXcorr": df["PLXcorr"], "e_Plx": df["parallax_error"]})
    # MassAge falla (variable Mobs1 sin definir) si el error de la banda ancla es NaN: se exigen todos los
    # valores y errores. Se pierden 1-2 estrellas por grupo.
    req = ["Teff", "eTeff", "Gmag", "BPmag", "RPmag", "e_Gmag", "e_BPmag", "e_RPmag", "Jmag", "e_Jmag", "Hmag",
           "e_Hmag", "PLXcorr", "e_Plx"]
    return e.dropna(subset=req).query("eTeff > 0")


def correr_massage(entrada, nombre, carpeta, xopt="Teff", fase="pms", nmc=100):
    out = carpeta / nombre
    res = Path(str(out) + "_results.csv")
    if res.exists() and not REHACER:
        r = pd.read_csv(res, skipinitialspace=True)
        r.columns = [c.strip() for c in r.columns]
        return r
    ent = carpeta / f"{nombre}_in.csv"
    entrada.to_csv(ent, index=False)
    cfg = carpeta / f"{nombre}_config.csv"
    cfg.write_text(CONFIG.format(dir=MASSAGE_MODELOS, out=out, np=MASSAGE_NPROC, xopt=xopt, fase=fase, nmc=nmc))
    r = subprocess.run([MASSAGE_PY, "massage_running_PLX_par.py", str(ent), str(cfg)], cwd=MASSAGE_DIR,
                       capture_output=True, text=True)
    (carpeta / f"{nombre}_log.txt").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    if not res.exists():
        raise RuntimeError(f"MassAge falló para {nombre}; ver {nombre}_log.txt")
    r = pd.read_csv(res, skipinitialspace=True)
    r.columns = [c.strip() for c in r.columns]
    return r"""),

md("## 1) Catálogo Gaia DR3 + fidelity + paralaje corregida (por campo)"),
code(r"""ARI = TapPlus(url="https://gaia.ari.uni-heidelberg.de/tap")


def consulta(q, servicios=("esa", "ari"), intentos=4):
    # consulta TAP con reintentos; si el archivo de la ESA falla, se usa el espejo de ARI (mismas tablas gaiadr3)
    ultimo = None
    for servicio in servicios:
        for k in range(intentos):
            try:
                tap = Gaia if servicio == "esa" else ARI
                return tap.launch_job_async(q).get_results().to_pandas()
            except Exception as e:
                ultimo = e
                print(f"  {servicio}: intento {k + 1} falló ({type(e).__name__}); reintento en {30 * (k + 1)} s")
                time.sleep(30 * (k + 1))
    raise RuntimeError(f"consulta fallida en {servicios}: {ultimo}")


def descargar_campo(tic):
    out = AQUI / f"TIC_{tic}"
    out.mkdir(exist_ok=True)
    f1, f2 = out / "gaia_tmp.csv", out / "sample1.csv"
    if f2.exists() and not REHACER:
        return pd.read_csv(f2)
    ra0, dec0 = MACRO.loc[tic, "RA"], MACRO.loc[tic, "DEC"]
    if f1.exists() and not REHACER:
        g = pd.read_csv(f1)
    else:
        q = f'''
        SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
               phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error, phot_rp_mean_flux_over_error, ruwe
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra0}, {dec0}, {RADIO_DEG}))
        AND parallax_error / parallax < 0.2 AND parallax >= 1
        '''
        g = consulta(q)
        g.columns = [c.lower() for c in g.columns]
        g.to_csv(f1, index=False)
    for b, n in (("g", "Gmag"), ("bp", "BPmag"), ("rp", "RPmag")):
        g[f"e_{n}"] = 1.0857 / g[f"phot_{b}_mean_flux_over_error"]
    fid, cor = [], []
    ids = g["source_id"].astype(int).tolist()
    for i in range(0, len(ids), 10000):
        s = ",".join(map(str, ids[i:i + 10000]))
        fid.append(consulta(f"SELECT source_id, fidelity_v2 FROM gaiaedr3_complements.spurious_astrometry "
                            f"WHERE source_id IN ({s})", servicios=("ari",)))
        cor.append(consulta(f"SELECT source_id, parallax_corr FROM gaiadr3.gaia_source_corrections "
                            f"WHERE source_id IN ({s})", servicios=("ari",)))
        time.sleep(2)
    d = g.merge(pd.concat(fid), on="source_id", how="left").merge(pd.concat(cor), on="source_id", how="left")
    d = d.rename(columns={"phot_g_mean_mag": "Gmag", "phot_bp_mean_mag": "BPmag", "phot_rp_mean_mag": "RPmag",
                          "parallax_corr": "PLXcorr"})
    d.to_csv(f2, index=False)
    return d


campos = {}
for tic in OBJETIVOS:
    campos[tic] = descargar_campo(tic)
    print(f"TIC {tic} ({OBJETIVOS[tic]}): {len(campos[tic])} fuentes en {RADIO_DEG}°")"""),
md(r"""## 2) Cortes de calidad, pre-secuencia principal y cinemático

Además se identifica la fuente Gaia del propio sistema (la más cercana en 2″) para saber qué cortes pasa.
Muchas binarias eclipsantes tienen RUWE alto y las primarias de tipo A no pasan el corte de PMS; por eso la
pertenencia del sistema a un grupo se decide por proximidad en el espacio de cinco parámetros (sección 3),
no por estar en la muestra."""),
code(r"""def cortes(d):
    calidad = (d["ruwe"] < 1.4) | (d["fidelity_v2"] > 0.2)
    m_rp = d["RPmag"] - 5 * np.log10(1000 / d["PLXcorr"]) + 5
    pms = m_rp < 2.1 * (d["BPmag"] - d["RPmag"]) + 2.6
    cin = (d["PLXcorr"] > 2) & (d["PLXcorr"] < 3.6) & (np.hypot(d["pmra"], d["pmdec"]) < 5.425)
    return calidad, pms, cin


def fuente_objetivo(tic, d):
    # la fuente Gaia DR3 del sistema es la de Macro_21aug2026.csv (columna Source)
    k = d.index[d["source_id"].astype("int64") == int(MACRO.loc[tic, "Source"])]
    return k[0] if len(k) else None


def fila_macro(tic):
    # el sistema con los valores de la literatura, en el formato de las tablas de Gaia
    m = MACRO.loc[tic]
    return pd.Series(dict(source_id=int(m["Source"]), ra=m["RA"], dec=m["DEC"], PLXcorr=m["PLXcorr"],
                          parallax_error=m["e_Plx"], pmra=m["pmRA"], pmdec=m["pmDE"], ruwe=m["RUWE"],
                          Gmag=m["Gmag"], BPmag=m["BPmag"], RPmag=m["RPmag"], e_Gmag=m["e_Gmag"],
                          e_BPmag=m["e_BPmag"], e_RPmag=m["e_RPmag"]))


muestras, objetivo_fila = {}, {}
filas = []
for tic, d in campos.items():
    cal, pms, cin = cortes(d)
    sel = cal & pms & cin
    muestras[tic] = d.loc[sel].reset_index(drop=True)
    k = fuente_objetivo(tic, d)
    # astrometría del sistema: la de Macro (literatura) en todos los casos
    objetivo_fila[tic] = fila_macro(tic)
    filas.append(dict(TIC=tic, sistema=OBJETIVOS[tic], fuentes=len(d), calidad=int(cal.sum()), pms=int((cal & pms).sum()),
                      cinematico=int(sel.sum()),
                      objetivo_en_gaia=k is not None,
                      objetivo_pasa=("" if k is None else
                                     ",".join(n for n, m in (("calidad", cal), ("pms", pms), ("cinemático", cin)) if m[k])),
                      ruwe_objetivo=d.loc[k, "ruwe"] if k is not None else np.nan,
                      plx_objetivo=d.loc[k, "PLXcorr"] if k is not None else np.nan))
pd.DataFrame(filas)"""),
code(r"""fig, axs = plt.subplots(len(OBJETIVOS), 3, figsize=(16, 4.3 * len(OBJETIVOS)), squeeze=False)
for fila, (tic, d) in zip(axs, campos.items()):
    cal, pms, cin = cortes(d)
    s = muestras[tic]
    o = objetivo_fila[tic]
    fila[0].scatter(d.loc[cal & pms, "PLXcorr"], np.hypot(d.loc[cal & pms, "pmra"], d.loc[cal & pms, "pmdec"]),
                    s=2, c="#c0392b", label="calidad + PMS")
    fila[0].scatter(s["PLXcorr"], np.hypot(s["pmra"], s["pmdec"]), s=2, c="#2a78d6", label="+ cinemático")
    fila[0].axvline(2, c="k"); fila[0].axvline(3.6, c="k"); fila[0].axhline(5.425, c="k")
    fila[0].set_xscale("log"); fila[0].set_yscale("log"); fila[0].set_xlim(1, 10)
    fila[0].set_xlabel("PLXcorr [mas]"); fila[0].set_ylabel("|μ| [mas/año]")
    fila[1].scatter(d["ra"], d["dec"], s=0.5, c="0.8"); fila[1].scatter(s["ra"], s["dec"], s=2, c="k")
    fila[2].scatter(d["pmra"], d["pmdec"], s=0.5, c="0.8"); fila[2].scatter(s["pmra"], s["pmdec"], s=2, c="k")
    fila[2].set_xlim(-6, 6); fila[2].set_ylim(-6, 6)
    if o is not None:
        fila[0].plot(o["PLXcorr"], np.hypot(o["pmra"], o["pmdec"]), "*", c="gold", mec="k", ms=16)
        fila[1].plot(o["ra"], o["dec"], "*", c="gold", mec="k", ms=16)
        fila[2].plot(o["pmra"], o["pmdec"], "*", c="gold", mec="k", ms=16)
    fila[1].invert_xaxis(); fila[1].set_xlabel("ra"); fila[1].set_ylabel("dec")
    fila[2].set_xlabel("pmra"); fila[2].set_ylabel("pmdec")
    fila[0].set_title(f"TIC {tic} ({OBJETIVOS[tic]}): {len(s)} estrellas tras los cortes", loc="left")
    fila[0].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(AQUI / "cortes.png", dpi=110); plt.show()"""),
md(r"""## 3) HDBSCAN

**Barrido de `min_cluster_size`** (de 1000 a 10, en pasos de 5) y elección en el plateau más largo dentro de
`MCS_RANGO`: un plateau es un intervalo de `min_cluster_size` donde el número de cúmulos no cambia, es decir,
una solución estable."""),
code(r"""def escalar(df, scalers=None):
    cols = ["ra", "dec", "PLXcorr", "pmra", "pmdec"]
    if scalers is None:
        scalers = {c: RobustScaler().fit(df[[c]]) for c in cols}
    X = np.column_stack([scalers[c].transform(df[[c]]) for c in cols])
    return X, scalers


def barrido(X):
    tamanos = np.arange(1000, 10, -5)
    n = []
    for mcs in tamanos:
        lab = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=1, cluster_selection_method="leaf").fit(X).labels_
        n.append(len(set(lab)) - (1 if -1 in lab else 0))
    return tamanos, np.array(n)


def elegir_mcs(tamanos, n):
    sel = (tamanos >= MCS_RANGO[0]) & (tamanos <= MCS_RANGO[1]) & (n >= 2)
    t, k = tamanos[sel][::-1], n[sel][::-1]   # de menor a mayor tamaño
    mejor, ini = (0, None), 0
    for i in range(1, len(t) + 1):
        if i == len(t) or k[i] != k[ini]:
            largo = t[i - 1] - t[ini]
            if largo > mejor[0]:
                mejor = (largo, t[ini])
            ini = i
    return int(mejor[1]) if mejor[1] is not None else 100


barridos, mcs_elegido, X_esc, escaladores = {}, {}, {}, {}
for tic, s in muestras.items():
    X_esc[tic], escaladores[tic] = escalar(s)
    barridos[tic] = barrido(X_esc[tic])
    mcs_elegido[tic] = MCS_MANUAL.get(tic, elegir_mcs(*barridos[tic]))

fig, axs = plt.subplots(1, len(OBJETIVOS), figsize=(5 * len(OBJETIVOS), 4), squeeze=False)
for ax, tic in zip(axs[0], OBJETIVOS):
    t, n = barridos[tic]
    ax.plot(t, n); ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(mcs_elegido[tic], c="r", ls="--", label=f"mcs = {mcs_elegido[tic]}")
    ax.set_xlabel("min_cluster_size"); ax.set_ylabel("cúmulos recuperados")
    ax.set_title(f"TIC {tic} ({OBJETIVOS[tic]})"); ax.legend()
fig.tight_layout(); fig.savefig(AQUI / "hdbscan_barrido.png", dpi=110); plt.show()
mcs_elegido"""),
code(r"""resultados = {}
for tic, s in muestras.items():
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs_elegido[tic], min_samples=1, cluster_selection_method="leaf",
                         metric="euclidean").fit(X_esc[tic])
    s = s.copy()
    s["Cl"], s["Cl_P"] = cl.labels_, cl.probabilities_
    s.to_csv(AQUI / f"TIC_{tic}" / f"hdbscan_mcs{mcs_elegido[tic]}.csv", index=False)
    resultados[tic] = s
    n = s.loc[s["Cl"] >= 0].groupby("Cl").size()
    print(f"TIC {tic}: {len(n)} cúmulos (mcs = {mcs_elegido[tic]}); miembros con P > {P_CORTE}: "
          f"{(s['Cl_P'] > P_CORTE).sum()}")"""),
md(r"""### ¿A qué grupo pertenece cada sistema?

Se proyecta la fuente Gaia del sistema con los mismos escaladores y se buscan sus 15 vecinos más cercanos en
el espacio de cinco parámetros entre los miembros con $P>0.37$. El grupo asignado es el mayoritario entre
los vecinos. Para saber si la asignación es razonable se compara la distancia del sistema a sus vecinos con
la distancia típica entre miembros del mismo grupo (`d_rel`; ≲ 1–2 es consistente con ser miembro)."""),
code(r"""asignacion = {}
filas = []
for tic, s in resultados.items():
    o = objetivo_fila[tic]
    miembros = s.loc[(s["Cl"] >= 0) & (s["Cl_P"] > P_CORTE)].reset_index(drop=True)
    Xm, _ = escalar(miembros, escaladores[tic])
    if o is None or not np.all(np.isfinite(o[["ra", "dec", "PLXcorr", "pmra", "pmdec"]].astype(float))):
        filas.append(dict(TIC=tic, grupo=np.nan, nota="sin astrometría Gaia del sistema")); continue
    Xo, _ = escalar(pd.DataFrame([o]), escaladores[tic])
    nn = NearestNeighbors(n_neighbors=15).fit(Xm)
    dist, idx = nn.kneighbors(Xo)
    labs = miembros.loc[idx[0], "Cl"].values
    g = int(pd.Series(labs).mode()[0])
    Xg = Xm[miembros["Cl"] == g]
    d_int = np.median(NearestNeighbors(n_neighbors=16).fit(Xg).kneighbors(Xg)[0][:, 1:].mean(axis=1))
    d_rel = dist[0].mean() / d_int
    en_muestra = bool(((s["source_id"] == o["source_id"]) & (s["Cl"] == g)).any())
    asignacion[tic] = g
    filas.append(dict(TIC=tic, sistema=OBJETIVOS[tic], grupo=g, miembros_grupo=int((miembros["Cl"] == g).sum()),
                      fraccion_vecinos_en_grupo=float((labs == g).mean()), d_rel=float(d_rel),
                      objetivo_en_muestra_y_grupo=en_muestra,
                      plx_objetivo=float(o["PLXcorr"]), plx_grupo=float(miembros.loc[miembros["Cl"] == g, "PLXcorr"].median()),
                      pmra_objetivo=float(o["pmra"]), pmra_grupo=float(miembros.loc[miembros["Cl"] == g, "pmra"].median()),
                      pmdec_objetivo=float(o["pmdec"]), pmdec_grupo=float(miembros.loc[miembros["Cl"] == g, "pmdec"].median())))
tabla_asig = pd.DataFrame(filas)
tabla_asig.to_csv(AQUI / "asignacion_grupos.csv", index=False)
tabla_asig.round(3)"""),
code(r"""fig, axs = plt.subplots(len(OBJETIVOS), 3, figsize=(16, 4.5 * len(OBJETIVOS)), squeeze=False)
for fila, (tic, s) in zip(axs, resultados.items()):
    m = s.loc[(s["Cl"] >= 0) & (s["Cl_P"] > P_CORTE)]
    o = objetivo_fila[tic]
    g = asignacion.get(tic)
    for ax, (x, y) in zip(fila, (("ra", "dec"), ("pmra", "pmdec"), ("BPmag-RPmag", "M_RP"))):
        if x == "BPmag-RPmag":
            X = lambda t: t["BPmag"] - t["RPmag"]
            Y = lambda t: t["RPmag"] - 5 * np.log10(1000 / t["PLXcorr"]) + 5
        else:
            X, Y = (lambda t, x=x: t[x]), (lambda t, y=y: t[y])
        ax.scatter(X(s), Y(s), s=1, c="0.85")
        for k, gg in m.groupby("Cl"):
            ax.scatter(X(gg), Y(gg), s=4 if k != g else 9, label=f"Grp{k}" + (" ← sistema" if k == g else ""),
                       zorder=3 if k == g else 2, edgecolor="k" if k == g else "none", linewidth=0.3)
        if o is not None:
            ax.plot(X(pd.DataFrame([o])).iloc[0], Y(pd.DataFrame([o])).iloc[0], "*", c="gold", mec="k", ms=18, zorder=5)
        ax.set_xlabel(x); ax.set_ylabel(y)
    fila[0].invert_xaxis(); fila[2].invert_yaxis()
    fila[1].set_xlim(-6, 6); fila[1].set_ylim(-6, 6)
    fila[0].set_title(f"TIC {tic} ({OBJETIVOS[tic]}): grupo {g}", loc="left")
    fila[2].legend(fontsize=7, markerscale=2)
fig.tight_layout(); fig.savefig(AQUI / "grupos_hdbscan.png", dpi=110); plt.show()"""),
md(r"""## 4) Cruce con el TIC 8.2 y diagrama HR del grupo

Cruce posicional de 1″ con `IV/39/tic82` (CDS XMatch, modo "mejor": se queda el más cercano). Del TIC se
toman $T_{\rm eff}$, $L$, 2MASS $J,H,K$ y ALLWISE $W1$–$W4$. Isócronas PARSEC de 1, 3, 5, 10 y 50 Myr
(archivo de modelos de MassAge)."""),
code(r"""def cruzar_tic(df):
    t = Table.from_pandas(df[["source_id", "ra", "dec"]])
    r = XMatch.query(cat1=t, cat2="vizier:IV/39/tic82", max_distance=1 * u.arcsec, colRA1="ra", colDec1="dec").to_pandas()
    r = r.sort_values("angDist").drop_duplicates("source_id")
    cols = ["source_id", "TIC", "Teff", "e_Teff", "Lum", "e_Lum", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag",
            "W1mag", "e_W1mag", "W2mag", "e_W2mag", "W3mag", "e_W3mag", "W4mag", "e_W4mag"]
    return df.merge(r[cols], on="source_id", how="left")


grupos = {}
for tic, s in resultados.items():
    g = asignacion.get(tic)
    if g is None:
        continue
    m = s.loc[(s["Cl"] == g) & (s["Cl_P"] > P_CORTE)].copy()
    f = AQUI / f"TIC_{tic}" / "grupo_TIC82.csv"
    if f.exists() and not REHACER:
        grupos[tic] = pd.read_csv(f)
    else:
        grupos[tic] = cruzar_tic(m)
        grupos[tic].to_csv(f, index=False)
    print(f"TIC {tic}: {len(grupos[tic])} miembros, {grupos[tic]['Teff'].notna().sum()} con Teff del TIC")

# primaria de cada sistema con MassAge: tipo espectral de la literatura (SpT_PM13), en fase "pms" (como el
# taller) y en "all" (pre-secuencia y secuencia principal), para ver cuánto depende de la fase elegida
primarias = {}
for tic in grupos:
    primarias[tic] = {}
    for fase in ("pms", "all"):
        r = correr_massage(entrada_primaria(tic), f"massage_primaria_{fase}", AQUI / f"TIC_{tic}",
                           xopt="SpT_PM13", fase=fase)
        primarias[tic][fase] = r.iloc[0]
pd.DataFrame({(tic, f): primarias[tic][f][["T_median", "lgL_median", "mass_median", "age_median", "age_low",
                                            "age_upp", "Av_median"]] for tic in primarias for f in primarias[tic]}).T.round(3)

iso = pd.read_csv(Path(MASSAGE_MODELOS) / "MODELS" / "PARSEC" / "PARSEC_logage_001_V3.csv")
fig, axs = plt.subplots(1, len(grupos), figsize=(5.5 * len(grupos), 5), squeeze=False)
for ax, (tic, gdf) in zip(axs[0], grupos.items()):
    for edad, col in ((1, "#c0392b"), (3, "#eb6834"), (5, "#eda100"), (10, "#1baf7a"), (50, "#2a78d6")):
        lg = np.log10(edad * 1e6)
        k = iso.loc[np.isclose(iso["logAge"], iso["logAge"].iloc[(iso["logAge"] - lg).abs().argmin()])]
        ax.plot(k["logTe"], k["logL"], c=col, lw=1, label=f"{edad} Myr")
    ok = gdf["Teff"].notna() & gdf["Lum"].notna()
    ax.scatter(np.log10(gdf.loc[ok, "Teff"]), np.log10(gdf.loc[ok, "Lum"]), s=8, c="k", label="miembros")
    rp = primarias[tic]["all"]
    ax.errorbar(np.log10(rp["T_median"]), rp["lgL_median"], xerr=rp["T_mad"] / rp["T_median"] / np.log(10),
                yerr=rp["lgL_mad"], fmt="*", c="gold", mec="k", ms=18, label="primaria (SpT literatura, MassAge)")
    ax.invert_xaxis(); ax.set_xlim(4.4, 3.4); ax.set_ylim(-2.5, 3.5)
    ax.set_xlabel("log Teff"); ax.set_ylabel("log L/L☉"); ax.set_title(f"TIC {tic} ({OBJETIVOS[tic]})")
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(AQUI / "hr_grupos.png", dpi=110); plt.show()"""),
md(r"""## 5) Discos y 6) rayos X: indicadores de juventud del grupo

- **Discos:** fuentes con buena fotometría ($\sigma_K,\sigma_{W3},\sigma_{W4}<0.5$) en el diagrama
  $K-W3$ frente a $K-W4$ (Hernández et al. 2023); un exceso indica disco.
- **Rayos X:** cruce con eRASS1 (Freund et al. 2024, `J/A+A/684/A121/like_ctp`) a 15″;
  $L_X=4\pi F_X d^2$ y $\log(L_X/L_{\rm bol})$. Las estrellas jóvenes están cerca de la saturación
  ($\approx-3$)."""),
code(r"""indic = {}
fig, axs = plt.subplots(2, len(grupos), figsize=(5.5 * len(grupos), 9), squeeze=False)
for j, (tic, gdf) in enumerate(grupos.items()):
    b = gdf.loc[(gdf["e_Kmag"] < 0.5) & (gdf["e_W3mag"] < 0.5) & (gdf["e_W4mag"] < 0.5)]
    axs[0][j].scatter(gdf["Kmag"] - gdf["W4mag"], gdf["Kmag"] - gdf["W3mag"], s=4, c="0.7", label="todos")
    axs[0][j].scatter(b["Kmag"] - b["W4mag"], b["Kmag"] - b["W3mag"], s=12, facecolor="none", edgecolor="b", label="buena W3W4")
    axs[0][j].set_xlabel("K − W4"); axs[0][j].set_ylabel("K − W3"); axs[0][j].set_title(f"TIC {tic}")
    axs[0][j].legend(fontsize=7)
    t = Table.from_pandas(gdf[["source_id", "ra", "dec"]])
    try:
        x = XMatch.query(cat1=t, cat2="vizier:J/A+A/684/A121/like_ctp", max_distance=15 * u.arcsec,
                         colRA1="ra", colDec1="dec").to_pandas().sort_values("angDist").drop_duplicates("source_id")
    except Exception as e:
        print(f"TIC {tic}: eRASS1 no disponible ({e})"); x = pd.DataFrame(columns=["source_id", "FX"])
    xg = gdf.merge(x[["source_id", "FX"]], on="source_id")
    xg["Lx"] = 4 * np.pi * xg["FX"] * (3.086e18 * 1000 / xg["PLXcorr"]) ** 2
    xg["logLx_Lbol"] = np.log10(xg["Lx"] / 3.828e33 / xg["Lum"])
    axs[1][j].scatter(xg["Teff"], xg["logLx_Lbol"], s=12, c="#c0392b")
    axs[1][j].axhline(-3, c="k", lw=1); axs[1][j].set_xlabel("Teff"); axs[1][j].set_ylabel("log Lx/Lbol")
    indic[tic] = dict(n_W3W4=len(b), frac_disco=float(((b["Kmag"] - b["W3mag"]) > 1.0).mean()) if len(b) else np.nan,
                      n_rayosX=len(xg), mediana_logLx_Lbol=float(xg["logLx_Lbol"].median()) if len(xg) else np.nan)
fig.tight_layout(); fig.savefig(AQUI / "discos_rayosX.png", dpi=110); plt.show()
pd.DataFrame(indic).T.round(3)"""),
md(r"""## 7) MassAge

Configuración del taller: PARSEC, fase `pms`, interpolación cúbica, 0.01–10 M☉, 0.01–1000 Myr, 100
iteraciones Monte Carlo, ley de extinción c89 con $R_V=3.1$, ancla $J$, $T_{\rm eff}$ lineal. Se corre para
los miembros con $T_{\rm eff}$ y $J$ del TIC (requisito de MassAge) y para la primaria de cada sistema."""),
code(r"""edades = {}
for tic, gdf in grupos.items():
    t0 = time.time()
    edades[tic] = correr_massage(entrada_massage(gdf), "massage_grupo", AQUI / f"TIC_{tic}",
                                 nmc=MASSAGE_NMC.get(tic, NMC_DEFECTO))
    print(f"TIC {tic}: MassAge en {len(edades[tic])} miembros ({(time.time() - t0) / 60:.1f} min)")"""),
md(r"""### Edad del grupo y edad asignada a cada sistema

- Se usan los miembros con edad finita y masa de 0.1 a 1.5 M☉: en ese intervalo las estrellas siguen en la
  pre-secuencia principal a la edad de Orión y la edad de MassAge está bien acotada.
- **Edad del grupo:** mediana de las edades medianas de los miembros; error: desviación estándar de la mediana
  en 2000 remuestreos bootstrap; dispersión: percentiles 16–84 de las edades individuales.
- Se compara con la edad individual de la primaria del catálogo previo (MIST) y con la de MassAge para la
  primaria con el **tipo espectral de la literatura** (`Macro_21aug2026.csv`, `SpT_PM13`) en fase `pms` y `all`."""),
code(r"""rng = np.random.default_rng(1)
filas = []
for tic, e in edades.items():
    e.columns = [c.strip() for c in e.columns]
    ok = np.isfinite(e["age_median"]) & (e["mass_median"] > 0.1) & (e["mass_median"] < 1.5)
    a = e.loc[ok, "age_median"].values
    boot = [np.median(rng.choice(a, len(a))) for _ in range(2000)] if len(a) > 3 else [np.nan]
    filas.append(dict(TIC=tic, sistema=OBJETIVOS[tic], grupo=asignacion[tic], n_edades=len(a),
                      edad_grupo_Myr=float(np.median(a)) if len(a) else np.nan,
                      err_edad_grupo_Myr=float(np.std(boot)),
                      p16_Myr=float(np.percentile(a, 16)) if len(a) else np.nan,
                      p84_Myr=float(np.percentile(a, 84)) if len(a) else np.nan,
                      edad_primaria_MIST_catalogo=float(CATALOGO.loc[tic, "age_median_1"]),
                      edad_primaria_PARSEC_pms=float(primarias[tic]["pms"]["age_median"]),
                      edad_primaria_PARSEC_all=float(primarias[tic]["all"]["age_median"]),
                      SpT_literatura=float(MACRO.loc[tic, "SpT"]), ref_SpT=MACRO.loc[tic, "REF"],
                      Teff_primaria_SpT=float(primarias[tic]["all"]["T_median"]),
                      Teff_primaria_catalogo=float(CATALOGO.loc[tic, "T_mean_1"])))
final = pd.DataFrame(filas)
final.to_csv(AQUI / "edades_asignadas.csv", index=False)
final.round(2)"""),
code(r"""fig, axs = plt.subplots(1, len(edades), figsize=(5.2 * len(edades), 4.2), squeeze=False)
for ax, (tic, e) in zip(axs[0], edades.items()):
    ok = np.isfinite(e["age_median"]) & (e["mass_median"] > 0.1) & (e["mass_median"] < 1.5)
    ax.hist(e.loc[ok, "age_median"], bins=np.logspace(-1, 2.5, 30), color="0.6")
    f = final.loc[final["TIC"] == tic].iloc[0]
    ax.axvline(f["edad_grupo_Myr"], c="k", lw=2, label=f"grupo: {f['edad_grupo_Myr']:.1f} ± {f['err_edad_grupo_Myr']:.1f} Myr")
    ax.axvspan(f["p16_Myr"], f["p84_Myr"], color="k", alpha=0.1)
    ax.axvline(f["edad_primaria_MIST_catalogo"], c="#c0392b", ls="--",
               label=f"primaria, MIST: {f['edad_primaria_MIST_catalogo']:.0f} Myr")
    for fase, ls_ in (("pms", ":"), ("all", "-.")):
        v = f[f"edad_primaria_PARSEC_{fase}"]
        if np.isfinite(v):
            ax.axvline(v, c="#2a78d6", ls=ls_, label=f"primaria, PARSEC {fase}: {v:.1f} Myr")
    ax.set_xscale("log"); ax.set_xlabel("edad [Myr]"); ax.set_ylabel("miembros")
    ax.set_title(f"TIC {tic} ({OBJETIVOS[tic]})"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(AQUI / "edades_grupos.png", dpi=110); plt.show()"""),
]
nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata = {"kernelspec": {"display_name": "Python (cumulos)", "language": "python", "name": "cumulos"},
               "language_info": {"name": "python"}}
nbf.write(nb, "/home/luciam/Documents/starry/Resultados_Orion/edades_vecindad/edades_por_vecindad.ipynb")
