"""Confirma la pertenencia a las Pléyades de las candidatas a binaria eclipsante y prepara la lista para TESSExtractor web.

Candidatas: (a) los 4 sistemas ajustados con starry en Resultados_pleyades_lucia; (b) las 11 curvas que se marcaron como EB o candidata en la
selección visual (cotrending/seleccion_visual.csv). Pruebas de pertenencia, con las propiedades de las miembros de alta probabilidad (Cl_P > 0.7) como referencia:
  1. astrometría: distancia de Mahalanobis en (paralaje corregido, pmra, pmdec) con la dispersión intrínseca del cúmulo más los errores de cada estrella;
     y desviación de la velocidad tangencial, Δv_t = 4.74 Δμ / ϖ;
  2. velocidad radial de Gaia DR3 frente a la mediana del cúmulo, cuando existe;
  3. posición en el diagrama color-magnitud (M_G frente a BP−RP) respecto de la secuencia mediana del cúmulo.
Salida: candidatas_pleyades.csv (todo) y candidatas_tessextractor.txt (confirmadas y probables, con sectores).
"""
from pathlib import Path
import warnings
import numpy as np, pandas as pd
from astropy.table import Table
from scipy.stats import chi2

warnings.filterwarnings("ignore")
BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades"); AQUI = BASE / "candidatas"
M = pd.read_csv(BASE / "membresia/pleyades_miembros_TIC82.csv").dropna(subset=["TIC"]); M["TIC"] = M["TIC"].astype(int)
H = pd.read_csv(BASE / "membresia/pleyades_hdbscan_mcs75.csv")              # todas las fuentes tras los cortes, con Cl y Cl_P
SEL = pd.read_csv(BASE / "cotrending/seleccion_visual.csv")
LUCIA = {26017840: "starry", 35159657: "starry", 125754991: "starry", 346541355: "starry"}
GAIA_DE_TIC = {26017840: 67920926352940032, 346541355: 71877862543976192}      # de TIC 8.2 (las dos que no están en la lista de miembros)

marc = SEL[SEL.clase.isin(["EB", "CAND"])].set_index("TIC")
tics = sorted(set(LUCIA) | set(marc.index))
filas = []
for t in tics:
    r = M[M.TIC == t]
    if len(r):
        g = int(r["source_id"].iloc[0]); fila = H[H.source_id == g]
    else:
        g = GAIA_DE_TIC[t]; fila = H[H.source_id == g]
    f = fila.iloc[0]
    filas.append(dict(TIC=t, source_id=g, en_lista_miembros=bool(len(r)), Cl=f["Cl"], Cl_P=f["Cl_P"], ra=f["ra"], dec=f["dec"], plx=f["PLXcorr"], plx_err=f["parallax_error"],
                      pmra=f["pmra"], pmdec=f["pmdec"], pmra_err=f["pmra_error"], pmdec_err=f["pmdec_error"], ruwe=f["ruwe"], G=f["Gmag"], BP=f["BPmag"], RP=f["RPmag"],
                      origen=("Lucia (starry)" if t in LUCIA else "") + (" + " if (t in LUCIA and t in marc.index) else "") + (f"selección visual ({marc.loc[t, 'clase']})" if t in marc.index else "")))
C = pd.DataFrame(filas)

# --- referencia: miembros de alta probabilidad -------------------------------------------------------------------------------------------
nuc = H[(H["Cl"] == 2) & (H["Cl_P"] > 0.7)].copy()
mu = nuc[["PLXcorr", "pmra", "pmdec"]].median().to_numpy()
cov = np.cov(nuc[["PLXcorr", "pmra", "pmdec"]].to_numpy().T)
print(f"núcleo de referencia: {len(nuc)} estrellas; centro (plx, pmra, pmdec) = {np.round(mu, 3)}; sigmas = {np.round(np.sqrt(np.diag(cov)), 3)}")
d2, dvt = [], []
for _, r in C.iterrows():
    S = cov + np.diag([r.plx_err ** 2, r.pmra_err ** 2, r.pmdec_err ** 2])
    x = np.array([r.plx, r.pmra, r.pmdec]) - mu
    d2.append(float(x @ np.linalg.solve(S, x)))
    dvt.append(4.74 * np.hypot(r.pmra - mu[1], r.pmdec - mu[2]) / r.plx)
C["d_mahalanobis"] = np.sqrt(d2); C["p_astrom"] = 1 - chi2.cdf(d2, 3); C["dvt_kms"] = dvt
sig_vt = float(np.median(4.74 * np.hypot(nuc.pmra - mu[1], nuc.pmdec - mu[2]) / nuc.PLXcorr))
C["dvt_sobre_tipico"] = C["dvt_kms"] / sig_vt

# --- diagrama color-magnitud ---------------------------------------------------------------------------------------------------------
nuc["color"], nuc["MG"] = nuc.BPmag - nuc.RPmag, nuc.Gmag - 5 * np.log10(1000 / nuc.PLXcorr) + 5
bins = np.arange(nuc.color.min(), nuc.color.max() + 0.2, 0.2); k = np.digitize(nuc.color, bins)
sec = nuc.groupby(k).agg(c=("color", "median"), m=("MG", "median"), n=("MG", "size")); sec = sec[sec.n >= 8]
C["color"] = C.BP - C.RP; C["MG"] = C.G - 5 * np.log10(1000 / C.plx) + 5
C["dMG_secuencia"] = C["MG"] - np.interp(C["color"], sec.c, sec.m)            # > 0: más débil que la secuencia; < 0: más brillante (binarias hasta -0.75)

# --- velocidad radial de Gaia DR3 -------------------------------------------------------------------------------------------------------
try:
    from astroquery.gaia import Gaia
    ids = ",".join(str(i) for i in C.source_id)
    job = Gaia.launch_job(f"SELECT source_id, radial_velocity, radial_velocity_error, rv_nb_transits, phot_variable_flag FROM gaiadr3.gaia_source WHERE source_id IN ({ids})")
    rv = job.get_results().to_pandas().rename(columns={"radial_velocity": "rv", "radial_velocity_error": "rv_err"})
    C = C.merge(rv[["source_id", "rv", "rv_err", "rv_nb_transits", "phot_variable_flag"]], on="source_id", how="left")
    # mediana de RV del cúmulo: miembros del núcleo que la tienen en Gaia (subconjunto brillante)
    ref = nuc.source_id.astype("int64").tolist()[:600]
    jr = Gaia.launch_job("SELECT radial_velocity FROM gaiadr3.gaia_source WHERE radial_velocity IS NOT NULL AND source_id IN (" + ",".join(map(str, ref)) + ")")
    rvs = jr.get_results()["radial_velocity"].data.astype(float)
    RV0, RVs = float(np.nanmedian(rvs)), float(1.4826 * np.nanmedian(np.abs(rvs - np.nanmedian(rvs))))
    print(f"RV de referencia (Gaia, {len(rvs)} estrellas del núcleo): {RV0:.2f} ± {RVs:.2f} km/s")
    C["drv"] = C["rv"] - RV0; C["rv_ref"] = RV0; C["rv_sig_ref"] = RVs
except Exception as e:
    print("sin velocidad radial de Gaia:", str(e)[:150]); C["rv"] = np.nan; C["drv"] = np.nan; C["rv_err"] = np.nan; C["rv_nb_transits"] = np.nan; C["phot_variable_flag"] = ""

# --- veredicto ---------------------------------------------------------------------------------------------------------------------------
def veredicto(r):
    """La RV de Gaia de una binaria (sobre todo eclipsante, de periodo corto) es el promedio de pocos tránsitos y puede estar desplazada decenas de km/s por el
    movimiento orbital: solo rechaza cuando su error es pequeño (< 3 km/s); si el error es grande, queda como duda."""
    fallos, dudas = [], []
    if r.p_astrom < 0.003 or r.dvt_sobre_tipico > 3.5:
        fallos.append("astrometría fuera del cúmulo")
    elif r.p_astrom < 0.05:
        dudas.append("astrometría en el borde")
    if np.isfinite(r.drv):
        sig = np.hypot(r.rv_err if np.isfinite(r.rv_err) else 0.0, r.rv_sig_ref)
        if abs(r.drv) > 3 * sig:
            (fallos if (np.isfinite(r.rv_err) and r.rv_err < 3) else dudas).append(
                f"RV discrepante ({r.drv:+.1f} km/s, error {r.rv_err:.1f}{'' if r.rv_err < 3 else '; no concluyente en una binaria'})")
    if r.dMG_secuencia > 0.8:
        dudas.append("más débil que la secuencia")
    if r.dMG_secuencia < -1.2:
        dudas.append("muy por encima de la secuencia")
    if r.ruwe > 1.4:
        dudas.append(f"RUWE {r.ruwe:.1f} (posible compañero no resuelto)")
    if fallos:
        return "no miembro probable", "; ".join(fallos + dudas)
    if dudas:
        return "probable", "; ".join(dudas)
    ok = "astrometría, RV y CMD consistentes" if (np.isfinite(r.drv) and r.rv_err < 3) else ("astrometría y CMD consistentes; RV poco precisa (binaria)" if np.isfinite(r.drv) else "astrometría y CMD consistentes (sin RV en Gaia)")
    return "confirmada", ok
v = C.apply(veredicto, axis=1, result_type="expand"); C["veredicto"], C["motivo"] = v[0], v[1]
C.to_csv(AQUI / "candidatas_pleyades.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 70)
print(C[["TIC", "origen", "Cl_P", "en_lista_miembros", "d_mahalanobis", "p_astrom", "dvt_kms", "drv", "dMG_secuencia", "ruwe", "veredicto", "motivo"]].round(3).to_string(index=False))


# --- sectores TESS (tess-point) y lista para TESSExtractor web ---------------------------------------------------------------------------
from tess_stars2px import tess_stars2px_function_entry
TIC8 = {26017840: (52.494767, 23.371827, 14.422)}                    # de TIC 8.2 y de la muestra anterior (no están en la lista de miembros)
filas_sec = []
for _, r in C.iterrows():
    m = M[M.TIC == r.TIC]
    ra, dec, tm = (m["ra"].iloc[0], m["dec"].iloc[0], m["Tmag"].iloc[0]) if len(m) else (TIC8.get(r.TIC, (r.ra, r.dec, np.nan)))
    res = tess_stars2px_function_entry(int(r.TIC), float(ra), float(dec))
    for sec in sorted(set(int(x) for x in res[3] if int(x) < 200)):          # tess-point también devuelve apuntados hipotéticos (p. ej. 1751)
        filas_sec.append(dict(TIC=int(r.TIC), RA=ra, Dec=dec, Sector=sec, Tmag=tm, veredicto=r.veredicto))
SEC = pd.DataFrame(filas_sec)
SEC.to_csv(AQUI / "candidatas_sectores.csv", index=False)
ok = SEC[SEC.veredicto.isin(["confirmada", "probable"])]
with open(AQUI / "candidatas_tessextractor.txt", "w") as f:
    f.write("# TIC       RA          Dec        Sector Tmag     (candidatas confirmadas y probables; sectores observados según tess-point)\n")
    for r in ok.itertuples():
        f.write(f"  {r.TIC:<9d} {r.RA:<11.6f} {r.Dec:<10.6f} {r.Sector:<6d} {r.Tmag:.3f}\n")
print("\nsectores por candidata:"); print(SEC.groupby(["TIC", "veredicto"]).Sector.apply(lambda x: ",".join(map(str, x))).to_string())
print(f"\nlista TESSExtractor: {ok.TIC.nunique()} estrellas, {len(ok)} pares TIC-sector")
