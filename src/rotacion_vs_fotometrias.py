"""¿La rotación de 427395536 (sistema 16) viene de la binaria o de un vecino? Periodo de rotación en distintas fotometrías, con los eclipses enmascarados.

Fotometrías: propia (TESSExtractor, Flux_CBV_SS), QLP (SAP y KSPSAP), TGLC local y oficial (apertura y PSF; la PSF separa a los vecinos) y CDIPS.
Si la señal es del objetivo, el periodo y la amplitud se conservan al pasar de apertura a PSF; si viene de un vecino, la amplitud cae en la PSF.
Controles: 11300044 (sistema 8) y 72829000 (sistema 10), donde Prot = Porb.
Salida: rotacion_fotometrias.csv, figuras/rotacion_fotometrias.png y vecinos_gaia_427395536.csv.
"""
import glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.timeseries import LombScargle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion"); AQUI = RAIZ / "diagnostico_resultados"
T = pd.read_csv(AQUI / "tabla_maestra.csv"); PRI = pd.read_csv(RAIZ / "priors_v2.csv")
GAIA = {427395536: 3017252428529514752, 72829000: 3216557873341545216, 11300044: 3216540903927294592}
SISTEMA = {427395536: 16, 72829000: 10, 11300044: 8}
PERIODO_OBJETIVO = {427395536: 0.7372, 72829000: 2.65, 11300044: 2.099}     # Prot de consenso del pipeline


def binear(t, f, minutos=30):
    k = np.floor((t - t.min()) / (minutos / 1440)).astype(int)
    g = pd.DataFrame(dict(k=k, t=t, f=f)).groupby("k").mean()
    return g.t.to_numpy(), g.f.to_numpy()


def cargar(tic, s):
    """Devuelve {nombre: (t, flujo normalizado)} de las fotometrías disponibles para (tic, sector)."""
    out = {}
    f = RAIZ / "Light_Curve_Files" / f"tic {tic}_S_{s}_LC.txt"
    if f.exists():
        d = pd.read_csv(f, sep=r"\s+").rename(columns={"#Time": "t"}); d = d.dropna(subset=["t", "Flux_CBV_SS"])
        out["propia (CBV)"] = (d.t.to_numpy(), d.Flux_CBV_SS.to_numpy())
    f = RAIZ / "Light_Curve_Files_TGLC/local" / f"tic {tic}_S_{s}_TGLC.txt"
    if f.exists():
        d = pd.read_csv(f, sep=r"\s+", comment="#"); d = d[(d.TESS_flags == 0) & (d.TGLC_flags == 0)]
        out["TGLC PSF"] = (d.time.to_numpy(), d.cal_psf_flux.to_numpy()); out["TGLC apertura"] = (d.time.to_numpy(), d.cal_aper_flux.to_numpy())
    else:
        g = glob.glob(str(RAIZ / f"Light_Curve_Files_TGLC/hlsp_tglc*{GAIA[tic]}-s{s:04d}*.fits"))
        if g:
            d = fits.open(g[0])[1].data; ok = (d["TESS_flags"] == 0) & (d["TGLC_flags"] == 0)
            out["TGLC PSF"] = (d["time"][ok], d["cal_psf_flux"][ok]); out["TGLC apertura"] = (d["time"][ok], d["cal_aper_flux"][ok])
    g = glob.glob(str(RAIZ / f"Light_Curve_Files_MAST/hlsp_qlp_tess_ffi_s{s:04d}-{tic:016d}_tess_v01_llc.fits"))
    if g:
        d = fits.open(g[0])[1].data; ok = d["QUALITY"] == 0
        out["QLP SAP"] = (d["TIME"][ok], d["SAP_FLUX"][ok])
        det = "KSPSAP_FLUX" if "KSPSAP_FLUX" in d.columns.names else "DET_FLUX"          # S94+ es producto TGLC: DET_FLUX en vez de KSPSAP_FLUX
        out["QLP KSP"] = (d["TIME"][ok], d[det][ok])
    g = glob.glob(str(RAIZ / f"Light_Curve_Files_MAST/hlsp_cdips*{GAIA[tic]:019d}-s{s:04d}*.fits"))
    if g:
        d = fits.open(g[0])[1].data; m = d["IRM1"]; ok = np.isfinite(m)
        out["CDIPS"] = (d["TMID_BJD"][ok] - 2457000.0, 10 ** (-0.4 * (m[ok] - np.nanmedian(m[ok]))))
    res = {}
    for k, (t, f_) in out.items():
        t, f_ = np.asarray(t, float), np.asarray(f_, float); ok = np.isfinite(t) & np.isfinite(f_) & (f_ > 0)
        if ok.sum() > 100:
            res[k] = (t[ok], f_[ok] / np.median(f_[ok]))
    return res


def mascara_eclipses(t, tic, s):
    r = T[(T.TIC == tic) & (T.Sector == s)]
    if r.empty:
        return np.ones(len(t), bool)
    r = r.iloc[0]; P = r.P
    P_ = pd.read_csv(Path(r["dir"]) / f"TIC_{tic}_S{s}_Posteriors.csv", index_col=0); t0 = float(P_["B_t0"].mean())
    pr = PRI[(PRI.TIC == tic) & (PRI.Sector == s)]; dur = float(pr.Duration.iloc[0]) if len(pr) else 0.25
    W = min(dur / P, 0.25)
    f2 = r.get("fase_sec_ref", 0.5); f2 = 0.5 if not np.isfinite(f2) else float(f2)
    ph = ((t - t0) / P + 0.5) % 1 - 0.5; ph2 = ((t - t0) / P - f2 + 0.5) % 1 - 0.5
    return ~((np.abs(ph) < 0.8 * W) | (np.abs(ph2) < 0.8 * W))


def periodograma(t, f, pmin=0.2, pmax=12.0):
    tb, fb = binear(t, f); fr = np.linspace(1 / pmax, 1 / pmin, 6000)
    ls = LombScargle(tb, fb); pw = ls.power(fr); i = int(np.argmax(pw))
    return 1 / fr, pw, 1 / fr[i], float(pw[i]), ls, tb, fb


def amplitud(ls, tb, fb, P):
    c = ls.model_parameters(1 / P); return float(np.hypot(c[1], c[2]))


filas, curvas = [], {}
for tic in GAIA:
    for s in sorted(T[T.TIC == tic].Sector.unique()):
        for nombre, (t, f) in cargar(tic, int(s)).items():
            ok = mascara_eclipses(t, tic, int(s)); t, f = t[ok], f[ok]
            if len(t) < 100:
                continue
            per, pw, Pp, pot, ls, tb, fb = periodograma(t, f)
            a0 = amplitud(ls, tb, fb, PERIODO_OBJETIVO[tic])
            filas.append(dict(TIC=tic, sistema=SISTEMA[tic], sector=int(s), fotometria=nombre, n=len(t), P_pico=Pp, potencia=pot,
                              P_objetivo=PERIODO_OBJETIVO[tic], amp_en_Pobjetivo_ppt=1e3 * a0, rms_ppt=1e3 * np.std(fb)))
            curvas[(tic, int(s), nombre)] = (per, pw)
D = pd.DataFrame(filas); D.to_csv(AQUI / "rotacion_fotometrias.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(D.round(3).to_string(index=False))

# --- figura: periodogramas del sistema 16 por sector, una línea por fotometría -------------------------------------------------------------
COL = {"propia (CBV)": "#2a78d6", "TGLC PSF": "#eb6834", "TGLC apertura": "#1baf7a", "QLP SAP": "#8a8a85", "QLP KSP": "#bdbcb6", "CDIPS": "#4a3aa7"}
sec16 = sorted(D[D.TIC == 427395536].sector.unique())
fig, ax = plt.subplots(1, len(sec16), figsize=(5.2 * len(sec16), 3.8), sharey=False, squeeze=False)
for a, s in zip(ax[0], sec16):
    for nom in COL:
        if (427395536, s, nom) in curvas:
            per, pw = curvas[(427395536, s, nom)]; a.plot(per, pw, color=COL[nom], lw=1.4 if "PSF" in nom or "propia" in nom else 1.0, label=nom)
    a.axvline(0.7372, color="k", ls=":", lw=0.8); a.set_xscale("log"); a.set_xlabel("periodo [d]"); a.set_title(f"TIC 427395536 · S{s}", fontsize=10, loc="left")
ax[0, 0].set_ylabel("potencia Lomb–Scargle"); ax[0, 0].legend(fontsize=7.5)
fig.suptitle("Sistema 16: periodograma de la rotación (eclipses enmascarados) por fotometría; línea punteada = 0.737 d", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(); (AQUI / "figuras").mkdir(exist_ok=True); fig.savefig(AQUI / "figuras/rotacion_fotometrias.png", dpi=130, bbox_inches="tight")

# --- vecinos en Gaia alrededor del sistema 16 ------------------------------------------------------------------------------------------------
try:
    from astroquery.gaia import Gaia
    r0 = T[T.TIC == 427395536].iloc[0]
    q = (f"SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag AS G, g.bp_rp, g.parallax, g.pmra, g.pmdec, g.ruwe, g.phot_variable_flag, "
         f"DISTANCE(POINT(g.ra,g.dec), POINT(r.ra,r.dec))*3600 AS sep_arcsec FROM gaiadr3.gaia_source AS g, gaiadr3.gaia_source AS r "
         f"WHERE r.source_id = 3017252428529514752 AND 1 = CONTAINS(POINT(g.ra,g.dec), CIRCLE(r.ra,r.dec, 60/3600.)) AND g.phot_g_mean_mag < 19 ORDER BY sep_arcsec")
    V = Gaia.launch_job(q).get_results().to_pandas()
    V["flujo_rel"] = 10 ** (-0.4 * (V.G - V.G.iloc[0]))
    V.to_csv(AQUI / "vecinos_gaia_427395536.csv", index=False)
    print("\nVecinos en Gaia (≤ 60″, G < 19):", len(V) - 1, "| fuentes con flujo ≥ 5 % del objetivo dentro de 42″:", int(((V.sep_arcsec > 0.5) & (V.sep_arcsec < 42) & (V.flujo_rel > 0.05)).sum()))
    print(V[(V.sep_arcsec < 63) & (V.flujo_rel > 0.03)][["source_id", "G", "bp_rp", "sep_arcsec", "flujo_rel", "parallax", "pmra", "pmdec", "ruwe", "phot_variable_flag"]].round(3).to_string(index=False))
except Exception as e:
    print("sin vecinos de Gaia:", str(e)[:150])
