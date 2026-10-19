"""Paso 1 de la metodología de Orión (1_obtencion_priors_curva + pruebas de 1b) aplicado a las candidatas de las Pléyades.

Por cada curva de TESSExtractor web (Flux_CBV_SS): wotan biweight (ventana 0.5 d) -> BLS -> decisión P / 2P por diferencia primario-secundario (como en Orión)
-> profundidades primaria y secundaria -> prueba par-impar -> Lomb-Scargle con eclipses enmascarados.
Diferencias respecto de 1_ de Orión: malla de BLS uniforme en frecuencia y 10 veces más fina (la lineal en periodo de 2000 puntos no resuelve periodos cortos
con 25 d de base); se descartan los archivos 'sector 1751' (apuntado hipotético, flujo NaN) y los que no pasan el control de calidad.
Salida: priors_pleyades.csv, resumen_por_tic.csv y figuras/<TIC>.png.
"""
import re, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
from astropy.timeseries import BoxLeastSquares, LombScargle
from wotan import flatten
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades"); AQUI = BASE / "pipeline"
LCDIR = BASE / "tessextractor_local/tess_extractor_corr/Light_Curve_Files"
VENTANA, PMIN, PMAX = 0.5, 0.3, 15.0
EXCLUIR = {(385591142, 70): "CBV desbocada (flujo entre -0.8 y 2.7)"}


def cargar(f):
    d = pd.read_csv(f, sep=r"\s+").rename(columns={"#Time": "t"}).dropna(subset=["t", "Flux_CBV_SS"])
    d = d.sort_values("t").drop_duplicates("t"); return d.t.to_numpy(float), d.Flux_CBV_SS.to_numpy(float)


def par_impar(t, plano, t0, P, hw=0.03):
    n = np.floor((t - t0) / P + 0.5).astype(int); phc = ((t - t0) / P + 0.5) % 1 - 0.5; en = np.abs(phc) < hw
    ph = ((t - t0) % P) / P; fuera = (ph > 0.15) & (ph < 0.35)
    a, c = plano[en & (n % 2 == 0)], plano[en & (n % 2 == 1)]
    if len(a) < 5 or len(c) < 5 or fuera.sum() < 5:
        return np.nan
    return abs(np.nanmedian(a) - np.nanmedian(c)) / np.hypot(np.std(a) / np.sqrt(len(a)), np.std(c) / np.sqrt(len(c)))


def profundidad(ph, y, centro, hw=0.03, lo=0.15, hi=0.35):
    fuera = (ph > lo) & (ph < hi); en = np.abs(((ph - centro + 0.5) % 1) - 0.5) < hw
    if en.sum() < 5 or fuera.sum() < 5:
        return np.nan, np.nan
    return float(np.nanmedian(y[fuera]) - np.nanmedian(y[en])), float(np.nanstd(y[en]) / np.sqrt(en.sum()))


filas, grafs = [], {}
for f in sorted(glob.glob(str(LCDIR / "*_LC.txt"))):
    m = re.search(r"tic (\d+)_S_(\d+)_LC", f); tic, s = int(m[1]), int(m[2])
    if s > 200 or (tic, s) in EXCLUIR:
        continue
    t, y = cargar(f); y = y / np.nanmedian(y)
    plano, tend = flatten(t, y, method="biweight", window_length=VENTANA, break_tolerance=0.5, return_trend=True)
    ok = np.isfinite(plano); t, y, plano, tend = t[ok], y[ok], plano[ok], tend[ok]
    base = t.max() - t.min()
    fr = np.arange(1 / PMAX, 1 / PMIN, 0.1 / base / 10); periodos = np.sort(1 / fr)           # paso en frecuencia = 0.1 / base / 10
    dur = np.linspace(0.005, min(0.9 * PMIN, 0.25), 40)
    r = BoxLeastSquares(t, plano).power(periodos, dur)
    i = int(np.argmax(r.power)); Pb, t0, du = r.period[i], r.transit_time[i], r.duration[i]
    snr_bls = float((r.depth[i] / r.depth_err[i])) if r.depth_err[i] > 0 else np.nan
    ph2 = ((t - t0) / (2 * Pb)) % 1
    prim, sec = np.nanmedian(plano[np.abs(ph2) < 0.05]), np.nanmedian(plano[np.abs(ph2 - 0.5) < 0.05])
    sig_Pb = par_impar(t, plano, t0, Pb)                       # par-impar al periodo de BLS: si difieren, hay dos eclipses distintos encimados (como en 1b de Orión)
    duplicar = abs(prim - sec) > 0.005 or (np.isfinite(sig_Pb) and sig_Pb > 5)
    P = 2 * Pb if duplicar else Pb
    ph = ((t - t0) % P) / P
    hw = float(np.clip(0.25 * du / P, 0.002, 0.03))            # ventana del núcleo del eclipse: Orión usa ±0.03 fija, que diluye eclipses angostos en periodos largos
    dp, dpe = profundidad(ph, plano, 0.0, hw=hw); ds, dse = profundidad(ph, plano, 0.5, hw=hw)
    # par-impar: profundidad de eclipses primarios en ciclos pares e impares
    n = np.floor((t - t0) / P + 0.5).astype(int); phc = ((t - t0) / P + 0.5) % 1 - 0.5; en = np.abs(phc) < 0.03
    dpar, dimp, sig_pi = np.nan, np.nan, np.nan
    if (en & (n % 2 == 0)).sum() > 5 and (en & (n % 2 == 1)).sum() > 5:
        fuera = (ph > 0.15) & (ph < 0.35); b = np.nanmedian(plano[fuera])
        a, c = plano[en & (n % 2 == 0)], plano[en & (n % 2 == 1)]
        dpar, dimp = b - np.nanmedian(a), b - np.nanmedian(c)
        sig_pi = abs(dpar - dimp) / np.hypot(np.std(a) / np.sqrt(len(a)), np.std(c) / np.sqrt(len(c)))
    # rotación con eclipses enmascarados (curva sin aplanar, como en Orión)
    w = 3.0 * du / P; fase = ((t - t0) / P) % 1
    mk = (fase < w) | (fase > 1 - w) | (np.abs(fase - 0.5) < w)
    tr, yr = (t[~mk], y[~mk]) if (~mk).sum() >= 20 else (t, y)
    frl = np.linspace(1 / 15.0, 1 / 0.2, 4000); pw = LombScargle(tr, yr).power(frl); j = int(np.argmax(pw))
    filas.append(dict(TIC=tic, Sector=s, n=len(t), base_d=base, mad_ppt=1e3 * 1.4826 * np.nanmedian(np.abs(plano - np.nanmedian(plano))),
                      P_bls=Pb, P_orb=P, duplicado=bool(P == 2 * Pb), sig_par_impar_Pbls=sig_Pb, t0=t0, Duration=du, snr_bls=snr_bls,
                      Depth_primary=dp, Depth_primary_err=dpe, Depth_secondary=ds, Depth_secondary_err=dse,
                      signif_secundario=(ds / dse if dse and np.isfinite(dse) and dse > 0 else np.nan),
                      D_par=dpar, D_impar=dimp, signif_par_impar=sig_pi, P_rot=1 / frl[j], pot_rot=float(pw[j])))
    grafs[(tic, s)] = (t, plano, t0, P, periodos, r.power, Pb)

D = pd.DataFrame(filas); D.to_csv(AQUI / "priors_pleyades.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(D[["TIC", "Sector", "mad_ppt", "P_bls", "P_orb", "snr_bls", "Depth_primary", "Depth_secondary", "signif_secundario", "signif_par_impar", "P_rot", "pot_rot"]].round(4).to_string(index=False))

# resumen por TIC: consistencia del periodo entre sectores
res = []
for tic, g in D.groupby("TIC"):
    Pm = g.P_bls.median(); cons = int(((g.P_bls / Pm - 1).abs() < 0.01).sum())
    res.append(dict(TIC=tic, n_sec=len(g), P_bls_mediana=Pm, sectores_consistentes=cons, P_bls_min=g.P_bls.min(), P_bls_max=g.P_bls.max(),
                    snr_mediano=g.snr_bls.median(), prof_prim_ppt=1e3 * g.Depth_primary.median(), prof_sec_ppt=1e3 * g.Depth_secondary.median()))
R = pd.DataFrame(res); R.to_csv(AQUI / "resumen_por_tic.csv", index=False)
print("\n", R.round(4).to_string(index=False))

for tic in D.TIC.unique():
    ss = sorted(s for (x, s) in grafs if x == tic)
    fig, ax = plt.subplots(len(ss), 3, figsize=(14, 2.5 * len(ss)), squeeze=False)
    for k, s in enumerate(ss):
        t, plano, t0, P, per, pw, Pb = grafs[(tic, s)]; ph = ((t - t0) / P + 0.25) % 1 - 0.25
        ax[k, 0].plot(t, plano, ".", ms=1.2, color="#2a78d6", alpha=0.6); ax[k, 0].set_title(f"TIC {tic} · S{s} · aplanada", fontsize=9, loc="left")
        ax[k, 1].plot(per, pw, color="#eb6834", lw=0.8); ax[k, 1].axvline(Pb, color="k", ls=":", lw=0.8); ax[k, 1].set_xscale("log"); ax[k, 1].set_title(f"BLS · pico {Pb:.4f} d", fontsize=9, loc="left")
        ax[k, 2].plot(ph, plano, ".", ms=1.5, color="#4a3aa7", alpha=0.5); ax[k, 2].set_title(f"plegada en P = {P:.4f} d", fontsize=9, loc="left")
        lo, hi = np.nanpercentile(plano, [0.5, 99.5]); ax[k, 2].set_ylim(lo - 0.002, hi + 0.002)
    fig.tight_layout(); fig.savefig(AQUI / f"figuras/{tic}.png", dpi=90); plt.close(fig)
print("figuras:", len(D.TIC.unique()))
