"""Comparación de las curvas propias (Flux_CBV_SS) con TGLC para los sistemas que difieren entre sectores.

Usa las mismas funciones de 1b_profundidades_eclipse_v2.ipynb (normalizar_sin_eclipses, medir) y los
mismos P, t0 y duración de priors_v2.csv, así la única diferencia es la fotometría.
TGLC en MAST (HLSP) solo cubre S5/S6 para estos objetos.
"""
import json
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

RUTA_ORION = Path("/home/luciam/Documents/starry/Resultados_Orion")
RUTA_TGLC = RUTA_ORION / "Light_Curve_Files_TGLC"
DIR_OUT = RUTA_ORION / "resultados_pipeline" / "profundidades_v2" / "tglc"
DIR_OUT.mkdir(parents=True, exist_ok=True)

# funciones del notebook 1b (celdas de configuración y de medición)
nb = json.load(open(RUTA_ORION / "1b_profundidades_eclipse_v2.ipynb"))
for i in (4, 5):
    exec("".join(nb["cells"][i]["source"]))

v2 = pd.read_csv(RUTA_ORION / "priors_v2.csv")
cons = pd.read_csv(RUTA_ORION / "resultados_pipeline" / "profundidades_v2" / "consistencia_sectores.csv")
revisar = cons.loc[(cons["marca_primario"] != "consistente") | (cons["marca_secundario"] == "difiere_revisar")]
COLOR_SECTOR = {5: "#2a78d6", 6: "#2a78d6", 32: "#eb6834", 33: "#eb6834", 87: "#eda100", 98: "#1baf7a"}
COLOR_TGLC = {"cal_psf_flux": "black", "cal_aper_flux": "#8e44ad"}


def cargar_tglc(tic, sector):
    # una curva por fuente Gaia asociada al TIC (TIC 264741359 tiene dos)
    salida = []
    for f in sorted(glob.glob(str(RUTA_TGLC / f"*s{sector:04d}*_llc.fits"))):
        h = fits.open(f)
        if str(h[0].header["TICID"]) != str(tic):
            continue
        d = h[1].data
        ok = (d["TGLC_flags"] == 0) & (d["TESS_flags"] == 0) & np.isfinite(d["time"])
        salida.append(dict(gaia=h[0].header["GAIADR3"], tmag=h[0].header["TESSMAG"], t=d["time"][ok].astype(float),
                           **{c: d[c][ok].astype(float) for c in COLOR_TGLC}))
    return sorted(salida, key=lambda s: s["tmag"])


def plegar(t, f, r):
    m, F, ph = medir(t, f, r["P_orb"], r["t0"], r["Duration"])
    c1 = m["centro_primario"] if np.isfinite(m["centro_primario"]) else 0.0
    c2 = m["fase_secundario"] if np.isfinite(m["fase_secundario"]) else 0.5
    h1 = (((ph - c1) + 0.5) % 1 - 0.5) * r["P_orb"] * 24
    h2 = (((ph - c2) + 0.5) % 1 - 0.5) * r["P_orb"] * 24
    return m, F, h1, h2


def mediana_bines(x, y, ancho, rango):
    bordes = np.arange(-rango, rango + ancho, ancho)
    xc, yc = [], []
    for a, b in zip(bordes[:-1], bordes[1:]):
        sel = (x >= a) & (x < b)
        if sel.sum() >= 2:
            xc.append(0.5 * (a + b)); yc.append(np.nanmedian(y[sel]))
    return np.array(xc), np.array(yc)


filas = []
for _, fc in revisar.iterrows():
    tic = int(fc["TIC"])
    g = v2.loc[v2["TIC"] == tic].sort_values("Sector")
    dur_h = g["Duration"].median() * 24
    rangos = (0.85 * dur_h, 1.25 * dur_h)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in g.iterrows():
            s = int(r["Sector"])
            curvas = [("propia", None, *cargar_curva(tic, s))]
            fuentes = cargar_tglc(tic, s)
            for src in fuentes:
                for col in COLOR_TGLC:
                    curvas.append((col, src, src["t"], src[col]))
            for nombre, src, t, f in curvas:
                if len(t) < 100:
                    continue
                m, F, h1, h2 = plegar(t, f, r)
                filas.append(dict(TIC=tic, Sector=s, fotometria=nombre,
                                  gaia=src["gaia"] if src else np.nan, tmag_gaia=src["tmag"] if src else np.nan,
                                  D1=m["D_primario"], D1_err=m["D_primario_err"],
                                  D2=m["D_secundario"], D2_err=m["D_secundario_err"],
                                  sec_detectado=m["secundario_detectado"], sigma_punto=m["sigma_punto"],
                                  D2_D1=m["D_secundario"] / m["D_primario"]))
                if src is not None and src["gaia"] != fuentes[0]["gaia"]:
                    continue   # en la figura solo la fuente Gaia más brillante
                cad_h = np.median(np.diff(t)) * 24
                ancho = max(dur_h / 30, 1.2 * cad_h)
                if nombre == "propia":
                    col, ls, lw, et = COLOR_SECTOR.get(s, "0.4"), "-", 2.0, f"S{s} propia"
                else:
                    col, ls, lw = COLOR_TGLC[nombre], "--", 1.6
                    et = f"S{s} TGLC {'PSF' if 'psf' in nombre else 'apert.'}"
                for ax, h, D, rango in ((axs[0], h1, m["D_primario"], rangos[0]),
                                        (axs[1], h2, m["D_secundario"], rangos[1])):
                    sel = np.abs(h) < rango
                    if nombre != "cal_aper_flux":
                        ax.plot(h[sel], F[sel], ".", ms=1.5, color=col, alpha=0.15, zorder=1)
                    xb, yb = mediana_bines(h, F, ancho, rango)
                    ax.plot(xb, yb, ls=ls, color=col, lw=lw, zorder=3, label=f"{et}: D={D:.4f}")
    for ax, nom, rango in ((axs[0], "primario", rangos[0]), (axs[1], "secundario", rangos[1])):
        ax.axhline(1, color="0.7", lw=0.8)
        ax.set_xlim(-rango, rango)
        ax.set_xlabel("horas desde el centro del eclipse")
        ax.set_title(f"eclipse {nom}", fontsize=10)
        ax.legend(fontsize=7, loc="lower left")
    axs[0].set_ylabel("flujo normalizado")
    lo = np.nanmin([l.get_ydata().min() for l in axs[1].get_lines() if len(l.get_ydata()) > 3 and l.get_linestyle() != "None"])
    axs[1].set_ylim(lo - 0.3 * (1 - lo), 1 + 0.4 * (1 - lo))
    lo = np.nanmin([l.get_ydata().min() for l in axs[0].get_lines() if len(l.get_ydata()) > 3 and l.get_linestyle() != "None"])
    axs[0].set_ylim(lo - 0.15 * (1 - lo), 1 + 0.25 * (1 - lo))
    fig.suptitle(f"sistema {int(fc['indice'])} · TIC {tic}   primario: {fc['marca_primario']} · "
                 f"secundario: {fc['marca_secundario']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(DIR_OUT / f"TIC_{tic}_propia_vs_tglc.png", dpi=110)
    plt.close(fig)

tabla = pd.DataFrame(filas)
tabla.to_csv(DIR_OUT / "profundidades_propia_vs_tglc.csv", index=False)
pd.set_option("display.width", 200)
print(tabla.drop(columns=["gaia"]).round(4).to_string(index=False))
