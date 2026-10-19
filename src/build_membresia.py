"""Construye 1_membresia_hdbscan.ipynb (kernel: Python (cumulos))."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
celdas = [
md(r"""# Pléyades 1: membresía con HDBSCAN

Misma metodología que para Orión (*Taller: caracterización global de las estrellas*, Parte 1), aplicada a
las Pléyades:

1. **Catálogo:** Gaia DR3 con $\sigma_\varpi/\varpi<0.2$ y $\varpi\ge1$ mas, más `fidelity_v2` y
   `parallax_corr` de ARI. Se reutiliza `~/Downloads/pleyades_sample14.csv` (mismo script del taller,
   radio 14.8°) recortado a **7°** alrededor del centro de las Pléyades, la mitad del radio total que
   reportan Kounkel & Covey (2019).
2. **Cortes:**
   - calidad del taller: `(ruwe<1.4 || fidelity_v2>0.2)`;
   - **sin corte de pre-secuencia principal**: el corte del taller, $M_{RP}<2.1(BP-RP)+2.6$, está pensado para
     regiones de 1–10 Myr; a los ~125 Myr de las Pléyades eliminaría el 84 % de sus miembros (solo pasan las
     estrellas con $BP-RP<1$);
   - precorte cinemático en paralaje solamente, $5<\varpi_{\rm corr}<10$ mas (100–200 pc), sin corte en
     movimiento propio para no presuponer dónde está el cúmulo.
3. **HDBSCAN** en $(\alpha,\delta,\varpi_{\rm corr},\mu_\alpha,\mu_\delta)$ con `RobustScaler`, `min_samples=1`,
   `cluster_selection_method="leaf"`; barrido de `min_cluster_size` de 1000 a 10 y elección en el plateau.
4. **Elección del grupo:** el más numeroso y cinemáticamente más concordante (menor dispersión interna en
   paralaje y movimiento propio)."""),
code(r"""import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import RobustScaler
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.simplefilter("ignore")
AQUI = Path("/home/luciam/Documents/starry/Resultados_Pleyades/membresia")
CATALOGO_14 = Path("/home/luciam/Downloads/pleyades_sample14.csv")
CENTRO = SkyCoord(56.65833333 * u.deg, 24.17805556 * u.deg)   # SkyCoord.from_name("Pleiades")
RADIO_DEG = 7.0
PLX_RANGO = (5.0, 10.0)
P_CORTE = 0.37
MCS_MANUAL = None          # fijar a mano tras ver el barrido
MCS_RANGO = (30, 400)"""),
md("## 1) Muestra y cortes"),
code(r"""d = pd.read_csv(CATALOGO_14)
sep = CENTRO.separation(SkyCoord(d["ra"].values * u.deg, d["dec"].values * u.deg)).deg
d = d.loc[sep < RADIO_DEG].reset_index(drop=True)
# error de magnitud de Gaia: 1.0857 / (flux_over_error); en el catálogo del taller Gerr = 1.0857 * flux_over_error
for c, n in (("Gerr", "e_Gmag"), ("Bperr", "e_BPmag"), ("Rperr", "e_RPmag")):
    d[n] = 1.0857 ** 2 / d[c]
calidad = (d["ruwe"] < 1.4) | (d["fidelity_v2"] > 0.2)
cin = (d["PLXcorr"] > PLX_RANGO[0]) & (d["PLXcorr"] < PLX_RANGO[1])
s = d.loc[calidad & cin].reset_index(drop=True)
s.to_csv(AQUI / "pleyades_7deg_cortes.csv", index=False)
print(f"{len(d)} fuentes en {RADIO_DEG}°; calidad: {calidad.sum()}; + {PLX_RANGO[0]}<plx<{PLX_RANGO[1]}: {len(s)}")"""),
md("## 2) HDBSCAN: barrido de `min_cluster_size` y ajuste"),
code(r"""cols = ["ra", "dec", "PLXcorr", "pmra", "pmdec"]
X = np.column_stack([RobustScaler().fit_transform(s[[c]]) for c in cols])
tamanos = np.arange(1000, 10, -5)
n = []
for mcs in tamanos:
    lab = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=1, cluster_selection_method="leaf").fit(X).labels_
    n.append(len(set(lab)) - (1 if -1 in lab else 0))
n = np.array(n)


def elegir_mcs(t, k):
    sel = (t >= MCS_RANGO[0]) & (t <= MCS_RANGO[1]) & (k >= 2)
    t, k = t[sel][::-1], k[sel][::-1]
    mejor, ini = (0, None), 0
    for i in range(1, len(t) + 1):
        if i == len(t) or k[i] != k[ini]:
            if t[i - 1] - t[ini] > mejor[0]:
                mejor = (t[i - 1] - t[ini], t[ini])
            ini = i
    return int(mejor[1]) if mejor[1] is not None else 100


MCS = MCS_MANUAL or elegir_mcs(tamanos, n)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tamanos, n); ax.set_xscale("log"); ax.set_yscale("log")
ax.axvline(MCS, c="r", ls="--", label=f"mcs = {MCS}")
ax.set_xlabel("min_cluster_size"); ax.set_ylabel("cúmulos recuperados"); ax.legend()
fig.tight_layout(); fig.savefig(AQUI / "hdbscan_barrido.png", dpi=120); plt.show()

cl = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=1, cluster_selection_method="leaf").fit(X)
s["Cl"], s["Cl_P"] = cl.labels_, cl.probabilities_
s.to_csv(AQUI / f"pleyades_hdbscan_mcs{MCS}.csv", index=False)
print(f"mcs = {MCS}: {len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0)} grupos")"""),
md(r"""## 3) Elección del grupo

Para cada grupo (miembros con $P>0.37$): número de miembros, mediana y dispersión robusta (MAD·1.4826) en
paralaje y movimiento propio, y la dispersión de velocidad tangencial en km/s
($v_t = 4.74\,|\mu-\langle\mu\rangle|/\varpi$), que no depende de la distancia del grupo. El grupo elegido es el más numeroso entre
los de menor dispersión cinemática."""),
code(r"""def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


filas = []
for k, g in s.loc[(s["Cl"] >= 0) & (s["Cl_P"] > P_CORTE)].groupby("Cl"):
    dmu = np.hypot(g["pmra"] - g["pmra"].median(), g["pmdec"] - g["pmdec"].median())
    vt = 4.74 * dmu / g["PLXcorr"]
    filas.append(dict(Cl=k, n=len(g), plx=g["PLXcorr"].median(), sig_plx=mad(g["PLXcorr"]),
                      pmra=g["pmra"].median(), pmdec=g["pmdec"].median(),
                      sig_pmra=mad(g["pmra"]), sig_pmdec=mad(g["pmdec"]), sig_vt_kms=float(np.median(vt) * 1.4826),
                      ra=g["ra"].median(), dec=g["dec"].median()))
grupos = pd.DataFrame(filas).sort_values("n", ascending=False)
# puntaje: más miembros y menor dispersión de velocidad tangencial
grupos["puntaje"] = np.log10(grupos["n"]) - np.log10(grupos["sig_vt_kms"])
ELEGIDO = int(grupos.sort_values("puntaje", ascending=False)["Cl"].iloc[0])
grupos.to_csv(AQUI / "grupos_resumen.csv", index=False)
print("grupo elegido:", ELEGIDO)
grupos.round(3)"""),
code(r"""miembros = s.loc[(s["Cl"] == ELEGIDO) & (s["Cl_P"] > P_CORTE)].copy()
miembros.to_csv(AQUI / "pleyades_miembros.csv", index=False)
fig, axs = plt.subplots(1, 4, figsize=(22, 5))
axs[0].scatter(s["ra"], s["dec"], s=0.5, c="0.85")
axs[1].scatter(s["pmra"], s["pmdec"], s=0.5, c="0.85")
axs[2].scatter(s["PLXcorr"], np.hypot(s["pmra"], s["pmdec"]), s=0.5, c="0.85")
axs[3].scatter(s["BPmag"] - s["RPmag"], s["Gmag"] - 5 * np.log10(1000 / s["PLXcorr"]) + 5, s=0.5, c="0.85")
for k, g in s.loc[(s["Cl"] >= 0) & (s["Cl_P"] > P_CORTE)].groupby("Cl"):
    kw = dict(s=6 if k == ELEGIDO else 2, label=f"Grp{k}" + (" (elegido)" if k == ELEGIDO else ""),
              zorder=3 if k == ELEGIDO else 2)
    axs[0].scatter(g["ra"], g["dec"], **kw)
    axs[1].scatter(g["pmra"], g["pmdec"], **kw)
    axs[2].scatter(g["PLXcorr"], np.hypot(g["pmra"], g["pmdec"]), **kw)
    axs[3].scatter(g["BPmag"] - g["RPmag"], g["Gmag"] - 5 * np.log10(1000 / g["PLXcorr"]) + 5, **kw)
axs[0].invert_xaxis(); axs[0].set_xlabel("ra"); axs[0].set_ylabel("dec")
axs[1].set_xlabel("pmra"); axs[1].set_ylabel("pmdec")
axs[2].set_xlabel("PLXcorr"); axs[2].set_ylabel("|μ|")
axs[3].invert_yaxis(); axs[3].set_xlabel("BP − RP"); axs[3].set_ylabel("M_G")
axs[3].legend(fontsize=8, markerscale=3)
fig.tight_layout(); fig.savefig(AQUI / "grupos_hdbscan.png", dpi=110); plt.show()
print(f"miembros del grupo elegido: {len(miembros)}")"""),
]
nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata = {"kernelspec": {"display_name": "Python (cumulos)", "language": "python", "name": "cumulos"},
               "language_info": {"name": "python"}}
nbf.write(nb, "/home/luciam/Documents/starry/Resultados_Pleyades/membresia/1_membresia_hdbscan.ipynb")
