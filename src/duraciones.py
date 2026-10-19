"""Prueba independiente de e sin ω: ancho a media profundidad (FWHM) del eclipse primario y del secundario, medido igual
en los datos (flujo aplanado de la corrida) y en el modelo mediano de starry (q50), plegados con la efeméride del ajuste.

Primer orden en e:  T_sec / T_prim ≈ (1 + e sin ω) / (1 − e sin ω)   =>   e sin ω ≈ (r − 1) / (r + 1).
Lo importante es la comparación datos vs modelo con el mismo método: si el modelo tiene r ≠ 1 y los datos no, la
e sin ω del ajuste no viene de las duraciones observadas. Error: bootstrap de los puntos (100 réplicas).
"""
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion")
AQUI = Path(__file__).resolve().parent
RNG = np.random.default_rng(3)
NBIN = 1500


def plegar(t, t0, P):
    return ((t - t0) / P + 0.25) % 1 - 0.25           # primario en 0, secundario cerca de 0.5


def fwhm(ph, f, centro, semi):
    m = np.abs(ph - centro) < semi
    if m.sum() < 15:
        return np.nan
    x, y = ph[m], f[m]
    bins = np.linspace(centro - semi, centro + semi, 61)
    idx = np.digitize(x, bins)
    xb = np.array([x[idx == i].mean() for i in range(1, len(bins)) if (idx == i).sum() > 0])
    yb = np.array([np.median(y[idx == i]) for i in range(1, len(bins)) if (idx == i).sum() > 0])
    if len(xb) < 10:
        return np.nan
    base = np.median(yb[np.abs(xb - centro) > 0.8 * semi]) if (np.abs(xb - centro) > 0.8 * semi).sum() > 2 else 1.0
    j = np.argmin(yb); prof = base - yb[j]
    if prof <= 0:
        return np.nan
    nivel = base - prof / 2
    izq, der = np.where(yb[:j] > nivel)[0], np.where(yb[j:] > nivel)[0]
    if len(izq) == 0 or len(der) == 0:
        return np.nan
    i1, i2 = izq[-1], j + der[0]
    xl = np.interp(nivel, [yb[i1 + 1], yb[i1]], [xb[i1 + 1], xb[i1]])
    xr = np.interp(nivel, [yb[i2 - 1], yb[i2]], [xb[i2 - 1], xb[i2]])
    return xr - xl


def medir(ph, f, c2, semi):
    return fwhm(ph, f, 0.0, semi), fwhm(ph, f, c2, semi)


T = pd.read_csv(RAIZ / "diagnostico_resultados/tabla_maestra.csv")
filas = []
for _, r in T.iterrows():
    d = Path(r.dir); base = f"TIC_{r.TIC}_S{r.Sector}"
    tr = pd.read_csv(d / f"{base}_trending.csv"); mo = pd.read_csv(d / f"{base}_q50.csv")
    s = pd.read_csv(d / f"{base}_Summary.csv", index_col=0)
    P, t0 = s.loc["B_porb", "mean"], s.loc["B_t0", "mean"]
    c2 = 0.5 + 2 / np.pi * r.ecosw                          # fase del secundario (primer orden)
    semi = min(0.12, max(0.02, 1.5 * r.rA_a / np.pi * (1 + r.k)))   # ventana ~1.5 duraciones
    phd, fd = plegar(tr.time.values, t0, P), tr.flat_flux.values
    phm, fm = plegar(mo.time.values, t0, P), mo.flux.values
    w1d, w2d = medir(phd, fd, c2, semi); w1m, w2m = medir(phm, fm, c2, semi)
    boot = []
    for _ in range(100):
        i = RNG.integers(0, len(fd), len(fd)); a, b = medir(phd[i], fd[i], c2, semi)
        boot.append(b / a if a > 0 else np.nan)
    rd, rm = w2d / w1d, w2m / w1m
    filas.append(dict(sistema=r.sistema, TIC=r.TIC, Sector=r.Sector, D1=r.D1, D2=r.D2,
                      W1_datos=w1d, W2_datos=w2d, r_datos=rd, r_datos_sd=np.nanstd(boot), r_modelo=rm,
                      esinw_dur_datos=(rd - 1) / (rd + 1), esinw_dur_modelo=(rm - 1) / (rm + 1),
                      esinw_starry=r.esinw, esinw_starry_sd=r.esinw_sd))
D = pd.DataFrame(filas)
D["esinw_dur_datos_sd"] = D.r_datos_sd / 2                     # d[(r-1)/(r+1)]/dr ≈ 1/2 cerca de r = 1
D.round(4).to_csv(AQUI / "esinw_duraciones_por_sector.csv", index=False)
pd.set_option("display.width", 250)
print(D.round(3).to_string(index=False))
