"""Validación del cotrending: ¿conserva los eclipses y la rotación, y reduce el ruido sistemático?

(1) Binarias eclipsantes conocidas (membresia/binarias_eclipsantes_conocidas.csv): se pliega con el periodo catalogado
    y se compara antes/después: profundidad del mínimo y scatter robusto alrededor de la curva plegada.
(2) Rotación: pico del periodograma de Lomb-Scargle (0.2-8 d) antes/después, para todas las estrellas con salida.
Salida: validacion_EB.csv, validacion_rotacion.csv y figuras/validacion_EB.png.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades")
ORIG = BASE / "tessextractor_local/resultados/dat"
COR = BASE / "cotrending/dat_cotrended"
AQUI = BASE / "cotrending"


def flujo(f, col=1):
    d = np.loadtxt(f, ndmin=2)
    d = d[np.isfinite(d[:, 0]) & np.isfinite(d[:, col])]
    return d[:, 0], 10 ** (-0.4 * (d[:, col] - np.median(d[:, col])))


def plegar(t, f, P, nb=100):
    ph = (t / P) % 1
    k = np.floor(ph * nb).astype(int)
    g = pd.DataFrame(dict(k=k, f=f)).groupby("k").f.median().reindex(range(nb)).interpolate(limit_direction="both")
    modelo = np.interp(ph, (np.arange(nb) + 0.5) / nb, g.to_numpy(), period=1)
    r = f - modelo
    scat = 1.4826 * np.median(np.abs(r - np.median(r)))
    suav = g.rolling(3, center=True, min_periods=1).median().to_numpy()
    prof = np.median(suav) - suav.min()
    return prof, scat, (np.arange(nb) + 0.5) / nb, g.to_numpy(), int(np.argmin(suav))


def validar_EB():
    cat = pd.read_csv(BASE / "membresia/binarias_eclipsantes_conocidas.csv")
    cat["P"] = cat[["P_VSX", "P_TESS_EB", "P_Gaia_EB"]].bfill(axis=1).iloc[:, 0]
    filas, plots = [], []
    for r in cat.dropna(subset=["TIC", "P"]).itertuples():
        for f in sorted(COR.glob(f"TIC_{int(r.TIC)}_S_*_LC.dat")):
            s = int(re.search(r"_S_(\d+)_", f.name)[1]); fo = ORIG / f.name
            if not fo.exists():
                continue
            t0, f0 = flujo(fo, 1); t1, f1 = flujo(f, 1)
            P = float(r.P)
            p0, s0, x, y0, _ = plegar(t0, f0, P); p1, s1, _, y1, _ = plegar(t1, f1, P)
            filas.append(dict(TIC=int(r.TIC), Sector=s, Tmag=r.Tmag, P=P, prof_antes_ppt=1e3 * p0, prof_despues_ppt=1e3 * p1,
                              razon_prof=p1 / p0 if p0 > 0 else np.nan, scat_fold_antes_ppt=1e3 * s0, scat_fold_despues_ppt=1e3 * s1,
                              razon_scat=s1 / s0))
            plots.append((int(r.TIC), s, P, x, y0, y1, p0))
    d = pd.DataFrame(filas); d.to_csv(AQUI / "validacion_EB.csv", index=False)
    if plots:
        plots = sorted(plots, key=lambda p: -p[6])[:12]
        n = len(plots); nc = 3; nr = int(np.ceil(n / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(12, 2.8 * nr), squeeze=False)
        for a, (tic, s, P, x, y0, y1, _) in zip(ax.ravel(), plots):
            a.plot(x, y0, "o-", ms=2, lw=0.6, color="0.5", label="antes"); a.plot(x, y1, "o-", ms=2, lw=0.8, color="C3", label="cotrending")
            a.set_title(f"TIC {tic} S{s} P={P:.3f} d", fontsize=8)
        ax[0, 0].legend(fontsize=7); fig.tight_layout(); (AQUI / "figuras").mkdir(exist_ok=True)
        fig.savefig(AQUI / "figuras/validacion_EB.png", dpi=100); plt.close(fig)
    return d


def validar_rotacion():
    filas = []
    for f in sorted(COR.glob("TIC_*_S_*_LC.dat")):
        fo = ORIG / f.name
        if not fo.exists():
            continue
        t0, f0 = flujo(fo, 1); t1, f1 = flujo(f, 1)
        res = {}
        for nom, t, y in (("antes", t0, f0), ("despues", t1, f1)):
            ok = np.abs(y - 1) < 5 * 1.4826 * np.median(np.abs(y - np.median(y)))
            fr = np.linspace(1 / 8, 5, 1500)
            p = LombScargle(t[ok][::3], y[ok][::3]).power(fr)
            i = int(np.argmax(p)); res[nom] = (1 / fr[i], p[i], np.std(y[ok]))
        m = re.match(r"TIC_(\d+)_S_(\d+)_", f.name)
        filas.append(dict(TIC=int(m[1]), Sector=int(m[2]), P_antes=res["antes"][0], P_despues=res["despues"][0],
                          pot_antes=res["antes"][1], pot_despues=res["despues"][1],
                          std_antes=res["antes"][2], std_despues=res["despues"][2]))
    d = pd.DataFrame(filas)
    d["dP_rel"] = (d.P_despues - d.P_antes).abs() / d.P_antes
    d.to_csv(AQUI / "validacion_rotacion.csv", index=False)
    return d


if __name__ == "__main__":
    e = validar_EB()
    if len(e):
        print(e.round(3).to_string())
        print("razón de profundidad mediana:", e.razon_prof.median().round(3), "| razón de scatter mediana:", e.razon_scat.median().round(3))
    r = validar_rotacion()
    print(f"rotación: {len(r)} curvas; mismo periodo (±3 %) en {(r.dP_rel < 0.03).mean():.0%}; "
          f"potencia LS mediana antes {r.pot_antes.median():.3f}, después {r.pot_despues.median():.3f}")
