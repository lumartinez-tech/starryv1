"""Cotrending de las curvas de TESSExtractor de las Pléyades con vectores base (PCA) de las propias Pléyades.

Por sector (los datos de un sector comparten la misma rejilla de tiempos) y por grupo de magnitud:
  1. Matriz estrellas x cadencias de flujo relativo (10^(-0.4 (m - mediana))), huecos interpolados solo para la base.
  2. Base: SVD de las curvas normalizadas por su ruido punto a punto; se descartan iterativamente las estrellas
     variables (residuo grande tras los primeros componentes) y se recalcula.
  3. Para cada estrella se ajustan los K primeros componentes con regresión robusta asimétrica (se ignoran
     caídas < -3 sigma, que pueden ser eclipses, y subidas > 5 sigma). K (0..K_MAX) se elige por validación cruzada
     en bloques contiguos de tiempo (regla de parsimonia), para no ajustar variabilidad astrofísica.
  4. Se escribe el .dat corregido con el mismo formato (time, magnitude, magnitude_err, ...) y un CSV de diagnóstico.
Uso: python cotrending_pleyades.py [sectores...]
"""
import re, sys, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades")
DAT = BASE / "tessextractor_local/resultados/dat"
SALIDA = BASE / "cotrending"
DAT_OUT = SALIDA / "dat_cotrended"; DAT_OUT.mkdir(parents=True, exist_ok=True)
FIG = SALIDA / "figuras"; FIG.mkdir(exist_ok=True)
SECTORES_MIN = 30          # estrellas mínimas por sector para construir una base
GRUPOS = [(0, 12.0), (12.0, 14.0), (14.0, 99.0)]   # Tmag
K_MAX, N_BLOQUES = 6, 5

MIEMBROS = pd.read_csv(BASE / "membresia/pleyades_tessextractor_miembros.txt", sep=r"\s+", comment="#",
                       names=["TIC", "RA", "Dec", "Sector", "Tmag"]).drop_duplicates("TIC").set_index("TIC")


def cargar_sector(s):
    archivos = {}
    for f in DAT.glob(f"TIC_*_S_{s}_LC.dat"):
        if f.stat().st_size > 1000:
            archivos[int(re.match(r"TIC_(\d+)_S", f.name)[1])] = f
    tics = sorted(archivos)
    datos = {t: np.loadtxt(archivos[t], ndmin=2) for t in tics}
    # rejilla maestra: la primera curva con la longitud más común. Las estrellas con esa longitud se alinean por
    # índice de cadencia (las marcas de tiempo varían unos segundos o minutos entre cámaras, pero la cadencia i es la misma);
    # el resto, por el tiempo más cercano con tolerancia de 0.45 cadencias.
    longitudes = np.array([len(d) for d in datos.values()])
    moda = np.bincount(longitudes).argmax()
    maestra = next(d[:, 0] for d in datos.values() if len(d) == moda)
    dt = np.median(np.diff(maestra))
    malla = maestra
    F = np.full((len(tics), len(malla)), np.nan); E = np.full_like(F, np.nan); M0 = np.full_like(F, np.nan)
    for i, t in enumerate(tics):
        d = datos[t]
        if len(d) == moda:
            j = np.arange(moda)
        else:
            j = np.clip(np.searchsorted(malla, d[:, 0]), 1, len(malla) - 1)
            j = np.where(np.abs(malla[j - 1] - d[:, 0]) < np.abs(malla[j] - d[:, 0]), j - 1, j)
            bien = np.abs(malla[j] - d[:, 0]) < 0.45 * dt
            j, d = j[bien], d[bien]
        ok = np.isfinite(d[:, 1]) & np.isfinite(d[:, 2])
        j, d = j[ok], d[ok]
        f = 10 ** (-0.4 * (d[:, 1] - np.median(d[:, 1])))
        F[i, j] = f; E[i, j] = d[:, 2] * f / 1.0857; M0[i, j] = d[:, 1]
    return tics, malla, F, E, M0


def sigma_pp(y):
    d = np.diff(y[np.isfinite(y)])
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if len(d) > 10 else np.nan


def rellenar(y):
    ok = np.isfinite(y)
    return np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok]) if ok.sum() > 2 else np.zeros_like(y)


def construir_base(X, n_comp=8, n_iter=3):
    """X: estrellas x tiempo (flujo relativo - 1). Devuelve componentes (n_comp x tiempo), estrellas usadas."""
    sig = np.array([sigma_pp(x) for x in X])
    buenos = np.isfinite(sig) & (np.isfinite(X).mean(axis=1) > 0.8)
    Z = np.array([np.clip(rellenar(x) / s, -8, 8) if b else np.zeros(X.shape[1]) for x, s, b in zip(X, sig, buenos)])
    usar = buenos.copy()
    for _ in range(n_iter):
        U, S, Vt = np.linalg.svd(Z[usar] - Z[usar].mean(axis=1, keepdims=True), full_matrices=False)
        V = Vt[:n_comp]
        res = np.array([np.std(z - (z @ V.T) @ V) if b else np.inf for z, b in zip(Z - Z.mean(axis=1, keepdims=True), usar | buenos)])
        lim = np.nanmedian(res[usar]) + 3 * 1.4826 * np.nanmedian(np.abs(res[usar] - np.nanmedian(res[usar])))
        usar = buenos & (res < lim)
    U, S, Vt = np.linalg.svd(Z[usar] - Z[usar].mean(axis=1, keepdims=True), full_matrices=False)
    return Vt[:n_comp], usar, S[:n_comp] ** 2 / np.sum(S ** 2)


def ajuste_robusto(y, A, n_iter=6):
    """Mínimos cuadrados con recorte asimétrico. y: tiempo, A: tiempo x p (con columna de 1). Devuelve coeficientes."""
    ok = np.isfinite(y)
    m = ok.copy()
    for _ in range(n_iter):
        coef = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
        r = y - A @ coef
        s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m])))
        m = ok & (r > -3 * s) & (r < 5 * s)
    return coef, m


def elegir_K(y, V):
    """K por validación cruzada en bloques contiguos; regla de parsimonia (menor K con CV <= 1.01 x mínimo)."""
    n = len(y); ok = np.isfinite(y)
    bloques = np.array_split(np.arange(n), N_BLOQUES)
    cv = np.zeros(K_MAX + 1)
    for K in range(K_MAX + 1):
        A = np.column_stack([np.ones(n)] + [V[k] for k in range(K)]) if K else np.ones((n, 1))
        err = []
        for b in bloques:
            tr = np.ones(n, bool); tr[b] = False
            coef, m = ajuste_robusto(np.where(tr, y, np.nan), A)
            te = np.zeros(n, bool); te[b] = True; te &= ok
            r = (y - A @ coef)[te]
            s = 1.4826 * np.median(np.abs(r - np.median(r))) if te.sum() > 10 else np.nan
            err.append(np.mean(np.clip(r, -5 * s, 3 * s) ** 2) if np.isfinite(s) else np.nan)   # mismo recorte asimétrico
        cv[K] = np.nanmean(err)
    mejor = np.nanmin(cv)
    K = int(next(k for k in range(K_MAX + 1) if cv[k] <= 1.01 * mejor))
    return K, cv


def sigma_binado(t, y, horas=3.0):
    ok = np.isfinite(y)
    k = np.floor((t[ok] - t[ok].min()) / (horas / 24)).astype(int)
    g = pd.Series(y[ok]).groupby(k).median().to_numpy()
    d = np.diff(g)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if len(d) > 10 else np.nan


def procesar_sector(s):
    tics, t, F, E, M0 = cargar_sector(s)
    if len(tics) < SECTORES_MIN:
        print(f"S{s}: solo {len(tics)} estrellas, se omite"); return []
    tmag = MIEMBROS.reindex(tics).Tmag.to_numpy()
    filas = []; bases = {}
    for lo, hi in GRUPOS:
        sel = np.flatnonzero((tmag > lo) & (tmag <= hi))
        if len(sel) < SECTORES_MIN:
            print(f"S{s} Tmag ({lo},{hi}]: {len(sel)} estrellas < {SECTORES_MIN}; se omite"); continue
        X = F[sel] - 1
        V, usadas, var = construir_base(X)
        bases[(lo, hi)] = (V, var, len(sel), int(usadas.sum()))
        np.savez(SALIDA / f'base_S{s}_{lo}_{hi}.npz', V=V, t=t, tics=np.array(tics)[sel], usadas=usadas)
        print(f"S{s} Tmag ({lo},{hi}]: {len(sel)} estrellas, {int(usadas.sum())} en la base, var. expl. 4 primeros = {100 * var[:4].sum():.1f} %")
        for i in sel:
            y = F[i] - 1
            K, cv = elegir_K(y, V)
            A = np.column_stack([np.ones(len(t))] + [V[k] for k in range(K)]) if K else np.ones((len(t), 1))
            coef, m = ajuste_robusto(y, A)
            sist = A[:, 1:] @ coef[1:] if K else np.zeros(len(t))
            fc = (F[i] - sist); fc = fc / np.nanmedian(fc)
            with np.errstate(invalid='ignore', divide='ignore'):
                mc = np.nanmedian(M0[i]) - 2.5 * np.log10(fc)
            filas.append(dict(TIC=tics[i], Sector=s, Tmag=tmag[i], grupo=f"({lo},{hi}]", K=K,
                              sig_pp_antes_ppt=1e3 * sigma_pp(F[i]), sig_pp_despues_ppt=1e3 * sigma_pp(fc),
                              sig_3h_antes_ppt=1e3 * sigma_binado(t, F[i]), sig_3h_despues_ppt=1e3 * sigma_binado(t, fc),
                              cv0=cv[0], cvK=cv[K]))
            ok = np.isfinite(M0[i]) & np.isfinite(mc)       # flujo corregido <= 0 -> se descarta ese punto
            np.savetxt(DAT_OUT / f"TIC_{tics[i]}_S_{s}_LC.dat",
                       np.c_[t[ok], mc[ok], E[i][ok] / F[i][ok] * 1.0857, M0[i][ok], sist[ok]],
                       header="time magnitude_cotrended magnitude_err magnitude_original sistematico_restado_flujo", fmt="%.8f")
    # figura de la base
    fig, ax = plt.subplots(len(bases), 1, figsize=(10, 2.6 * max(len(bases), 1)), sharex=True, squeeze=False)
    for a, ((lo, hi), (V, var, n, nb)) in zip(ax[:, 0], bases.items()):
        for k in range(4):
            a.plot(t, V[k] / np.std(V[k]) + 5 * k, lw=0.6)
        a.set_ylabel(f"Tmag ({lo},{hi}]\n{nb}/{n} en la base", fontsize=8)
    ax[-1, 0].set_xlabel("tiempo [BTJD]"); fig.suptitle(f"Vectores base, S{s}"); fig.tight_layout()
    fig.savefig(FIG / f"base_S{s}.png", dpi=100); plt.close(fig)
    return filas


if __name__ == "__main__":
    secs = [int(a) for a in sys.argv[1:]] or [42, 43, 44, 70, 71]
    todo = []
    for s in secs:
        todo += procesar_sector(s)
    d = pd.DataFrame(todo)
    d["razon_3h"] = d.sig_3h_despues_ppt / d.sig_3h_antes_ppt
    d.to_csv(SALIDA / "diagnostico_cotrending.csv", index=False)
    print(d.groupby(["Sector", "grupo"]).agg(n=("TIC", "size"), K_med=("K", "median"), razon_3h_med=("razon_3h", "median")))
