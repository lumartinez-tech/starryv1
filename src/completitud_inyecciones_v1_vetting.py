"""Completitud de la búsqueda de tránsitos por inyección y recuperación, con el pipeline del paso 1 de las Pléyades.

Pipeline de detección (igual que 1_priors_pleyades.py): curva de TESSExtractor (Flux_CBV_SS) -> wotan biweight 0.5 d -> BLS.
Añadido (criterios de la clase de Artem, Activity 2): una señal solo cuenta como detectada si además es consistente en términos de errores:
  - SDE del BLS >= umbral calibrado con curvas nulas (flujo invertido y bloques barajados) para 1 % de falsas alarmas;
  - al menos 2 tránsitos con >= 3 puntos dentro;
  - S/N con ruido rojo >= 7: la incertidumbre de la profundidad usa la dispersión de las medias de la curva en ventanas de la duración
    del tránsito (incluye el ruido correlacionado), dividida entre sqrt(n_tránsitos);
  - profundidades individuales consistentes entre sí (chi2 con p > 0.01) y pares/impares compatibles (< 3 sigma);
  - beta (RMS binneado frente a la expectativa de ruido blanco, como red_noise_analysis de Activity 2) <= 3.
Uso: python completitud_inyecciones.py [N_inyecciones_por_curva] [--nulos]
Salida: resultados/inyecciones.csv, resultados/nulos.csv, resultados/detecciones_reales.csv
"""
import glob, re, sys, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.stats import chi2 as chi2d
from wotan import flatten

warnings.filterwarnings("ignore")
BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades")
LCDIR = BASE / "tessextractor_local/tess_extractor_corr/Light_Curve_Files"
SAL = BASE / "completitud/resultados"; SAL.mkdir(parents=True, exist_ok=True)
TICS = [67787319, 114084277, 258067178, 302545345, 353927958, 385589362, 385591142, 440695918, 440700411]
EXCLUIR = {(385591142, 70)}
PARAM = pd.read_csv("/home/luciam/Documents/massage_workshop/massage.py/massageXtesss_pleyades.csv").drop_duplicates("TIC").set_index("TIC")
VENTANA, PMIN, PMAX = 0.5, 0.5, 12.0
DURS = np.array([0.03, 0.05, 0.08, 0.12, 0.18, 0.25])
SNR_MIN, P_CONS, OE_MAX, BETA_MAX = 7.0, 0.01, 3.0, 3.0
N_INY = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200


def cargar(tic, sector):
    d = pd.read_csv(LCDIR / f"tic {tic}_S_{sector}_LC.txt", sep=r"\s+").rename(columns={"#Time": "t"}).dropna(subset=["t", "Flux_CBV_SS"])
    d = d.sort_values("t").drop_duplicates("t")
    t, y = d.t.to_numpy(float), d.Flux_CBV_SS.to_numpy(float)
    return t, y / np.nanmedian(y)


CURVAS = []
for tic in TICS:
    for f in sorted(glob.glob(str(LCDIR / f"tic {tic}_S_*_LC.txt"))):
        s = int(re.search(r"_S_(\d+)_LC", f)[1])
        if s > 200 or (tic, s) in EXCLUIR:
            continue
        t, y = cargar(tic, s)
        CURVAS.append(dict(tic=tic, sector=s, t=t, y=y))


def trapecio(t, t0, P, depth, T, tau):
    """Tránsito trapezoidal periódico (Activity 2): profundidad, duración total T y ingreso/egreso tau."""
    x = np.abs(((t - t0 + 0.5 * P) % P) - 0.5 * P)
    f = np.zeros_like(t)
    lleno = x <= T / 2 - tau
    rampa = (x > T / 2 - tau) & (x < T / 2)
    f[lleno] = depth
    f[rampa] = depth * (T / 2 - x[rampa]) / tau
    return 1.0 - f


def aplanar(t, y):
    plano, _ = flatten(t, y, method="biweight", window_length=VENTANA, break_tolerance=0.5, return_trend=True)
    ok = np.isfinite(plano)
    return t[ok], plano[ok]


def bls_pico(t, plano, sig):
    base = t.max() - t.min()
    frec = np.arange(1 / PMAX, 1 / PMIN, 0.01 / base); per = np.sort(1 / frec)
    r = BoxLeastSquares(t, plano, dy=np.full_like(plano, sig)).power(per, DURS)
    p = r.power; i = int(np.argmax(p))
    return float(r.period[i]), float(r.transit_time[i]), float(r.duration[i]), float((p[i] - np.mean(p)) / np.std(p))


N_ITER = 3


def buscar_iterativo(t, y):
    """wotan + BLS iterativo: si el pico más fuerte no es consistente, se enmascara y se busca el siguiente (como el TPS de Kepler/TESS)."""
    t, plano = aplanar(t, y)
    sig = 1.4826 * np.nanmedian(np.abs(plano - np.nanmedian(plano)))
    tm, pm = t.copy(), plano.copy(); picos = []
    for it in range(N_ITER):
        if len(tm) < 100:
            break
        P, t0, dur, sde = bls_pico(tm, pm, sig)
        v = vetar(t, plano, P, t0, dur, sig)            # el vetting usa la curva completa (todos los tránsitos)
        picos.append(dict(P=P, t0=t0, dur=dur, SDE=sde, **v))
        if v["ok"]:
            break
        ph = ((tm - t0 + 0.5 * P) % P) - 0.5 * P
        k = np.abs(ph) > 1.0 * dur
        tm, pm = tm[k], pm[k]
    return picos, sig


def vetar(t, plano, P, t0, dur, sig):
    """Consistencia de la señal en términos de errores (criterios de Activity 2)."""
    ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    ep = np.round((t - t0) / P).astype(int)
    dentro = np.abs(ph) < 0.4 * dur
    base_ = np.nanmedian(plano[np.abs(ph) > 1.0 * dur])
    # ruido de las medias en ventanas de la duración (incluye ruido rojo), con las ventanas de los tránsitos excluidas
    k = np.floor((t - t.min()) / dur).astype(int)
    exc = np.abs(ph) < 1.5 * dur
    dfb = pd.DataFrame(dict(k=k[~exc], y=plano[~exc])).groupby("k").agg(m=("y", "mean"), n=("y", "size"))
    dfb = dfb[dfb.n >= 3]
    if len(dfb) < 10:
        return dict(ok=False, motivo="pocas_ventanas", snr=np.nan, p_cons=np.nan, oe=np.nan, beta=np.nan, n_tr=0, depth=np.nan)
    sT = 1.4826 * np.median(np.abs(dfb.m - np.median(dfb.m)))
    cad = np.median(np.diff(t)); m_ = max(dur / cad, 1.0)
    beta = float(sT / (sig / np.sqrt(m_)))
    d_i, e_i, n_i = [], [], []
    for e in np.unique(ep[dentro]):
        s = dentro & (ep == e)
        if s.sum() >= 3:
            d_i.append(base_ - np.mean(plano[s])); e_i.append(e); n_i.append(s.sum())
    d_i, e_i, n_i = np.array(d_i), np.array(e_i), np.array(n_i)
    n = len(d_i)
    if n < 2:
        return dict(ok=False, motivo="menos_de_2_transitos", snr=np.nan, p_cons=np.nan, oe=np.nan, beta=beta, n_tr=n, depth=np.nanmean(d_i) if n else np.nan)
    err_i = sT * np.sqrt(np.maximum(1.0, m_ / n_i))                 # error de la media de n_i puntos con ruido correlacionado a la escala de la duración
    # consistencia robusta: se descartan épocas a más de 4 sigma de la mediana (flares, bordes, luz dispersa); más de 20 % descartado = inconsistente
    med_d = np.median(d_i); z = np.abs(d_i - med_d) / err_i
    bueno = z < 4.0
    frac_malas = 1 - bueno.mean()
    d_i, e_i, err_i = d_i[bueno], e_i[bueno], err_i[bueno]
    if len(d_i) < 2:
        return dict(ok=False, motivo="menos_de_2_transitos", snr=np.nan, p_cons=np.nan, oe=np.nan, beta=beta, n_tr=len(d_i), depth=np.nan)
    w = 1 / err_i ** 2
    dep = float(np.sum(w * d_i) / np.sum(w)); sdep = float(np.sum(w) ** -0.5)
    snr = dep / sdep
    chi = float(np.sum(((d_i - dep) / err_i) ** 2)); p = float(chi2d.sf(chi, len(d_i) - 1))
    par, imp = d_i[e_i % 2 == 0], d_i[e_i % 2 == 1]
    ep_, ei_ = err_i[e_i % 2 == 0], err_i[e_i % 2 == 1]
    if len(par) and len(imp):
        dp_, di_ = np.average(par, weights=1 / ep_ ** 2), np.average(imp, weights=1 / ei_ ** 2)
        oe = abs(dp_ - di_) / np.sqrt(np.sum(ep_ ** -2) ** -1 + np.sum(ei_ ** -2) ** -1)
    else:
        oe = 0.0
    incons = (p <= P_CONS) or (frac_malas > 0.2)
    ok = (snr >= SNR_MIN) and (not incons) and (oe < OE_MAX) and (beta <= BETA_MAX)
    motivo = "ok" if ok else ",".join(k_ for k_, c in (("snr_bajo", snr < SNR_MIN), ("inconsistente", incons), ("par_impar", oe >= OE_MAX), ("ruido_rojo", beta > BETA_MAX)) if c)
    return dict(ok=bool(ok), motivo=motivo, snr=float(snr), p_cons=p, oe=float(oe), beta=beta, n_tr=len(d_i), depth=dep, f_malas=float(frac_malas))


def coincide(P, Pinj):
    for nombre, f in (("exacto", 1.0), ("mitad", 0.5), ("doble", 2.0)):
        if abs(P / (Pinj * f) - 1) < 0.01:
            return nombre
    return "no"


def tarea(a):
    """a = (índice de curva, semilla, modo). modo: 'iny' inyecta; 'nulo' invierte y baraja bloques; 'real' sin cambios."""
    ic, seed, modo = a
    c = CURVAS[ic]; rng = np.random.default_rng(seed)
    t, y = c["t"], c["y"].copy()
    par = PARAM.loc[c["tic"]]
    fila = dict(tic=c["tic"], sector=c["sector"], modo=modo, seed=seed)
    if modo == "iny":
        P = float(np.exp(rng.uniform(np.log(PMIN), np.log(PMAX)))); dep = float(np.exp(rng.uniform(np.log(5e-4), np.log(0.10))))
        k = np.sqrt(dep); b = float(rng.uniform(0, 1 - k)) if k < 1 else 0.0
        T = 13.0 / 24 * (P / 365.25) ** (1 / 3) * (par.rho) ** (-1 / 3) * np.sqrt(max((1 + k) ** 2 - b ** 2, 1e-3))
        tau = min(0.5 * T, T * k / ((1 + k) * max(1 - b ** 2, 0.05)))
        t0 = float(t.min() + rng.uniform(0, P))
        y = y * trapecio(t, t0, P, dep, T, tau)
        fila.update(P_inj=P, depth_inj=dep, T_inj=T, b=b, t0_inj=t0, Rp_RJ=k * par.Rad * 9.731)
    elif modo == "nulo":
        # flujo invertido (los eclipses reales pasan a ser picos) y bloques de 0.5 d barajados (conserva el ruido rojo)
        yi = 2 - y
        k_ = np.floor((t - t.min()) / 0.5).astype(int)
        bloques = [yi[k_ == u] for u in np.unique(k_)]
        orden = rng.permutation(len(bloques)); y = np.concatenate([bloques[j] for j in orden])[: len(t)]
        if len(y) < len(t):
            y = np.concatenate([y, np.full(len(t) - len(y), 1.0)])
    try:
        picos, sig = buscar_iterativo(t, y)
    except Exception as e:                       # noqa
        fila.update(error=str(e)[:60]); return fila
    fila["sigma"] = sig; fila["n_picos"] = len(picos)
    for k_, pk in enumerate(picos):
        for nombre, val in pk.items():
            fila[f"{nombre}_{k_}"] = val
        if modo == "iny":
            fila[f"coincide_{k_}"] = coincide(pk["P"], fila["P_inj"])
    return fila


if __name__ == "__main__":
    nulos = "--nulos" in sys.argv
    print(f"{len(CURVAS)} curvas; {N_INY} inyecciones por curva")
    with Pool(7) as pool:
        if not (SAL / "detecciones_reales.csv").exists():
            real = pool.map(tarea, [(i, 0, "real") for i in range(len(CURVAS))])
            pd.DataFrame(real).to_csv(SAL / "detecciones_reales.csv", index=False)
        print("reales listas", flush=True)
        if not (SAL / "nulos.csv").exists():
            nul = pool.map(tarea, [(i, 1000 + 17 * j + i, "nulo") for i in range(len(CURVAS)) for j in range(10)], chunksize=4)
            pd.DataFrame(nul).to_csv(SAL / "nulos.csv", index=False)
        print("nulos listos", flush=True)
        if not nulos:
            iny = pool.map(tarea, [(i, 50000 + 100000 * i + j, "iny") for i in range(len(CURVAS)) for j in range(N_INY)], chunksize=8)
            pd.DataFrame(iny).to_csv(SAL / "inyecciones.csv", index=False)
            print("inyecciones listas", flush=True)
