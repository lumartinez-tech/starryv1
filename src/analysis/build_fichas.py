"""Construye 3_fichas_inspeccion.ipynb, versión 2 (kernel: Python (cumulos)).

Ficha de inspección visual por estrella para buscar binarias eclipsantes y tránsitos en las curvas de
TESSExtractor local (Resultados_Pleyades/tessextractor_local/resultados/dat/).

Versión 2 (la 1 queda en build_fichas_v1.py / fichas_v1/):
  - la rotación se quita con LOCoR (rotadores rápidos) y un filtro notch (todas), como en la búsqueda de
    planetas jóvenes con K2 y TESS (Rizzuto et al. 2017; THYME), en lugar de una mediana móvil;
  - búsqueda con TLS (Hippke & Heller 2019) en lugar de BLS;
  - periodos cercanos a la órbita de TESS marcados como instrumentales;
  - todo en flujo normalizado y curvas normalizadas guardadas por estrella.
Uso: python build_fichas.py
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
celdas = [
md(r"""# Pléyades 3: fichas de inspección visual (versión 2)

Una ficha por estrella con todos sus sectores, **en flujo normalizado** (cada sector dividido por su mediana).

**Cómo se quita la variabilidad de la estrella** (lo que separa un eclipse de la rotación con manchas):

1. **Periodo de rotación** por sector con Lomb–Scargle, sobre la curva con las caídas enmascaradas (para que los
   eclipses no definan el "periodo de rotación").
2. **LOCoR** (*Locally Optimized Combination of Rotations*; Rizzuto et al. 2017) para los rotadores rápidos
   ($P_{\rm rot} < 2$ d y amplitud > 2 ppt): cada ciclo de rotación se modela con la mediana, a la misma fase de
   rotación, de los ciclos vecinos (±2), que tienen casi las mismas manchas. Los puntos en caídas se excluyen del
   modelo, así que un eclipse no se borra aunque la binaria esté sincronizada ($P_{\rm orb}=P_{\rm rot}$).
3. **Filtro notch** (Rizzuto et al. 2017) en todas las curvas: en una ventana de 0.5 d alrededor de cada punto se
   ajusta un polinomio de grado 2 con y sin una caída en forma de caja (duraciones de 1 a 7 h); si el BIC prefiere
   la caída (ΔBIC > 10), la tendencia es la del modelo con caída. En una segunda pasada se ajusta el polinomio
   excluyendo los puntos de las caídas detectadas.

**Búsqueda:** TLS (*Transit Least Squares*; Hippke & Heller 2019) por campaña (S42–44, S70–71), sobre la curva
sin tendencia promediada a 15 min; el periodo se duplica si impares y pares difieren (> 3σ). Los periodos
cercanos a la órbita de TESS (12.5–14.5 d y 25–29.5 d) se marcan como probablemente instrumentales.

Los primeros y últimos 0.25 d de cada segmento continuo (rampas de inicio de órbita) se excluyen de la búsqueda.

**Validación en la ficha:** eventos comunes del sector (gris), impares contra pares, secundario en fase 0.5,
número de eventos y el periodo de rotación.

Uso: `ficha(TIC)`; la sección 5 genera todas las fichas y el resumen ordenado. Las curvas normalizadas quedan en
`curvas_normalizadas/TIC_<tic>.csv` (sector, tiempo, flujo normalizado, tendencia, flujo sin tendencia)."""),
code(r"""import contextlib
import io
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares, LombScargle
from transitleastsquares import transitleastsquares

warnings.simplefilter("ignore")
BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades")
DIR_DAT = BASE / "tessextractor_local" / "resultados" / "dat"
DIR_SALIDA = BASE / "inspeccion"
DIR_FICHAS = DIR_SALIDA / "fichas_v2"
DIR_CURVAS = DIR_SALIDA / "curvas_normalizadas"
DIR_CACHE = DIR_SALIDA / "cache"
for d in (DIR_FICHAS, DIR_CURVAS, DIR_CACHE):
    d.mkdir(parents=True, exist_ok=True)

MIEMBROS = pd.read_csv(BASE / "membresia" / "pleyades_miembros_TIC82.csv").dropna(subset=["TIC"])
MIEMBROS["TIC"] = MIEMBROS["TIC"].astype(int)
MIEMBROS = MIEMBROS.drop_duplicates("TIC").set_index("TIC")
CONOCIDAS = pd.read_csv(BASE / "membresia" / "binarias_eclipsantes_conocidas.csv").dropna(subset=["TIC"])
CONOCIDAS["TIC"] = CONOCIDAS["TIC"].astype(int)
CONOCIDAS = CONOCIDAS.set_index("TIC")
# notas de la inspección visual (se muestran en la ficha)
NOTAS = {
    385589411: "22 Tau (A0Vn), doble visual. Sin eclipses: las caídas de S42/S71 son eventos comunes del sector.",
    440695975: "HD 23964A (B9.5 V p Si Sr Cr), binaria espectroscópica (SB9 197). Modulación de ~1.6 d por rotación (α² CVn).",
    84331646: "Candidata nueva (inspección visual de la versión 1): P ≈ 10.60 d, ~1 %, sin secundario visible.",
}

# eventos comunes del sector
PASO_COMUN, FRAC_COMUN, SIGMA_COMUN = 1 / 48, 0.30, 5.0
# rotación y tendencia
P_ROT_RAPIDA, AMP_ROT_MIN = 2.0, 2e-3          # LOCoR si P_rot < 2 d y amplitud > 2 ppt
LOCOR_VECINOS = 2
VENTANA_NOTCH, DBIC_NOTCH = 0.5, 10.0
DURACIONES_NOTCH = np.array([1, 2, 3, 5, 7]) / 24
BIN_TENDENCIA = 10 / 1440
# búsqueda
PERIODOS = (0.25, 15.0)
BIN_BUSQUEDA = 15 / 1440
SALTO_CAMPANA = 60.0
BORDE_SEGMENTO = 0.25                          # d excluidos al inicio y al final de cada segmento
ORBITA_TESS = ((12.5, 14.5), (25.0, 29.5))"""),
md("## 1) Lectura, eventos comunes, rotación y tendencia"),
code(r"""PATRON = re.compile(r"TIC_(\d+)_S_(\d+)_LC\.dat$")


def inventario():
    filas = [dict(TIC=int(m[1]), Sector=int(m[2]), archivo=f)
             for f in DIR_DAT.glob("TIC_*_S_*_LC.dat") if (m := PATRON.search(f.name))]
    return pd.DataFrame(filas, columns=["TIC", "Sector", "archivo"])


def cargar_sector(archivo):
    # columnas de TESSExtractor: time magnitude magnitude_err flux_ap_no_corr flux_annulus
    d = np.loadtxt(archivo, usecols=(0, 1, 2), ndmin=2)
    d = d[np.all(np.isfinite(d), axis=1)]
    d = d[np.argsort(d[:, 0])]
    t, m, e = d.T
    f = 10 ** (-0.4 * (m - np.median(m)))          # flujo normalizado a la mediana del sector
    return t, f, e * f / 1.0857


def binear(t, y, paso):
    k = np.floor((t - t.min()) / paso).astype(int)
    g = pd.DataFrame(dict(k=k, t=t, y=y)).groupby("k")
    return g["t"].mean().to_numpy(), g["y"].median().to_numpy()


def sigma_robusta(r):
    r = r[np.isfinite(r)]
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def tendencia_rapida(t, f, ventana=0.75, n_iter=3, paso=0.02):
    # mediana móvil con caídas enmascaradas (versión 1); aquí solo para los eventos comunes y la máscara inicial
    malla = np.arange(t.min(), t.max() + paso, paso)
    masc = np.zeros(len(f), bool)
    for _ in range(n_iter):
        tu, fu = t[~masc], f[~masc]
        i0, i1 = np.searchsorted(tu, malla - ventana / 2), np.searchsorted(tu, malla + ventana / 2)
        g = np.array([np.median(fu[a:b]) if b - a > 5 else np.nan for a, b in zip(i0, i1)])
        ok = np.isfinite(g)
        tr = np.interp(t, malla[ok], g[ok])
        r = f / tr - 1
        s = sigma_robusta(r)
        masc = r < -3 * s
    return tr, s


def eventos_comunes(sector, inv):
    # intervalos de PASO_COMUN en los que > FRAC_COMUN de las estrellas del sector tiene una caída > SIGMA_COMUN
    sel = inv.loc[inv["Sector"] == sector]
    cache = DIR_CACHE / f"eventos_comunes_S{sector}_n{len(sel)}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    for viejo in DIR_CACHE.glob(f"eventos_comunes_S{sector}_n*.csv"):
        viejo.unlink()
    con_datos, con_caida = pd.Series(dtype=float), pd.Series(dtype=float)
    for f in sel["archivo"]:
        try:
            t, fl, _ = cargar_sector(f)
            tr, s = tendencia_rapida(t, fl)
        except Exception:
            continue
        k = np.floor(t / PASO_COMUN).astype(int)
        caida = (fl / tr - 1) < -SIGMA_COMUN * s
        con_datos = con_datos.add(pd.Series(1, index=np.unique(k)), fill_value=0)
        con_caida = con_caida.add(pd.Series(1, index=np.unique(k[caida])), fill_value=0)
    if len(sel) < 15 or con_datos.empty:
        ev = pd.DataFrame(columns=["t_ini", "t_fin", "frac", "n"])
    else:
        frac = con_caida.reindex(con_datos.index).fillna(0) / con_datos
        k = frac.index[(frac > FRAC_COMUN) & (con_datos >= 10)].to_numpy()
        ev = pd.DataFrame(dict(t_ini=k * PASO_COMUN, t_fin=(k + 1) * PASO_COMUN, frac=frac.loc[k].to_numpy(),
                               n=con_datos.loc[k].to_numpy()))
    ev.to_csv(cache, index=False)
    return ev


def en_eventos(t, ev, margen=PASO_COMUN):
    m = np.zeros(len(t), bool)
    for a, b in zip(ev["t_ini"], ev["t_fin"]):
        m |= (t >= a - margen) & (t <= b + margen)
    return m


def rotacion(t, f, masc):
    # Lomb–Scargle sobre la curva sin las caídas; amplitud = rango del perfil plegado (5–95 %)
    tb, fb = binear(t[~masc], f[~masc], 30 / 1440)
    if len(tb) < 50:
        return np.nan, 0.0, 0.0
    fr = np.linspace(1 / 15, 1 / 0.1, 20000)
    pw = LombScargle(tb, fb).power(fr)
    i = np.argmax(pw)
    P = 1 / fr[i]
    k = ((tb / P) % 1 * 20).astype(int)
    perfil = np.array([np.median(fb[k == j]) for j in range(20) if np.any(k == j)])
    return P, float(pw[i]), float(np.percentile(perfil, 95) - np.percentile(perfil, 5))


def locor(t, f, P, masc0):
    # cada ciclo de rotación se modela con la mediana, a la misma fase, de los ciclos vecinos (±LOCOR_VECINOS);
    # los puntos en caídas no entran al modelo (los bins vacíos se interpolan en fase)
    nb = int(np.clip(P / (15 / 1440), 16, 200))
    c = np.floor((t - t.min()) / P).astype(int)
    ph = ((t - t.min()) / P) % 1
    k = np.minimum((ph * nb).astype(int), nb - 1)
    centros = (np.arange(nb) + 0.5) / nb
    masc = masc0.copy()
    for _ in range(2):
        nc = c.max() + 1
        prof = np.full((nc, nb), np.nan)
        g = pd.DataFrame(dict(c=c[~masc], k=k[~masc], f=f[~masc])).groupby(["c", "k"])["f"].median()
        prof[g.index.get_level_values(0), g.index.get_level_values(1)] = g.to_numpy()
        modelo = np.full(len(f), np.nan)
        for ci in np.unique(c):
            vec = [ci + d for d in range(-LOCOR_VECINOS, LOCOR_VECINOS + 1) if d != 0 and 0 <= ci + d < nc]
            if not vec:
                continue
            y = np.nanmedian(prof[vec], axis=0)
            ok = np.isfinite(y)
            if ok.sum() < 3:
                continue
            m = c == ci
            modelo[m] = np.interp(ph[m], np.r_[centros[ok] - 1, centros[ok], centros[ok] + 1], np.r_[y[ok], y[ok], y[ok]])
        r = f / modelo - 1
        masc = masc0 | (r < -3 * sigma_robusta(r))
    return modelo


def _ajuste(A, y):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    return c, max(float(r @ r), 1e-30)


def _base(x, n_izq, n_der):
    # parábola si hay datos a ambos lados del punto; si no (bordes de huecos), recta para no extrapolar
    if min(n_izq, n_der) >= 3:
        return np.column_stack([np.ones(len(x)), x, x * x])
    return np.column_stack([np.ones(len(x)), x])


def notch(t, y):
    # filtro notch (Rizzuto et al. 2017) sobre la curva promediada a 10 min: en cada punto, polinomio de grado 2
    # con y sin caída (caja centrada, duraciones DURACIONES_NOTCH); segunda pasada sin los puntos de las caídas
    tb, yb = binear(t, y, BIN_TENDENCIA)
    n = len(tb)
    i0 = np.searchsorted(tb, tb - VENTANA_NOTCH / 2)
    i1 = np.searchsorted(tb, tb + VENTANA_NOTCH / 2)
    caida = np.zeros(n, bool)
    tr = np.full(n, np.nan)
    for j in range(n):
        a, b = i0[j], i1[j]
        m = b - a
        if m < 8:
            continue
        x = tb[a:b] - tb[j]
        A = _base(x, j - a, b - j - 1)
        kp = A.shape[1]
        c0, rss0 = _ajuste(A, yb[a:b])
        mejor_bic, mejor_c, mejor_d = m * np.log(rss0 / m) + kp * np.log(m), c0, 0.0
        for d in DURACIONES_NOTCH:
            box = (np.abs(x) < d / 2).astype(float)
            if box.sum() < 2 or box.sum() > m - 5:
                continue
            c1, rss1 = _ajuste(np.column_stack([A, box]), yb[a:b])
            bic1 = m * np.log(rss1 / m) + (kp + 1) * np.log(m)
            if c1[kp] < 0 and bic1 < mejor_bic - DBIC_NOTCH:
                mejor_bic, mejor_c, mejor_d = bic1, c1, d
        lo, hi = np.percentile(yb[a:b], [1, 99])
        tr[j] = np.clip(mejor_c[0], lo, hi)          # sin extrapolar fuera del rango de los datos locales
        if mejor_d > 0:
            caida[(tb >= tb[j] - mejor_d / 2) & (tb <= tb[j] + mejor_d / 2)] = True
    # segunda pasada: polinomio sin los bins de caídas (notch) ni los < -3σ
    ok = np.isfinite(tr)
    r = yb / np.interp(tb, tb[ok], tr[ok]) - 1
    excl = caida | (r < -3 * sigma_robusta(r))
    tr2 = np.full(n, np.nan)
    for j in range(n):
        a, b = i0[j], i1[j]
        sel = ~excl[a:b]
        if sel.sum() < 6:
            continue
        x = tb[a:b][sel] - tb[j]
        c0, _ = _ajuste(_base(x, (x < 0).sum(), (x > 0).sum()), yb[a:b][sel])
        lo, hi = np.percentile(yb[a:b][sel], [1, 99])
        tr2[j] = np.clip(c0[0], lo, hi)
    ok = np.isfinite(tr2)
    if ok.sum() < 5:
        ok = np.isfinite(tr); tr2 = tr
    return np.interp(t, tb[ok], tr2[ok])


def preparar_sector(sector, archivo, inv):
    t, f, e = cargar_sector(archivo)
    ev = eventos_comunes(sector, inv)
    comun = en_eventos(t, ev)
    tr0, s0 = tendencia_rapida(t, f)
    masc0 = comun | ((f / tr0 - 1) < -3 * s0)
    P_rot, pot, amp = rotacion(t, f, masc0)
    rapida = bool(np.isfinite(P_rot) and P_rot < P_ROT_RAPIDA and amp > AMP_ROT_MIN and pot > 0.1)
    if rapida:
        mod = locor(t, f, P_rot, masc0)
        mod = np.where(np.isfinite(mod), mod, tr0)
    else:
        mod = np.ones_like(f)
    tr = mod * notch(t, f / mod)
    r = f / tr - 1
    # bordes de cada segmento continuo (rampas de inicio de órbita y extremos mal definidos): fuera de la búsqueda
    corte = np.where(np.diff(t) > 0.5)[0]
    ini, fin = np.r_[t[0], t[corte + 1]], np.r_[t[corte], t[-1]]
    borde = np.zeros(len(t), bool)
    for a_, b_ in zip(ini, fin):
        borde |= ((t >= a_) & (t < a_ + BORDE_SEGMENTO)) | ((t > b_ - BORDE_SEGMENTO) & (t <= b_))
    return dict(sector=sector, t=t, f=f, tr=tr, r=r, sig=sigma_robusta(r), ev=ev, masc=comun | borde,
                P_rot=P_rot, pot_rot=pot, amp_rot=amp, locor=rapida)"""),
md("## 2) Búsqueda con TLS, plegado y profundidades"),
code(r"""def correr_tls(t, r):
    tb, yb = binear(t, 1 + r, BIN_BUSQUEDA)
    T = tb.max() - tb.min()
    pmax = min(PERIODOS[1], T / 2)
    if len(tb) < 100 or pmax <= PERIODOS[0]:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        res = transitleastsquares(tb, yb).power(
            period_min=PERIODOS[0], period_max=pmax, oversampling_factor=2, duration_grid_step=1.1,
            R_star_min=0.1, R_star_max=5, M_star_min=0.1, M_star_max=5, use_threads=1,
            show_progress_bar=False, T0_fit_margin=0.01)
    if not np.isfinite(res.period):
        return None
    return dict(P=float(res.period), dur=float(res.duration), SDE=float(res.SDE), periodos=np.asarray(res.periods),
                sde=np.asarray(res.power), n_eventos=int(res.distinct_transit_count),
                modelo=BoxLeastSquares(tb, yb - 1), T=T)


def t0_en(modelo, P, dur):
    return float(modelo.power([P], [min(max(dur, 0.02), 0.9 * P)], objective="likelihood").transit_time[0])


def fase(t, P, t0):
    return ((t - t0) / P + 0.25) % 1 - 0.25          # primario en 0, rango [-0.25, 0.75)


def profundidad(ph, r, centro, ancho):
    d = np.abs(((ph - centro + 0.5) % 1) - 0.5) < ancho / 2
    if d.sum() < 3:
        return np.nan, np.nan
    return -np.median(r[d]), 1.2533 * np.std(r[d]) / np.sqrt(d.sum())


def plegar(camp, t0s, P):
    ph, rr, ss = [], [], []
    for c, t0 in zip(camp, t0s):
        if not np.isfinite(t0):
            continue
        for s in c:
            ok = ~s["masc"]
            ph.append(fase(s["t"][ok], P, t0)); rr.append(s["r"][ok]); ss.append(np.full(ok.sum(), s["sector"]))
    return tuple(map(np.concatenate, (ph, rr, ss)))


def guardar_curva(tic, sec):
    pd.concat([pd.DataFrame(dict(sector=s["sector"], tiempo_BTJD=s["t"], flujo_norm=s["f"], tendencia=s["tr"],
                                 flujo_sin_tendencia=1 + s["r"], evento_comun=s["masc"].astype(int)))
               for s in sec]).to_csv(DIR_CURVAS / f"TIC_{tic}.csv", index=False, float_format="%.7f")


def analizar(tic, inv=None):
    inv = inventario() if inv is None else inv
    filas = inv.loc[inv["TIC"] == tic].sort_values("Sector")
    if filas.empty:
        raise ValueError(f"TIC {tic}: no hay curvas en {DIR_DAT}")
    sec = [preparar_sector(s, f, inv) for s, f in zip(filas["Sector"], filas["archivo"])]
    guardar_curva(tic, sec)
    camp, actual = [], [sec[0]]
    for a, b in zip(sec[:-1], sec[1:]):
        if b["t"].min() - a["t"].max() > SALTO_CAMPANA:
            camp.append(actual); actual = []
        actual.append(b)
    camp.append(actual)
    busq = []
    for c in camp:
        t = np.concatenate([s["t"][~s["masc"]] for s in c]); r = np.concatenate([s["r"][~s["masc"]] for s in c])
        busq.append(correr_tls(t, r))
    validos = [b for b in busq if b is not None]
    if not validos:
        raise ValueError(f"TIC {tic}: TLS sin resultado")
    mejor = max(validos, key=lambda b: b["SDE"])
    P, dur = mejor["P"], mejor["dur"]
    P_tls, q = P, dur / P
    # t0 por campaña en el eclipse más profundo del plegado en 2P (la paridad no se sigue entre campañas)
    t0s = [t0_en(b["modelo"], P, dur) if b is not None else np.nan for b in busq]
    for j, (c, t0) in enumerate(zip(camp, t0s)):
        if not np.isfinite(t0):
            continue
        t = np.concatenate([s["t"][~s["masc"]] for s in c]); r = np.concatenate([s["r"][~s["masc"]] for s in c])
        da, _ = profundidad(fase(t, 2 * P, t0), r, 0.0, q / 2)
        db, _ = profundidad(fase(t, 2 * P, t0), r, 0.5, q / 2)
        if np.isfinite(db) and (not np.isfinite(da) or db > da):
            t0s[j] = t0 + P
    ph2, rr, ss = plegar(camp, t0s, 2 * P)
    dimp, eimp = profundidad(ph2, rr, 0.0, q / 2)
    dpar, epar = profundidad(ph2, rr, 0.5, q / 2)
    sig_ip = abs(dimp - dpar) / np.hypot(eimp, epar)
    duplicado = bool(np.isfinite(sig_ip) and sig_ip > 3)
    if duplicado:
        P, q = 2 * P, q / 2
    ph, _, _ = plegar(camp, t0s, P)
    d1, e1 = profundidad(ph, rr, 0.0, q)
    d2, e2 = profundidad(ph, rr, 0.5, q)
    orbita = any(a <= P <= b for a, b in ORBITA_TESS)
    rot = pd.DataFrame([dict(P=s["P_rot"], pot=s["pot_rot"], amp=s["amp_rot"]) for s in sec])
    ir = rot["pot"].idxmax()
    res = dict(TIC=tic, sectores=",".join(str(s["sector"]) for s in sec), P=P, P_TLS=P_tls,
               periodo_duplicado=duplicado, dur_h=dur * 24, SDE_TLS=mejor["SDE"], n_eventos=mejor["n_eventos"],
               D_prim=d1, e_D_prim=e1, snr_prim=d1 / e1 if e1 > 0 else np.nan,
               D_sec=d2, e_D_sec=e2, sig_sec=d2 / e2 if e2 > 0 else np.nan,
               D_impar=dimp, D_par=dpar, sig_impar_par=sig_ip, P_orbita_TESS=orbita,
               P_rot=rot.loc[ir, "P"], amp_rot_ppt=rot.loc[ir, "amp"] * 1e3,
               sectores_LOCoR=sum(s["locor"] for s in sec),
               rms_ppt=np.median([s["sig"] for s in sec]) * 1e3,
               frac_enmascarada=np.mean(np.concatenate([s["masc"] for s in sec])))
    ph_otro, _, _ = plegar(camp, t0s, P / 2 if duplicado else 2 * P)
    return res, dict(sec=sec, camp=camp, busq=busq, t0s=t0s, ph=ph, ph_otro=ph_otro, ph_ip=ph2, r=rr, s=ss)"""),
md("## 3) La ficha"),
code(r"""COLORES = dict(zip([18, 42, 43, 44, 58, 70, 71], plt.cm.tab10.colors))


def lim_y(y, paso_pts=15, margen=0.15):
    # límites a partir de los datos promediados (ignoran puntos aislados pero incluyen los eclipses)
    y = y[np.isfinite(y)]
    yb = np.array([np.median(y[i:i + paso_pts]) for i in range(0, len(y), paso_pts)]) if len(y) > paso_pts else y
    lo, hi = min(np.min(yb), np.percentile(y, 0.5)), max(np.max(yb), np.percentile(y, 99.5))
    d = (hi - lo) or 1e-3
    return lo - margen * d, hi + margen * d


def plegada(ax, ph, r, s, titulo, P, nbins=200):
    for sec in np.unique(s):
        m = s == sec
        ax.plot(ph[m], 1 + r[m], ".", ms=1.5, alpha=0.35, color=COLORES.get(sec, "0.5"))
    b = np.linspace(-0.25, 0.75, nbins + 1)
    k = np.digitize(ph, b) - 1
    med = np.array([np.median(r[k == i]) if np.any(k == i) else np.nan for i in range(nbins)])
    ax.plot(0.5 * (b[1:] + b[:-1]), 1 + med, "k-", lw=1.2)
    ax.axvline(0, c="r", lw=0.6, ls=":"); ax.axvline(0.5, c="b", lw=0.6, ls=":")
    ax.set_xlim(-0.25, 0.75); ax.set_ylim(*lim_y(np.r_[1 + med[np.isfinite(med)], 1 + r]))
    ax.set_xlabel("fase"); ax.set_ylabel("flujo normalizado"); ax.set_title(f"{titulo}  (P = {P:.5f} d)", fontsize=10)


def ficha(tic, guardar=True, mostrar=True, inv=None):
    res, d = analizar(tic, inv)
    camp = d["camp"]
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1.2, 1.2], hspace=0.45, wspace=0.3)
    anchos = [max(c[-1]["t"].max() - c[0]["t"].min(), 5) for c in camp]
    g0 = gs[0, :].subgridspec(1, len(camp), width_ratios=anchos, wspace=0.08)
    g1 = gs[1, :].subgridspec(1, len(camp), width_ratios=anchos, wspace=0.08)
    for j, c in enumerate(camp):
        a0, a1 = fig.add_subplot(g0[0, j]), fig.add_subplot(g1[0, j])
        for s in c:
            col = COLORES.get(s["sector"], "0.4")
            a0.plot(s["t"], s["f"], ".", ms=1, color=col, alpha=0.4)
            tr_ = s["tr"].copy(); tr_[np.r_[False, np.diff(s["t"]) > 0.5]] = np.nan
            a0.plot(s["t"], tr_, "-", color="k", lw=0.7)
            tb, rb = binear(s["t"], s["r"], 30 / 1440)
            a1.plot(s["t"], 1 + s["r"], ".", ms=1, color=col, alpha=0.3)
            a1.plot(tb, 1 + rb, ".", ms=3, color=col)
            for a, b in zip(s["ev"]["t_ini"], s["ev"]["t_fin"]):
                if s["t"].min() - 0.1 < a < s["t"].max() + 0.1:
                    for ax in (a0, a1):
                        ax.axvspan(a, b, color="0.6", alpha=0.35, lw=0)
            met = "LOCoR+notch" if s["locor"] else "notch"
            a0.annotate(f"S{s['sector']}: P_rot {s['P_rot']:.2f} d, {met}", (s["t"].min(), 0),
                        xycoords=("data", "axes fraction"), xytext=(2, 2), textcoords="offset points",
                        fontsize=7, color=col)
        sects = ", ".join(f"S{s['sector']}" for s in c)
        a0.set_title(f"flujo normalizado y tendencia ({sects})", fontsize=9)
        a1.set_title(f"flujo normalizado sin tendencia ({sects})", fontsize=9)
        f_all = np.concatenate([s["f"] for s in c]); r_all = np.concatenate([s["r"] for s in c])
        a0.set_ylim(*lim_y(f_all)); a1.set_ylim(*lim_y(1 + r_all))
        a1.set_xlabel("BTJD")
        if j == 0:
            a0.set_ylabel("flujo normalizado"); a1.set_ylabel("flujo / tendencia")
        t0 = d["t0s"][j]
        if np.isfinite(t0):
            tmin, tmax = c[0]["t"].min(), c[-1]["t"].max()
            n = np.arange(np.ceil((tmin - t0) / res["P"]), np.floor((tmax - t0) / res["P"]) + 1)
            for tt in t0 + n * res["P"]:
                a1.axvline(tt, c="r", lw=0.4, alpha=0.5, ymax=0.06)
    ab = fig.add_subplot(gs[2, 0:2])
    for j, b in enumerate(d["busq"]):
        if b is not None:
            ab.plot(b["periodos"], b["sde"], lw=0.6, label=", ".join(f"S{s['sector']}" for s in camp[j]))
    P, Pt = res["P"], res["P_TLS"]
    for m_, ls in ((1, "-"), (2, "--"), (0.5, ":")):
        ab.axvline(Pt * m_, c="r", lw=0.8, ls=ls, alpha=0.6)
    for a, b in ORBITA_TESS:
        ab.axvspan(a, b, color="0.85", lw=0, zorder=0)
    if tic in CONOCIDAS.index and np.isfinite(CONOCIDAS.loc[tic, "P_VSX"]):
        ab.axvline(CONOCIDAS.loc[tic, "P_VSX"], c="g", lw=1.2, alpha=0.6, label="P de VSX")
    ab.set_xscale("log"); ab.set_xlabel("periodo [d]"); ab.set_ylabel("SDE (TLS)")
    ab.set_title(f"TLS: P = {Pt:.5f} d, SDE = {res['SDE_TLS']:.1f} (gris: periodos de la órbita de TESS)", fontsize=10)
    ab.legend(fontsize=8)
    plegada(fig.add_subplot(gs[2, 2]), d["ph"], d["r"], d["s"], "plegada en P" + (" = 2 P_TLS" if res["periodo_duplicado"] else ""), P)
    if res["periodo_duplicado"]:
        plegada(fig.add_subplot(gs[2, 3]), d["ph_otro"], d["r"], d["s"], "plegada en P/2", P / 2)
    else:
        plegada(fig.add_subplot(gs[2, 3]), d["ph_otro"], d["r"], d["s"], "plegada en 2P", 2 * P)
    qb = res["dur_h"] / 24 / Pt
    az = fig.add_subplot(gs[3, 0])
    w = max(1.5 * qb, 0.03)
    for centro, nombre, col in ((0.0, "impar", "C0"), (0.5, "par", "C3")):
        x = ((d["ph_ip"] - centro + 0.5) % 1 - 0.5) * 2
        m = np.abs(x) < w
        az.plot(x[m], 1 + d["r"][m], ".", ms=2, alpha=0.4, color=col, label=nombre)
    az.set_xlim(-w, w); az.legend(fontsize=8); az.set_xlabel("fase (P_TLS) alrededor del eclipse")
    az.set_ylabel("flujo normalizado"); az.set_title(f"impares vs pares: {res['sig_impar_par']:.1f}σ", fontsize=10)
    a2 = fig.add_subplot(gs[3, 1])
    q = res["dur_h"] / 24 / P
    x = d["ph"] - 0.5
    m = np.abs(x) < max(3 * q, 0.06)
    a2.plot(x[m], 1 + d["r"][m], ".", ms=2, alpha=0.4, color="0.3")
    if m.sum() > 20:
        xb, yb = binear(x[m] + 1, d["r"][m], max(q / 4, 0.002))
        a2.plot(xb - 1, 1 + yb, "k.-", ms=3, lw=0.8)
    a2.set_xlabel("fase − 0.5"); a2.set_ylabel("flujo normalizado")
    a2.set_title(f"secundario: D = {res['D_sec'] * 1e3:.2f} ± {res['e_D_sec'] * 1e3:.2f} ppt", fontsize=10)
    ai = fig.add_subplot(gs[3, 2:]); ai.axis("off")
    info = [f"TIC {tic}"]
    if tic in MIEMBROS.index:
        mm = MIEMBROS.loc[tic]
        info.append(f"Tmag = {mm['Tmag']:.2f}   G = {mm['Gmag']:.2f}   BP−RP = {mm['BPmag'] - mm['RPmag']:.2f}   "
                    f"Rcont = {mm['Rcont']:.3f}   ϖ = {mm['PLXcorr']:.2f} mas   RUWE = {mm['ruwe']:.2f}")
    if tic in CONOCIDAS.index:
        k = CONOCIDAS.loc[tic]
        info.append(f"VSX: {k['VSX']}   P_VSX = {k['P_VSX']:.5f} d")
    if tic in NOTAS:
        info.append(f"Nota: {NOTAS[tic]}")
    info += [f"rotación: P_rot = {res['P_rot']:.3f} d, amplitud {res['amp_rot_ppt']:.1f} ppt; LOCoR en "
             f"{res['sectores_LOCoR']} de {len(d['sec'])} sectores",
             f"rms por punto ≈ {res['rms_ppt']:.2f} ppt   enmascarado (eventos comunes): {100 * res['frac_enmascarada']:.1f} %",
             f"P = {P:.5f} d" + (" (= 2 × P_TLS: impares ≠ pares)" if res["periodo_duplicado"] else " (= P_TLS)")
             + f"   duración = {res['dur_h']:.1f} h   SDE_TLS = {res['SDE_TLS']:.1f}   eventos = {res['n_eventos']}",
             f"D_prim = {res['D_prim'] * 1e3:.2f} ± {res['e_D_prim'] * 1e3:.2f} ppt (S/N {res['snr_prim']:.1f})   "
             f"D_sec = {res['D_sec'] * 1e3:.2f} ± {res['e_D_sec'] * 1e3:.2f} ppt ({res['sig_sec']:.1f}σ)",
             f"impares {res['D_impar'] * 1e3:.2f} ppt vs pares {res['D_par'] * 1e3:.2f} ppt ({res['sig_impar_par']:.1f}σ)"]
    if res["P_orbita_TESS"]:
        info.append("⚠ periodo cercano a la órbita de TESS: probablemente instrumental")
    if res["D_sec"] < -3 * res["e_D_sec"]:
        info.append("⚠ el flujo sube en fase 0.5: variabilidad sinusoidal, no eclipse")
    ai.text(0, 1, "\n".join(info), va="top", fontsize=10, family="monospace")
    fig.suptitle(f"TIC {tic}", fontsize=15, y=0.93)
    if guardar:
        fig.savefig(DIR_FICHAS / f"ficha_TIC_{tic}.png", dpi=110, bbox_inches="tight")
    if mostrar:
        plt.show()
    else:
        plt.close(fig)
    return res"""),
md("## 4) Ver una estrella"),
code(r"""INV = inventario()
print(f"{INV['TIC'].nunique()} estrellas con curva, {len(INV)} curvas (TIC, sector)")
res = ficha(84331646, inv=INV)       # candidata nueva
pd.Series(res).to_frame().T"""),
code(r"""# lista ordenada (si ya existe el resumen) para elegir qué ver
ruta_res = DIR_SALIDA / "resumen_tls.csv"
if ruta_res.exists():
    resumen = pd.read_csv(ruta_res)
    display(resumen.head(30).round(4))
# ficha(TIC) para ver cualquier otra"""),
md(r"""## 5) Todas las estrellas

Genera las fichas de todas las estrellas (en paralelo) y `resumen_tls.csv`. El orden (`puntaje`) es la S/N del
primario, puesta en cero si el flujo sube en fase 0.5 (variabilidad sinusoidal), si el periodo cae en la órbita de
TESS o si hay menos de 3 eventos. Es solo una ayuda: la decisión es visual. También se puede correr fuera del notebook con
`python generar_todas_fichas.py`."""),
code(r"""N_PROC = 5
CORRER_TODO = False      # cambiar a True para correr


def _uno(tic):
    try:
        return ficha(int(tic), mostrar=False, inv=INV)
    except Exception as e:
        return dict(TIC=int(tic), error=str(e))


def resumen_ordenado(filas):
    r = pd.DataFrame(filas)
    r["Tmag"] = r["TIC"].map(MIEMBROS["Tmag"]); r["VSX"] = r["TIC"].map(CONOCIDAS["VSX"])
    r["nota"] = r["TIC"].map(NOTAS)
    ok = ((r["D_sec"] > -3 * r["e_D_sec"]) & ~r["P_orbita_TESS"].fillna(False).astype(bool)
          & (r["n_eventos"] >= 3))
    r["puntaje"] = np.where(ok, r["snr_prim"], 0.0)
    return r.sort_values(["puntaje", "snr_prim"], ascending=False)


if CORRER_TODO:
    INV = inventario()
    for s in sorted(INV["Sector"].unique()):
        eventos_comunes(s, INV)
    with ProcessPoolExecutor(N_PROC) as ex:
        filas = list(ex.map(_uno, sorted(INV["TIC"].unique()), chunksize=2))
    resumen = resumen_ordenado(filas)
    resumen.to_csv(DIR_SALIDA / "resumen_tls.csv", index=False)
    display(resumen.head(30).round(4))"""),
]
nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata = {"kernelspec": {"display_name": "Python (cumulos)", "language": "python", "name": "cumulos"},
               "language_info": {"name": "python"}}
nbf.write(nb, "/home/luciam/Documents/starry/Resultados_Pleyades/inspeccion/3_fichas_inspeccion.ipynb")
print("ok")
