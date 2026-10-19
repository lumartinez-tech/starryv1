"""Construye 3_fichas_inspeccion.ipynb (kernel: Python (cumulos)).

Ficha de inspección visual por estrella para buscar binarias eclipsantes en las curvas de TESSExtractor local
(Resultados_Pleyades/tessextractor_local/resultados/dat/), siguiendo las prácticas de los catálogos de
binarias eclipsantes de Kepler y TESS y de los reportes de validación de SPOC/QLP.
Uso: python build_fichas.py
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
celdas = [
md(r"""# Pléyades 3: fichas de inspección visual de binarias eclipsantes

Una ficha por estrella con todos sus sectores, pensada para que la morfología de un eclipse sea evidente:

1. **Flujo normalizado** (no magnitudes) y eje vertical acotado por los datos promediados, no por puntos aislados.
2. **Curva cruda con su tendencia** (mediana móvil robusta en tiempo, ventana de 0.75 d, en tres pasadas que
   enmascaran las caídas para que los eclipses no arrastren la tendencia).
3. **Eventos comunes del sector** sombreados en gris: intervalos de 30 min en los que más del 30 % de las
   estrellas del mismo sector tiene una caída > 5σ (luz dispersa, descargas de momento, bordes de órbita).
   Se excluyen del análisis.
4. **Periodograma BLS** (Kovács et al. 2002), que busca caídas en forma de caja, en lugar de Lomb–Scargle, que
   con dos eclipses parecidos da P/2 y en estrellas con manchas da el periodo de rotación.
5. **Plegado con el primario en fase 0** (fase −0.25 a 0.75; el secundario de una órbita circular cae en 0.5),
   en P y en 2P, con puntos promediados. Cada campaña (S42–44, S70–71) se pliega con su propio $t_0$ para no
   arrastrar errores de fase entre años.
6. **Pares contra impares** (plegado en 2P): si las profundidades difieren, el periodo verdadero es 2P; si
   además aparece un secundario en fase 0.5 en el plegado en P, el periodo es P.

Uso: `ficha(TIC)` para una estrella; la última sección calcula el BLS de todas y ordena las candidatas.
Las curvas se pueden revisar mientras la corrida de TESSExtractor sigue avanzando: la ficha usa los sectores
que ya existan."""),
code(r"""import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares

warnings.simplefilter("ignore")
BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades")
DIR_DAT = BASE / "tessextractor_local" / "resultados" / "dat"
DIR_SALIDA = BASE / "inspeccion"
DIR_FICHAS = DIR_SALIDA / "fichas"
DIR_CACHE = DIR_SALIDA / "cache"
for d in (DIR_FICHAS, DIR_CACHE):
    d.mkdir(parents=True, exist_ok=True)

MIEMBROS = pd.read_csv(BASE / "membresia" / "pleyades_miembros_TIC82.csv").dropna(subset=["TIC"])
MIEMBROS["TIC"] = MIEMBROS["TIC"].astype(int)
MIEMBROS = MIEMBROS.drop_duplicates("TIC").set_index("TIC")
CONOCIDAS = pd.read_csv(BASE / "membresia" / "binarias_eclipsantes_conocidas.csv").dropna(subset=["TIC"])
CONOCIDAS["TIC"] = CONOCIDAS["TIC"].astype(int)
CONOCIDAS = CONOCIDAS.set_index("TIC")
# notas de la inspección (se muestran en la ficha)
NOTAS = {
    385589411: "22 Tau (A0Vn), doble visual. Sin eclipses: las caídas de S42/S71 son eventos comunes del sector.",
    440695975: "HD 23964A (B9.5 V p Si Sr Cr), binaria espectroscópica (SB9 197). Modulación de ~1.6 d por rotación (α² CVn).",
}

VENTANA_TENDENCIA = 0.75                                     # d
PASO_COMUN, FRAC_COMUN, SIGMA_COMUN = 1 / 48, 0.30, 5.0      # eventos comunes del sector
PERIODOS = (0.2, 15.0)                                       # d
DURACIONES = np.array([0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35])   # d
FACTOR_FREC, N_MAX_PERIODOS = 1.0, 200_000
SALTO_CAMPANA = 60.0                                         # d entre campañas (S42–44 / S70–71)
BIN_BLS = 10 / 1440                                          # se promedia a 10 min antes del BLS"""),
md("## 1) Lectura, tendencia y eventos comunes del sector"),
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
    f = 10 ** (-0.4 * (m - np.median(m)))
    return t, f, e * f / 1.0857


def tendencia(t, f, ventana=VENTANA_TENDENCIA, n_iter=3, paso=0.02):
    # mediana móvil en tiempo evaluada en una malla y luego interpolada; en cada pasada se enmascaran las
    # caídas (< -3σ) para que los eclipses no arrastren la tendencia
    malla = np.arange(t.min(), t.max() + paso, paso)
    masc = np.zeros(len(f), bool)
    for _ in range(n_iter):
        tu, fu = t[~masc], f[~masc]
        i0, i1 = np.searchsorted(tu, malla - ventana / 2), np.searchsorted(tu, malla + ventana / 2)
        g = np.array([np.median(fu[a:b]) if b - a > 5 else np.nan for a, b in zip(i0, i1)])
        ok = np.isfinite(g)
        tr = np.interp(t, malla[ok], g[ok])
        r = f / tr - 1
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        masc = r < -3 * s
    return tr, s


def eventos_comunes(sector, inv):
    # intervalos de PASO_COMUN en los que > FRAC_COMUN de las estrellas del sector tiene una caída > SIGMA_COMUN;
    # se guarda en caché y se recalcula cuando aumenta el número de estrellas del sector
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
            tr, s = tendencia(t, fl)
        except Exception:
            continue
        k = np.floor(t / PASO_COMUN).astype(int)
        caida = (fl / tr - 1) < -SIGMA_COMUN * s
        con_datos = con_datos.add(pd.Series(1, index=np.unique(k)), fill_value=0)
        con_caida = con_caida.add(pd.Series(1, index=np.unique(k[caida])), fill_value=0)
    if len(sel) < 15 or con_datos.empty:
        ev = pd.DataFrame(columns=["t_ini", "t_fin", "frac", "n"])
    else:
        frac = (con_caida.reindex(con_datos.index).fillna(0) / con_datos)
        k = frac.index[(frac > FRAC_COMUN) & (con_datos >= 10)].to_numpy()
        ev = pd.DataFrame(dict(t_ini=k * PASO_COMUN, t_fin=(k + 1) * PASO_COMUN, frac=frac.loc[k].to_numpy(),
                               n=con_datos.loc[k].to_numpy()))
    ev.to_csv(cache, index=False)
    return ev


def en_eventos(t, ev, margen=PASO_COMUN):
    m = np.zeros(len(t), bool)
    for a, b in zip(ev["t_ini"], ev["t_fin"]):
        m |= (t >= a - margen) & (t <= b + margen)
    return m"""),
md("## 2) BLS por campaña, plegado y profundidades"),
code(r"""def binear(t, y, paso):
    k = np.floor((t - t.min()) / paso).astype(int)
    g = pd.DataFrame(dict(k=k, t=t, y=y)).groupby("k")
    return g["t"].mean().to_numpy(), g["y"].median().to_numpy()


def correr_bls(t, r):
    tb, rb = binear(t, r, BIN_BLS)
    T = tb.max() - tb.min()
    modelo = BoxLeastSquares(tb, rb)
    periodos, potencia, duracion = [], [], []
    for pa, pb in ((PERIODOS[0], 1.0), (1.0, min(PERIODOS[1], T / 2))):
        if pb <= pa:
            continue
        dur = DURACIONES[DURACIONES < 0.9 * pa]               # astropy exige duración < periodo mínimo
        df = FACTOR_FREC * dur.min() / T ** 2
        n = int(np.clip((1 / pa - 1 / pb) / df, 2000, N_MAX_PERIODOS))
        p = 1 / np.linspace(1 / pb, 1 / pa, n)
        res = modelo.power(p, dur, objective="likelihood")
        periodos.append(p); potencia.append(res.power); duracion.append(res.duration)
    p, pw, du = map(np.concatenate, (periodos, potencia, duracion))
    o = np.argsort(p)
    p, pw, du = p[o], pw[o], du[o]
    sde = (pw - np.mean(pw)) / np.std(pw)
    i = np.argmax(sde)
    return dict(modelo=modelo, P=p[i], dur=du[i], SDE=sde[i], periodos=p, sde=sde, T=T)


def t0_en(modelo, P, dur):
    return float(modelo.power([P], [min(dur, 0.9 * P)], objective="likelihood").transit_time[0])


def fase(t, P, t0):
    return ((t - t0) / P + 0.25) % 1 - 0.25          # primario en 0, rango [-0.25, 0.75)


def profundidad(ph, r, centro, ancho):
    d = np.abs(((ph - centro + 0.5) % 1) - 0.5) < ancho / 2
    if d.sum() < 3:
        return np.nan, np.nan
    return -np.median(r[d]), 1.2533 * np.std(r[d]) / np.sqrt(d.sum())


def analizar(tic, inv=None):
    inv = inventario() if inv is None else inv
    filas = inv.loc[inv["TIC"] == tic].sort_values("Sector")
    if filas.empty:
        raise ValueError(f"TIC {tic}: no hay curvas en {DIR_DAT}")
    sec = []
    for s, f in zip(filas["Sector"], filas["archivo"]):
        t, fl, e = cargar_sector(f)
        tr, sig = tendencia(t, fl)
        ev = eventos_comunes(s, inv)
        sec.append(dict(sector=s, t=t, f=fl, tr=tr, r=fl / tr - 1, sig=sig, ev=ev, masc=en_eventos(t, ev)))
    # campañas: sectores separados por menos de SALTO_CAMPANA días
    camp, actual = [], [sec[0]]
    for a, b in zip(sec[:-1], sec[1:]):
        if b["t"].min() - a["t"].max() > SALTO_CAMPANA:
            camp.append(actual); actual = []
        actual.append(b)
    camp.append(actual)
    blss = []
    for c in camp:
        t = np.concatenate([s["t"][~s["masc"]] for s in c]); r = np.concatenate([s["r"][~s["masc"]] for s in c])
        b = correr_bls(t, r) if len(t) > 100 else None
        blss.append(b)
    validos = [b for b in blss if b is not None]
    mejor = max(validos, key=lambda b: b["SDE"])
    P, dur = mejor["P"], mejor["dur"]
    P_bls, q = P, dur / P
    # cada campaña con su propio t0; como las campañas están separadas un año, la paridad no se puede
    # seguir entre ellas: en cada una el t0 se pone en el eclipse más profundo del plegado en 2P
    t0s = [t0_en(b["modelo"], P, dur) if b is not None else np.nan for b in blss]
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
    # impares ≠ pares: el BLS encontró P/2 (dos eclipses distintos); el periodo es 2P
    duplicado = bool(np.isfinite(sig_ip) and sig_ip > 3)
    if duplicado:
        P, q = 2 * P, q / 2
    ph, _, _ = plegar(camp, t0s, P)
    d1, e1 = profundidad(ph, rr, 0.0, q)
    d2, e2 = profundidad(ph, rr, 0.5, q)
    res = dict(TIC=tic, sectores=",".join(str(s["sector"]) for s in sec), P=P, P_BLS=P_bls,
               periodo_duplicado=duplicado, dur_h=dur * 24, SDE=mejor["SDE"],
               D_prim=d1, e_D_prim=e1, snr_prim=d1 / e1 if e1 > 0 else np.nan, D_sec=d2, e_D_sec=e2, sig_sec=d2 / e2 if e2 > 0 else np.nan,
               D_impar=dimp, D_par=dpar, sig_impar_par=sig_ip,
               rms_ppt=np.median([s["sig"] for s in sec]) * 1e3,
               frac_enmascarada=np.mean(np.concatenate([s["masc"] for s in sec])))
    ph_otro, _, _ = plegar(camp, t0s, P / 2 if duplicado else 2 * P)
    return res, dict(sec=sec, camp=camp, blss=blss, mejor=mejor, t0s=t0s, ph=ph, ph_otro=ph_otro,
                     ph_ip=ph2, r=rr, s=ss)


def plegar(camp, t0s, P):
    ph, rr, ss = [], [], []
    for c, t0 in zip(camp, t0s):
        if not np.isfinite(t0):
            continue
        for s in c:
            ok = ~s["masc"]
            ph.append(fase(s["t"][ok], P, t0)); rr.append(s["r"][ok]); ss.append(np.full(ok.sum(), s["sector"]))
    return tuple(map(np.concatenate, (ph, rr, ss)))"""),
md("## 3) La ficha"),
code(r"""COLORES = dict(zip([18, 42, 43, 44, 58, 70, 71], plt.cm.tab10.colors))


def lim_y(y, paso_pts=15, margen=0.15):
    # límites a partir de los datos promediados (ignoran puntos aislados pero incluyen los eclipses)
    n = max(len(y) // paso_pts, 1)
    yb = np.array([np.median(y[i:i + paso_pts]) for i in range(0, len(y), paso_pts)]) if n > 1 else y
    lo, hi = min(np.min(yb), np.percentile(y, 0.5)), max(np.max(yb), np.percentile(y, 99.5))
    d = (hi - lo) or 1e-3
    return lo - margen * d, hi + margen * d


def plegada(ax, ph, r, s, titulo, P, nbins=200, mark_sec=True):
    for sec in np.unique(s):
        m = s == sec
        ax.plot(ph[m], 1 + r[m], ".", ms=1.5, alpha=0.35, color=COLORES.get(sec, "0.5"), label=f"S{sec}")
    b = np.linspace(-0.25, 0.75, nbins + 1)
    k = np.digitize(ph, b) - 1
    med = np.array([np.median(r[k == i]) if np.any(k == i) else np.nan for i in range(nbins)])
    ax.plot(0.5 * (b[1:] + b[:-1]), 1 + med, "k-", lw=1.2)
    ax.axvline(0, c="r", lw=0.6, ls=":")
    if mark_sec:
        ax.axvline(0.5, c="b", lw=0.6, ls=":")
    ax.set_xlim(-0.25, 0.75); ax.set_ylim(*lim_y(np.r_[1 + med[np.isfinite(med)], 1 + r]))
    ax.set_xlabel("fase"); ax.set_title(f"{titulo}  (P = {P:.5f} d)", fontsize=10)


def ficha(tic, guardar=True, mostrar=True, inv=None):
    res, d = analizar(tic, inv)
    camp, sec = d["camp"], d["sec"]
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1.2, 1.2], hspace=0.45, wspace=0.28)
    anchos = [max(c[-1]["t"].max() - c[0]["t"].min(), 5) for c in camp]
    g0 = gs[0, :].subgridspec(1, len(camp), width_ratios=anchos, wspace=0.08)
    g1 = gs[1, :].subgridspec(1, len(camp), width_ratios=anchos, wspace=0.08)
    for j, c in enumerate(camp):
        a0, a1 = fig.add_subplot(g0[0, j]), fig.add_subplot(g1[0, j])
        for s in c:
            col = COLORES.get(s["sector"], "0.4")
            a0.plot(s["t"], s["f"], ".", ms=1, color=col, alpha=0.4)
            tr_ = s["tr"].copy(); tr_[np.r_[False, np.diff(s["t"]) > 0.5]] = np.nan   # sin líneas en los huecos
            a0.plot(s["t"], tr_, "-", color="k", lw=0.8)
            tb, rb = binear(s["t"], s["r"], 30 / 1440)
            a1.plot(s["t"], 1 + s["r"], ".", ms=1, color=col, alpha=0.3)
            a1.plot(tb, 1 + rb, ".", ms=3, color=col)
            for a, b in zip(s["ev"]["t_ini"], s["ev"]["t_fin"]):
                if s["t"].min() - 0.1 < a < s["t"].max() + 0.1:
                    for ax in (a0, a1):
                        ax.axvspan(a, b, color="0.6", alpha=0.35, lw=0)
        sects = ", ".join(f"S{s['sector']}" for s in c)
        a0.set_title(f"crudo con tendencia ({sects})", fontsize=9); a1.set_title(f"sin tendencia ({sects})", fontsize=9)
        f_all, r_all = np.concatenate([s["f"] for s in c]), np.concatenate([s["r"] for s in c])
        a0.set_ylim(*lim_y(f_all)); a1.set_ylim(*lim_y(1 + r_all))
        a1.set_xlabel("BTJD")
        if j == 0:
            a0.set_ylabel("flujo normalizado"); a1.set_ylabel("flujo / tendencia")
        # marcas de los eclipses primarios predichos
        t0 = d["t0s"][j]
        if np.isfinite(t0):
            tmin, tmax = c[0]["t"].min(), c[-1]["t"].max()
            n = np.arange(np.ceil((tmin - t0) / res["P"]), np.floor((tmax - t0) / res["P"]) + 1)
            for tt in t0 + n * res["P"]:
                a1.axvline(tt, c="r", lw=0.4, alpha=0.5, ymax=0.06)
    # periodograma
    ab = fig.add_subplot(gs[2, 0:2])
    for j, b in enumerate(d["blss"]):
        if b is not None:
            ab.plot(b["periodos"], b["sde"], lw=0.6, label=", ".join(f"S{s['sector']}" for s in camp[j]))
    P, Pb = res["P"], res["P_BLS"]
    for m_, ls in ((1, "-"), (2, "--"), (0.5, ":")):
        ab.axvline(Pb * m_, c="r", lw=0.8, ls=ls, alpha=0.6)
    if tic in CONOCIDAS.index and np.isfinite(CONOCIDAS.loc[tic, "P_VSX"]):
        ab.axvline(CONOCIDAS.loc[tic, "P_VSX"], c="g", lw=1.2, alpha=0.6, label="P de VSX")
    ab.set_xscale("log"); ab.set_xlabel("periodo [d]"); ab.set_ylabel("SDE (BLS)")
    ab.set_title(f"BLS: P = {Pb:.5f} d, SDE = {res['SDE']:.1f} (rojo: P_BLS, 2P --, P/2 ··)", fontsize=10)
    ab.legend(fontsize=8)
    plegada(fig.add_subplot(gs[2, 2]), d["ph"], d["r"], d["s"], "plegada en P" + (" = 2 P_BLS" if res["periodo_duplicado"] else ""), P)
    if res["periodo_duplicado"]:
        plegada(fig.add_subplot(gs[2, 3]), d["ph_otro"], d["r"], d["s"], "plegada en P/2 (= P_BLS)", P / 2)
    else:
        plegada(fig.add_subplot(gs[2, 3]), d["ph_otro"], d["r"], d["s"], "plegada en 2P", 2 * P)
    # pares contra impares y zoom del secundario
    q = res["dur_h"] / 24 / P
    qb = res["dur_h"] / 24 / Pb
    az = fig.add_subplot(gs[3, 0])
    w = max(1.5 * qb, 0.03)
    for centro, nombre, col in ((0.0, "impar", "C0"), (0.5, "par", "C3")):
        x = ((d["ph_ip"] - centro + 0.5) % 1 - 0.5) * 2        # fase en unidades de P_BLS
        m = np.abs(x) < w
        az.plot(x[m], 1 + d["r"][m], ".", ms=2, alpha=0.4, color=col, label=nombre)
    az.set_xlim(-w, w); az.legend(fontsize=8); az.set_xlabel("fase (P_BLS) alrededor del eclipse")
    az.set_title(f"impares vs pares: {res['sig_impar_par']:.1f}σ", fontsize=10)
    a2 = fig.add_subplot(gs[3, 1])
    x = (d["ph"] - 0.5)
    m = np.abs(x) < max(3 * q, 0.06)
    a2.plot(x[m], 1 + d["r"][m], ".", ms=2, alpha=0.4, color="0.3")
    if m.sum() > 20:
        xb, yb = binear(x[m] + 1, d["r"][m], max(q / 4, 0.002))
        a2.plot(xb - 1, 1 + yb, "k.-", ms=3, lw=0.8)
    a2.set_xlabel("fase − 0.5"); a2.set_title(f"secundario: D = {res['D_sec'] * 1e3:.2f} ± {res['e_D_sec'] * 1e3:.2f} ppt", fontsize=10)
    # información
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
    info += [f"sectores: {res['sectores']}   rms por punto ≈ {res['rms_ppt']:.2f} ppt   "
             f"enmascarado por eventos comunes: {100 * res['frac_enmascarada']:.1f} %",
             f"S/N del primario = {res['snr_prim']:.1f}",
             f"P = {P:.5f} d" + (f" (= 2 × P_BLS = 2 × {Pb:.5f}: impares ≠ pares)" if res["periodo_duplicado"] else " (= P_BLS)")
             + f"   duración = {res['dur_h']:.1f} h   SDE = {res['SDE']:.1f}",
             f"D_prim = {res['D_prim'] * 1e3:.2f} ± {res['e_D_prim'] * 1e3:.2f} ppt   "
             f"D_sec (fase 0.5) = {res['D_sec'] * 1e3:.2f} ± {res['e_D_sec'] * 1e3:.2f} ppt ({res['sig_sec']:.1f}σ)",
             f"impares {res['D_impar'] * 1e3:.2f} ppt vs pares {res['D_par'] * 1e3:.2f} ppt ({res['sig_impar_par']:.1f}σ)",
             "",
             "Lectura: si impares ≠ pares (>3σ) el periodo se duplica automáticamente; el secundario queda en",
             "fase 0.5 (o desplazado si la órbita es excéntrica);",
             "caídas que coinciden con zonas grises → instrumentales; modulación sinusoidal → rotación/elipsoidal."]
    ai.text(0, 1, "\n".join(info), va="top", fontsize=10, family="monospace", wrap=True)
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
res = ficha(125754991, inv=INV)       # V1229 Tau, control
pd.Series(res).to_frame().T"""),
code(r"""# navegador: elige un TIC de la lista (ordenada por S/N del primario si ya existe resumen_bls.csv; si no, por Tmag)
ruta_res = DIR_SALIDA / "resumen_bls.csv"
if ruta_res.exists():
    orden = pd.read_csv(ruta_res).sort_values("snr_prim", ascending=False)["TIC"].tolist()
else:
    orden = sorted(INV["TIC"].unique(), key=lambda t: MIEMBROS["Tmag"].get(t, 99))
try:
    import ipywidgets as w
    w.interact_manual(lambda TIC: ficha(TIC, guardar=True, inv=INV), TIC=w.Dropdown(options=orden))
except ImportError:
    print("ipywidgets no está instalado: usa ficha(TIC). Primeros de la lista:", orden[:20])"""),
md(r"""## 5) BLS de todas las estrellas y lista de candidatas

Corre `analizar()` para todas las estrellas con curva (en paralelo) y guarda `resumen_bls.csv`. Conviene correrlo
cuando termine TESSExtractor; antes, calcula solo con los sectores que ya existan. Las primeras pasadas también
calculan los eventos comunes de cada sector (quedan en `cache/`).

La lista se ordena por la señal a ruido de la profundidad del primario plegado (`snr_prim`), no por el SDE del
BLS: en estrellas sin eclipses (rotación, ruido) el SDE también puede salir alto. Criterio orientativo de
candidata: `snr_prim` alto, profundidad mayor que ~3 veces el rms por punto y poca fracción enmascarada; la decisión final es visual, con la ficha."""),
code(r"""N_PROC = 3
CORRER_TODO = False      # cambiar a True para correr


def _uno(tic):
    try:
        return analizar(tic, INV)[0]
    except Exception as e:
        return dict(TIC=tic, error=str(e))


if CORRER_TODO:
    INV = inventario()
    for s in sorted(INV["Sector"].unique()):      # eventos comunes primero (una vez por sector)
        eventos_comunes(s, INV)
    tics = sorted(INV["TIC"].unique())
    with ProcessPoolExecutor(N_PROC) as ex:
        filas = list(ex.map(_uno, tics, chunksize=4))
    resumen = pd.DataFrame(filas)
    resumen["Tmag"] = resumen["TIC"].map(MIEMBROS["Tmag"])
    resumen["VSX"] = resumen["TIC"].map(CONOCIDAS["VSX"])
    resumen = resumen.sort_values("snr_prim", ascending=False)
    resumen.to_csv(DIR_SALIDA / "resumen_bls.csv", index=False)
    print(f"{len(resumen)} estrellas; errores: {resumen.get('error', pd.Series(dtype=str)).notna().sum()}")
    display(resumen.head(30).round(4))"""),
code(r"""# fichas (PNG en fichas/) para las N mejores del resumen, sin mostrarlas
N_FICHAS = 50
if (DIR_SALIDA / "resumen_bls.csv").exists() and CORRER_TODO:
    for tic in pd.read_csv(DIR_SALIDA / "resumen_bls.csv").dropna(subset=["snr_prim"])["TIC"].head(N_FICHAS):
        ficha(int(tic), mostrar=False, inv=INV)
    print(f"fichas en {DIR_FICHAS}")"""),
]
nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata = {"kernelspec": {"display_name": "Python (cumulos)", "language": "python", "name": "cumulos"},
               "language_info": {"name": "python"}}
nbf.write(nb, "/home/luciam/Documents/starry/Resultados_Pleyades/inspeccion/3_fichas_inspeccion.ipynb")
print("ok")
