"""Comprobación modelo vs datos en el eclipse primario y secundario, por TIC y sector.

Para cada (TIC, sector) usa la última corrida disponible (por fecha) entre starry_por_tic, starry_correcciones
(+ ronda2) y starry_edad_vecindad. Pliega con P y t0 de la solución, mide la profundidad de datos y modelo
en cada eclipse, la fase del secundario (datos vs modelo) y el residuo como fracción de la profundidad.
Salida: ajuste_primario_secundario.csv y una figura por TIC en figuras/.
Uso: python comprobar_ajuste.py
"""
import glob, os, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion")
AQUI = Path(__file__).resolve().parent
FIG = AQUI / "figuras"; FIG.mkdir(exist_ok=True)
PRIORS = pd.read_csv(RAIZ / "priors_v2.csv")
BASES = [("starry_por_tic", "orig"), ("starry_correcciones", "corr"),
         ("starry_correcciones/ronda2", "r2"), ("starry_edad_vecindad", "edadvec")]


def corridas():
    filas = []
    for base, ver in BASES:
        for f in glob.glob(str(RAIZ / base / "TIC_*/S*/TIC_*_S*_Summary.csv")):
            m = re.search(r"TIC_(\d+)_S(\d+)_Summary", f)
            d = Path(f).parent
            if (d / f"TIC_{m[1]}_S{m[2]}_q50.csv").exists() and (d / f"TIC_{m[1]}_S{m[2]}_trending.csv").exists():
                filas.append(dict(TIC=int(m[1]), Sector=int(m[2]), ver=ver, dir=d, mtime=os.path.getmtime(f)))
    c = pd.DataFrame(filas).sort_values("mtime")
    return c.groupby(["TIC", "Sector"]).tail(1).sort_values(["TIC", "Sector"]).reset_index(drop=True)


def curva_fase(x, y, paso):
    k = np.floor((x + 0.5) / paso).astype(int)
    g = pd.DataFrame(dict(k=k, x=x, y=y)).groupby("k")
    return g["x"].mean().to_numpy(), g["y"].median().to_numpy(), g["y"].count().to_numpy()


def fase_minimo(ph, y, excluir, paso=0.004):
    """Fase del mínimo de la curva plegada fuera del primario (|ph| > excluir), mediana móvil de 5 bins."""
    xb, yb, nb = curva_fase(ph, y, paso)
    ok = (np.abs(xb) > excluir) & (nb >= 3)
    xb, yb = xb[ok], yb[ok]
    if len(yb) < 15:
        return np.nan, np.nan
    ys = pd.Series(yb).rolling(5, center=True, min_periods=3).median().to_numpy()
    i = np.nanargmin(ys)
    return xb[i], ys[i]


def sigma_empirico(ph, res, W, excluir):
    """σ_emp: dispersión (MAD) de la mediana del residuo en ventanas de semiancho W/4 centradas en fases sin eclipse.
    Es el mismo estadístico que se mide en el núcleo del eclipse, así que incluye el ruido correlacionado (manchas)."""
    cs = np.arange(-0.5 + W / 4, 0.5 - W / 4, W / 4)
    vals = []
    for c in cs:
        if any(abs(((c - e + 0.5) % 1) - 0.5) < 1.5 * W for e in excluir):
            continue
        sel = np.abs(((ph - c + 0.5) % 1) - 0.5) < W / 4
        if sel.sum() >= 5:
            vals.append(np.median(res[sel]))
    vals = np.array(vals)
    return (1.4826 * np.median(np.abs(vals - np.median(vals))), len(vals)) if len(vals) >= 5 else (np.nan, len(vals))


def medir(c):
    tic, sec, d = int(c.TIC), int(c.Sector), c.dir
    S = pd.read_csv(d / f"TIC_{tic}_S{sec}_Summary.csv", index_col=0)["mean"]
    D = pd.read_csv(d / f"TIC_{tic}_S{sec}_trending.csv").sort_values("time")
    Q = pd.read_csv(d / f"TIC_{tic}_S{sec}_q50.csv").sort_values("time").rename(columns={"flux": "modelo"})
    M = pd.merge_asof(D, Q, on="time", direction="nearest", tolerance=1e-4).dropna()
    P, t0 = S["B_porb"], S["B_t0"]
    r = PRIORS.loc[(PRIORS.TIC == tic) & (PRIORS.Sector == sec)]
    r = r.iloc[0] if len(r) else None
    dur = np.nan
    if r is not None:
        dur = r.Duration if r.Duration / P < 0.25 else PRIORS.loc[PRIORS.TIC == tic, "Duration"].median()
    W = dur / P if np.isfinite(dur) else 0.05
    x = (M.time.to_numpy() - t0) / P
    ph = (x + 0.5) % 1 - 0.5                        # fase del primario en [-0.5, 0.5)
    y, mo = M.flat_flux.to_numpy(), M.modelo.to_numpy()
    fuera0 = np.abs(ph) > 1.0 * W
    base = np.median(y[fuera0])
    # fase del secundario: datos y modelo (mínimo de la curva plegada fuera del primario)
    ph2d, _ = fase_minimo(ph, y, 1.5 * W)
    ph2m, _ = fase_minimo(ph, mo, 1.5 * W)
    ph2p = r.fase_secundario if (r is not None and np.isfinite(r.fase_secundario)
                                 and abs(r.P_orb / P - 1) < 0.01) else np.nan
    out = dict(TIC=tic, Sector=sec, version=c.ver, P=P, ecc=S.get("B_ecc"), w=S.get("B_w"), inc=S.get("B_inc"),
               W=W, fase_sec_datos=ph2d % 1 if np.isfinite(ph2d) else np.nan,
               fase_sec_modelo=ph2m % 1 if np.isfinite(ph2m) else np.nan,
               fase_sec_priors=ph2p % 1 if np.isfinite(ph2p) else np.nan)
    sig_fuera = 1.4826 * np.median(np.abs((y - mo)[fuera0] - np.median((y - mo)[fuera0])))
    out["rms_fuera_ppt"] = 1e3 * sig_fuera
    ventanas = {"prim": ph}
    # fase 'observada' del secundario: la medida independiente de 1b; si no hay, el mínimo de los datos
    ph2c = ph2p if np.isfinite(ph2p) else ph2d
    out['fase_sec_ref'] = ph2c % 1 if np.isfinite(ph2c) else np.nan
    if np.isfinite(ph2c):
        ventanas["sec"] = (ph - ph2c + 0.5) % 1 - 0.5
    excl = [0.0] + ([ph2c] if np.isfinite(ph2c) else [])
    sig_emp, n_emp = sigma_empirico(ph, y - mo, W, excl)
    out["sigma_emp_ppt"], out["n_ventanas_emp"] = 1e3 * sig_emp, n_emp
    out["signif_emp_1b"] = r.signif_emp_secundario if r is not None else np.nan
    out["sec_detectado_1b"] = bool(r.secundario_detectado) if r is not None else False
    for nombre, f in ventanas.items():
        nuc, ecl = np.abs(f) < W / 4, np.abs(f) < W / 2
        if nuc.sum() < 5:
            continue
        dd, dm = base - np.median(y[nuc]), base - np.median(mo[nuc])
        res = np.median((y - mo)[nuc])
        out[f"prof_datos_{nombre}_ppt"] = 1e3 * dd
        out[f"prof_modelo_{nombre}_ppt"] = 1e3 * dm
        out[f"res_{nombre}_ppt"] = 1e3 * res
        out[f"res_{nombre}_pct"] = 100 * res / dd if dd > 0 else np.nan     # >0: datos más someros; <0: modelo más somero
        out[f"res_{nombre}_sigma"] = res / (sig_fuera / np.sqrt(nuc.sum()))
        out[f"chi2_{nombre}"] = float(np.mean(((y - mo)[ecl] / sig_fuera) ** 2))
        out[f"signif_{nombre}"] = dd / sig_emp if np.isfinite(sig_emp) else np.nan   # profundidad en unidades de σ_emp
        out[f"z_res_{nombre}"] = res / sig_emp if np.isfinite(sig_emp) else np.nan   # residuo en unidades de σ_emp
    return out, (M, ph, ventanas, W, base, P)


def clasificar(o):
    """Criterio: (1) el secundario solo se evalúa si 1b lo detectó (signif_emp > 5); (2) el residuo solo cuenta si es
    significativo frente al ruido empírico (|z| > 3); (3) si lo es, se clasifica por fracción de la profundidad
    (<5 % bueno, 5–15 % regular, > 15 % malo); (4) la fase del secundario del modelo debe estar a < 0.02 de la observada."""
    est = []
    for n in ("prim", "sec"):
        p, z = o.get(f"res_{n}_pct"), o.get(f"z_res_{n}")
        if n == "sec" and not o.get("sec_detectado_1b", False):
            est.append("secundario no detectado en 1b"); continue
        if p is None or not np.isfinite(p):
            est.append("sin dato"); continue
        if z is None or not np.isfinite(z):                 # sin ventanas libres de eclipse: solo criterio relativo
            a = abs(p)
            est.append(("bueno" if a < 5 else "regular" if a < 15 else "malo") + " (sin σ_emp)"); continue
        if abs(z) < 3:
            est.append("bueno (dentro del ruido)"); continue
        a = abs(p)
        est.append("bueno" if a < 5 else "regular" if a < 15 else "malo")
    dfase = o.get("fase_sec_ref"), o.get("fase_sec_modelo")
    if all(np.isfinite(v) for v in dfase):
        d = (dfase[1] - dfase[0] + 0.5) % 1 - 0.5
        o["dfase_sec"] = d
        if abs(d) > 0.02 and o.get("sec_detectado_1b", False):
            est[1] = "malo (fase)"
    o["ajuste_prim"], o["ajuste_sec"] = est
    return o


def dibujar(tic, lista):
    n = len(lista)
    fig, ax = plt.subplots(2 * n, 2, figsize=(10, 3.3 * n), sharex="col",
                           gridspec_kw=dict(height_ratios=[3, 1] * n), squeeze=False)
    for i, (o, (M, ph, vent, W, base, P)) in enumerate(lista):
        y, mo = M.flat_flux.to_numpy(), M.modelo.to_numpy()
        for j, nombre in enumerate(("prim", "sec")):
            a, b = ax[2 * i, j], ax[2 * i + 1, j]
            if nombre not in vent:
                a.text(0.5, 0.5, "secundario no medible", ha="center", va="center", transform=a.transAxes)
                continue
            f = vent[nombre]
            ok = np.abs(f) < 2.5 * W
            xb, yb, _ = curva_fase(f[ok], y[ok], max(W / 40, 0.0008))
            xm, ym, _ = curva_fase(f[ok], mo[ok], max(W / 40, 0.0008))
            a.plot(f[ok], y[ok], ".", color="0.8", ms=2, rasterized=True)
            a.plot(xb, yb, "o", color="C0", ms=3, label="datos (mediana en bins)")
            a.plot(xm, ym, "-", color="C3", lw=1.6, label="modelo")
            if nombre == 'sec' and np.isfinite(o.get('dfase_sec', np.nan)) and abs(o['dfase_sec']) > 0.02:
                a.axvline(o['dfase_sec'], color='C3', ls=':', lw=1)
                a.text(0.02, 0.05, f"secundario del modelo: fase {o['fase_sec_modelo']:.3f} (observada {o['fase_sec_ref']:.3f})", transform=a.transAxes, fontsize=7, color='C3')
            rb, _, _ = curva_fase(f[ok], (y - mo)[ok], max(W / 40, 0.0008))
            b.axhline(0, color="k", lw=0.6)
            b.plot(xb, curva_fase(f[ok], (y - mo)[ok], max(W / 40, 0.0008))[1] * 1e3, "o", color="C0", ms=3)
            b.set_ylabel("res [ppt]", fontsize=8)
            pct = o.get(f"res_{nombre}_pct", np.nan)
            a.set_title(f"S{o['Sector']} {'primario' if nombre == 'prim' else 'secundario'}: "
                        f"{o['ajuste_' + nombre]} (res = {pct:.1f} % de la prof.)", fontsize=9)
            a.set_ylabel("flujo", fontsize=8)
            if i == 0 and j == 0:
                a.legend(fontsize=7, loc="lower right")
            lo = min(np.nanmin(yb), np.nanmin(ym)); hi = base + 0.2 * (base - lo)
            a.set_ylim(lo - 0.1 * (base - lo), hi)
        ax[2 * i + 1, 0].set_xlabel("fase respecto al primario (días/P)", fontsize=8)
        ax[2 * i + 1, 1].set_xlabel("fase respecto al secundario", fontsize=8)
    fig.suptitle(f"TIC {tic}", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / f"TIC_{tic}_ajuste.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    filas, por_tic = [], {}
    for c in corridas().itertuples():
        try:
            o, aux = medir(c)
            o = clasificar(o)
        except Exception as e:
            o, aux = dict(TIC=c.TIC, Sector=c.Sector, version=c.ver, error=str(e)), None
            print("ERROR", c.TIC, c.Sector, e)
        filas.append(o)
        if aux is not None:
            por_tic.setdefault(int(c.TIC), []).append((o, aux))
    for tic, lista in por_tic.items():
        dibujar(tic, lista)
    pd.DataFrame(filas).to_csv(AQUI / "ajuste_primario_secundario.csv", index=False)
    print(len(filas), "corridas;", len(por_tic), "figuras")
