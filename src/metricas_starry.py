"""Métricas de calidad de una corrida de starry (convergencia, residuos y parámetros).

Se usa en comparacion_original_vs_corregida.ipynb para las corridas de starry_por_tic/ y de
starry_correcciones/ con exactamente el mismo cálculo.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_ORION = Path("/home/luciam/Documents/starry/Resultados_Orion")
PRIORS = pd.read_csv(RUTA_ORION / "priors_v2.csv")


def corridas(carpeta):
    """(TIC, sector, OUT_DIR) de cada notebook ejecutado en `carpeta`."""
    filas = []
    for nbp in sorted(Path(carpeta).glob("TIC_*/full_*.ipynb")):
        m = re.match(r"full_(\d+)_s(\d+)", nbp.name)
        nb = json.load(open(nbp))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        if not any(c.get("execution_count") for c in code):
            continue
        src = "".join("".join(c["source"]) for c in code)
        out = re.search(r'OUT_DIR = Path\("([^"]+)"\)', src)
        if out and Path(out.group(1)).exists():
            errores = [o for c in code for o in c.get("outputs", []) if o["output_type"] == "error"]
            filas.append(dict(TIC=int(m.group(1)), Sector=int(m.group(2)), out=Path(out.group(1)),
                              notebook=nbp.name, errores=len(errores)))
    return pd.DataFrame(filas)


def metricas(tic, sector, out):
    """Diccionario de métricas de una corrida."""
    import arviz as az
    f = dict(TIC=tic, Sector=sector)
    summ = list(out.glob("*_Summary.csv"))
    if not summ:
        return f
    S = pd.read_csv(summ[0], index_col=0)
    f["rhat_max"] = S["r_hat"].max()
    f["ess_min"] = S["ess_bulk"].min()
    for p in ["A_r", "B_r", "B_inc", "B_ecc", "B_w", "B_amp", "A_amp", "B_porb", "B_t0"]:
        if p in S.index:
            f[p] = S.loc[p, "mean"]
            f[p + "_sd"] = S.loc[p, "sd"]
    f["k"] = f["B_r"] / f["A_r"]
    if "B_ecc" in f:
        w = np.radians(f["B_w"])
        f["ecosw"], f["esinw"] = f["B_ecc"] * np.cos(w), f["B_ecc"] * np.sin(w)
        # i > 90 es el espejo de 180 - i: se compara siempre la inclinación reducida
        f["B_inc_red"] = f["B_inc"] if f["B_inc"] <= 90 else 180 - f["B_inc"]
    tr = list(out.glob("*_trace.nc"))
    if tr:
        T = az.from_netcdf(tr[0])
        div = T.sample_stats["diverging"].values
        f["div_frac"] = float(div.mean())
    q50, trd = list(out.glob("*_q50.csv")), list(out.glob("*_trending.csv"))
    r = PRIORS.loc[(PRIORS.TIC == tic) & (PRIORS.Sector == sector)]
    if q50 and trd and len(r):
        r = r.iloc[0]
        Q, D = pd.read_csv(q50[0]), pd.read_csv(trd[0])
        M = pd.merge_asof(D.sort_values("time"), Q.sort_values("time").rename(columns={"flux": "modelo"}),
                          on="time", direction="nearest", tolerance=1e-4).dropna()
        P, t0 = f.get("B_porb", r.P_orb), f.get("B_t0", r.t0)
        dur = r.Duration if r.Duration / P < 0.25 else PRIORS.loc[PRIORS.TIC == tic, "Duration"].median()
        W = dur / P
        # fase del secundario según los datos (1b); si el 1b usó otro periodo, 0.5
        phi2 = r.fase_secundario if (np.isfinite(r.fase_secundario) and abs(r.P_orb / P - 1) < 0.01) else 0.5
        x = (M.time - t0) / P
        ph1 = (x + 0.5) % 1 - 0.5
        ph2 = (x - phi2 + 0.5) % 1 - 0.5
        res = M.flat_flux - M.modelo
        fuera = (np.abs(ph1) > 0.75 * W) & (np.abs(ph2) > 0.75 * W)
        s_out = 1.4826 * np.median(np.abs(res[fuera] - np.median(res[fuera])))
        f["rms_fuera_ppt"] = 1e3 * s_out
        for nombre, ph in (("prim", ph1), ("sec", ph2)):
            nuc = np.abs(ph) < W / 4
            ecl = np.abs(ph) < W / 2
            if nuc.sum() > 3:
                f[f"res_{nombre}_ppt"] = 1e3 * np.median(res[nuc])
                f[f"res_{nombre}_sigma"] = np.median(res[nuc]) / (s_out / np.sqrt(nuc.sum()))
                f[f"chi2_{nombre}"] = float(np.mean((res[ecl] / s_out) ** 2))
    return f


def tabla(carpeta):
    c = corridas(carpeta)
    filas = []
    for r in c.itertuples():
        m = metricas(r.TIC, r.Sector, r.out)
        m.update(errores_nb=r.errores, carpeta=str(r.out))
        filas.append(m)
    return pd.DataFrame(filas)
