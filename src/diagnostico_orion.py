"""Diagnóstico de los resultados de starry en Orión: tabla maestra, banderas de casos atípicos y figuras.

Lee las corridas de starry_por_tic, starry_correcciones (+ ronda2, ronda3, rondaC) y starry_edad_vecindad, elige la
corrida vigente de cada (TIC, sector) con una regla explícita, deriva parámetros físicos desde los posteriors y marca
los casos inconsistentes. Pensado para ejecutarse con el kernel python3 del entorno starry (Python 3.9).
"""
import glob, os, re, sys, warnings
from collections import namedtuple
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings("ignore")
RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion")
AQUI = RAIZ / "diagnostico_resultados"
sys.path.insert(0, str(RAIZ / "ajuste_primario_secundario"))
BASES = [("starry_por_tic", "orig"), ("starry_correcciones", "corr"), ("starry_correcciones/ronda2", "r2"),
         ("starry_edad_vecindad", "edadvec"), ("starry_correcciones/ronda3", "r3"), ("starry_correcciones/rondaC", "rC")]
VIGENTES = ("orig", "corr", "r2", "edadvec")        # versiones que definen la corrida vigente por fecha
RONDAS_NUEVAS = ("r3", "rC")                        # solo reemplazan a la vigente si mejoran (ver elegir)
RSUN_POR_AU = 215.032

# ---------------------------------------------------------------- estilo (paleta de referencia del skill dataviz)
AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
TINTA, TINTA2, SUPERFICIE = "#0b0b0b", "#52514e", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("azul_seq", SEQ)
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
MARCAS = {5: "v", 6: "o", 32: "s", 33: "P", 87: "D", 98: "^"}


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "text.color": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "regular", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8.5, "figure.dpi": 110, "lines.linewidth": 1.6,
    })


estilo()


# ---------------------------------------------------------------- lectura de corridas
Fila = namedtuple("Fila", "TIC Sector ver dir")


def candidatas():
    filas = []
    for base, ver in BASES:
        for f in glob.glob(str(RAIZ / base / "TIC_*/S*/TIC_*_S*_Summary.csv")):
            m = re.search(r"TIC_(\d+)_S(\d+)_Summary", f)
            d = Path(f).parent
            if (d / f"TIC_{m[1]}_S{m[2]}_Posteriors.csv").exists():
                filas.append(dict(TIC=int(m[1]), Sector=int(m[2]), ver=ver, dir=d, mtime=os.path.getmtime(f)))
    return pd.DataFrame(filas).sort_values("mtime").reset_index(drop=True)


def q16_50_84(x):
    return np.percentile(x, [16, 50, 84])


def resumen_corrida(c):
    tic, sec, d = int(c.TIC), int(c.Sector), Path(c.dir)
    S = pd.read_csv(d / f"TIC_{tic}_S{sec}_Summary.csv", index_col=0)
    P = pd.read_csv(d / f"TIC_{tic}_S{sec}_Posteriors.csv", index_col=0)
    o = dict(TIC=tic, Sector=sec, ver=c.ver, dir=str(d), mtime=c.mtime,
             rhat_max=S["r_hat"].max(), rhat_peor=S["r_hat"].idxmax(), ess_min=S["ess_bulk"].min(), ess_peor=S["ess_bulk"].idxmin())
    der = pd.DataFrame(index=P.index)
    der["A_m"], der["B_m"], der["A_r"], der["B_r"] = P["A_m"], P["B_m"], P["A_r"], P["B_r"]
    der["P"], der["inc"], der["e"], der["w"] = P["B_porb"], P["B_inc"], P["B_ecc"], P["B_w"]
    der["ecosw"], der["esinw"] = P["B_ecosw"], P["B_esinw"]
    der["A_amp"], der["B_amp"] = P["A_amp"], P["B_amp"]
    der["q"] = P["B_m"] / P["A_m"]
    der["k"] = P["B_r"] / P["A_r"]
    der["a"] = P["a_AU"] * RSUN_POR_AU
    der["rA_a"], der["rB_a"] = der["A_r"] / der["a"], der["B_r"] / der["a"]
    der["contacto"] = (der["A_r"] + der["B_r"]) / ((1 - der["e"]) * der["a"])     # > 1: los discos se solapan en el periastro
    for v in der.columns:
        x = der[v].to_numpy(float)
        media = float(np.mean(x))                                      # las figuras usan la MEDIA posterior
        q025, q16, med, q84, q975 = np.percentile(x, [2.5, 16, 50, 84, 97.5])
        o[v], o[v + "_sd"], o[v + "_med"] = media, float(np.std(x)), med
        o[v + "_q025"], o[v + "_q16"], o[v + "_q84"], o[v + "_q975"] = q025, q16, q84, q975
        o[v + "_lo"], o[v + "_hi"] = max(media - q16, 0.0), max(q84 - media, 0.0)
    # ω es una variable circular: media circular en [-180, 180) y desviación circular
    o["w"] = float(stats.circmean(der["w"].to_numpy(float), high=180, low=-180))
    o["w_circ_sd"] = float(np.degrees(stats.circstd(np.radians(P["B_w"].to_numpy()))))
    # convergencia: divergencias
    tr = d / f"TIC_{tic}_S{sec}_trace.nc"
    o["div_frac"] = np.nan
    if tr.exists():
        try:
            import arviz as az
            T = az.from_netcdf(str(tr)); o["div_frac"] = float(T.sample_stats["diverging"].values.mean())
        except Exception:
            pass
    cz = d / f"TIC_{tic}_S{sec}_circularizacion.csv"
    if cz.exists():
        z = pd.read_csv(cz).iloc[0]
        for k in ("clasificacion_ecosw", "signif_ecosw", "abs_ecosw_95", "ecosw_fase_1b"):
            o[k] = z.get(k, np.nan)
    return o


def evaluar_ajuste(c):
    import comprobar_ajuste as CA
    row = Fila(int(c.TIC), int(c.Sector), c.ver, Path(c.dir))
    try:
        o, _ = CA.medir(row); o = CA.clasificar(o)
    except Exception as e:
        return dict(error_ajuste=str(e))
    return {k: o.get(k) for k in ("res_prim_pct", "z_res_prim", "res_sec_pct", "z_res_sec", "ajuste_prim", "ajuste_sec", "dfase_sec",
                                  "fase_sec_ref", "fase_sec_modelo", "sec_detectado_1b", "signif_emp_1b", "sigma_emp_ppt")}


import json
PRIOR_PATRONES = {"A_m": r"m_A, sd_m_A = ([-\d.eE+]+), ([-\d.eE+]+)", "A_r": r"r_A, sd_r_A = ([-\d.eE+]+), ([-\d.eE+]+)",
                  "B_m": r"m_B, sd_m_B = ([-\d.eE+]+), ([-\d.eE+]+)", "B_r": r"r_B, sd_r_B = ([-\d.eE+]+), ([-\d.eE+]+)",
                  "A_amp": r"amp_A, sd_amp_A = ([-\d.eE+]+), ([-\d.eE+]+)", "B_amp": r"amp_B, sd_amp_B = ([-\d.eE+]+), ([-\d.eE+]+)",
                  "A_inc": r"inc_A, sd_inc_A = ([-\d.eE+]+), ([-\d.eE+]+)", "B_inc": r"inc_B, sd_inc_B = ([-\d.eE+]+), ([-\d.eE+]+)"}


def priors_por_directorio():
    """Mapa directorio de salida -> priors (valor, sd) leídos del notebook que escribió ahí."""
    mapa = {}
    for base, _ in BASES:
        for f in glob.glob(str(RAIZ / base / "TIC_*/full_*.ipynb")):
            try:
                nb = json.load(open(f))
                src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
            except Exception:
                continue
            m = re.search(r'OUT_DIR = Path\("([^"]+)"\)', src)
            if not m:
                continue
            pri = {}
            for k, pat in PRIOR_PATRONES.items():
                mm = re.search(pat, src)
                if mm:
                    pri[k] = (float(mm[1]), float(mm[2]))
            mapa[os.path.normpath(m[1])] = (os.path.getmtime(f), pri)
    return {k: v[1] for k, v in mapa.items()}


def conv_ok(r, rhat=1.02, ess=400):
    return bool(r["rhat_max"] <= rhat and r["ess_min"] >= ess)


def elegir(todas):
    """Corrida vigente por (TIC, sector): la última por fecha entre las versiones de VIGENTES que converge (R̂ máx ≤ 1.05 y ESS mín ≥ 100);
    si ninguna converge, la de menor R̂. Una corrida de la ronda 3 (A) la
    reemplaza solo si su R̂ máx no es peor y su ESS mín no es menor; una de la ronda C, solo si la vigente tenía la fase del secundario a más de 0.02 de la observada y la nueva
    la deja a menos de 0.02 sin empeorar el R̂ máx más de 0.05. Devuelve (vigentes, alternativas)."""
    vig, alt = [], []
    for (tic, sec), g in todas.groupby(["TIC", "Sector"]):
        base = g[g["ver"].isin(VIGENTES)].sort_values("mtime")
        if base.empty:
            base = g.sort_values("mtime")
        ok = base[(base["rhat_max"] <= 1.05) & (base["ess_min"] >= 100)]
        elegida = ok.iloc[-1] if len(ok) else base.sort_values("rhat_max").iloc[0]     # última que converge; si ninguna, la de menor R̂
        for _, n in g[g["ver"].isin(RONDAS_NUEVAS)].sort_values("mtime").iterrows():
            if n["ver"] == "r3":
                mejor = n["rhat_max"] <= elegida["rhat_max"] and n["ess_min"] >= elegida["ess_min"]
            else:
                antes = elegida.get("dfase_sec", np.nan)
                mejor = ((abs(n.get("dfase_sec", np.inf)) < 0.02) and (n["rhat_max"] <= elegida["rhat_max"] + 0.05)
                         and (not np.isfinite(antes) or abs(antes) >= 0.02))     # solo si la vigente tenía la fase mal puesta
            alt.append(dict(TIC=tic, Sector=sec, ronda=n["ver"], rhat_max=n["rhat_max"], ess_min=n["ess_min"], div_frac=n["div_frac"],
                            rhat_vigente=elegida["rhat_max"], ess_vigente=elegida["ess_min"], ver_vigente=elegida["ver"],
                            e_ronda=n["e"], e_vigente=elegida["e"], aceptada=bool(mejor)))
            if mejor:
                elegida = n
        vig.append(elegida)
    return pd.DataFrame(vig).reset_index(drop=True), pd.DataFrame(alt)


def tabla_maestra(forzar=False):
    ruta = AQUI / "tabla_maestra.csv"
    c = candidatas()
    filas = []
    for _, r in c.iterrows():
        try:
            filas.append(resumen_corrida(r))
        except Exception as e:
            print("sin leer", r.TIC, r.Sector, r.ver, e)
    todas = pd.DataFrame(filas)
    # métricas de ajuste (modelo vs datos) de cada candidata
    aj = [evaluar_ajuste(r) for _, r in todas.iterrows()]
    todas = pd.concat([todas, pd.DataFrame(aj)], axis=1)
    vig, alt = elegir(todas)
    # datos externos
    pri = pd.read_csv(RAIZ / "priors_v2.csv").drop_duplicates(["TIC", "Sector"]).set_index(["TIC", "Sector"])
    mas = pd.read_csv(RAIZ / "resultados_pipeline/resumen_por_tic_sector.csv").drop_duplicates(["TIC", "Sector"]).set_index(["TIC", "Sector"])
    tm = pd.read_csv(RAIZ / "datos_estrellas_massage_orion/Macro_21aug2026.csv", usecols=["TIC", "Tmag"]).drop_duplicates("TIC").set_index("TIC")["Tmag"]
    ind = pd.read_csv(RAIZ / "resultados_pipeline/periodos/indice_sistemas.csv").set_index("TIC")
    idx = list(zip(vig.TIC, vig.Sector))
    for col, nom in (("P_orb", "P_orb_1b"), ("P_rot", "P_rot_1b"), ("Depth_primary", "D1"), ("Depth_secondary", "D2"),
                     ("secundario_detectado", "sec_det"), ("signif_emp_secundario", "signif_sec_1b"), ("ecosw", "ecosw_1b"), ("ecosw_err", "ecosw_1b_err")):
        vig[nom] = [pri[col].get(i, np.nan) if i in pri.index else np.nan for i in idx]
    for col, nom in (("M_A", "M_A_massage"), ("M_A_err", "M_A_massage_err"), ("R_A", "R_A_massage"), ("R_A_err", "R_A_massage_err"),
                     ("M_B", "M_B_massage"), ("M_B_err", "M_B_massage_err"), ("R_B", "R_B_massage"), ("Teff_A", "Teff_A"), ("Teff_B", "Teff_B"),
                     ("edad_A_Myr", "edad_Myr")):
        vig[nom] = [mas[col].get(i, np.nan) if (i in mas.index and col in mas.columns) else np.nan for i in idx]
    # edad: la de massage, salvo en los sistemas cuya edad se reasignó por vecindad (las corridas edadvec ya usan la nueva)
    iv = pd.read_csv(RAIZ / "starry_edad_vecindad/indice_edad_vecindad.csv")
    nueva = {(int(t), int(s)): float(e) for t, s, e in zip(iv.TIC, iv.Sector, iv.edad_Myr)}
    vig["edad_fuente"] = "massage"
    for i, (t, s) in enumerate(idx):
        if (int(t), int(s)) in nueva:
            vig.loc[i, "edad_Myr"] = nueva[(int(t), int(s))]; vig.loc[i, "edad_fuente"] = "vecindad"
    vig["Tmag"] = vig.TIC.map(tm)
    mp = priors_por_directorio()
    for k in PRIOR_PATRONES:
        vig["prior_" + k] = [mp.get(os.path.normpath(d), {}).get(k, (np.nan, np.nan))[0] for d in vig["dir"]]
        vig["prior_" + k + "_sd"] = [mp.get(os.path.normpath(d), {}).get(k, (np.nan, np.nan))[1] for d in vig["dir"]]
    vig["sistema"] = vig.TIC.map(ind["indice"]).astype("Int64")
    vig["P_rot_consenso"] = vig.TIC.map(ind["P_rot"])
    vig["P_orb_consenso"] = vig.TIC.map(ind["P_orb"])
    vig["etiqueta"] = [f"{int(s)}" if pd.notna(s) else str(t)[-4:] for s, t in zip(vig["sistema"], vig.TIC)]
    vig = banderas(vig)
    vig.to_csv(ruta, index=False); todas.to_csv(AQUI / "todas_las_corridas.csv", index=False); alt.to_csv(AQUI / "rondas_A_C.csv", index=False)
    return vig, todas, alt


# ---------------------------------------------------------------- banderas de casos atípicos
def banderas(T):
    T = T.copy()
    T["f_conv"] = (T["rhat_max"] > 1.02) | (T["ess_min"] < 400)
    T["f_div"] = T["div_frac"] > 0.05
    T["f_ajuste"] = T["ajuste_prim"].astype(str).str.startswith("malo") | T["ajuste_sec"].astype(str).str.startswith("malo") | \
        T["ajuste_sec"].astype(str).str.startswith("regular")
    T["f_fase"] = T["ajuste_sec"].astype(str).eq("malo (fase)")
    T["f_sec_no_det"] = T["ajuste_sec"].astype(str).str.contains("no detectado")
    T["f_contacto"] = T["contacto"] > 1.0
    T["f_k_gt1"] = T["k"] > 1.05
    T["f_e_alta_P_corto"] = (T["e"] - 2 * T["e_lo"] > 0.1) & (T["P"] < 3.0)
    # ¿los datos mueven el parámetro respecto al prior que usó la corrida? z en desviaciones del PRIOR; contracción = 1 - sd_post / sd_prior
    for p in ("A_m", "A_r", "B_m", "B_r", "B_amp"):
        sd_post = T[p + "_sd"] if p + "_sd" in T else np.nan
        sd_post = T["B_amp_sd"] if p == "B_amp" and "B_amp_sd" in T else sd_post
        media = T[p]
        T["z_prior_" + p] = (media - T["prior_" + p]) / T["prior_" + p + "_sd"]
        T["contr_" + p] = 1 - sd_post / T["prior_" + p + "_sd"]
    T["f_vs_prior"] = (T[["z_prior_A_m", "z_prior_A_r", "z_prior_B_m", "z_prior_B_r"]].abs() > 3).any(axis=1)
    T["f_vs_massage"] = T["f_vs_prior"]
    return T


# diferencia mínima (absoluta; relativa para radios) para que una inconsistencia de > 5σ cuente como física y no como ruido numérico
PISO = {"k": 0.05, "A_r": 0.05, "B_r": 0.05, "inc": 0.5, "e": 0.03, "ecosw": 0.01, "esinw": 0.03, "A_m": 0.10, "B_m": 0.10, "q": 0.05}
RELATIVOS = ("A_r", "B_r", "A_m", "B_m")
PARAMS_FUERTES = ("k", "A_r", "B_r", "inc", "e", "ecosw")


def consistencia(T, params=("k", "A_r", "B_r", "inc", "e", "ecosw", "esinw", "A_m", "B_m", "q")):
    """Por sistema y parámetro: máxima diferencia entre pares de sectores en unidades del error combinado (z_) y en valor
    absoluto o relativo (d_). Una inconsistencia es fuerte si z > 5 y d supera el piso físico de PISO."""
    filas = []
    for tic, g in T.groupby("TIC"):
        if len(g) < 2:
            continue
        fila = dict(TIC=tic, sistema=g["sistema"].iloc[0], etiqueta=g["etiqueta"].iloc[0], n_sectores=len(g))
        for p in params:
            v, s = g[p].to_numpy(float), g[p + "_sd"].to_numpy(float)
            pares = [(i, j) for i in range(len(v)) for j in range(i + 1, len(v))]
            z = [abs(v[i] - v[j]) / np.hypot(s[i], s[j]) for i, j in pares]
            d = [abs(v[i] - v[j]) / (0.5 * abs(v[i] + v[j]) if p in RELATIVOS else 1.0) for i, j in pares]
            fila["z_" + p] = np.nanmax(z) if z else np.nan
            fila["d_" + p] = np.nanmax(d) if d else np.nan
            fila["fuerte_" + p] = bool(any((zi > 5) and (di > PISO[p]) for zi, di in zip(z, d)))
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("sistema")


def inconsistentes(C, params=PARAMS_FUERTES):
    return C[[("fuerte_" + p) for p in params]].any(axis=1)


# Sistemas donde la modulación a Porb/2 es variación elipsoidal orbital (3, 4, 11) o no es interpretable (19): su "Prot" no es una rotación.
NO_ROTACION = {11412481: "elipsoidal", 50897998: "elipsoidal", 138829403: "elipsoidal", 427347825: "no interpretable"}


def por_sistema(T):
    cols = ["P", "e", "q", "k", "A_m", "B_m", "A_r", "B_r", "inc", "rA_a", "rB_a", "a"]
    g = T.groupby("TIC")
    S = g[cols].mean()                                   # media entre sectores de las medias posteriores
    S["n_sectores"] = g.size()
    for c in ("sistema", "etiqueta", "Tmag", "edad_Myr", "P_rot_consenso", "P_orb_consenso", "D1", "D2"):
        S[c] = g[c].first()
    S["rot_valida"] = ~S["TIC"].isin(NO_ROTACION) if "TIC" in S else ~S.index.isin(NO_ROTACION)
    S["prot_porb"] = (S["P_rot_consenso"] / S["P_orb_consenso"]).where(S["rot_valida"])      # solo rotaciones medibles
    return S.reset_index()


# ---------------------------------------------------------------- figuras
def _cat_color(T):
    return np.where(T["flag_atipica"], NARANJA, AZUL)


def etiquetar(ax, x, y, textos, **kw):
    for xi, yi, t in zip(x, y, textos):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(str(t), (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=TINTA2, **kw)


def leyenda_sectores(ax, sectores, loc="best"):
    h = [plt.Line2D([], [], marker=MARCAS.get(s, "o"), ls="", color=TINTA2, mfc="none", ms=6, label=f"S{s}") for s in sorted(sectores)]
    ax.legend(handles=h, title="sector", loc=loc, ncol=1)


def fig_convergencia(T):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    mal = T["f_conv"]
    for s, g in T.groupby("Sector"):
        a.scatter(g["ess_min"], g["rhat_max"], marker=MARCAS.get(s, "o"), s=42, facecolors=np.where(g["f_conv"], NARANJA, AZUL),
                  edgecolors=TINTA2, linewidths=0.6, alpha=0.9)
    a.set_xscale("log"); a.set_yscale("log"); a.axhline(1.02, color=GRIS, ls=":"); a.axvline(400, color=GRIS, ls=":")
    a.set_xlabel("ESS mínimo (bulk)"); a.set_ylabel(r"$\hat{R}$ máximo"); a.set_title("Convergencia de cada corrida vigente")
    etiquetar(a, T.loc[mal, "ess_min"], T.loc[mal, "rhat_max"], [f"{e}/S{s}" for e, s in zip(T.loc[mal, "etiqueta"], T.loc[mal, "Sector"])])
    leyenda_sectores(a, T["Sector"].unique(), "upper right")
    n_ok, n_mal = int((~mal).sum()), int(mal.sum())
    b.barh(["cumplen\nR̂ ≤ 1.02 y ESS ≥ 400", "no cumplen"], [n_ok, n_mal], color=[AZUL, NARANJA], height=0.5)
    for i, v in enumerate([n_ok, n_mal]):
        b.text(v + 0.5, i, str(v), va="center")
    b.set_xlabel("corridas"); b.set_title(f"{len(T)} corridas, {T['TIC'].nunique()} sistemas"); b.grid(axis="y", visible=False)
    fig.tight_layout(); return fig


def fig_ajuste(T):
    sistemas = T.drop_duplicates("TIC").sort_values("sistema")
    secs = sorted(T["Sector"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.34 * len(sistemas) + 2.2), sharey=True)
    for a, col, tit in ((ax[0], "res_prim_pct", "Eclipse primario"), (ax[1], "res_sec_pct", "Eclipse secundario")):
        M = np.full((len(sistemas), len(secs)), np.nan); mask = np.zeros_like(M, bool)
        for i, tic in enumerate(sistemas["TIC"]):
            for j, s in enumerate(secs):
                r = T[(T["TIC"] == tic) & (T["Sector"] == s)]
                if len(r):
                    M[i, j] = r[col].iloc[0]
                    if col == "res_sec_pct" and "no detectado" in str(r["ajuste_sec"].iloc[0]):
                        mask[i, j] = True
        M = np.where(mask, np.nan, M)
        im = a.imshow(np.clip(M, -30, 30), cmap=CMAP_DIV, vmin=-30, vmax=30, aspect="auto")
        for i, j in zip(*np.where(mask)):
            a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#e6e5e1", hatch="///", edgecolor="#cfcec9", lw=0))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if mask[i, j]:
                    a.text(j, i, "n.d.", ha="center", va="center", fontsize=7, color=TINTA2)
                elif np.isfinite(M[i, j]):
                    a.text(j, i, f"{M[i, j]:+.0f}" if abs(M[i, j]) >= 1 else f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                           color="white" if abs(M[i, j]) > 18 else TINTA)
        a.set_xticks(range(len(secs))); a.set_xticklabels([f"S{s}" for s in secs]); a.set_title(tit); a.grid(False)
    ax[0].set_yticks(range(len(sistemas))); ax[0].set_yticklabels([f"{e}  ·  TIC {t}" for e, t in zip(sistemas["etiqueta"], sistemas["TIC"])], fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("residuo = (datos − modelo) / profundidad  [%]\n(+: datos más someros que el modelo)")
    fig.suptitle("Ajuste modelo vs datos por sistema y sector   (n.d. = secundario no detectado en 1b)", x=0.01, ha="left", fontsize=11)
    return fig


def fig_e_P(T):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw=dict(width_ratios=[1.5, 1]))
    EMAX = 0.62
    a.axvspan(3, 6, color="#cde2fb", alpha=0.5, lw=0)
    a.text(3.05, 0.57, "P_circ empírico: envolvente ~3 d,\nnúcleo frío ~6 d (Kepler y TESS,\nZanazzi 2021)", fontsize=7.5, color=TINTA2, va="top")
    for _, r in T.iterrows():
        col = NARANJA if r["flag_atipica"] else AZUL
        c = lambda v: min(v, EMAX)
        intervalo(a, r["P"], c(r["e"]), c(r["e_q16"]), c(r["e_q84"]), c(r["e_q025"]), c(r["e_q975"]), False, col, MARCAS.get(int(r["Sector"]), "o"), 6)
    fuera = T[T["e"] > EMAX]
    for _, r in fuera.iterrows():
        a.annotate(f"{r['etiqueta']}/S{int(r['Sector'])}: e = {r['e']:.2f} ↑", (r["P"], EMAX), xytext=(8, -3), textcoords="offset points", fontsize=7.5, color=NARANJA)
    at = T[(T["flag_atipica"] | (T["e"] > 0.15)) & (T["e"] <= EMAX)]
    etiquetar(a, at["P"], at["e"], [f"{e}/S{s}" for e, s in zip(at["etiqueta"], at["Sector"])])
    a.set_xscale("log"); a.set_xticks([1, 2, 3, 5, 7]); a.set_xticklabels(["1", "2", "3", "5", "7"]); a.minorticks_off()
    a.set_xlabel(r"$P_{\rm orb}$ [d]"); a.set_ylabel("e  (media; barra gruesa 68 %, fina 95 %)"); a.set_title("Excentricidad vs periodo orbital  (naranja = corrida con problema)")
    a.set_ylim(-0.02, EMAX + 0.02); leyenda_sectores(a, T["Sector"].unique(), "upper left")
    for s, g in T.groupby("Sector"):
        b.errorbar(g["ecosw"], g["esinw"], xerr=g["ecosw_sd"], yerr=g["esinw_sd"], fmt="none", ecolor="#b8b7b2", elinewidth=0.8)  # media ± 1 sd
        b.scatter(g["ecosw"], g["esinw"], marker=MARCAS.get(s, "o"), s=40, facecolors=np.where(g["flag_atipica"], NARANJA, AZUL), edgecolors=TINTA2, linewidths=0.6, zorder=3)
    b.axhline(0, color=GRIS, lw=0.6); b.axvline(0, color=GRIS, lw=0.6)
    th = np.linspace(0, 2 * np.pi, 200)
    for r in (0.1, 0.3, 0.6):
        b.plot(r * np.cos(th), r * np.sin(th), color=GRIS, lw=0.6, ls=":")
    b.set_aspect("equal"); b.set_xlabel(r"$e\cos\omega$"); b.set_ylabel(r"$e\sin\omega$"); b.set_title("Plano (e cos ω, e sin ω); círculos e = 0.1, 0.3, 0.6")
    fig.tight_layout(); return fig


def fig_omega(T):
    sig = T[(T["clasificacion_ecosw"] == "excentrica")]
    fig = plt.figure(figsize=(12, 4.8))
    a = fig.add_subplot(1, 3, 1, projection="polar"); b = fig.add_subplot(1, 3, 2); c = fig.add_subplot(1, 3, 3)
    a.scatter(np.radians(sig["w"]), sig["e"], c=np.where(sig["flag_atipica"], NARANJA, AZUL), s=34, edgecolors=TINTA2, linewidths=0.6)
    a.set_title(r"$\omega$ y $e$ (solo con $e\cos\omega$ significativa)", loc="left", fontsize=10); a.set_rlabel_position(135); a.grid(True)
    b.hist(sig["w"], bins=np.linspace(-180, 180, 13), color=AZUL, edgecolor=SUPERFICIE); b.set_xlabel(r"$\omega$ [°]"); b.set_ylabel("corridas")
    b.set_title(f"Distribución de ω ({len(sig)} corridas excéntricas)")
    c.errorbar(T["ecosw_1b"], T["ecosw"], yerr=T["ecosw_sd"], fmt="o", ms=4, color=AZUL, ecolor="#b8b7b2", mec=TINTA2, mew=0.5)
    lim = np.nanmax(np.abs(np.r_[T["ecosw_1b"], T["ecosw"]])) * 1.1
    c.plot([-lim, lim], [-lim, lim], color=GRIS, lw=0.8); c.set_xlabel(r"$e\cos\omega$ de la fase del secundario (1b)"); c.set_ylabel(r"$e\cos\omega$ del posterior")
    c.set_title("Consistencia con la fotometría (1b)")
    dd = T[(T["ecosw"] - T["ecosw_1b"]).abs() > 3 * T["ecosw_sd"].clip(lower=0.003)]
    etiquetar(c, dd["ecosw_1b"], dd["ecosw"], [f"{e}/S{s}" for e, s in zip(dd["etiqueta"], dd["Sector"])])
    fig.tight_layout(); return fig


def fig_rotacion(S):
    S = S.reset_index(drop=True); V, X = S[S["rot_valida"]], S[~S["rot_valida"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    a.plot([0.5, 10], [0.5, 10], color=TINTA2, lw=1); a.plot([0.5, 10], [0.25, 5], color=GRIS, lw=0.8, ls=":"); a.plot([0.5, 10], [1, 20], color=GRIS, lw=0.8, ls=":")
    a.scatter(V["P_orb_consenso"], V["P_rot_consenso"], s=52, c=V["e"], cmap=CMAP_SEQ, vmin=0, edgecolors=TINTA2, linewidths=0.7, zorder=3, label="rotación medida (color = e)")
    a.scatter(X["P_orb_consenso"], X["P_rot_consenso"], s=52, facecolors="none", edgecolors=GRIS, linewidths=1.2, zorder=3, label="modulación orbital (no es rotación)")
    etiquetar(a, S["P_orb_consenso"], S["P_rot_consenso"], S["etiqueta"]); a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.6, 9); a.set_ylim(0.5, 8)
    a.set_xlabel(r"$P_{\rm orb}$ [d]"); a.set_ylabel(r"$P_{\rm rot}$ [d] (consenso)"); a.set_title("Rotación vs órbita"); a.legend(loc="upper left", fontsize=8)
    b.axhline(1, color=TINTA2, lw=1); b.axhline(0.5, color=GRIS, ls=":"); b.axhline(7 / 8, color=GRIS, ls=":")
    b.scatter(V["P_orb_consenso"], V["P_rot_consenso"] / V["P_orb_consenso"], s=52, color=AZUL, edgecolors=TINTA2, linewidths=0.7, zorder=3)
    b.scatter(X["P_orb_consenso"], X["P_rot_consenso"] / X["P_orb_consenso"], s=52, facecolors="none", edgecolors=GRIS, linewidths=1.2, zorder=3)
    etiquetar(b, S["P_orb_consenso"], S["P_rot_consenso"] / S["P_orb_consenso"], S["etiqueta"]); b.text(9, 0.88, "8/7", fontsize=7.5, color=TINTA2, ha="right")
    b.set_xscale("log"); b.set_xlabel(r"$P_{\rm orb}$ [d]"); b.set_ylabel(r"$P_{\rm rot}/P_{\rm orb}$"); b.set_title("Cociente Prot/Porb  (1 = sincronizado; huecos = variación elipsoidal, no rotación)")
    fig.tight_layout(); return fig


def fig_masas(T, S):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    a, b, c, d = ax
    for s, g in T.groupby("Sector"):
        a.errorbar(g["A_m"], g["B_m"], xerr=g["A_m_sd"], yerr=g["B_m_sd"], fmt="none", ecolor="#d3d2cd", elinewidth=0.7)
        a.scatter(g["A_m"], g["B_m"], marker=MARCAS.get(s, "o"), s=36, facecolors=np.where(g["flag_atipica"], NARANJA, AZUL), edgecolors=TINTA2, linewidths=0.5, zorder=3)
    xs = np.array([0.08, 15.0])
    for q, ls in ((1, "-"), (0.5, "--"), (0.25, ":"), (0.1, ":")):
        a.plot(xs, q * xs, color=GRIS, lw=0.8, ls=ls); a.text(0.13, q * 0.13 * 1.12, f"q={q}", fontsize=7.5, color=TINTA2)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.1, 15); a.set_ylim(0.02, 6)
    a.set_xlabel(r"$M_A$ [M$_\odot$]"); a.set_ylabel(r"$M_B$ [M$_\odot$]"); a.set_title("Masas de las componentes"); leyenda_sectores(a, T["Sector"].unique(), "lower right")
    b.scatter(S["P"], S["q"], s=48, color=AZUL, edgecolors=TINTA2, linewidths=0.6, zorder=3); etiquetar(b, S["P"], S["q"], S["etiqueta"])
    b.set_xscale("log"); b.set_xticks([1, 2, 3, 5, 7]); b.set_xticklabels(["1", "2", "3", "5", "7"]); b.minorticks_off()
    b.set_xlabel(r"$P_{\rm orb}$ [d]"); b.set_ylabel(r"$q = M_B/M_A$"); b.set_title("q vs periodo (por sistema)")
    for nom, ax_, col, lab in (("A", c, "A_m", r"$M_A$"), ("B", d, "B_m", r"$M_B$")):
        x, y = T["prior_" + col], T[col]
        ax_.errorbar(x, y, xerr=T["prior_" + col + "_sd"], yerr=T[col + "_sd"], fmt="none", ecolor="#d3d2cd", elinewidth=0.7)
        ax_.scatter(x, y, s=34, facecolors=np.where(T["z_prior_" + col].abs() > 3, NARANJA, AZUL), edgecolors=TINTA2, linewidths=0.5, zorder=3)
        lo = max(0.02, np.nanmin(np.r_[x, y]) * 0.8); hi = np.nanmax(np.r_[x, y]) * 1.3
        ax_.plot([lo, hi], [lo, hi], color=GRIS, lw=0.8); ax_.set_xscale("log"); ax_.set_yscale("log"); ax_.set_xlim(lo, hi); ax_.set_ylim(lo, hi)
        ax_.set_xlabel(f"{lab} del prior (massage)"); ax_.set_ylabel(f"{lab} de starry (posterior)"); ax_.set_title(f"{lab}: posterior vs prior")
        dd = T[T["z_prior_" + col].abs() > 3]; etiquetar(ax_, dd["prior_" + col], dd[col], [f"{e}/S{s}" for e, s in zip(dd["etiqueta"], dd["Sector"])])
    fig.tight_layout(); return fig


def fig_radios(T, S):
    fig, ax = plt.subplots(1, 4, figsize=(18.5, 4.4)); a, b, c, d = ax
    for s, g in T.groupby("Sector"):
        a.errorbar(g["A_r"], g["B_r"], xerr=g["A_r_sd"], yerr=g["B_r_sd"], fmt="none", ecolor="#d3d2cd", elinewidth=0.7)
        a.scatter(g["A_r"], g["B_r"], marker=MARCAS.get(s, "o"), s=36, facecolors=np.where(g["flag_atipica"], NARANJA, AZUL), edgecolors=TINTA2, linewidths=0.5, zorder=3)
    mx = max(T["A_r"].max(), T["B_r"].max()) * 1.05
    for k, ls in ((1, "-"), (0.5, "--")):
        a.plot([0, mx], [0, k * mx], color=GRIS, lw=0.8, ls=ls); a.text(mx * 0.97, k * mx * 0.96, f"k={k}", fontsize=7.5, color=TINTA2, ha="right")
    a.set_xlim(0, mx); a.set_ylim(0, mx); a.set_xlabel(r"$R_A$ [R$_\odot$]"); a.set_ylabel(r"$R_B$ [R$_\odot$]"); a.set_title("Radios de las componentes")
    b.errorbar(T["q"], T["k"], xerr=T["q_sd"], yerr=T["k_sd"], fmt="none", ecolor="#d3d2cd", elinewidth=0.7)
    b.scatter(T["q"], T["k"], s=34, facecolors=np.where(T["flag_atipica"], NARANJA, AZUL), edgecolors=TINTA2, linewidths=0.5, zorder=3)
    qq = np.linspace(0.05, 1.1, 50); b.plot(qq, qq, color=GRIS, lw=0.8); b.plot(qq, qq ** 0.8, color=GRIS, lw=0.8, ls=":")
    b.text(1.05, 1.07, "k = q", fontsize=7.5, color=TINTA2); b.set_xlabel(r"$q = M_B/M_A$"); b.set_ylabel(r"$k = R_B/R_A$"); b.set_title("Razón de radios vs razón de masas")
    dd = T[T["f_k_gt1"]]; etiquetar(b, dd["q"], dd["k"], [f"{e}/S{s}" for e, s in zip(dd["etiqueta"], dd["Sector"])])
    for ax_, col, lab in ((c, "A_r", r"$R_A$"), (d, "B_r", r"$R_B$")):   # posterior vs prior; naranja: |z| > 3
        ax_.errorbar(T["prior_" + col], T[col], xerr=T["prior_" + col + "_sd"], yerr=T[col + "_sd"], fmt="none", ecolor="#d3d2cd", elinewidth=0.7)
        ax_.scatter(T["prior_" + col], T[col], s=34, facecolors=np.where(T["z_prior_" + col].abs() > 3, NARANJA, AZUL), edgecolors=TINTA2, linewidths=0.5, zorder=3)
        lim = np.nanmax(np.r_[T["prior_" + col], T[col]]) * 1.05; ax_.plot([0, lim], [0, lim], color=GRIS, lw=0.8)
        ax_.set_xlabel(f"{lab} del prior"); ax_.set_ylabel(f"{lab} de starry (posterior)"); ax_.set_title(f"{lab}: posterior vs prior")
        dd = T[T["z_prior_" + col].abs() > 3]; etiquetar(ax_, dd["prior_" + col], dd[col], [f"{e}/S{s}" for e, s in zip(dd["etiqueta"], dd["Sector"])])
    fig.tight_layout(); return fig


def fig_contraccion(T):
    ps = [("A_m", r"$M_A$"), ("A_r", r"$R_A$"), ("B_m", r"$M_B$"), ("B_r", r"$R_B$"), ("B_amp", "amp B")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (p, lab) in enumerate(ps):
        v = T["z_prior_" + p].dropna(); a.scatter(np.full(len(v), i) + np.random.default_rng(1).uniform(-0.18, 0.18, len(v)), v, s=22, color=AZUL, edgecolors=TINTA2, linewidths=0.4, alpha=0.9, zorder=3)
        w = T["contr_" + p].dropna(); b.scatter(np.full(len(w), i) + np.random.default_rng(2).uniform(-0.18, 0.18, len(w)), w, s=22, color=AZUL, edgecolors=TINTA2, linewidths=0.4, alpha=0.9, zorder=3)
    for ax_ in (a, b):
        ax_.set_xticks(range(len(ps))); ax_.set_xticklabels([l for _, l in ps]); ax_.grid(axis="x", visible=False)
    a.axhline(0, color=TINTA2, lw=0.8); a.axhspan(-3, 3, color="#cde2fb", alpha=0.45, lw=0); a.set_ylabel("(posterior − prior) / σ del prior"); a.set_title("¿Cuánto se aleja el posterior del prior?   (banda = ±3σ)")
    b.axhline(0, color=TINTA2, lw=0.8); b.set_ylabel(r"contracción $1-\sigma_{post}/\sigma_{prior}$"); b.set_title("¿Cuánto aprenden los datos?  (0 = nada; 1 = todo)")
    b.set_ylim(-0.5, 1.02); fig.tight_layout(); return fig


def fig_geometria(T):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    for s, g in T.groupby("Sector"):
        a.scatter(g["P"], g["rA_a"], marker=MARCAS.get(s, "o"), s=34, color=AZUL, edgecolors=TINTA2, linewidths=0.5, label=None, zorder=3)
        a.scatter(g["P"], g["rB_a"], marker=MARCAS.get(s, "o"), s=34, color=AQUA, edgecolors=TINTA2, linewidths=0.5, zorder=3)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"$P_{\rm orb}$ [d]"); a.set_ylabel("radio fraccionario  $R/a$")
    a.set_title("Radios fraccionarios vs periodo"); a.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=AZUL, label="A"), plt.Line2D([], [], marker="o", ls="", color=AQUA, label="B")], loc="upper right")
    x = T["rA_a"] + T["rB_a"]
    b.scatter(x, T["inc"], s=36, facecolors=np.where(T["flag_atipica"], NARANJA, AZUL), edgecolors=TINTA2, linewidths=0.5, zorder=3)
    xs = np.linspace(x.min() * 0.9, x.max() * 1.05, 100); b.plot(xs, np.degrees(np.arccos(np.clip(xs, 0, 1))), color=GRIS, lw=0.9)
    b.text(xs[-1], np.degrees(np.arccos(xs[-1])) + 0.4, "límite de eclipse (cos i = ΣR/a)", fontsize=7.5, color=TINTA2, ha="right")
    b.set_xlabel(r"$(R_A+R_B)/a$"); b.set_ylabel(r"$i$ [°]"); b.set_title("Inclinación vs tamaño del sistema")
    dd = T[T["f_contacto"]]; etiquetar(b, (dd["rA_a"] + dd["rB_a"]), dd["inc"], [f"{e}/S{s}" for e, s in zip(dd["etiqueta"], dd["Sector"])])
    fig.tight_layout(); return fig


def fig_por_sector(T, params=(("k", r"$k=R_B/R_A$"), ("A_r", r"$R_A$ [R$_\odot$]"), ("B_r", r"$R_B$ [R$_\odot$]"), ("inc", r"$i$ [°]"),
                              ("e", r"$e$"), ("B_m", r"$M_B$ [M$_\odot$]"))):
    sistemas = T.drop_duplicates("TIC").sort_values("sistema")["TIC"].tolist()
    pos = {t: i for i, t in enumerate(sistemas)}
    fig, ax = plt.subplots(len(params), 1, figsize=(12.5, 2.0 * len(params)), sharex=True)
    for a, (p, lab) in zip(ax, params):
        for s, g in T.groupby("Sector"):
            x = g["TIC"].map(pos) + {5: -0.24, 6: -0.15, 32: -0.05, 33: 0.05, 87: 0.15, 98: 0.24}.get(s, 0)
            a.errorbar(x, g[p], yerr=g[p + "_sd"], fmt=MARCAS.get(s, "o"), ms=4.2, color=AZUL, mfc=np.where(True, AZUL, AZUL) if False else AZUL, ecolor="#b8b7b2", elinewidth=0.8, mec=TINTA2, mew=0.4)
        a.set_ylabel(lab)
    ax[-1].set_xticks(range(len(sistemas))); ax[-1].set_xticklabels([T.loc[T.TIC == t, "etiqueta"].iloc[0] for t in sistemas], fontsize=8)
    ax[-1].set_xlabel("sistema"); ax[0].legend(handles=[plt.Line2D([], [], marker=MARCAS.get(s, "o"), ls="", color=AZUL, mec=TINTA2, label=f"S{s}") for s in sorted(T["Sector"].unique())],
                                                ncol=6, loc="upper right", fontsize=8)
    ax[0].set_title("Parámetros por sistema y sector (media ± sd): la dispersión vertical dentro de un sistema es la inconsistencia entre sectores")
    fig.tight_layout(); return fig


def fig_consistencia(C):
    cols = [c for c in C.columns if c.startswith("z_")]
    M = C[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10.5, 0.33 * len(C) + 1.8))
    im = ax.imshow(np.clip(M, 0, 8), cmap=CMAP_SEQ, vmin=0, vmax=8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 5 else TINTA)
    for i in range(M.shape[0]):
        for j, c in enumerate(cols):
            if bool(C.iloc[i].get("fuerte_" + c[2:], False)) and c[2:] in PARAMS_FUERTES:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=NARANJA, lw=2))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[2:] for c in cols]); ax.set_yticks(range(len(C)))
    ax.set_yticklabels([f"{e}  ·  TIC {t}" for e, t in zip(C["etiqueta"], C["TIC"])], fontsize=8); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("máx. |Δ| entre sectores / error combinado  (σ)")
    ax.set_title("Consistencia entre sectores  (contorno naranja: > 5σ y con diferencia física, ver PISO)"); fig.tight_layout(); return fig


RELACIONES = [("P", "e"), ("P", "q"), ("q", "k"), ("A_m", "A_r"), ("B_m", "B_r"), ("P", "rA_a"), ("q", "e"), ("Tmag", "q"), ("edad_Myr", "e"), ("prot_porb", "e")]


# prot_porb queda fuera: de los 15 sistemas con rotación medible, 13 están sincronizados (≈ 1 ± 2 %) y el rango de esos valores es ruido
def correlaciones(S, vars_=("P", "e", "q", "k", "A_m", "B_m", "A_r", "B_r", "inc", "rA_a", "rB_a", "Tmag", "edad_Myr")):
    n = len(vars_); R = np.full((n, n), np.nan); Pv = np.full((n, n), np.nan); N = np.zeros((n, n), int)
    for i, a in enumerate(vars_):
        for j, b in enumerate(vars_):
            if a == b:
                R[i, j], Pv[i, j], N[i, j] = 1.0, 0.0, S[a].notna().sum(); continue
            d = S[[a, b]].dropna()
            if len(d) >= 6:
                r, p = stats.spearmanr(d[a], d[b]); R[i, j], Pv[i, j], N[i, j] = r, p, len(d)
    return pd.DataFrame(R, vars_, vars_), pd.DataFrame(Pv, vars_, vars_), pd.DataFrame(N, vars_, vars_)


TIPOS = {
    frozenset(["A_m", "Tmag"]): "selección (más brillante = más masiva)", frozenset(["A_r", "Tmag"]): "selección", frozenset(["B_m", "Tmag"]): "selección",
    frozenset(["q", "B_m"]): "definicional (q = M_B/M_A)", frozenset(["q", "A_m"]): "definicional", frozenset(["k", "B_r"]): "definicional (k = R_B/R_A)",
    frozenset(["k", "A_r"]): "definicional", frozenset(["q", "B_r"]): "derivada (B_r ~ M_B)", frozenset(["A_m", "A_r"]): "relación masa-radio esperada",
    frozenset(["B_m", "B_r"]): "relación masa-radio esperada", frozenset(["q", "k"]): "relación masa-radio relativa esperada",
    frozenset(["A_m", "B_m"]): "esperada (escala de masas)", frozenset(["A_r", "B_r"]): "esperada (escala de masas)",
    frozenset(["P", "rA_a"]): "geométrica (Kepler: R/a ∝ P⁻²ᐟ³)", frozenset(["P", "rB_a"]): "geométrica (Kepler)",
    frozenset(["inc", "rA_a"]): "efecto de selección (solo vemos sistemas que eclipsan)", frozenset(["inc", "rB_a"]): "efecto de selección",
    frozenset(["P", "inc"]): "efecto de selección", frozenset(["rA_a", "rB_a"]): "geométrica", frozenset(["B_m", "rB_a"]): "derivada", frozenset(["A_m", "rA_a"]): "derivada",
    frozenset(["B_r", "rB_a"]): "definicional", frozenset(["A_r", "rA_a"]): "definicional", frozenset(["A_m", "edad_Myr"]): "selección (muestra de masivas más jóvenes)",
    frozenset(["B_m", "edad_Myr"]): "derivada", frozenset(["A_r", "edad_Myr"]): "derivada", frozenset(["Tmag", "edad_Myr"]): "selección",
}


def fig_correlaciones(S):
    R, Pv, N = correlaciones(S)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    im = a.imshow(R.to_numpy(), cmap=CMAP_DIV, vmin=-1, vmax=1)
    for i in range(len(R)):
        for j in range(len(R)):
            if np.isfinite(R.iloc[i, j]) and i != j:
                sig = Pv.iloc[i, j] < 0.05
                a.text(j, i, f"{R.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, fontweight="bold" if sig else "normal", color="white" if abs(R.iloc[i, j]) > 0.7 else TINTA)
    a.set_xticks(range(len(R))); a.set_xticklabels(R.columns, rotation=60, ha="right", fontsize=8); a.set_yticks(range(len(R))); a.set_yticklabels(R.index, fontsize=8); a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.04, pad=0.02, label="ρ de Spearman"); a.set_title("Correlaciones entre parámetros (negrita: p < 0.05)")
    pares = []
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            if np.isfinite(R.iloc[i, j]) and Pv.iloc[i, j] < 0.05:
                x, y = R.index[i], R.columns[j]
                pares.append(dict(x=x, y=y, rho=R.iloc[i, j], p=Pv.iloc[i, j], n=int(N.iloc[i, j]), tipo=TIPOS.get(frozenset([x, y]), "informativa")))
    P_ = pd.DataFrame(pares).sort_values("rho", key=np.abs, ascending=False).reset_index(drop=True) if pares else pd.DataFrame(columns=["x", "y", "rho", "p", "n", "tipo"])
    n_pares = len(R) * (len(R) - 1) // 2
    info = P_[P_["tipo"] == "informativa"]
    txt = ("INFORMATIVAS (p < 0.05; no son definiciones ni efectos de selección):\n" + "\n".join(f"  {r.x:>9s} – {r.y:<9s} ρ = {r.rho:+.2f}  p = {r.p:.3f}  n = {r.n}" for r in info.itertuples())
           + f"\n\nTRIVIALES O ESPERADAS (no aportan evidencia nueva):\n" + "\n".join(f"  {r.x:>9s} – {r.y:<9s} ρ = {r.rho:+.2f}  [{r.tipo}]" for r in P_[P_["tipo"] != "informativa"].head(11).itertuples())
           + f"\n\nCuidado: {n_pares} pares con n = 19; con p < 0.05 se esperan ~{0.05 * n_pares:.0f} falsos positivos.\nUmbral de Bonferroni: p < {0.05 / n_pares:.4f}.")
    b.axis("off"); b.text(0, 1, txt, va="top", family="monospace", fontsize=7.8)
    fig.tight_layout(); return fig, P_


# nombre corto (para la ecuación), etiqueta del eje y escala logarítmica de cada variable por sistema
VARS = {
    "P": ("P", r"$P_{\rm orb}$ [d]", False), "e": ("e", "e  (media entre sectores)", False), "q": ("q", r"$q=M_B/M_A$", False),
    "B_m": (r"M_B", r"$M_B$ [M$_\odot$]", True), "A_m": (r"M_A", r"$M_A$ [M$_\odot$]", True),
    "P_rot_consenso": (r"P_rot", r"$P_{\rm rot}$ [d]", False), "edad_Myr": ("edad", "edad [Myr]", True),
    "abs_ecosw": ("|e cos ω|", r"$|e\cos\omega|$  (fase del eclipse secundario)", True),
}


def _p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _num(v):
    return f"{v:.2g}" if abs(v) < 0.01 or abs(v) >= 100 else f"{v:.2f}"


def ic_bootstrap_rho(x, y, B=4000, semilla=1):
    """IC 95 % de ρ de Spearman remuestreando sistemas con reemplazo."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(semilla)
    r = [stats.spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return np.nanpercentile(r, [2.5, 97.5])


def holm(ps):
    """p-values ajustados por Holm (1979) para una familia de pruebas."""
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps)
    adj = np.maximum.accumulate([(m - k) * ps[i] for k, i in enumerate(o)])
    out = np.empty(m); out[o] = np.minimum(adj, 1); return out


def recta_theil_sen(ax, x, y, nx, ny, logx=False, logy=False, mad_k=2.5, p_holm=None):
    """Recta de Theil–Sen (en log10 si el eje es logarítmico) con la banda del IC 95 % de la pendiente.

    Devuelve los handles para la leyenda, una máscara de atípicos (> mad_k MAD de la recta) y el título de la leyenda
    con ρ de Spearman (no cambia con log, porque usa rangos).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    tx, ty = (np.log10(x) if logx else x), (np.log10(y) if logy else y)
    sl, ic, lo, hi = stats.theilslopes(ty, tx, 0.95)
    rho, p = stats.spearmanr(x, y)
    xs = np.linspace(tx.min(), tx.max(), 100)
    mx, my = np.median(tx), np.median(ty)
    banda = np.array([my + s * (xs - mx) for s in (lo, hi)])
    back_x = 10**xs if logx else xs
    back = (lambda v: 10**v) if logy else (lambda v: v)
    ax.fill_between(back_x, back(banda.min(0)), back(banda.max(0)), color=GRIS, alpha=0.18, lw=0, zorder=1)
    ax.plot(back_x, back(ic + sl * xs), color=TINTA2, lw=1.4, zorder=2)
    res = ty - (ic + sl * tx)
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    atip = np.abs(res - np.median(res)) > mad_k * mad if mad > 0 else np.zeros_like(res, bool)
    lx, ly = (f"log {nx}" if logx else nx), (f"log {ny}" if logy else ny)
    signo = "−" if sl < 0 else "+"
    h = [plt.Line2D([], [], color=TINTA2, lw=1.4, label=f"Theil–Sen: {ly} = {_num(ic).replace('-', '−')} {signo} {_num(abs(sl))}·{lx}"),
         matplotlib.patches.Patch(color=GRIS, alpha=0.35, lw=0, label=f"IC 95 % de la pendiente: [{_num(lo)}, {_num(hi)}]".replace("-", "−"))]
    sig = "significativa" if p < 0.05 else "no significativa"
    tau, p_tau = stats.kendalltau(x, y)
    lo_r, hi_r = ic_bootstrap_rho(x, y)
    titulo = (f"Spearman ρ = {rho:+.2f} [IC 95 % bootstrap {lo_r:+.2f}, {hi_r:+.2f}], {_p(p)} ({sig}), n = {len(x)}\n"
              f"Kendall τ = {tau:+.2f}, {_p(p_tau)}")
    if p_holm is not None:
        titulo += f";  p de Holm ({p_holm[1]} pruebas) = {p_holm[0]:.3f}" if p_holm[0] >= 0.001 else f";  p de Holm ({p_holm[1]} pruebas) < 0.001"
    titulo = titulo.replace("= -", "= −").replace(" -", " −")
    return h, atip, titulo


# ---------------------------------------------------------------- muestra completa: Orión + Pléyades
PLE = RAIZ.parent / "Resultados_Pleyades"
PLE_VIEJAS = RAIZ.parent / "Resultados_pleyades_lucia/Resultados_pleyades/Posteriors"
EDAD_PLEYADES = 125.0                                      # Myr, límite de litio (Stauffer et al. 1998), la de los notebooks
# Corrida que se usa por sistema: las nuevas (starry_por_tic, oct 2026) donde existen; si no, las de ene-feb 2026.
# 26017840: la única corrida nueva (S42) da e = 0.84 a P = 1.39 d (periastro casi en contacto), así que se usa la anterior.
PLE_CORRIDAS = {346541355: "nuevas", 26017840: "viejas", 35159657: "nuevas", 125754991: "nuevas"}
# P2 (125754991) y P3 (35159657) se volvieron a correr el 13-oct con el prior uniforme en (h, k); de las corridas nuevas
# solo entran los sectores que convergen (R̂ ≤ 1.02 y ESS ≥ 400, como en Orión).
# P_LS de 26017840 alterna entre P_orb y 2 P_orb según el sector: no se interpreta como rotación
PLE_NO_ROTACION = {26017840: "no interpretable"}
# Validación astrométrica (Resultados_Pleyades/candidatas/candidatas_pleyades.csv): 346541355 tiene p_astrom = 0,
# Δv_t = 5.4 km/s (7σ) y paralaje 6.57 mas: no miembro probable. Se marca con "?" en las figuras.
PLE_NO_MIEMBRO = {346541355: "no miembro probable (p_astrom = 0)"}


def _convergio(summary, rhat=1.02, ess=400):
    s = pd.read_csv(summary, index_col=0)
    return bool(s.r_hat.max() <= rhat and s.ess_bulk.min() >= ess)


def _resumen_posteriors(P):
    P = P.assign(q=P["B_m"] / P["A_m"], P=P["B_porb"], e=P["B_ecc"],
                 ecosw=P["B_ecc"] * np.cos(np.radians(P["B_w"])))
    return P[["P", "e", "ecosw", "q", "A_m", "B_m", "A_r", "B_r"]].mean()


def ecosw_por_sistema(T):
    """|e cos ω| de Orión por sistema: media entre sectores del ajuste de starry. Solo se cuenta como medición si el
    eclipse secundario se detectó en algún sector (si no, la fase no está observada y e cos ω sale del modelo)."""
    g = T.groupby("TIC")
    E = pd.DataFrame(dict(ecosw=g["ecosw"].mean(), ecosw_sd=np.maximum(g["ecosw_sd"].mean(), g["ecosw"].std().fillna(0)),
                          sec_det=g["sec_det"].apply(lambda v: v.astype(bool).any())))
    E.loc[~E.sec_det, ["ecosw", "ecosw_sd"]] = np.nan
    E["ecosw_fuente"] = np.where(E.sec_det, "starry", "secundario no detectado")
    return E.reset_index()


def _ecosw_fase_pleyades(tic):
    """e cos ω = (π/2)(fase del secundario − 0.5), mediana de los sectores con secundario consistente (como en
    genera_starry_pleyades.py). Se usa donde la corrida de starry tenía ω fijo cerca de 90°."""
    m = pd.read_csv(PLE / "pipeline/interpolador/profundidades_v2_pleyades.csv")
    g = m[(m.TIC == tic) & m.secundario_detectado.astype(bool)]
    g = g[(g.fase_secundario - g.fase_secundario.median()).abs() < 0.02]
    if len(g) == 0:
        return np.nan, np.nan
    return g.ecosw.median(), max(g.ecosw.std(), 0.002) if len(g) > 1 else 0.01


def pleyades_por_sistema():
    """Una fila por sistema de las Pléyades con las mismas columnas que por_sistema (media entre sectores)."""
    pri = pd.concat([pd.read_csv(PLE / "pipeline/priors_pleyades.csv"), pd.read_csv(PLE / "pipeline/priors_346541355.csv")])
    filas = []
    for tic, cual in PLE_CORRIDAS.items():
        if cual == "nuevas":
            fs = [f for f in sorted(glob.glob(str(PLE / f"starry_por_tic/TIC_{tic}/S*/TIC_{tic}_S*_Posteriors.csv")))
                  if _convergio(f.replace("_Posteriors.csv", "_Summary.csv"))]
        else:
            fs = [f for f in glob.glob(str(PLE_VIEJAS / "*_Posteriors.csv")) if f"_{tic}_" in os.path.basename(f)]
        r = pd.concat([_resumen_posteriors(pd.read_csv(f)) for f in fs], axis=1).mean(axis=1)
        # P_rot: mediana de los periodogramas por sector, sin los > 10 d (sistemático de TESS ~14 d)
        r["P_rot_consenso"] = pri.loc[(pri.TIC == tic) & (pri.P_rot < 10), "P_rot"].median()
        r["TIC"], r["n_sectores"], r["edad_Myr"], r["corrida"] = tic, len(fs), EDAD_PLEYADES, cual
        if cual == "nuevas":       # prior uniforme en (h, k): e cos ω del ajuste
            P_ = pd.concat([pd.read_csv(f) for f in fs])
            ec = P_["B_ecc"] * np.cos(np.radians(P_["B_w"]))
            r["ecosw"], r["ecosw_sd"], r["ecosw_fuente"] = ec.mean(), ec.std(), "starry"
        else:                      # ω casi fijo en el prior viejo: e cos ω de la fase del secundario
            r["ecosw"], r["ecosw_sd"] = _ecosw_fase_pleyades(tic)
            r["ecosw_fuente"] = "fase del secundario" if np.isfinite(r["ecosw"]) else "secundario no detectado"
        filas.append(r)
    S = pd.DataFrame(filas).sort_values("P").reset_index(drop=True)
    S["etiqueta"] = [f"P{i + 1}" + ("?" if t in PLE_NO_MIEMBRO else "") for i, t in enumerate(S["TIC"])]
    S["miembro"] = ~S["TIC"].isin(PLE_NO_MIEMBRO)
    S["rot_valida"] = ~S["TIC"].isin(PLE_NO_ROTACION)
    S["prot_porb"] = (S["P_rot_consenso"] / S["P"]).where(S["rot_valida"])
    S["cumulo"] = "Pléyades"
    return S


def orion_sin_ajuste(S):
    """Sistemas de Orión con periodo y rotación pero sin corrida de starry (2 y 17): solo entran en P_rot y edad."""
    ind = pd.read_csv(RAIZ / "resultados_pipeline/periodos/indice_sistemas.csv")
    ind = ind[~ind.TIC.isin(S.TIC)]
    hr = pd.read_csv(RAIZ / "resumen_diagramas_hr_por_tic.csv").groupby("TIC")["edad_A_Myr"].first()
    return pd.DataFrame(dict(TIC=ind.TIC, sistema=ind.indice, etiqueta=ind.indice.astype(str), P=ind.P_orb,
                             P_rot_consenso=ind.P_rot, edad_Myr=ind.TIC.map(hr), rot_valida=~ind.TIC.isin(NO_ROTACION),
                             cumulo="Orión", corrida="sin starry"))


def muestra_completa(S, T):
    """Orión (con y sin ajuste de starry) + Pléyades, una fila por sistema, con |e cos ω|."""
    O = S.drop(columns=["ecosw", "ecosw_sd"], errors="ignore").merge(ecosw_por_sistema(T), on="TIC", how="left")
    O = O.assign(cumulo="Orión", corrida="starry")
    M = pd.concat([O, orion_sin_ajuste(S), pleyades_por_sistema()], ignore_index=True)
    M["abs_ecosw"] = M["ecosw"].abs()
    M["ecosw_signif"] = M["abs_ecosw"] > 3 * M["ecosw_sd"]
    return M


CUMULOS = {"Orión": dict(color=AZUL, marker="o"), "Pléyades": dict(color=NARANJA, marker="s")}


def fig_par(S, x, y, excluir=None, extra=None, p_holm=None):
    """Una relación entre dos parámetros por sistema, con recta de Theil–Sen y leyenda completa.

    S puede traer la columna "cumulo" (Orión / Pléyades): cada cúmulo lleva su color y símbolo, la recta se ajusta a
    toda la muestra y el título de la leyenda da además ρ de Orión sola. Los atípicos se marcan con un anillo.
    excluir: (máscara, etiqueta) de sistemas que se muestran huecos y no entran en el ajuste.
    extra: función ax -> lista de handles para curvas de referencia.
    """
    nx, lab_x, logx = VARS[x]; ny, lab_y, logy = VARS[y]
    if "cumulo" not in S:
        S = S.assign(cumulo="Orión")
    d = S[[x, y, "etiqueta", "cumulo"]].copy()
    exc = (excluir[0] if excluir is not None else pd.Series(False, index=S.index)).reindex(d.index).fillna(False).astype(bool)
    ok = np.isfinite(d[x].astype(float)) & np.isfinite(d[y].astype(float))
    d, exc = d[ok], exc[ok]
    f = d[~exc]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    hs, atip, titulo = recta_theil_sen(ax, f[x], f[y], nx, ny, logx, logy, p_holm=p_holm)
    atip = pd.Series(atip, index=f.index)
    cums = [c for c in CUMULOS if c in set(d["cumulo"])]
    if len(cums) > 1:
        fo = f[f["cumulo"] == "Orión"]
        if len(fo) > 3:
            r_o, p_o = stats.spearmanr(fo[x], fo[y])
            titulo += f"\nsolo Orión: ρ = {r_o:+.2f}, {_p(p_o)}, n = {len(fo)}".replace("= -", "= −")
    h = []
    for c in cums:
        st = CUMULOS[c]
        fc, gc = f[f["cumulo"] == c], d[exc & (d["cumulo"] == c)]
        ax.scatter(fc[x], fc[y], s=48, color=st["color"], marker=st["marker"], edgecolors=SUPERFICIE, linewidths=1, zorder=3)
        ax.scatter(gc[x], gc[y], s=48, marker=st["marker"], facecolors="none", edgecolors=st["color"], linewidths=1.4, zorder=3)
        h.append(plt.Line2D([], [], marker=st["marker"], ls="", ms=7, color=st["color"], mec=SUPERFICIE,
                            label=f"{c} (media entre sectores), n = {len(fc)}"))
    if atip.any():
        fa = f[atip]
        ax.scatter(fa[x], fa[y], s=150, facecolors="none", edgecolors=TINTA, linewidths=1.1, zorder=4)
        h.append(plt.Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=TINTA, mew=1.1,
                            label=f"atípico: > 2.5 MAD de la recta ({', '.join(fa['etiqueta'].astype(str))})"))
    if exc.any():
        h.append(plt.Line2D([], [], marker="o", ls="", ms=7, mfc="none", mec=TINTA2, mew=1.4,
                            label=f"{excluir[1]} ({', '.join(d['etiqueta'][exc].astype(str))})"))
    if extra is not None:
        h += extra(ax)
    etiquetar(ax, d[x], d[y], d["etiqueta"])
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel(lab_x); ax.set_ylabel(lab_y)
    ax.set_title(f"{lab_y.split(' [')[0].split('  ')[0]} vs {lab_x.split(' [')[0].split('  ')[0]}")
    ax.legend(handles=h + hs, title=titulo, loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=8, title_fontsize=8.5,
              alignment="left")
    fig.tight_layout(); return fig


def fig_ecosw(M, x, y, piso=1e-4):
    """Relación con |e cos ω| (la componente de la excentricidad que fija la fase del eclipse secundario).

    Las órbitas compatibles con circular (|e cos ω| < 3σ) se dibujan como límite superior en 3σ; no se ajusta recta,
    porque casi todos los valores son límites y una correlación entre límites no tiene sentido.
    """
    nx, lab_x, logx = VARS[x]; ny, lab_y, logy = VARS[y]
    c = "abs_ecosw"
    d = M[np.isfinite(M[x].astype(float)) & np.isfinite(M[y].astype(float))].copy()
    sin_sec = M[M[c].isna() & M["ecosw_fuente"].eq("secundario no detectado")]
    d[c] = np.where(d["ecosw_signif"], d[c], np.maximum(3 * d["ecosw_sd"], piso))
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    h = []
    for cum, st in CUMULOS.items():
        g = d[d["cumulo"] == cum]
        if g.empty:
            continue
        det, lim = g[g["ecosw_signif"]], g[~g["ecosw_signif"]]
        ax.scatter(det[x], det[y], s=52, color=st["color"], marker=st["marker"], edgecolors=SUPERFICIE, linewidths=1, zorder=4)
        flecha = dict(uplims=True) if y == c else dict(xuplims=True)
        err = {("yerr" if y == c else "xerr"): 0.3 * lim[c]}
        ax.errorbar(lim[x], lim[y], fmt=st["marker"], ms=6, mfc="none", mec=st["color"], mew=1.3, ecolor=st["color"],
                    elinewidth=1, capsize=0, zorder=3, **flecha, **err)
        h.append(plt.Line2D([], [], marker=st["marker"], ls="", ms=7, color=st["color"], mec=SUPERFICIE,
                            label=f"{cum}: órbita excéntrica, |e cos ω| > 3σ (n = {len(det)})"))
        h.append(plt.Line2D([], [], marker=st["marker"], ls="", ms=6, mfc="none", mec=st["color"], mew=1.3,
                            label=f"{cum}: compatible con circular, límite superior 3σ (n = {len(lim)})"))
    etiquetar(ax, d[x], d[y], d["etiqueta"])
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel(lab_x); ax.set_ylabel(lab_y)
    ax.set_title(f"{lab_y.split(' [')[0].split('  ')[0]} vs {lab_x.split(' [')[0].split('  ')[0]}")
    exc = d.loc[d["ecosw_signif"], "etiqueta"].astype(str).tolist()
    titulo = (f"Excéntricas: {', '.join(exc) or 'ninguna'};  el resto, circulares en e cos ω.\n"
              "e ≥ |e cos ω|: la fase del secundario fija e cos ω; e sin ω (duraciones) queda mal restringida.")
    if len(sin_sec):
        titulo += f"\nSin secundario detectado (sin medición): {', '.join(sin_sec['etiqueta'].astype(str))}"
    ax.legend(handles=h, title=titulo, loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=8, title_fontsize=8,
              alignment="left")
    fig.tight_layout(); return fig


PARES_ORBITALES = [("P", "B_m"), ("P", "abs_ecosw"), ("P", "P_rot_consenso"), ("P", "q"), ("P", "edad_Myr"),
                   ("abs_ecosw", "q"), ("abs_ecosw", "B_m"), ("abs_ecosw", "edad_Myr")]


def figs_pares_orbitales(S):
    """Las ocho relaciones orbitales: {nombre de archivo: figura}. S es muestra_completa (o por_sistema sin |e cos ω|)."""
    figs = {}
    con_recta = [(x, y) for x, y in PARES_ORBITALES if "abs_ecosw" not in (x, y)]
    pv = []
    for x, y in con_recta:
        d = S[S["rot_valida"].astype(bool)] if y == "P_rot_consenso" else S
        d = d[[x, y]].dropna(); pv.append(stats.spearmanr(d[x], d[y])[1])
    p_holm = dict(zip(con_recta, holm(pv)))
    for x, y in PARES_ORBITALES:
        nombre = f"14_{x}_vs_{y}".replace("_consenso", "").replace("_Myr", "").replace("abs_", "")
        if "abs_ecosw" in (x, y):
            if "abs_ecosw" in S:
                figs[nombre] = fig_ecosw(S, x, y)
            continue
        kw = dict(p_holm=(p_holm[(x, y)], len(con_recta)))
        if y == "P_rot_consenso":
            kw["excluir"] = (~S["rot_valida"].astype(bool), "P_LS no es rotación (elipsoidal o no interpretable); fuera del ajuste")

            def ref(ax):
                xs = np.array([S["P"].min(), S["P"].max()])
                ax.plot(xs, xs, color=GRIS, lw=1.2, ls=(0, (4, 3)), zorder=1)
                return [plt.Line2D([], [], color=GRIS, lw=1.2, ls=(0, (4, 3)), label=r"$P_{\rm rot}=P_{\rm orb}$ (sincronizada)")]
            kw["extra"] = ref
        figs[nombre] = fig_par(S, x, y, **kw)
    return figs


def fig_relaciones(S, pares):
    info = pares[pares["tipo"] == "informativa"].head(6)
    esp = pares[pares["tipo"].str.contains("masa-radio")].head(6 - len(info)) if len(info) < 6 else pares.iloc[0:0]
    top = pd.concat([info, esp])
    if top.empty:
        return None
    nr = int(np.ceil(len(top) / 3)); fig, ax = plt.subplots(nr, 3, figsize=(14, 4.6 * nr), squeeze=False); ax = ax.ravel()
    for a, (_, r) in zip(ax, top.iterrows()):
        d = S[[r.x, r.y, "etiqueta"]].dropna()
        hs, atip, titulo = recta_theil_sen(a, d[r.x], d[r.y], r.x, r.y)
        a.scatter(d[r.x][~atip], d[r.y][~atip], s=44, color=AZUL, edgecolors=SUPERFICIE, linewidths=1, zorder=3)
        a.scatter(d[r.x][atip], d[r.y][atip], s=58, color=NARANJA, edgecolors=SUPERFICIE, linewidths=1, zorder=4)
        if atip.any():
            hs.append(plt.Line2D([], [], marker="o", ls="", ms=7, color=NARANJA, mec=SUPERFICIE,
                                 label=f"atípico > 2.5 MAD ({', '.join(d['etiqueta'][atip].astype(str))})"))
        etiquetar(a, d[r.x], d[r.y], d["etiqueta"]); a.set_xlabel(r.x); a.set_ylabel(r.y)
        a.set_title(f"{r.x} vs {r.y}   [{r.tipo}]", fontsize=9)
        a.legend(handles=hs, title=titulo, loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=7.5, title_fontsize=7.5,
                 alignment="left")
    for a in ax[len(top):]:
        a.axis("off")
    fig.suptitle("Relaciones entre parámetros por sistema", x=0.01, ha="left"); fig.tight_layout(); return fig


def tabla_atipicos(T, C):
    filas = []
    cons = C.set_index("TIC")
    for _, r in T.sort_values(["sistema", "Sector"]).iterrows():
        motivos, acciones = [], []
        if r["f_conv"]:
            motivos.append(f"convergencia (R̂ {r['rhat_max']:.2f}, ESS {r['ess_min']:.0f})")
            acciones.append("A: más muestreo; si hay degeneración de masas, reparametrizar")
        if r["f_div"]:
            motivos.append(f"divergencias {100 * r['div_frac']:.0f} %"); acciones.append("revisar geometría de masas")
        if r["f_fase"]:
            motivos.append(f"fase del secundario ({r['fase_sec_modelo']:.3f} vs {r['fase_sec_ref']:.3f})"); acciones.append("C: prior de e cosω con la fase de 1b y e < 0.6")
        elif r["f_ajuste"]:
            motivos.append(f"ajuste del eclipse ({r['ajuste_prim']} / {r['ajuste_sec']})"); acciones.append("estudiar caso (profundidad)")
        if r["f_sec_no_det"]:
            motivos.append("secundario no detectado en 1b"); acciones.append("no evaluable: decisión del asesor")
        if r["f_contacto"]:
            motivos.append("discos solapados en el periastro"); acciones.append("restringir e (periastro > R_A+R_B)")
        if r["f_k_gt1"]:
            motivos.append(f"k = {r['k']:.2f} > 1"); acciones.append("revisar asignación primaria/secundaria")
        if r["f_e_alta_P_corto"]:
            motivos.append(f"e = {r['e']:.2f} con P = {r['P']:.2f} d"); acciones.append("posible tercer cuerpo; discutir")
        if r["f_vs_massage"]:
            motivos.append("el posterior se aleja > 3σ del prior en masa o radio"); acciones.append("revisar priors de massage")
        if r["TIC"] in cons.index:
            z = cons.loc[r["TIC"]]
            malos = [p for p in PARAMS_FUERTES if bool(z.get("fuerte_" + p, False))]
            if malos:
                motivos.append("inconsistente entre sectores: " + ", ".join(malos)); acciones.append("discutir: dilución distinta por sector / geometría compartida")
        if motivos:
            filas.append(dict(sistema=r["etiqueta"], TIC=int(r["TIC"]), sector=int(r["Sector"]), motivos="; ".join(motivos), acciones="; ".join(dict.fromkeys(acciones))))
    return pd.DataFrame(filas)


def marcar_atipicas(T, C):
    T = banderas(T)
    inc = set()
    cons = C.set_index("TIC")
    T["f_inconsistente"] = [bool(r["TIC"] in cons.index and any(bool(cons.loc[r["TIC"]].get("fuerte_" + p, False)) for p in PARAMS_FUERTES)) for _, r in T.iterrows()]
    # naranja en las figuras = problema de la propia corrida (convergencia, ajuste o física); la inconsistencia entre sectores y el alejamiento
    # del prior se muestran aparte (mapas de consistencia y contracción)
    T["flag_atipica"] = T[["f_conv", "f_div", "f_ajuste", "f_fase", "f_contacto", "f_k_gt1"]].any(axis=1)
    return T


# ================================================================ visualización de posteriores
# Referencias de diseño: forest/caterpillar (ArviZ, MCMCvis), quantile dotplots (Kay et al. 2016), rank plots (Vehtari et al. 2021),
# pair plots con divergencias y comparación prior-posterior (Gabry et al. 2019), matriz de dispersión (corner, Foreman-Mackey 2016).
CADENAS = [AZUL, NARANJA, AQUA, GRIS]


def intervalo(ax, pos, media, q16, q84, q025, q975, horizontal=True, color=AZUL, marker="o", ms=5.5, zorder=3, relleno=None):
    """Una estimación posterior: punto = MEDIA; trazo grueso = intervalo creíble de 68 %; trazo fino = 95 %."""
    if horizontal:      # pos en el eje y; valores en el eje x
        ax.plot([q025, q975], [pos, pos], color=color, lw=1.0, solid_capstyle="round", zorder=zorder)
        ax.plot([q16, q84], [pos, pos], color=color, lw=3.0, solid_capstyle="round", zorder=zorder)
        ax.plot([media], [pos], ls="", marker=marker, ms=ms, mfc=relleno or SUPERFICIE, mec=color, mew=1.4, zorder=zorder + 1)
    else:               # pos en el eje x; valores en el eje y
        ax.plot([pos, pos], [q025, q975], color=color, lw=1.0, solid_capstyle="round", zorder=zorder)
        ax.plot([pos, pos], [q16, q84], color=color, lw=3.0, solid_capstyle="round", zorder=zorder)
        ax.plot([pos], [media], ls="", marker=marker, ms=ms, mfc=relleno or SUPERFICIE, mec=color, mew=1.4, zorder=zorder + 1)


DESPLAZ = {5: -0.30, 6: -0.18, 32: -0.06, 33: 0.06, 87: 0.18, 98: 0.30}
PARAMS_FOREST = (("k", r"$k=R_B/R_A$"), ("A_r", r"$R_A$ [R$_\odot$]"), ("B_r", r"$R_B$ [R$_\odot$]"),
                 ("inc", r"$i$ [°]"), ("e", r"$e$"), ("q", r"$q=M_B/M_A$"))


def fig_forest(T, params=PARAMS_FOREST):
    """Un panel por parámetro; cada sistema es una fila y dentro de ella cada sector es un marcador. Si los marcadores de una fila
    no se solapan, el parámetro cambia de un sector a otro (inconsistencia)."""
    sist = T.drop_duplicates("TIC").sort_values("sistema")
    pos = {t: i for i, t in enumerate(sist["TIC"])}
    nc = 3; nr = int(np.ceil(len(params) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(15.5, 0.36 * len(sist) * nr + 1.8), sharey=True, squeeze=False)
    for a, (p, lab) in zip(ax.ravel(), params):
        for _, r in T.iterrows():
            y = pos[r["TIC"]] + DESPLAZ.get(int(r["Sector"]), 0) * 1.2
            col = NARANJA if r["flag_atipica"] else AZUL
            intervalo(a, y, r[p], r[p + "_q16"], r[p + "_q84"], r[p + "_q025"], r[p + "_q975"], True, col, MARCAS.get(int(r["Sector"]), "o"), 5)
        for i in range(len(sist)):
            if i % 2 == 0:
                a.axhspan(i - 0.5, i + 0.5, color="#f1f0ec", zorder=0, lw=0)
        a.set_xlabel(lab); a.grid(axis="y", visible=False)
    ax[0, 0].set_yticks(range(len(sist))); ax[0, 0].set_yticklabels([f"{e} · TIC {t}" for e, t in zip(sist["etiqueta"], sist["TIC"])], fontsize=8)
    ax[0, 0].invert_yaxis()
    h = [plt.Line2D([], [], marker=MARCAS.get(s, "o"), ls="", color=TINTA2, mfc=SUPERFICIE, mew=1.2, ms=6, label=f"S{s}") for s in sorted(T["Sector"].unique())]
    ax[0, 0].legend(handles=h, title="sector", ncol=2, loc="lower right", fontsize=7.5)
    fig.suptitle("Parámetros por sistema y sector.  Punto = media posterior · trazo grueso = intervalo 68 % · trazo fino = 95 %   (naranja = corrida con problema)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); return fig


def puntos_cuantiles(ax, x, n=20, y0=0.0, color=AZUL, tam=None):
    """Quantile dotplot (Kay et al. 2016): n puntos, cada uno con probabilidad 1/n; se apilan en intervalos del eje x."""
    q = np.quantile(x, (np.arange(n) + 0.5) / n)
    ancho = max((q.max() - q.min()) / 12, 1e-6)
    ix = np.floor((q - q.min()) / ancho).astype(int)
    alt = np.zeros(ix.max() + 1, int); ys = []
    for i in ix:
        ys.append(alt[i]); alt[i] += 1
    s = tam or 28
    ax.scatter(q, y0 + np.array(ys) * 0.9, s=s, color=color, edgecolors=TINTA2, linewidths=0.4, zorder=3)
    return q


def muestras_corrida(r):
    P = pd.read_csv(Path(r["dir"]) / f"TIC_{int(r['TIC'])}_S{int(r['Sector'])}_Posteriors.csv", index_col=0)
    return P


def fig_puntos_e(T, umbral=0.1, n=20):
    """¿Es excéntrica? Cada fila es una corrida donde la pregunta no está clara (P(e > 0.1) entre 5 y 95 %) o con e > 0.1 a P < 3 d.
    Los 20 puntos son 20 cuantiles igualmente probables del posterior de e: contar los puntos a la derecha de la línea da P(e > 0.1)."""
    pe = []
    for _, r in T.iterrows():
        pe.append(float(np.mean(muestras_corrida(r)["B_ecc"].to_numpy() > umbral)))
    T = T.assign(p_e=pe)
    sel = T[((T["p_e"] > 0.05) & (T["p_e"] < 0.95)) | (T["f_e_alta_P_corto"] & ~T["f_fase"])].sort_values(["sistema", "Sector"])
    if sel.empty:
        return None
    fig, ax = plt.subplots(len(sel), 1, figsize=(10, 0.8 * len(sel) + 1.4), sharex=True, squeeze=False)
    for a, (_, r) in zip(ax[:, 0], sel.iterrows()):
        e = muestras_corrida(r)["B_ecc"].to_numpy()
        puntos_cuantiles(a, e, n, color=NARANJA if r["flag_atipica"] else AZUL)
        a.axvline(umbral, color=GRIS, ls=":", lw=1); a.set_yticks([]); a.grid(False)
        a.text(1.005, 0.5, f"P(e>{umbral}) = {int(round(100 * r['p_e']))} %", transform=a.transAxes, va="center", fontsize=8, color=TINTA2)
        a.set_ylabel(f"{r['etiqueta']}/S{int(r['Sector'])}\nP={r['P']:.2f} d", rotation=0, ha="right", va="center", fontsize=8)
        a.spines["left"].set_visible(False)
    ax[-1, 0].set_xlabel("excentricidad e"); ax[-1, 0].set_xlim(-0.01, 0.62)
    fig.suptitle(f"Posterior de e como {n} puntos (cada punto = {100 // n} % de probabilidad); línea = e = {umbral}.  Casos ambiguos o con e alta a periodo corto",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); return fig


def fig_corner(r, variables=("A_m", "B_m", "A_r", "B_r", "B_inc", "B_ecc")):
    """Matriz de dispersión (estilo corner.py): diagonal = marginal, abajo = pares, con las divergencias en naranja."""
    import arviz as az
    d = Path(r["dir"])
    trace = az.from_netcdf(str(d / f"TIC_{int(r['TIC'])}_S{int(r['Sector'])}_trace.nc"))
    E, div, nch, ndr = _escalares(trace)
    v = [x for x in variables if x in E.columns]
    n = len(v); fig, ax = plt.subplots(n, n, figsize=(2.0 * n + 1, 2.0 * n + 1))
    idx = np.random.default_rng(0).choice(len(E), size=min(len(E), 2500), replace=False); dv = np.flatnonzero(div)
    for i in range(n):
        for j in range(n):
            a = ax[i, j]
            if j > i:
                a.axis("off"); continue
            if i == j:
                a.hist(E[v[i]], bins=40, color=AZUL, alpha=0.8, edgecolor=SUPERFICIE); a.axvline(E[v[i]].mean(), color=TINTA2, lw=1, ls="--"); a.set_yticks([])
            else:
                a.plot(E[v[j]].to_numpy()[idx], E[v[i]].to_numpy()[idx], ".", ms=2, color="#9aa7b8", alpha=0.5, rasterized=True)
                if len(dv):
                    a.plot(E[v[j]].to_numpy()[dv], E[v[i]].to_numpy()[dv], ".", ms=3, color=NARANJA, alpha=0.7, rasterized=True)
            a.grid(False); a.tick_params(labelsize=7)
            if i == n - 1: a.set_xlabel(v[j], fontsize=8)
            else: a.set_xticklabels([])
            if j == 0 and i > 0: a.set_ylabel(v[i], fontsize=8)
            elif j > 0: a.set_yticklabels([])
    fig.suptitle(f"Corner · sistema {r['etiqueta']} · TIC {int(r['TIC'])} · S{int(r['Sector'])}   (naranja = divergencias; línea = media)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); return fig


def _escalares(trace):
    """DataFrame de las variables escalares del posterior (cadena-mayor) y vector de divergencias."""
    post = trace.posterior; cols = {}
    for v in post.data_vars:
        x = post[v].values
        if x.ndim == 2:
            cols[v] = x.reshape(-1)
        elif x.ndim == 3:
            for i in range(x.shape[2]):
                cols[f"{v}[{i}]"] = x[:, :, i].reshape(-1)
    div = trace.sample_stats["diverging"].values.reshape(-1).astype(bool)
    return pd.DataFrame(cols), div, post.sizes["chain"], post.sizes["draw"]


def _tendencia_modelo(r):
    d = Path(r["dir"]); tic, sec = int(r["TIC"]), int(r["Sector"])
    D = pd.read_csv(d / f"TIC_{tic}_S{sec}_trending.csv").sort_values("time")
    Q = {k: pd.read_csv(d / f"TIC_{tic}_S{sec}_{k}.csv").sort_values("time").rename(columns={"flux": k}) for k in ("q16", "q50", "q84")}
    M = D
    for k in Q:
        M = pd.merge_asof(M, Q[k], on="time", direction="nearest", tolerance=1e-4)
    return M.dropna()


def _panel_eclipse(a, M, P, t0, fase0, semi, titulo):
    ph = ((M["time"].to_numpy() - t0) / P - fase0 + 0.5) % 1 - 0.5
    sel = np.abs(ph) < semi
    b = np.linspace(-semi, semi, 41); k = np.digitize(ph[sel], b) - 1
    df = pd.DataFrame(dict(k=k, ph=ph[sel], y=M["flat_flux"].to_numpy()[sel], q16=M["q16"].to_numpy()[sel], q50=M["q50"].to_numpy()[sel], q84=M["q84"].to_numpy()[sel]))
    g = df.groupby("k").agg(ph=("ph", "mean"), y=("y", "mean"), e=("y", lambda v: v.std() / np.sqrt(max(len(v), 1))), q16=("q16", "mean"), q50=("q50", "mean"), q84=("q84", "mean"))
    a.plot(ph[sel], M["flat_flux"].to_numpy()[sel], ".", ms=1.5, color="#cfcec9", rasterized=True, zorder=1)
    a.fill_between(g["ph"], g["q16"], g["q84"], color="#d9d8d2", zorder=2, lw=0, label="modelo 16–84 %")
    a.plot(g["ph"], g["q50"], color=TINTA2, lw=1.4, zorder=3, label="modelo (mediana)")
    a.errorbar(g["ph"], g["y"], yerr=g["e"], fmt="o", ms=3.2, color=AZUL, ecolor=AZUL, elinewidth=0.8, zorder=4, label="datos (media por bin)")
    lo, hi = np.nanmin(g["y"] - g["e"]), np.nanmax(g["y"] + g["e"]); d = (hi - lo) or 1e-3
    a.set_ylim(lo - 0.25 * d, hi + 0.25 * d); a.set_title(titulo, fontsize=9); a.set_xlabel("fase"); a.set_ylabel("flujo")


def _pdf_prior(mu, sd, x):
    from scipy.stats import norm
    z = 1 - norm.cdf(-mu / sd)
    return norm.pdf(x, mu, sd) / z


def fig_caso(r):
    """Ficha de diagnóstico de una corrida: (1,2) modelo y datos en cada eclipse; (3) rank plot por cadena del parámetro peor muestreado;
    (4) traza de ese parámetro; (5) par de parámetros con divergencias; (6) prior frente a posterior."""
    import arviz as az
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    d = Path(r["dir"]); tic, sec = int(r["TIC"]), int(r["Sector"])
    trace = az.from_netcdf(str(d / f"TIC_{tic}_S{sec}_trace.nc"))
    E, div, nch, ndr = _escalares(trace)
    peor = str(r["rhat_peor"]) if pd.notna(r.get("rhat_peor")) else "B_m"
    if peor not in E.columns:
        peor = next((c for c in E.columns if c.startswith(peor.split("[")[0])), E.columns[0])
    x = E[peor].to_numpy().reshape(nch, ndr)
    fig = plt.figure(figsize=(15, 8.6)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
    # modelo vs datos
    try:
        M = _tendencia_modelo(r); P = r["P"]
        P_ = muestras_corrida(r); t0 = float(P_["B_t0"].mean())
        pri = pd.read_csv(RAIZ / "priors_v2.csv"); pr = pri[(pri.TIC == tic) & (pri.Sector == sec)]
        dur = float(pr["Duration"].iloc[0]) if len(pr) else 0.2
        semi = float(np.clip(1.4 * dur / P, 0.04, 0.2))
        _panel_eclipse(fig.add_subplot(gs[0, 0]), M, P, t0, 0.0, semi, "Eclipse primario")
        f2 = r.get("fase_sec_ref"); f2 = 0.5 if not np.isfinite(f2) else float(f2)
        _panel_eclipse(fig.add_subplot(gs[0, 1]), M, P, t0, f2, semi, f"Eclipse secundario (fase observada {f2:.3f})")
        fig.axes[0].legend(fontsize=7, loc="lower left")
    except Exception as e:
        fig.add_subplot(gs[0, 0]).text(0.1, 0.5, f"sin modelo: {e}", fontsize=8)
    # rank plot por cadena
    from scipy.stats import rankdata
    sub = GridSpecFromSubplotSpec(nch, 1, subplot_spec=gs[0, 2], hspace=0.15)
    rk = rankdata(x.reshape(-1), method="average").reshape(nch, ndr)
    nb = 20; bins = np.linspace(0, nch * ndr, nb + 1); esp = ndr / nb
    for c in range(nch):
        a = fig.add_subplot(sub[c, 0]); h, _ = np.histogram(rk[c], bins=bins)
        a.bar(np.arange(nb), h, width=0.9, color=CADENAS[c % 4], alpha=0.85); a.axhline(esp, color=TINTA2, ls="--", lw=0.8)
        sd = np.sqrt(ndr * (1 / nb) * (1 - 1 / nb)); a.axhspan(esp - 1.96 * sd, esp + 1.96 * sd, color="#d9d8d2", alpha=0.7, lw=0, zorder=0)
        a.set_yticks([]); a.set_ylabel(f"c{c + 1}", rotation=0, labelpad=8, fontsize=8); a.grid(False)
        if c < nch - 1:
            a.set_xticks([])
        if c == 0:
            a.set_title(f"Rank plot de {peor}: debería ser plano", fontsize=9)
    a.set_xlabel("rango (bins)")
    # traza
    a = fig.add_subplot(gs[1, 0])
    for c in range(nch):
        a.plot(x[c], lw=0.5, color=CADENAS[c % 4], alpha=0.85, label=f"cadena {c + 1}")
    a.set_title(f"Traza de {peor}", fontsize=9); a.set_xlabel("iteración"); a.legend(fontsize=7, ncol=2, loc="upper right")
    # pares con divergencias
    a = fig.add_subplot(gs[1, 1])
    otros = [c for c in E.columns if c != peor and not c.startswith("u_") and not c.startswith("B_hk")]
    corr = {c: abs(stats.spearmanr(E[peor], E[c])[0]) for c in otros if E[c].std() > 0}
    par = max(corr, key=corr.get) if corr else otros[0]
    idx = np.random.default_rng(0).choice(len(E), size=min(len(E), 3000), replace=False)
    a.plot(E[par].to_numpy()[idx], E[peor].to_numpy()[idx], ".", ms=2.5, color="#9aa7b8", alpha=0.6, rasterized=True)
    dv = np.flatnonzero(div)
    if len(dv):
        a.plot(E[par].to_numpy()[dv], E[peor].to_numpy()[dv], "o", ms=3.2, mfc=NARANJA, mec="none", alpha=0.8, label=f"divergencias ({len(dv)})")
        a.legend(fontsize=7.5, loc="best")
    a.set_xlabel(par); a.set_ylabel(peor); a.set_title(f"Pares: {peor} vs {par}  (|ρ| = {corr.get(par, np.nan):.2f})", fontsize=9)
    # prior vs posterior
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 2], hspace=0.55, wspace=0.3)
    from scipy.stats import gaussian_kde
    for i, (p, lab) in enumerate((("A_m", "M_A"), ("B_m", "M_B"), ("A_r", "R_A"), ("B_r", "R_B"))):
        a = fig.add_subplot(sub[i // 2, i % 2])
        if p not in E.columns:
            continue
        v = E[p].to_numpy(); mu, sd = r.get("prior_" + p), r.get("prior_" + p + "_sd")
        lo = max(0, min(v.min(), (mu - 3 * sd) if np.isfinite(mu) else v.min())); hi = max(v.max(), (mu + 3 * sd) if np.isfinite(mu) else v.max())
        xs = np.linspace(lo, hi, 300); k = gaussian_kde(v)
        a.fill_between(xs, k(xs), color=AZUL, alpha=0.35, lw=0); a.plot(xs, k(xs), color=AZUL, lw=1.2, label="posterior")
        if np.isfinite(mu) and sd > 0:
            a.plot(xs, _pdf_prior(mu, sd, xs), color=TINTA2, lw=1.2, ls="--", label="prior")
        a.axvline(v.mean(), color=AZUL, lw=1, ls=":"); a.set_yticks([]); a.set_title(f"{lab}", fontsize=8.5, loc="left"); a.tick_params(labelsize=7); a.grid(False)
        if i == 0:
            a.legend(fontsize=6.5, loc="upper right")
    fig.suptitle(f"Sistema {r['etiqueta']} · TIC {tic} · S{sec} · versión {r['ver']}      R̂ = {r['rhat_max']:.2f} ({r['rhat_peor']})   ESS = {r['ess_min']:.0f}   "
                 f"divergencias = {100 * r['div_frac']:.1f} %      e (media) = {r['e']:.3f}", x=0.01, ha="left", fontsize=10.5)
    return fig


def hut(e):
    """Cociente Ω_pseudo-sincrónico / n (Hut 1981) para una órbita de excentricidad e."""
    e = np.asarray(e, float)
    return (1 + 7.5 * e ** 2 + 45 / 8 * e ** 4 + 5 / 16 * e ** 6) / ((1 + 3 * e ** 2 + 3 / 8 * e ** 4) * (1 - e ** 2) ** 1.5)


def fig_rotacion_pseudo(S):
    """Prot frente al periodo pseudo-sincrónico (que coincide con Porb si e = 0) y cociente Prot/P_pseudo. Huecos: sin rotación medible."""
    S = S.reset_index(drop=True).copy(); S["P_ps"] = S["P"] / hut(S["e"]); S["razon"] = S["P_rot_consenso"] / S["P_ps"]
    V, X = S[S["rot_valida"]], S[~S["rot_valida"]]; fuera = (V["razon"] - 1).abs() > 0.1
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.9))
    xs = np.array([0.5, 12])
    a.plot(xs, xs, color=TINTA2, lw=1); a.plot(xs, xs / 2, color=GRIS, lw=0.8, ls=":"); a.plot(xs, xs * 2, color=GRIS, lw=0.8, ls=":")
    a.scatter(V.loc[~fuera, "P_ps"], V.loc[~fuera, "P_rot_consenso"], s=56, color=AZUL, edgecolors=TINTA2, linewidths=0.7, zorder=3, label="Prot ≈ P_pseudo (±10 %)")
    a.scatter(V.loc[fuera, "P_ps"], V.loc[fuera, "P_rot_consenso"], s=56, color=NARANJA, edgecolors=TINTA2, linewidths=0.7, zorder=3, label="fuera de la línea")
    a.scatter(X["P_ps"], X["P_rot_consenso"], s=56, facecolors="none", edgecolors=GRIS, linewidths=1.2, zorder=3, label="modulación orbital (no es rotación)")
    etiquetar(a, S["P_ps"], S["P_rot_consenso"], S["etiqueta"]); a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.6, 9); a.set_ylim(0.5, 8)
    a.set_xlabel(r"$P_{\rm pseudo-sincr.}$ [d]  (= $P_{\rm orb}$ si e = 0)"); a.set_ylabel(r"$P_{\rm rot}$ [d] (consenso)"); a.set_title("Rotación vs periodo pseudo-sincrónico"); a.legend(loc="upper left", fontsize=8)
    b.axhline(1, color=TINTA2, lw=1); b.axhline(0.5, color=GRIS, ls=":")
    b.scatter(V.loc[~fuera, "P"], V.loc[~fuera, "razon"], s=56, color=AZUL, edgecolors=TINTA2, linewidths=0.7, zorder=3)
    b.scatter(V.loc[fuera, "P"], V.loc[fuera, "razon"], s=56, color=NARANJA, edgecolors=TINTA2, linewidths=0.7, zorder=3)
    b.scatter(X["P"], X["razon"], s=56, facecolors="none", edgecolors=GRIS, linewidths=1.2, zorder=3)
    etiquetar(b, S["P"], S["razon"], S["etiqueta"]); b.set_xscale("log"); b.set_yscale("log"); b.set_xticks([1, 2, 3, 5, 7]); b.set_xticklabels(["1", "2", "3", "5", "7"]); b.minorticks_off()
    b.set_xlabel(r"$P_{\rm orb}$ [d]"); b.set_ylabel(r"$P_{\rm rot}/P_{\rm pseudo}$"); b.set_title("Cociente Prot / P_pseudo  (1 = pseudo-sincronizado)")
    fig.tight_layout(); return fig
