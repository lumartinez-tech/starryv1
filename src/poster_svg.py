"""Cuatro gráficos en SVG para el póster: (e, q), (q_prior, q_posterior), (e, edad) y (P_rot, P_orb).

Datos de Orión: diagnostico_resultados/tabla_maestra.csv (una fila por TIC y sector, corrida vigente) y
resultados_pipeline/periodos/indice_sistemas.csv (periodos de consenso). Valor por sistema = media entre sectores de las medias
posteriores (como `por_sistema` en diagnostico_orion.py); las barras son los intervalos creíbles (68 % grueso, 95 % fino) promedio
de los sectores. Sin los sistemas 2, 16 y 17. Las Pléyades (sistemas 22-25, por periodo orbital) solo salen en (P_rot, P_orb),
porque sus ajustes de starry todavía no existen: P_orb y P_rot vienen del paso 1/1b con el secundario enmascarado donde está.
"""
import json, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from astropy.timeseries import LombScargle

warnings.filterwarnings("ignore")
RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion"); DIAG = RAIZ / "diagnostico_resultados"
PLEY = Path("/home/luciam/Documents/starry/Resultados_Pleyades"); PIPE = PLEY / "pipeline"
SALIDA = DIAG / "poster" / "svg"; SALIDA.mkdir(parents=True, exist_ok=True)
EXCLUIR = {2, 16, 17}
NO_ROTACION = {11412481, 50897998, 138829403, 427347825}      # Prot ≈ Porb/2: variación elipsoidal o no interpretable (diagnostico_orion.NO_ROTACION)
TOL_CONSENSO = 0.05

# ---------------------------------------------------------------- estilo (paleta del diagnóstico de Orión)
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8a85"
TINTA, TINTA2, SUPERFICIE = "#0b0b0b", "#52514e", "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("azul_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
matplotlib.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "text.color": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 11, "axes.labelsize": 12, "legend.frameon": False, "legend.fontsize": 9.5,
})

# ---------------------------------------------------------------- datos de Orión por sistema
T = pd.read_csv(DIAG / "tabla_maestra.csv")
T = T[~T["sistema"].isin(EXCLUIR) & T["sistema"].notna()].copy()
IND = pd.read_csv(RAIZ / "resultados_pipeline/periodos/indice_sistemas.csv").set_index("TIC")
CAT = pd.read_csv(RAIZ / "catalogo_completo_mist_sommers.csv").drop_duplicates("TIC").set_index("TIC")
VEC = pd.read_csv(RAIZ / "edades_vecindad/edades_asignadas.csv").set_index("TIC")

T["q_prior"] = T["prior_B_m"] / T["prior_A_m"]
T["q_prior_sd"] = T["q_prior"] * np.hypot(T["prior_B_m_sd"] / T["prior_B_m"], T["prior_A_m_sd"] / T["prior_A_m"])
g = T.groupby("TIC")
cols = ["e", "e_q16", "e_q84", "e_q025", "e_q975", "q", "q_q16", "q_q84", "q_q025", "q_q975", "q_prior", "q_prior_sd", "edad_Myr", "P"]
S = g[cols].mean()
S["sistema"] = g["sistema"].first().astype(int); S["edad_fuente"] = g["edad_fuente"].first(); S["n_sectores"] = g.size()
S["P_orb"] = IND["P_orb"].reindex(S.index); S["P_orb_err"] = IND["P_orb_err"].reindex(S.index)
S["P_rot"] = IND["P_rot"].reindex(S.index); S["P_rot_err"] = IND["P_rot_err"].reindex(S.index)
S["rot_valida"] = ~S.index.isin(NO_ROTACION)


def edad_err(tic, edad, fuente):
    # (inferior, superior) del intervalo de la edad. massage: σ del catálogo; vecindad: percentiles 16-84 del grupo; sin grupo: sin incertidumbre formal
    if fuente == "vecindad":
        v = VEC.loc[tic]
        if np.isfinite(v["p16_Myr"]) and np.isfinite(v["p84_Myr"]):
            return edad - v["p16_Myr"], v["p84_Myr"] - edad
        return np.nan, np.nan
    sd = CAT.loc[tic, "age_std_1"]
    return min(sd, 0.9 * edad), sd           # la barra inferior se trunca a 0.9 de la edad cuando σ ≥ edad (el eje es logarítmico)


ae = [edad_err(t, r.edad_Myr, r.edad_fuente) for t, r in S.iterrows()]
S["edad_lo"], S["edad_hi"] = [a[0] for a in ae], [a[1] for a in ae]
S = S.sort_values("sistema")
print(S[["sistema", "n_sectores", "e", "q", "q_prior", "edad_Myr", "edad_lo", "edad_hi", "P_orb", "P_rot"]].round(3).to_string())

# ---------------------------------------------------------------- Pléyades: P_orb y P_rot (1b con el secundario enmascarado donde está)
nb1b = json.load(open(RAIZ / "1b_profundidades_eclipse_v2.ipynb"))["cells"]
g1b = {}
exec("".join(nb1b[4]["source"]).replace("MOSTRAR_FIGURAS = True", "MOSTRAR_FIGURAS = False"), g1b)
exec("".join(nb1b[5]["source"]), g1b)
exec("".join(nb1b[10]["source"]).split("filas_rot = []")[0], g1b)            # periodo_rotacion
periodo_rotacion = g1b["periodo_rotacion"]
med = pd.read_csv(PIPE / "interpolador/profundidades_v2_pleyades.csv")
LC = {346541355: PLEY / "tessextractor_local/tess_extractor_ 346541355/Light_Curve_Files"}
LC_DEF = PLEY / "tessextractor_local/tess_extractor_corr/Light_Curve_Files"
fase_sis = med[med.secundario_detectado].groupby("TIC")["fase_secundario"].median()
filas = []
for r in med.itertuples():
    d = pd.read_csv(LC.get(r.TIC, LC_DEF) / f"tic {r.TIC}_S_{r.Sector}_LC.txt", sep=r"\s+").dropna(subset=["#Time", "Flux_CBV_SS"])
    d = d.sort_values("#Time").drop_duplicates("#Time"); t, f = d["#Time"].to_numpy(float), d["Flux_CBV_SS"].to_numpy(float)
    t0c = r.t0 + (r.centro_primario if np.isfinite(r.centro_primario) else 0.0) * r.P
    phi2 = fase_sis[r.TIC] if r.secundario_detectado else 0.5                  # fase del sistema (en 35159657 S71 el secundario medido es espurio)
    rot = periodo_rotacion(t, f, r.P, t0c, r.W, phi2)
    filas.append(dict(TIC=r.TIC, Sector=r.Sector, **rot))
rot = pd.DataFrame(filas); rot.to_csv(DIAG / "poster/rotacion_pleyades_por_sector.csv", index=False)


def consenso(gr):                       # igual que explore_priors_v2.ipynb
    picos = {}
    for _, r in gr.iterrows():
        if isinstance(r["P_rot_picos"], str) and r["P_rot_picos"]:
            picos[int(r["Sector"])] = list(zip([float(x) for x in r["P_rot_picos"].split(";")], [float(x) for x in r["P_rot_picos_pot"].split(";")]))
    mejor = None
    for c in sorted({P for p in picos.values() for P, _ in p}):
        coincide = {}
        for s, p in picos.items():
            cerca = [(P, w) for P, w in p if abs(P / c - 1) < TOL_CONSENSO]
            if cerca:
                coincide[s] = max(cerca, key=lambda x: x[1])
        clave = (len(coincide), sum(w for _, w in coincide.values()))
        if mejor is None or clave > mejor[0]:
            mejor = (clave, coincide)
    (n, _), coincide = mejor
    Ps = np.array([P for P, _ in coincide.values()])
    err = (Ps.max() - Ps.min()) / 2 if len(Ps) > 1 else np.nan
    return pd.Series({"P_rot": float(np.median(Ps)), "P_rot_err": float(err), "n_acuerdo": n, "n_picos": len(picos), "ambiguo": n < len(picos)})


PL = rot.groupby("TIC").apply(consenso)
pri1 = pd.concat([pd.read_csv(PIPE / "priors_pleyades.csv"), pd.read_csv(PIPE / "priors_346541355.csv")])
P_SIS = med.groupby("TIC")["P"].first()
for tic in PL.index:
    pp = pri1[(pri1.TIC == tic) & (pri1.Sector < 200)].P_orb
    pp = pp[(pp / P_SIS[tic] - 1).abs() < 0.02]                                  # sectores donde el paso 1 resolvió P (no 2P)
    PL.loc[tic, "P_orb"] = P_SIS[tic]; PL.loc[tic, "P_orb_err"] = (pp.max() - pp.min()) / 2
PL = PL.sort_values("P_orb"); PL["sistema"] = range(22, 22 + len(PL)); PL["cociente"] = PL.P_rot / PL.P_orb
PL.to_csv(DIAG / "poster/sistemas_pleyades_p_rot_p_orb.csv")
print(PL.round(4).to_string())


# ---------------------------------------------------------------- utilidades de gráfico
def barras(ax, x, y, xl, xh, yl, yh, x95=None, y95=None, color=None):
    ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="none", ecolor=color or GRIS, elinewidth=2.0, capsize=0, zorder=2)
    if x95 is not None:
        ax.errorbar(x, y, xerr=[[x - x95[0]], [x95[1] - x]], yerr=[[y - y95[0]], [y95[1] - y]], fmt="none", ecolor=color or GRIS, elinewidth=0.8, capsize=2, zorder=1)


def etiquetar(ax, x, y, textos, dx=6, dy=5):
    for xi, yi, t in zip(x, y, textos):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(str(t), (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=TINTA2)


def con_colorbar(fig, ax, x, y, c):
    norm = LogNorm(vmin=S["P_orb"].min(), vmax=S["P_orb"].max())
    sc = ax.scatter(x, y, c=c, cmap=CMAP, norm=norm, s=80, edgecolors=TINTA2, linewidths=0.8, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05); cb.set_label("Periodo orbital [d]"); cb.set_ticks([1, 2, 3, 5, 7]); cb.set_ticklabels(["1", "2", "3", "5", "7"])
    cb.outline.set_visible(False); cb.ax.minorticks_off(); return sc


NOTA_BARRAS = "Barras: intervalo creíble de 68 % (grueso) y 95 % (fino).\nValor = media entre sectores; números = sistema. Color = periodo orbital."


def nota(fig, txt):
    fig.text(0.01, -0.01, txt, ha="left", va="top", fontsize=8, color=TINTA2)


def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, format="svg", bbox_inches="tight")
    if os.environ.get("PNG_PRUEBA"):                       # copia PNG solo para revisar el diseño
        fig.savefig(Path(os.environ["PNG_PRUEBA"]) / nombre.replace(".svg", ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig); print("guardado", SALIDA / nombre)


# ---------------------------------------------------------------- 1) (e, q)
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for _, r in S.iterrows():
    barras(ax, r.e, r.q, r.e_q16, r.e_q84, r.q_q16, r.q_q84, (r.e_q025, r.e_q975), (r.q_q025, r.q_q975))
con_colorbar(fig, ax, S.e, S.q, S.P_orb); etiquetar(ax, S.e, S.q, S.sistema)
ax.set_xlabel("Excentricidad, e"); ax.set_ylabel("Razón de masas, q = M$_B$/M$_A$"); ax.set_xlim(-0.01, None); ax.set_ylim(0, None)
fig.tight_layout(); nota(fig, NOTA_BARRAS); guardar(fig, "e_vs_q.svg")

# ---------------------------------------------------------------- 2) (q_prior, q_posterior)
fig, ax = plt.subplots(figsize=(6.6, 5.4))
lim = [0.015, max(S.q_q975.max(), (S.q_prior + S.q_prior_sd).max()) * 1.15]
ax.plot(lim, lim, color=TINTA2, lw=1, ls="--", zorder=1); ax.text(lim[1] * 0.55, lim[1] * 0.45, "posterior = prior", fontsize=9, color=TINTA2, ha="right", rotation=45, rotation_mode="anchor")
for _, r in S.iterrows():
    ax.errorbar(r.q_prior, r.q, xerr=r.q_prior_sd, yerr=[[r.q - r.q_q16], [r.q_q84 - r.q]], fmt="none", ecolor=GRIS, elinewidth=2.0, zorder=2)
    ax.errorbar(r.q_prior, r.q, yerr=[[r.q - r.q_q025], [r.q_q975 - r.q]], fmt="none", ecolor=GRIS, elinewidth=0.8, capsize=2, zorder=1)
con_colorbar(fig, ax, S.q_prior, S.q, S.P_orb); etiquetar(ax, S.q_prior, S.q, S.sistema)
ax.set_xlabel("Razón de masas del prior, q$_{prior}$ = M$_B$/M$_A$"); ax.set_ylabel("Razón de masas del posterior, q$_{posterior}$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.minorticks_off(); ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1]); ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
fig.tight_layout(); nota(fig, "Prior: barra = ±1σ del prior (propaga σ de M$_A$ y M$_B$). Posterior: 68 % (grueso) y 95 % (fino).\nValor = media entre sectores; números = sistema. Color = periodo orbital. Escala logarítmica."); guardar(fig, "q_prior_vs_q_posterior.svg")

# ---------------------------------------------------------------- 3) (e, edad)
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for _, r in S.iterrows():
    if np.isfinite(r.edad_lo):
        ax.errorbar(r.e, r.edad_Myr, xerr=[[r.e - r.e_q16], [r.e_q84 - r.e]], yerr=[[r.edad_lo], [r.edad_hi]], fmt="none", ecolor=GRIS, elinewidth=2.0, zorder=2)
    else:
        ax.errorbar(r.e, r.edad_Myr, xerr=[[r.e - r.e_q16], [r.e_q84 - r.e]], fmt="none", ecolor=GRIS, elinewidth=2.0, zorder=2)
    ax.errorbar(r.e, r.edad_Myr, xerr=[[r.e - r.e_q025], [r.e_q975 - r.e]], fmt="none", ecolor=GRIS, elinewidth=0.8, capsize=2, zorder=1)
con_colorbar(fig, ax, S.e, S.edad_Myr, S.P_orb); etiquetar(ax, S.e, S.edad_Myr, S.sistema)
ax.set_yscale("log"); ax.set_xlim(-0.01, None)
ax.set_xlabel("Excentricidad, e"); ax.set_ylabel("Edad de la primaria [Myr]")
sin = S[~np.isfinite(S.edad_lo)]
txt = ("Excentricidad: 68 % (grueso) y 95 % (fino). Edad: ±1σ de massage (vecindad: percentiles 16-84;\n"
       "la barra inferior se trunca a 0.9×edad). Valor = media entre sectores; números = sistema. Color = periodo orbital."
       + (f"\nSin incertidumbre formal en la edad: sistema {', '.join(str(int(s)) for s in sin.sistema)}." if len(sin) else ""))
fig.tight_layout(); nota(fig, txt); guardar(fig, "e_vs_edad.svg")

# ---------------------------------------------------------------- 4) (P_rot, P_orb) con las Pléyades
fig, ax = plt.subplots(figsize=(7.2, 5.6))
xx = np.array([0.5, 12])
ax.plot(xx, xx, color=TINTA2, lw=1); ax.plot(xx, xx / 2, color=GRIS, lw=0.8, ls=":"); ax.plot(xx, xx * 2, color=GRIS, lw=0.8, ls=":")
for lab, f in (("P$_{rot}$ = P$_{orb}$", 1), ("P$_{rot}$ = P$_{orb}$/2", 0.5), ("P$_{rot}$ = 2 P$_{orb}$", 2)):
    pass
V, X = S[S.rot_valida], S[~S.rot_valida]
for d, mk in ((V, True), (X, False)):
    ax.errorbar(d.P_orb, d.P_rot, xerr=d.P_orb_err.fillna(0), yerr=d.P_rot_err.fillna(0), fmt="none", ecolor=GRIS, elinewidth=1.2, zorder=2)
ax.scatter(V.P_orb, V.P_rot, s=80, color=AZUL, edgecolors=TINTA2, linewidths=0.8, zorder=3, label="Orión: rotación medida")
ax.scatter(X.P_orb, X.P_rot, s=80, facecolors="none", edgecolors=GRIS, linewidths=1.6, zorder=3, label="Orión: modulación orbital (no es rotación)")
ax.errorbar(PL.P_orb, PL.P_rot, xerr=PL.P_orb_err.fillna(0), yerr=PL.P_rot_err.fillna(0), fmt="none", ecolor="#f2b79f", elinewidth=1.2, zorder=2)
ax.scatter(PL.P_orb, PL.P_rot, s=90, marker="D", color=NARANJA, edgecolors=TINTA2, linewidths=0.8, zorder=4, label="Pléyades (sistemas 22-25)")
etiquetar(ax, S.P_orb, S.P_rot, S.sistema); etiquetar(ax, PL.P_orb, PL.P_rot, PL.sistema.astype(int), dx=7, dy=-11)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.7, 10.5); ax.set_ylim(0.6, 10.5)
ax.set_xticks([1, 2, 3, 5, 7, 10]); ax.set_xticklabels(["1", "2", "3", "5", "7", "10"]); ax.set_yticks([1, 2, 3, 5, 7, 10]); ax.set_yticklabels(["1", "2", "3", "5", "7", "10"]); ax.minorticks_off()
ax.text(10.3, 10.3 * 0.86, "1:1", fontsize=9, color=TINTA2, ha="right", va="top")
ax.text(10.3, 10.3 / 2 * 0.86, "1:2", fontsize=9, color=GRIS, ha="right", va="top")
ax.text(5.15, 10.3 * 0.86, "2:1", fontsize=9, color=GRIS, ha="right", va="top")
ax.set_xlabel("Periodo orbital, P$_{orb}$ [d]"); ax.set_ylabel("Periodo de rotación, P$_{rot}$ [d]")
ax.legend(loc="upper left")
fig.tight_layout(); nota(fig, "Líneas: P$_{rot}$/P$_{orb}$ = 1:1, 1:2 y 2:1. Barras: P$_{orb}$ = ½ del rango entre sectores; P$_{rot}$ = ½ del rango entre los sectores que\nconcuerdan (consenso de Lomb-Scargle). Números = sistema. Pléyades: P$_{rot}$ con el secundario enmascarado donde está (consenso ambiguo)."); guardar(fig, "p_rot_vs_p_orb.svg")
