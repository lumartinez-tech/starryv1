"""¿Planeta gigante, enana café o estrella? Clasificación de las secundarias con los posteriores de starry.

Para cada sistema (sin 2, 16, 17) se juntan las muestras posteriores de todos los sectores (12 000 por sector) y se calcula:
  1. P(M_B < 13 M_Jup), P(13-80 M_Jup), P(> 80 M_Jup).
  2. Densidad media rho = M_B / R_B^3 y masa y radio con intervalos.
  3. Inflación del radio: R_B / R_BHAC15(M_B, edad del sistema) - 1, muestra por muestra (edad fija = la del prior).
  4. Temperatura de B desde la razón de brillo superficial observada J = D2/D1 (profundidades v2) frente a la Teff que da BHAC15 para M_B y la edad.
     J se invierte con Planck a 786.5 nm y Teff_A de massage. Es un diagnóstico para la discusión; no entra en el modelo.
Pléyades (22-25): aún no hay posteriores de starry, así que se propagan los priors (R_A, profundidad y edad 125 +- 8 Myr -> R_B -> M_B con BHAC15).
Salida: clasificacion/clasificacion_secundarias.csv y figuras (PNG y SVG).
"""
import glob, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

warnings.filterwarnings("ignore")
RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion"); DIAG = RAIZ / "diagnostico_resultados"
PLEY = Path("/home/luciam/Documents/starry/Resultados_Pleyades"); PIPE = PLEY / "pipeline"
SAL = DIAG / "poster" / "clasificacion"; SAL.mkdir(parents=True, exist_ok=True)
MODELOS = RAIZ / "datos_estrellas_massage_orion/modelos"
EXCLUIR = {2, 16, 17}
MJ, RJ = 1047.57, 9.731                  # Msun -> Mjup, Rsun -> Rjup
RHO_SUN = 1.41                            # g/cm3
H, C, KB, LAM = 6.62607e-34, 2.99792e8, 1.380649e-23, 786.5e-9
XPL = H * C / (LAM * KB)
M_D, M_H = 13.0 / MJ, 80.0 / MJ           # límites de deuterio y de hidrógeno (en Msun)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- BHAC15 con la clase Malla de Orión
nb4 = json.load(open(RAIZ / "4_pipeline_completo.ipynb"))["cells"]
g4 = {}
exec("".join(nb4[1]["source"]), g4)
exec("".join(nb4[8]["source"]).split("mallas = {")[0], g4)
BH = g4["Malla"]("BHAC15", g4["cargar_bhac15"](MODELOS / "iso_barafe/BHAC15_iso.txt"), "isocronas")


def modelo(masa, edad):
    """R [Rsun] y Teff [K] de BHAC15 a esa edad (NaN fuera de 0.01-1.4 Msun)."""
    iso = BH.isocrona(float(edad))
    if iso is None:
        return np.full_like(masa, np.nan), np.full_like(masa, np.nan)
    iso = iso.sort_values("mass"); lm = np.log10(iso["mass"].to_numpy())
    x = np.log10(np.clip(masa, 1e-4, None))
    R = 10 ** np.interp(x, lm, iso["logR"].to_numpy(), left=np.nan, right=np.nan)
    T = 10 ** np.interp(x, lm, iso["logTeff"].to_numpy(), left=np.nan, right=np.nan)
    return R, T


def T_desde_J(J, T1):
    return XPL / np.log1p(np.expm1(XPL / T1) / J)


def q(x, p):
    return np.nanpercentile(x, p)


def resumen(M, R, edad, T1, J_sectores, etiqueta):
    """M [Msun], R [Rsun]: arreglos de muestras."""
    ok = np.isfinite(M) & np.isfinite(R) & (M > 0) & (R > 0)
    M, R = M[ok], R[ok]
    Rm, Tm = modelo(M, edad)
    rho = RHO_SUN * M / R ** 3
    infl = R / Rm - 1
    T_J = [T_desde_J(j, T1) for j in J_sectores if np.isfinite(j) and j > 0]
    f = lambda a, ps=(16, 50, 84): [q(a, p) for p in ps]
    fila = dict(sistema=etiqueta["sistema"], TIC=etiqueta["TIC"], fuente=etiqueta["fuente"], n_sectores=etiqueta["n"], edad_Myr=edad,
                P_menor_13MJ=np.mean(M < M_D), P_13_80MJ=np.mean((M >= M_D) & (M < M_H)), P_mayor_80MJ=np.mean(M >= M_H),
                M_MJ_q16=f(M * MJ)[0], M_MJ_med=f(M * MJ)[1], M_MJ_q84=f(M * MJ)[2], M_MJ_q025=q(M * MJ, 2.5), M_MJ_q975=q(M * MJ, 97.5),
                R_RJ_q16=f(R * RJ)[0], R_RJ_med=f(R * RJ)[1], R_RJ_q84=f(R * RJ)[2], R_RJ_q025=q(R * RJ, 2.5), R_RJ_q975=q(R * RJ, 97.5),
                rho_gcm3_q16=f(rho)[0], rho_gcm3_med=f(rho)[1], rho_gcm3_q84=f(rho)[2],
                inflacion_q16=f(infl)[0], inflacion_med=f(infl)[1], inflacion_q84=f(infl)[2], P_inflada=np.mean(infl[np.isfinite(infl)] > 0),
                T_modelo_q16=f(Tm)[0], T_modelo_med=f(Tm)[1], T_modelo_q84=f(Tm)[2],
                T_J_min=min(T_J) if T_J else np.nan, T_J_max=max(T_J) if T_J else np.nan, T1=T1,
                J_obs=np.nanmean(J_sectores) if len(J_sectores) else np.nan)
    if np.mean(M > 1.4) > 0.2:      # BHAC15 llega a 1.4 Msun: lo derivado del modelo no vale
        for k in ("inflacion_q16", "inflacion_med", "inflacion_q84", "P_inflada", "T_modelo_q16", "T_modelo_med", "T_modelo_q84"):
            fila[k] = np.nan
        fila["fuera_de_BHAC15"] = True
    else:
        fila["fuera_de_BHAC15"] = False
    return fila


# ---------------------------------------------------------------- Orión: posteriores por sistema
T = pd.read_csv(DIAG / "tabla_maestra.csv")
T = T[~T["sistema"].isin(EXCLUIR) & T["sistema"].notna()]
filas = []
for tic, g in T.groupby("TIC"):
    Ms, Rs = [], []
    for _, r in g.iterrows():
        P = pd.read_csv(glob.glob(f"{r['dir']}/*Posteriors.csv")[0], index_col=0)
        Ms.append(P["B_m"].to_numpy()); Rs.append(P["B_r"].to_numpy())
    M, R = np.concatenate(Ms), np.concatenate(Rs)
    J = [d2 / d1 for d1, d2, det in zip(g["D1"], g["D2"], g["sec_det"]) if bool(det) and d1 > 0]
    filas.append(resumen(M, R, g["edad_Myr"].iloc[0], g["Teff_A"].iloc[0], J,
                         dict(sistema=int(g["sistema"].iloc[0]), TIC=int(tic), fuente="posterior de starry", n=len(g))))

# ---------------------------------------------------------------- Pléyades: priors propagados
inter = PIPE / "interpolador"
prim = pd.read_csv(inter / "primarias_pleyades.csv").set_index("TIC")
med = pd.read_csv(inter / "profundidades_v2_pleyades.csv")
orden = med.groupby("TIC")["P"].first().sort_values()
N = 6000
for k, tic in enumerate(orden.index):
    a = prim.loc[tic]; g = med[(med.TIC == tic) & (med.Calidad_depth == "ok")]
    w = 1 / g.D_primario_err ** 2; D = (w * g.D_primario).sum() / w.sum(); sD = max((w.sum()) ** -0.5, 0.01 * D)
    RA = np.clip(rng.normal(a.R1, a.R1_err, N), 0.05, None); Dm = np.clip(rng.normal(D, sD, N), 1e-4, 0.9)
    RB = RA * np.sqrt(Dm)
    edades = np.round(rng.normal(125, 8, N) / 3) * 3          # edad fija 125 ± 8 Myr (rejilla de 3 Myr para reusar isocronas)
    M = np.full(N, np.nan)
    for e in np.unique(edades):
        iso = BH.isocrona(float(e))
        if iso is None:
            continue
        iso = iso.sort_values("mass"); s = edades == e
        # primer cruce en R(M) creciente (BHAC15 es monótona en este rango)
        M[s] = 10 ** np.interp(np.log10(RB[s]), iso["logR"].to_numpy(), np.log10(iso["mass"].to_numpy()), left=np.nan, right=np.nan)
    dd = g.groupby("Sector")[["D_primario", "D_secundario"]].first()
    J = [d2 / d1 for d1, d2 in zip(dd.D_primario, dd.D_secundario) if d1 > 0 and g.secundario_detectado.any()]
    fila = resumen(M, RB, 125.0, a.Teff1, J, dict(sistema=22 + k, TIC=int(tic), fuente="prior propagado (sin ajuste de starry)", n=len(g)))
    fila["inflacion_q16"] = fila["inflacion_med"] = fila["inflacion_q84"] = fila["P_inflada"] = np.nan    # R_B define M_B: no es una prueba independiente
    filas.append(fila)

C_ = pd.DataFrame(filas).sort_values("M_MJ_med").reset_index(drop=True)
C_.to_csv(SAL / "clasificacion_secundarias.csv", index=False)
pd.set_option("display.width", 250)
print(C_[["sistema", "fuente", "P_menor_13MJ", "P_13_80MJ", "P_mayor_80MJ", "M_MJ_med", "M_MJ_q16", "M_MJ_q84", "R_RJ_med", "rho_gcm3_med", "inflacion_med", "P_inflada", "T_modelo_med", "T_J_min", "T_J_max"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- figuras
AZUL, NARANJA, GRIS, TINTA, TINTA2, SUP = "#2a78d6", "#eb6834", "#8a8a85", "#0b0b0b", "#52514e", "#fcfcfb"
COL = {"planeta": "#2a78d6", "enana": "#eb6834", "estrella": "#b8b7b2"}
matplotlib.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.facecolor": SUP, "axes.facecolor": SUP, "savefig.facecolor": SUP, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "text.color": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 11, "axes.labelsize": 12, "legend.frameon": False, "legend.fontsize": 9.5})


def guardar(fig, nombre):
    fig.savefig(SAL / f"{nombre}.svg", format="svg", bbox_inches="tight"); fig.savefig(SAL / f"{nombre}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def nota(fig, txt):
    fig.text(0.01, -0.01, txt, ha="left", va="top", fontsize=8, color=TINTA2)


# 1) probabilidades por clase
d = C_.sort_values("M_MJ_med").reset_index(drop=True); y = np.arange(len(d))
fig, ax = plt.subplots(figsize=(8, 0.34 * len(d) + 1.6))
ax.barh(y, d.P_menor_13MJ, color=COL["planeta"], label="planeta gigante (< 13 M$_{Jup}$)")
ax.barh(y, d.P_13_80MJ, left=d.P_menor_13MJ, color=COL["enana"], label="enana café (13–80 M$_{Jup}$)")
ax.barh(y, d.P_mayor_80MJ, left=d.P_menor_13MJ + d.P_13_80MJ, color=COL["estrella"], label="estrella (> 80 M$_{Jup}$)")
ax.set_yticks(y); ax.set_yticklabels([f"{int(s)}" + ("  (Pléyades*)" if s >= 22 else "") for s in d.sistema]); ax.set_ylim(-0.6, len(d) - 0.4)
ax.set_xlim(0, 1); ax.set_xlabel("Probabilidad según el posterior de M$_B$"); ax.set_ylabel("Sistema (ordenado por masa de B)")
for yi, (m, lo, hi) in enumerate(zip(d.M_MJ_med, d.M_MJ_q16, d.M_MJ_q84)):
    ax.text(1.01, yi, f"{m:.0f} ({lo:.0f}–{hi:.0f}) M$_{{Jup}}$", va="center", fontsize=8, color=TINTA2)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=8.5); ax.grid(axis="y", visible=False)
fig.tight_layout(); nota(fig, "Texto a la derecha: mediana y intervalo de 68 % de M$_B$. Posterior de starry juntando los sectores; edad fija por sistema.\n*Pléyades: sin ajuste de starry todavía; prior propagado (R$_A$, profundidad y edad 125 ± 8 Myr → R$_B$ → M$_B$ con BHAC15).")
guardar(fig, "01_probabilidades_clase")

# 2) masa-radio con BHAC15
fig, ax = plt.subplots(figsize=(8, 5.8))
mm = np.logspace(np.log10(0.011), np.log10(1.39), 300)
for edad, ls in ((1, ":"), (10, "--"), (125, "-"), (1000, "-.")):
    R_, _ = modelo(mm, edad); ax.plot(mm * MJ, R_ * RJ, color=GRIS, ls=ls, lw=1.2); ax.text(mm[-1] * MJ * 1.04, R_[-1] * RJ, f"{edad} Myr" if edad < 1000 else "1 Gyr", fontsize=8, color=TINTA2, ha="left", va="center")
for x, t in ((13, "13 M$_{Jup}$ (deuterio)"), (80, "80 M$_{Jup}$ (hidrógeno)")):
    ax.axvline(x, color=TINTA2, lw=0.8, ls=":"); ax.text(x * 1.04, 0.45, t, rotation=90, fontsize=8.5, color=TINTA2, va="bottom")
norm = LogNorm(vmin=0.5, vmax=200)
cm = LinearSegmentedColormap.from_list("azul_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
for _, r in d.iterrows():
    ax.errorbar(r.M_MJ_med, r.R_RJ_med, xerr=[[r.M_MJ_med - r.M_MJ_q16], [r.M_MJ_q84 - r.M_MJ_med]], yerr=[[r.R_RJ_med - r.R_RJ_q16], [r.R_RJ_q84 - r.R_RJ_med]], fmt="none", ecolor=GRIS, elinewidth=1.8, zorder=2)
    ax.errorbar(r.M_MJ_med, r.R_RJ_med, xerr=[[r.M_MJ_med - r.M_MJ_q025], [r.M_MJ_q975 - r.M_MJ_med]], yerr=[[r.R_RJ_med - r.R_RJ_q025], [r.R_RJ_q975 - r.R_RJ_med]], fmt="none", ecolor=GRIS, elinewidth=0.7, capsize=2, zorder=1)
pl = d[d.sistema >= 22]; ori = d[d.sistema < 22]
sc = ax.scatter(ori.M_MJ_med, ori.R_RJ_med, c=ori.edad_Myr, cmap=cm, norm=norm, s=70, edgecolors=TINTA2, linewidths=0.8, zorder=3)
ax.scatter(pl.M_MJ_med, pl.R_RJ_med, s=85, marker="D", color=NARANJA, edgecolors=TINTA2, linewidths=0.8, zorder=4, label="Pléyades (prior propagado)")
for _, r in d.iterrows():
    ax.annotate(f"{int(r.sistema)}", (r.M_MJ_med, r.R_RJ_med), xytext=(6, 5), textcoords="offset points", fontsize=9, color=TINTA2)
cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05); cb.set_label("Edad de Orión [Myr]"); cb.outline.set_visible(False)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8, 7000); ax.set_ylim(0.4, 35)
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xticks([10, 20, 50, 100, 300, 1000, 3000]); ax.set_yticks([0.5, 1, 2, 5, 10, 30]); ax.minorticks_off()
ax.set_xlabel("Masa de la secundaria, M$_B$ [M$_{Jup}$]"); ax.set_ylabel("Radio de la secundaria, R$_B$ [R$_{Jup}$]"); ax.legend(loc="upper right")
fig.tight_layout(); nota(fig, "Líneas grises: isocronas BHAC15 a 1, 10, 125 Myr y 1 Gyr. Barras: 68 % (grueso) y 95 % (fino) del posterior juntando sectores.\nPuntos de Orión: color = edad; números = sistema. Rombos naranja: Pléyades, con la masa derivada de R$_B$ (por eso quedan sobre la isocrona de 125 Myr).")
guardar(fig, "02_masa_radio_bhac15")

# 3) inflación del radio
fig, ax = plt.subplots(figsize=(7.4, 0.34 * int(ori.inflacion_med.notna().sum()) + 1.8))
o = ori[ori.inflacion_med.notna()].sort_values("inflacion_med").reset_index(drop=True)
for i, r in o.iterrows():
    ax.errorbar(100 * r.inflacion_med, i, xerr=[[100 * (r.inflacion_med - r.inflacion_q16)], [100 * (r.inflacion_q84 - r.inflacion_med)]], fmt="o", color=AZUL, ecolor=GRIS, elinewidth=2, mec=TINTA2, ms=7)
ax.axvline(0, color=TINTA2, lw=1)
for x, t in ((5, "5–14 % (Beatty 2026)"), ):
    ax.axvspan(5, 14, color="#cde2fb", alpha=0.5, lw=0); ax.text(9.5, len(o) - 0.2, t, ha="center", fontsize=8, color=TINTA2)
ax.set_yticks(range(len(o))); ax.set_yticklabels([int(s) for s in o.sistema]); ax.grid(axis="y", visible=False)
ax.set_xlim(-60, 140); ax.set_xticks([-40, -20, 0, 20, 40, 80, 120]); ax.set_xlabel("Inflación del radio, R$_B$ / R$_{BHAC15}$(M$_B$, edad) − 1  [%]"); ax.set_ylabel("Sistema")
fig.tight_layout(); nota(fig, "Punto: mediana; barra: intervalo de 68 %. Edad fija del prior; no incluye el error de la edad. Franja: inflación media de enanas cafés transitantes (5–14 %).\nSolo sistemas con posterior de starry (las Pléyades no tienen: R$_B$ define M$_B$).")
guardar(fig, "03_inflacion_radio")

# 4) Teff de B: desde J vs modelo
fig, ax = plt.subplots(figsize=(6.6, 5.6))
v = d[np.isfinite(d.T_J_min) & np.isfinite(d.T_modelo_med)]
xx = np.array([1500, 20000]); ax.plot(xx, xx, color=TINTA2, lw=1)
for _, r in v.iterrows():
    ax.errorbar(r.T_modelo_med, 0.5 * (r.T_J_min + r.T_J_max), xerr=[[r.T_modelo_med - r.T_modelo_q16], [r.T_modelo_q84 - r.T_modelo_med]], yerr=[[0.5 * (r.T_J_min + r.T_J_max) - r.T_J_min], [r.T_J_max - 0.5 * (r.T_J_min + r.T_J_max)]], fmt="none", ecolor=GRIS, elinewidth=1.8, zorder=2)
vo, vp = v[v.sistema < 22], v[v.sistema >= 22]
ax.scatter(vo.T_modelo_med, 0.5 * (vo.T_J_min + vo.T_J_max), s=70, color=AZUL, edgecolors=TINTA2, linewidths=0.8, zorder=3, label="Orión")
ax.scatter(vp.T_modelo_med, 0.5 * (vp.T_J_min + vp.T_J_max), s=85, marker="D", color=NARANJA, edgecolors=TINTA2, linewidths=0.8, zorder=4, label="Pléyades")
for _, r in v.iterrows():
    ax.annotate(f"{int(r.sistema)}", (r.T_modelo_med, 0.5 * (r.T_J_min + r.T_J_max)), xytext=(6, 5), textcoords="offset points", fontsize=9, color=TINTA2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2000, 16000); ax.set_ylim(2000, 16000)
ax.set_xticks([2500, 4000, 6000, 10000, 15000]); ax.set_yticks([2500, 4000, 6000, 10000, 15000])
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v_, _: f"{v_:g}"))
ax.minorticks_off(); ax.legend(loc="upper left")
ax.set_xlabel("T$_{eff}$ de B según BHAC15 (M$_B$, edad) [K]"); ax.set_ylabel("T$_{eff}$ de B desde la razón de profundidades D$_2$/D$_1$ [K]")
fig.tight_layout(); nota(fig, "Eje y: Planck a 786.5 nm con J = D$_2$/D$_1$ y T$_{eff}$ de la primaria (massage); barra = rango entre sectores. Eje x: 68 % del posterior.\nPor encima de la línea, B es más caliente de lo que implica su masa. La tercera luz no lo explica: reduce D$_2$ y D$_1$ por igual y J no cambia.\nPosibles causas: masa o edad equivocadas, D$_2$/D$_1$ ≠ J (eclipses parciales) u otro objeto que eclipsa.")
guardar(fig, "04_teff_J_vs_modelo")

# 5) masa-radio con color = excentricidad, forma = muestra y tamaño proporcional al radio de la primaria
Rexc = T.groupby("sistema")[["A_r", "e"]].mean()
ecc, RA = {}, {}
for _, r in d.iterrows():
    if r.sistema < 22:
        ecc[int(r.sistema)] = Rexc.loc[int(r.sistema), "e"]; RA[int(r.sistema)] = Rexc.loc[int(r.sistema), "A_r"]
    else:
        tic_ = int(r.TIC); ec = med[med.TIC == tic_].ecosw.median()
        ecc[int(r.sistema)] = abs(ec) if np.isfinite(ec) else np.nan        # cota inferior: |e cos w| de la fase del secundario
        RA[int(r.sistema)] = prim.loc[tic_, "R1"]
ESCALA = 7.0                                                                   # puntos de diámetro por R_sun de la primaria
fig, ax = plt.subplots(figsize=(8.6, 5.9))
for edad, ls in ((1, ":"), (10, "--"), (125, "-"), (1000, "-.")):
    R_, _ = modelo(mm, edad); ax.plot(mm * MJ, R_ * RJ, color=GRIS, ls=ls, lw=1.2, zorder=1)
    ax.text(mm[-1] * MJ * 1.04, R_[-1] * RJ, f"{edad} Myr" if edad < 1000 else "1 Gyr", fontsize=8, color=TINTA2, ha="left", va="center")
for x, t in ((13, "13 M$_{Jup}$ (deuterio)"), (80, "80 M$_{Jup}$ (hidrógeno)")):
    ax.axvline(x, color=TINTA2, lw=0.8, ls=":"); ax.text(x * 1.04, 0.45, t, rotation=90, fontsize=8.5, color=TINTA2, va="bottom")
for _, r in d.iterrows():
    ax.errorbar(r.M_MJ_med, r.R_RJ_med, xerr=[[r.M_MJ_med - r.M_MJ_q16], [r.M_MJ_q84 - r.M_MJ_med]], yerr=[[r.R_RJ_med - r.R_RJ_q16], [r.R_RJ_q84 - r.R_RJ_med]], fmt="none", ecolor=GRIS, elinewidth=1.6, alpha=0.8, zorder=2)
    ax.errorbar(r.M_MJ_med, r.R_RJ_med, xerr=[[r.M_MJ_med - r.M_MJ_q025], [r.M_MJ_q975 - r.M_MJ_med]], yerr=[[r.R_RJ_med - r.R_RJ_q025], [r.R_RJ_q975 - r.R_RJ_med]], fmt="none", ecolor=GRIS, elinewidth=0.6, capsize=1.5, alpha=0.7, zorder=1)
norm_e = plt.Normalize(vmin=0, vmax=0.35)
for _, r in d.iterrows():
    sis = int(r.sistema); e_ = ecc[sis]; marca = "D" if sis >= 22 else "o"
    tam = (ESCALA * RA[sis]) ** 2
    if np.isfinite(e_):
        ax.scatter(r.M_MJ_med, r.R_RJ_med, s=tam, marker=marca, c=[e_], cmap=cm, norm=norm_e, edgecolors=TINTA, linewidths=1.0, zorder=3, alpha=0.95)
    else:
        ax.scatter(r.M_MJ_med, r.R_RJ_med, s=tam, marker=marca, facecolors="none", edgecolors=TINTA2, linewidths=1.4, zorder=3)
    ax.annotate(f"{sis}", (r.M_MJ_med, r.R_RJ_med), xytext=(0, 0), textcoords="offset points", fontsize=7.5, color="white" if (np.isfinite(e_) and e_ > 0.17) else TINTA, ha="center", va="center", zorder=5)
sm = matplotlib.cm.ScalarMappable(norm=norm_e, cmap=cm); cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05); cb.set_label("Excentricidad, e"); cb.outline.set_visible(False)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8, 7000); ax.set_ylim(0.4, 35)
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xticks([10, 20, 50, 100, 300, 1000, 3000]); ax.set_yticks([0.5, 1, 2, 5, 10, 30]); ax.minorticks_off()
ax.set_xlabel("Masa de la secundaria, M$_B$ [M$_{Jup}$]"); ax.set_ylabel("Radio de la secundaria, R$_B$ [R$_{Jup}$]")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", mfc="#9ec5f4", mec=TINTA, ms=8, label="Orión"),
     Line2D([], [], marker="D", ls="", mfc="#9ec5f4", mec=TINTA, ms=8, label="Pléyades (prior propagado; e = cota inferior)"),
     Line2D([], [], marker="o", ls="", mfc="none", mec=TINTA2, ms=8, label="sin dato de e")]
ax.legend(handles=h, loc="upper left", fontsize=8, labelspacing=0.9, borderpad=0.6)
# escala de tamaños (diámetro proporcional a R_A) en la esquina inferior derecha, en coordenadas del eje
xs = [0.60, 0.72, 0.84, 0.95]
for xk, rr in zip(xs, (0.5, 1, 2, 5)):
    ax.scatter([xk], [0.085], s=(ESCALA * rr) ** 2, marker="o", facecolors="#e6e5e1", edgecolors=TINTA2, linewidths=1.0, transform=ax.transAxes, zorder=6, clip_on=False)
    ax.text(xk, 0.175, f"{rr:g} R$_☉$", transform=ax.transAxes, ha="center", va="center", fontsize=8, color=TINTA2, zorder=6)
ax.text(0.775, 0.225, "Tamaño: radio de la primaria R$_A$", transform=ax.transAxes, ha="center", fontsize=8.5, color=TINTA2, zorder=6)
fig.tight_layout(); nota(fig, "Color = excentricidad (Orión: media posterior; Pléyades: |e cos ω| de la fase del secundario, cota inferior; hueco = sin secundario detectado). Forma = muestra.\nTamaño (diámetro) proporcional al radio de la primaria R$_A$. Líneas grises: isocronas BHAC15 a 1, 10, 125 Myr y 1 Gyr. Barras: 68 % (grueso) y 95 % (fino). Números = sistema.")
guardar(fig, "05_masa_radio_excentricidad")

# 6) igual que la 02 (tamaño fijo, isocronas BHAC15 para leer la edad), con la barra de color = excentricidad
fig, ax = plt.subplots(figsize=(8, 5.8))
for edad, ls in ((1, ":"), (10, "--"), (125, "-"), (1000, "-.")):
    R_, _ = modelo(mm, edad); ax.plot(mm * MJ, R_ * RJ, color=GRIS, ls=ls, lw=1.2, zorder=1)
    ax.text(mm[-1] * MJ * 1.04, R_[-1] * RJ, f"{edad} Myr" if edad < 1000 else "1 Gyr", fontsize=8, color=TINTA2, ha="left", va="center")
for x, t in ((13, "13 M$_{Jup}$ (deuterio)"), (80, "80 M$_{Jup}$ (hidrógeno)")):
    ax.axvline(x, color=TINTA2, lw=0.8, ls=":"); ax.text(x * 1.04, 0.45, t, rotation=90, fontsize=8.5, color=TINTA2, va="bottom")
for _, r in d.iterrows():
    ax.errorbar(r.M_MJ_med, r.R_RJ_med, xerr=[[r.M_MJ_med - r.M_MJ_q16], [r.M_MJ_q84 - r.M_MJ_med]], yerr=[[r.R_RJ_med - r.R_RJ_q16], [r.R_RJ_q84 - r.R_RJ_med]], fmt="none", ecolor=GRIS, elinewidth=1.8, zorder=2)
    ax.errorbar(r.M_MJ_med, r.R_RJ_med, xerr=[[r.M_MJ_med - r.M_MJ_q025], [r.M_MJ_q975 - r.M_MJ_med]], yerr=[[r.R_RJ_med - r.R_RJ_q025], [r.R_RJ_q975 - r.R_RJ_med]], fmt="none", ecolor=GRIS, elinewidth=0.7, capsize=2, zorder=1)
for _, r in d.iterrows():
    sis = int(r.sistema); e_ = ecc[sis]; pl_ = sis >= 22
    kw = dict(s=95 if pl_ else 75, marker="D" if pl_ else "o", zorder=3)
    if np.isfinite(e_):
        ax.scatter(r.M_MJ_med, r.R_RJ_med, c=[e_], cmap=cm, norm=norm_e, edgecolors=TINTA, linewidths=0.9, **kw)
    else:
        ax.scatter(r.M_MJ_med, r.R_RJ_med, facecolors="none", edgecolors=TINTA2, linewidths=1.4, **kw)
    ax.annotate(f"{sis}", (r.M_MJ_med, r.R_RJ_med), xytext=(6, 5), textcoords="offset points", fontsize=9, color=TINTA2)
sm = matplotlib.cm.ScalarMappable(norm=norm_e, cmap=cm); cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05); cb.set_label("Excentricidad, e"); cb.outline.set_visible(False)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8, 7000); ax.set_ylim(0.4, 35)
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xticks([10, 20, 50, 100, 300, 1000, 3000]); ax.set_yticks([0.5, 1, 2, 5, 10, 30]); ax.minorticks_off()
ax.set_xlabel("Masa de la secundaria, M$_B$ [M$_{Jup}$]"); ax.set_ylabel("Radio de la secundaria, R$_B$ [R$_{Jup}$]")
h = [Line2D([], [], marker="o", ls="", mfc="#9ec5f4", mec=TINTA, ms=8, label="Orión"),
     Line2D([], [], marker="D", ls="", mfc="#9ec5f4", mec=TINTA, ms=8, label="Pléyades (prior propagado; e = cota inferior)"),
     Line2D([], [], marker="o", ls="", mfc="none", mec=TINTA2, ms=8, label="sin dato de e")]
ax.legend(handles=h, loc="upper left", fontsize=8.5)
fig.tight_layout(); nota(fig, "Color = excentricidad (Orión: media posterior; Pléyades: |e cos ω| de la fase del secundario, cota inferior; hueco = sin secundario detectado). Forma = muestra.\nLíneas grises: isocronas BHAC15 a 1, 10, 125 Myr y 1 Gyr (leer la edad). Barras: 68 % (grueso) y 95 % (fino) del posterior juntando sectores. Números = sistema.")
guardar(fig, "06_masa_radio_color_excentricidad")
print("figuras en", SAL)
