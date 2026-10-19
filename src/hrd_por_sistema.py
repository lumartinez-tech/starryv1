"""
Diagramas HR por sistema con las isócronas de cada modelo (MIST, PARSEC, Somers00, Somers17, BHAC15)
y prueba rápida de edades por fase (pms y pms+ms) con las mismas mallas de MassAge.

- Mallas: las de MassAge (~/Documents/massage/massage/models), con los mismos filtros de fase que hrd.py:
  phase_sel = 'pms' (solo pre-MS) o 'ms' (pre-MS + MS; es acumulativo).
- Primaria: T_eff y log L de MassAge (catálogo del pipeline, entrada T_eff; 8, 13 y 14 de la corrida pms).
- Edades de MassAge: edades_largo.csv (tabla_edades.py).
- Secundaria: T_eff_B y L_B del pipeline (resumen_por_tic_sector.csv) e isócrona BHAC15 (isocronas_interp.py) a edad_B.
- Prueba rápida: Monte Carlo de (T, log L) ~ N(mediana, std) independientes, interpolación lineal en (T, log L) -> edad,
  mediana y percentiles 16-84. NO rehace la extinción ni la covarianza T-L de MassAge: solo indica tendencias.

Salidas: figuras/sistema_XX.png, edades_prueba_fases.csv
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator

AQUI = Path(__file__).parent
MOD = Path("/home/luciam/Documents/massage/massage/models")
O = Path("/home/luciam/Documents/starry/Resultados_Orion")
sys.path.insert(0, str(O / "datos_estrellas_massage_orion"))
from isocronas_interp import load_isochrone_model  # noqa: E402

FIG = AQUI / "figuras"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
NMC = 300

# ---------------- mallas (mismos filtros que MassAge/hrd.py) ----------------
def malla(modelo, fase):
    if modelo == "MIST":
        d = pd.read_csv(MOD / "MIST/MIST_V2.csv")
        d["trk"] = (d.star_age.diff() < 0).cumsum()
        lim = {"pms": -1, "ms": 0}[fase]
        d = d[d.phase <= lim]
        return pd.DataFrame(dict(Te=10 ** d.log_Teff, L=d.log_L, edad=d.star_age / 1e6, M=d.trk))
    if modelo == "PARSEC":
        d = pd.read_csv(MOD / "PARSEC/PARSEC_logage_001_V3.csv")
        lim = {"pms": 0, "ms": 1}[fase]
        d = d[d.label <= lim]
        return pd.DataFrame(dict(Te=10 ** d.logTe, L=d.logL, edad=10 ** d.logAge / 1e6, M=d.Mass, logAge=d.logAge))
    if modelo in ("Somers00", "Somers17"):
        d = pd.read_csv(MOD / f"Somers/{'f000' if modelo == 'Somers00' else 'f017'}track_ms.csv")
        return pd.DataFrame(dict(Te=10 ** d["log(Teff)"], L=d["log(L/Lsun)"], edad=d["Age(Gyr)"] * 1e3, M=d.Mass))
    if modelo == "BHAC15":
        d = pd.read_csv(MOD / "BARAFFE/BARAFFE.csv")
        return pd.DataFrame(dict(Te=10 ** d.logTeff, L=d.LogLum, edad=d.age, M=d.mass))

MODELOS = ["MIST", "PARSEC", "Somers00", "Somers17", "BHAC15"]
COL = {"MIST": "#1f77b4", "PARSEC": "#2ca02c", "Somers00": "#9467bd", "Somers17": "#d62728", "BHAC15": "#ff7f0e"}
mallas = {(m, f): malla(m, f) for m in MODELOS for f in ("pms", "ms")}
for (m, f), g in mallas.items():
    g.query("edad >= 0.01 and edad <= 1000", inplace=True)

_TRACKS = {}
def _tracks(g):
    k = id(g)
    if k not in _TRACKS:
        tr = []
        for M, d in g.groupby("M"):
            d = d.sort_values("edad")
            if len(d) >= 2:
                tr.append((M, np.log10(d.edad.values), np.log10(d.Te.values), d.L.values))
        _TRACKS[k] = tr
    return _TRACKS[k]

def isocrona(g, edad):
    """Isócrona a una edad: para cada masa, interpola log T y log L en log edad.
    PARSEC ya viene en isócronas (malla de log edad de 0.01 dex): se toma la más cercana."""
    if "logAge" in g:
        la = g.logAge.values; k = np.abs(la - np.log10(edad * 1e6)).min()
        if k > 0.02: return None
        sub = g[np.isclose(la, la[np.abs(la - np.log10(edad * 1e6)).argmin()])].sort_values("M")
        return np.log10(sub.Te.values), sub.L.values
    le0 = np.log10(edad); pts = []
    for M, le, lT, lL in _tracks(g):
        if le[0] <= le0 <= le[-1]:
            pts.append((M, np.interp(le0, le, lT), np.interp(le0, le, lL)))
    if not pts:
        return None
    p = np.array(sorted(pts))
    return p[:, 1], p[:, 2]

def mallas_iso(m):
    """Para graficar: pms+ms si existe (Somers y BHAC15 no distinguen fase)."""
    return mallas[(m, "ms")]

# ---------------- estrellas ----------------
cat = pd.read_csv(O / "catalogo_completo_mist_sommers.csv")
muestra = pd.read_csv(O / "diagnostico_resultados/muestra_completa_orion_pleyades.csv")
cs = [c for c in muestra.columns if "sistema" in c.lower()][0]
muestra = muestra.rename(columns={cs: "sistema"})
pms = pd.read_csv(O / "datos_estrellas_massage_orion/corrida_pms_8_13_14_20261013/resumen_modelos.csv")
L = pd.read_csv(AQUI / "edades_largo.csv", dtype={"sistema": str})
res = pd.read_csv(O / "resultados_pipeline/resumen_por_tic_sector.csv")
TM = pd.read_csv("/home/luciam/Documents/starry/preliminares_revision1/datos/tabla_maestra.csv")
bh = load_isochrone_model("BHAC15", O / "datos_estrellas_massage_orion/modelos/iso_barafe/BHAC15_iso.txt")

def primaria(tic):
    r = cat[cat.TIC == tic]
    if len(r) and np.isfinite(r.T_median_1.iloc[0]):
        r = r.iloc[0]
        return r.T_median_1, r.T_std_1, r.lgL_median_1, r.lgL_std_1, "MassAge, entrada T_eff (MIST)"
    q = pms[(pms.TIC == tic) & (pms.modelo == "mist_pms")]
    if len(q):
        q = q.iloc[0]; return q["T"], np.nan, q.lgL, np.nan, "corrida pms 13-oct"
    return None

def edades_massage(sist, modelo):
    alias = {"Somers00": "SOMERS00", "Somers17": "SOMERS17", "BHAC15": "BARAFFE"}.get(modelo, modelo)
    q = L[(L.sistema == sist) & (L.modelo == alias)].dropna(subset=["edad_med"])
    return q.drop_duplicates(subset=["edad_med"])

def prueba(T, sT, lL, slL, modelo, fase):
    g = mallas[(modelo, fase)]
    if not np.isfinite(sT): sT = 0.03 * T
    if not np.isfinite(slL): slL = 0.1
    Ts = rng.normal(T, sT, NMC); Ls = rng.normal(lL, slL, NMC)
    # recorta la malla alrededor de la estrella para que la triangulación sea rápida
    box = g[(g.Te.between(T - 6 * sT, T + 6 * sT)) & (g.L.between(lL - 6 * slL, lL + 6 * slL))]
    if len(box) < 10:
        return np.nan, np.nan, np.nan, 0.0
    f = LinearNDInterpolator(np.column_stack([np.log10(box.Te), box.L]), np.log10(box.edad))
    e = 10 ** f(np.log10(Ts), Ls)
    ok = np.isfinite(e)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan, ok.mean()
    return np.median(e[ok]), np.percentile(e[ok], 16), np.percentile(e[ok], 84), ok.mean()

filas = []
orden = [str(i) for i in range(1, 22)]
for sist in orden:
    m = muestra[muestra.sistema.astype(str).str.replace(".0", "", regex=False) == sist]
    if not len(m): continue
    tic = int(m.TIC.iloc[0]); p = primaria(tic)
    if p is None:
        print(f"sistema {sist}: sin T_eff/L de MassAge, se omite"); continue
    T, sT, lL, slL, fuenteP = p
    rs = res[res.TIC == tic]
    sec = rs.dropna(subset=["Teff_B"])
    usada = L[(L.sistema == sist)].iloc[0:0]

    fig, axs = plt.subplots(1, 5, figsize=(24, 5.4), sharey=True)
    for ax, mod in zip(axs, MODELOS):
        g = mallas_iso(mod)
        for e_ref in (0.5, 1, 3, 10, 30, 100):
            iso = isocrona(g, e_ref)
            if iso is not None:
                ax.plot(10 ** iso[0], iso[1], color="0.82", lw=0.8, zorder=1)
                k = np.argmin(np.abs(10 ** iso[0] - T)); ax.annotate(f"{e_ref:g}", (10 ** iso[0][k], iso[1][k]), fontsize=7, color="0.55")
        for r in edades_massage(sist, mod).itertuples():
            ls = "-" if "pipeline" in r.fuente else ("--" if "SpT" in r.entrada else ":")
            iso = isocrona(g, r.edad_med)
            if iso is not None:
                ax.plot(10 ** iso[0], iso[1], color=COL[mod], lw=2, ls=ls, zorder=3,
                        label=f"{r.edad_med:.3g} Myr ({r.entrada}{', ' + r.fase if r.fase not in ('—', '?') else ''})")
            for e2 in (r.edad_lo, r.edad_up):
                if np.isfinite(e2) and e2 > 0:
                    iso = isocrona(g, e2)
                    if iso is not None:
                        ax.plot(10 ** iso[0], iso[1], color=COL[mod], lw=0.8, ls=ls, alpha=0.5, zorder=2)
        for fase in ("pms", "ms"):
            if mod in ("Somers00", "Somers17", "BHAC15") and fase == "pms":
                continue  # sin etiquetas de fase en estas mallas
            med, lo, up, frac = prueba(T, sT, lL, slL, mod, fase)
            filas.append(dict(sistema=sist, TIC=tic, modelo=mod, fase=("pms+ms" if fase == "ms" else "pms") if mod in ("MIST", "PARSEC") else "sin fases",
                              edad_med=med, edad_p16=lo, edad_p84=up, frac_en_malla=frac))
        ax.errorbar(T, lL, xerr=sT if np.isfinite(sT) else None, yerr=slL if np.isfinite(slL) else None,
                    fmt="*", ms=15, color="k", zorder=5, label="primaria")
        if mod == "BHAC15" and len(sec):
            s = sec.iloc[0]
            ax.plot(s.Teff_B, np.log10(s.L_B), "s", ms=9, mfc="none", mec="k", zorder=5, label=f"secundaria (S{int(s.Sector)})")
            q = TM[TM.TIC == tic].edad_Myr.dropna()
            eB = q.iloc[0] if len(q) else np.nan   # edad de las corridas vigentes (tabla_maestra), no la de resumen_por_tic_sector
            if np.isfinite(eB):
                sub = bh[np.isclose(bh.logAge, bh.logAge.iloc[(bh.logAge - np.log10(eB * 1e6)).abs().argmin()])]
                ax.plot(sub.Teff, sub.logL, color="k", lw=1.6, ls="-.", zorder=4,
                        label=f"BHAC15 para M_B (corridas vigentes): {eB:.3g} Myr (más cercana: {10 ** sub.logAge.iloc[0] / 1e6:.3g})")
        ax.set_title(mod, color=COL[mod], fontweight="bold")
        ax.set_xlabel("T_eff (K)")
        ax.legend(fontsize=7, loc="lower left")
        w = max(4 * (sT if np.isfinite(sT) else 0.03 * T), 0.25 * T)
        xmin = T - w; xmax = T + w
        if mod == "BHAC15" and len(sec): xmin = min(xmin, sec.Teff_B.min() - 300)
        ax.set_xlim(xmax, max(xmin, 2000))
    axs[0].set_ylabel("log L/L☉")
    lo_y = lL - 1.5; hi_y = lL + 1.0
    if len(sec): lo_y = min(lo_y, np.log10(sec.L_B.min()) - 0.4)
    axs[0].set_ylim(lo_y, hi_y)
    eu = L[(L.sistema == sist)]
    fig.suptitle(f"Sistema {sist} (TIC {tic}) — T_eff = {T:.0f} K, log L = {lL:.2f}  [{fuenteP}].  "
                 "Gris: isócronas de referencia (Myr). Línea continua: pipeline (T_eff); discontinua: corridas de Jesús (SpT); punteada: corrida pms 13-oct.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"sistema_{int(sist):02d}.png", dpi=130)
    plt.close(fig)
    print("ok sistema", sist)

pd.DataFrame(filas).to_csv(AQUI / "edades_prueba_fases.csv", index=False)
