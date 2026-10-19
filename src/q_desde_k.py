"""q a partir del observable k = R_B/R_A, por dos caminos, para los 19 sistemas de Orión con starry.

1. Hayashi (analítico): para dos estrellas coetáneas, totalmente convectivas y en contracción de Kelvin–Helmholtz a T_eff
   ~ constante, R ∝ M^(2/3) T^(-4/3) t^(-1/3)  =>  q_H = k^(3/2) (T_B/T_A)^2  (la edad se cancela).
   T_B/T_A sale del ajuste: brillo superficial relativo J = (B_amp/A_amp)/k^2 en la banda de TESS, invertido con cuerpos
   negros integrados sobre 600–1000 nm (banda plana; aproximación), con T_A = Teff_A de MassAge (σ = 3 %, supuesto).
2. Isócrona: M_B invirtiendo R_B (posterior de starry) en BHAC15 a la edad del sistema; q_iso = M_B / M_A (MassAge).

Se usan las muestras de los posteriors de la corrida vigente de cada sector (tabla_maestra.csv) y se juntan los sectores
con el mismo peso. Validez de Hayashi: primaria fría (T_A < 6250 K) y la fórmula reproduce BHAC15 a ±15 % en (M_A, q_H,
edad); edad < 1 Myr se marca "precaución" (quema de deuterio, memoria de la condición inicial).
Salida: q_desde_k_por_sector.csv, q_desde_k_por_sistema.csv, correlaciones_q.csv.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "datos_estrellas_massage_orion"))
from isocronas_interp import IsochroneInterpolator, invert_radius_to_mass, load_isochrone_model  # noqa: E402

RNG = np.random.default_rng(42)
N_HAY, N_ISO = 2000, 150                      # muestras por sector (la inversión en la isócrona es lenta)
SIG_TA = 0.03
LAM = np.linspace(600e-9, 1000e-9, 200)       # banda de TESS aproximada como plana
h, c, kB = 6.62607e-34, 2.99792e8, 1.380649e-23


def banda(T):
    T = np.atleast_1d(T)[:, None]
    return (1 / (LAM**5 * np.expm1(h * c / (LAM * kB * T)))).sum(axis=1)


TGRID = np.linspace(1500, 40000, 4000)
BGRID = banda(TGRID)


def T_de_brillo(B):                            # inversa monótona de banda(T)
    return np.interp(B, BGRID, TGRID, left=np.nan, right=np.nan)


bhac = IsochroneInterpolator(load_isochrone_model("BHAC15", RAIZ / "datos_estrellas_massage_orion/modelos/iso_barafe/BHAC15_iso.txt"), "BHAC15")


def validez_hayashi(MA, q, edad):
    """Error relativo de k_Hayashi frente a k de BHAC15 en (M_A, q, edad)."""
    la = np.log10(edad * 1e6)
    RA, RB = bhac.radius(la, MA), bhac.radius(la, q * MA)
    TA, TB = bhac.teff(la, MA), bhac.teff(la, q * MA)
    k_iso, k_hay = RB / RA, q ** (2 / 3) * (TA / TB) ** (4 / 3)
    return k_hay / k_iso - 1


T = pd.read_csv(RAIZ / "diagnostico_resultados/tabla_maestra.csv")
filas, mues = [], {}
for _, r in T.iterrows():
    P = pd.read_csv(Path(r.dir) / f"TIC_{r.TIC}_S{r.Sector}_Posteriors.csv")
    s = P.sample(N_HAY, random_state=1, replace=len(P) < N_HAY)
    k = (s.B_r / s.A_r).to_numpy()
    J = (s.B_amp / s.A_amp).to_numpy() / k**2
    TA = r.Teff_A * (1 + SIG_TA * RNG.standard_normal(N_HAY))
    TB = T_de_brillo(J * banda(TA))
    qH = k**1.5 * (TB / TA) ** 2
    la = np.log10(r.edad_Myr * 1e6)
    si = s.iloc[:N_ISO]
    MA = r.prior_A_m + r.prior_A_m_sd * RNG.standard_normal(N_ISO)
    MB = np.array([invert_radius_to_mass(la, R, bhac) for R in si.B_r])
    qI = MB / MA
    mues[(r.sistema, r.Sector)] = (qH, qI, k)
    filas.append(dict(sistema=r.sistema, TIC=r.TIC, Sector=r.Sector, Teff_A=r.Teff_A, edad_Myr=r.edad_Myr,
                      k=np.median(k), k_sd=np.std(k), J=np.median(J), TB_TA=np.nanmedian(TB / TA),
                      q_H=np.nanmedian(qH), q_H_sd=np.nanstd(qH), q_iso=np.nanmedian(qI), q_iso_sd=np.nanstd(qI),
                      frac_iso_nan=np.isnan(qI).mean(), q_actual=r.q, q_actual_sd=r.q_sd))
SEC = pd.DataFrame(filas)
SEC.round(4).to_csv(AQUI / "q_desde_k_por_sector.csv", index=False)

S = pd.read_csv(RAIZ / "diagnostico_resultados/muestra_completa_orion_pleyades.csv")
S = S[(S.cumulo == "Orión") & S.k.notna()].set_index("sistema")
out = []
for sis, g in SEC.groupby("sistema"):
    qH = np.concatenate([mues[(sis, s)][0] for s in g.Sector]); qI = np.concatenate([mues[(sis, s)][1] for s in g.Sector])
    k = np.concatenate([mues[(sis, s)][2] for s in g.Sector])
    p = lambda v, x: np.nanpercentile(v, x)
    r = dict(sistema=int(sis), TIC=int(g.TIC.iloc[0]), n_sect=len(g), Teff_A=g.Teff_A.iloc[0], edad_Myr=g.edad_Myr.iloc[0],
             k=p(k, 50), k_lo=p(k, 16), k_hi=p(k, 84), TB_TA=g.TB_TA.median(),
             q_H=p(qH, 50), q_H_lo=p(qH, 16), q_H_hi=p(qH, 84),
             q_iso=p(qI, 50) if np.isfinite(qI).any() else np.nan,
             q_iso_lo=p(qI, 16) if np.isfinite(qI).any() else np.nan, q_iso_hi=p(qI, 84) if np.isfinite(qI).any() else np.nan,
             q_actual=S.loc[sis, "q"], e=S.loc[sis, "e"], abs_ecosw=S.loc[sis, "abs_ecosw"], P=S.loc[sis, "P"])
    MA = T.loc[T.sistema == sis, "prior_A_m"].iloc[0]
    try:
        r["err_hayashi_bhac"] = validez_hayashi(MA, r["q_H"], r["edad_Myr"]) if MA <= 1.4 else np.nan   # en el q que se afirma
    except Exception:
        r["err_hayashi_bhac"] = np.nan
    if r["Teff_A"] >= 6250 or not np.isfinite(r["err_hayashi_bhac"]) or abs(r["err_hayashi_bhac"]) > 0.15:
        r["hayashi"] = "no aplica"
    elif r["edad_Myr"] < 1:
        r["hayashi"] = "precaución (< 1 Myr)"
    else:
        r["hayashi"] = "aplica"
    out.append(r)
Q = pd.DataFrame(out).sort_values("sistema")
Q.round(4).to_csv(AQUI / "q_desde_k_por_sistema.csv", index=False)

corr = []
for y in ("e", "abs_ecosw", "P"):
    for x, sub in (("k", Q), ("q_actual", Q), ("q_iso", Q), ("q_H", Q[Q.hayashi != "no aplica"]), ("q_H", Q)):
        d = sub[[x, y]].dropna()
        rho, pv = stats.spearmanr(d[x], d[y]); tau, pt = stats.kendalltau(d[x], d[y])
        corr.append(dict(y=y, x=x, subconjunto="frías (Hayashi)" if len(sub) < len(Q) else "todas", n=len(d),
                         rho=rho, p_rho=pv, tau=tau, p_tau=pt))
C = pd.DataFrame(corr)
C.round(3).to_csv(AQUI / "correlaciones_q.csv", index=False)
pd.set_option("display.width", 250)
print(Q.round(3).to_string(index=False)); print(); print(C.round(3).to_string(index=False))
