"""Mejor restricción de q y e por sistema (Orión, 19 con starry) solo con lo ya medido.

e:  e cos ω = media de los sectores con secundario detectado; error = max(sd media, dispersión entre sectores)
    (los errores de starry por sector subestiman la dispersión). |e cos ω| es una cota inferior de e.
    e sin ω NO se usa: no viene de las duraciones (duraciones de datos y modelo coinciden, ver duraciones.py) y en el
    posterior se degenera con i, J y k; sale negativa en todos los sistemas por el prior de inclinación.
q:  jerarquía por sistema: Hayashi (primaria fría, la fórmula reproduce BHAC15 a ±15 %) > elipsoidal estable entre
    sectores > isócrona (dependiente del modelo). Se dan todas las estimaciones disponibles para comparar.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion")
T = pd.read_csv(RAIZ / "diagnostico_resultados/tabla_maestra.csv")
QK = pd.read_csv(AQUI.parent / "q_desde_k/q_desde_k_por_sistema.csv").set_index("sistema")
EL = pd.read_csv(AQUI / "q_elipsoidal_por_sector.csv")
DU = pd.read_csv(AQUI / "esinw_duraciones_por_sector.csv")

filas = []
for sis, g in T.groupby("sistema"):
    s = g[g.sec_det.astype(bool)]
    if len(s):
        ec = s.ecosw.mean(); ec_sd = max(s.ecosw_sd.mean(), s.ecosw.std() if len(s) > 1 else 0)
    else:
        ec, ec_sd = np.nan, np.nan
    q = QK.loc[sis]
    el = EL[EL.sistema == sis]
    q_el = el.q_alta_p50.median() if len(el) else np.nan
    q_el_disp = (el.q_alta_p50.max() / el.q_alta_p50.min() - 1) if len(el) > 1 else np.nan
    if q.hayashi in ("aplica", "precaución (< 1 Myr)"):
        qb, lo, hi, met = q.q_H, q.q_H_lo, q.q_H_hi, "Hayashi" + (" (precaución)" if q.hayashi != "aplica" else "")
    elif len(el) and q_el_disp < 0.3:
        qb, lo, hi = q_el, el.q_alta_p16.median(), el.q_alta_p84.median(); met = "elipsoidal"
    else:
        qb, lo, hi, met = q.q_iso, q.q_iso_lo, q.q_iso_hi, "isócrona (dep. modelo)"
    du = DU[DU.sistema == sis]
    filas.append(dict(sistema=int(sis), TIC=int(g.TIC.iloc[0]), P=g.P.median(), Teff_A=q.Teff_A, edad_Myr=q.edad_Myr,
                      n_sec_det=len(s), ecosw=ec, ecosw_sd=ec_sd, abs_ecosw=abs(ec), signif_ecosw=abs(ec) / ec_sd if ec_sd else np.nan,
                      e_starry=g.e.median(), esinw_starry=g.esinw.median(),
                      r_dur_datos=du.r_datos.median(), r_dur_modelo=du.r_modelo.median(),
                      q_H=q.q_H, q_iso=q.q_iso, q_elip=q_el, q_elip_disp_sectores=q_el_disp, q_actual=q.q_actual,
                      q_mejor=qb, q_mejor_lo=lo, q_mejor_hi=hi, metodo_q=met, k=q.k))
B = pd.DataFrame(filas).sort_values("sistema")
B.round(4).to_csv(AQUI / "mejor_q_y_e_por_sistema.csv", index=False)

corr = []
for y in ("abs_ecosw",):
    for x, sub, nom in (("q_mejor", B, "q mejor (mixto)"), ("q_H", B[B.metodo_q.str.startswith("Hayashi")], "q_H, solo frías"),
                        ("k", B, "k"), ("q_actual", B, "q actual"), ("P", B, "P")):
        d = sub[[x, y]].dropna()
        rho, p = stats.spearmanr(d[x], d[y]); tau, pt = stats.kendalltau(d[x], d[y])
        corr.append(dict(y=y, x=nom, n=len(d), rho=rho, p_rho=p, tau=tau, p_tau=pt))
C = pd.DataFrame(corr); C.round(3).to_csv(AQUI / "correlaciones_mejor_q_e.csv", index=False)
pd.set_option("display.width", 260)
print(B.round(4).to_string(index=False)); print(); print(C.round(3).to_string(index=False))
