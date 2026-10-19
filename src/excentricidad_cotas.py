"""Excentricidad por sistema (Orión y Pléyades) con el mismo prior: (√e cos ω, √e sin ω) ~ U(disco) ⇔ e ~ U(0, 1),
ω ~ U(−π, π). Excepción: P1 (26017840) usa la corrida vieja (e y ω muestreados directamente, ω casi fijo).

Por sistema se juntan las muestras de todos los sectores vigentes (mismo peso por sector):
- e: mediana y error (percentiles 16–84) del posterior.
- Cota inferior: |e cos ω| (e ≥ |e cos ω| siempre); se da también la cota a 2σ, max(0, |e cos ω| − 2σ).
- Cota superior: percentil 97.5 del posterior de e (condicionada a los priors de masa e inclinación).
- Comparación con el prior: prior e ~ U(0,1) → mediana 0.5, sd 0.289, intervalo 95 % [0.025, 0.975];
  contracción c = 1 − sd_post / 0.289.
- e cos ω y e sin ω del mismo posterior; e cos ω solo se toma como medido si el secundario se detectó.
"""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path("/home/luciam/Documents/starry")
AQUI = Path(__file__).resolve().parent
T = pd.read_csv(RAIZ / "Resultados_Orion/diagnostico_resultados/tabla_maestra.csv")
SD_PRIOR = 1 / np.sqrt(12)
N = 4000


def convergio(f):
    s = pd.read_csv(f.replace("_Posteriors.csv", "_Summary.csv"), index_col=0)
    return bool(s.r_hat.max() <= 1.02 and s.ess_bulk.min() >= 400)


def cargar(f):
    P = pd.read_csv(f).sample(N, random_state=0, replace=True)
    w = np.radians(P.B_w)
    return pd.DataFrame(dict(e=P.B_ecc, ecosw=P.B_ecc * np.cos(w), esinw=P.B_ecc * np.sin(w)))


corridas = []   # (sistema, cúmulo, TIC, archivos, secundario detectado en algún sector, prior)
for sis, g in T.groupby("sistema"):
    fs = [str(Path(r.dir) / f"TIC_{r.TIC}_S{r.Sector}_Posteriors.csv") for _, r in g.iterrows()]
    corridas.append((str(sis), "Orión", int(g.TIC.iloc[0]), fs, g.sec_det.astype(bool).any(), "disco (h, k)"))
PLE = {"P2": 125754991, "P3": 35159657, "P4?": 346541355}
for et, tic in PLE.items():
    fs = [f for f in sorted(glob.glob(str(RAIZ / f"Resultados_Pleyades/starry_por_tic/TIC_{tic}/S*/TIC_{tic}_S*_Posteriors.csv"))) if convergio(f)]
    corridas.append((et, "Pléyades", tic, fs, True, "disco (h, k)"))
corridas.append(("P1", "Pléyades", 26017840,
                 [str(RAIZ / "Resultados_pleyades_lucia/Resultados_pleyades/Posteriors/Tic_26017840_Posteriors.csv")],
                 False, "viejo: e, ω directos (ω ≈ 90° fijo)"))

filas = []
for sis, cum, tic, fs, sec, prior in corridas:
    S = pd.concat([cargar(f) for f in fs])
    pc = lambda v, q: np.percentile(v, q)
    ec, ec_sd = S.ecosw.median(), S.ecosw.std()
    filas.append(dict(sistema=sis, cumulo=cum, TIC=tic, n_sect=len(fs), prior_e=prior,
                      e=pc(S.e, 50), e_err_inf=pc(S.e, 50) - pc(S.e, 16), e_err_sup=pc(S.e, 84) - pc(S.e, 50),
                      ecosw=ec if sec else np.nan, ecosw_err=ec_sd if sec else np.nan,
                      esinw=S.esinw.median(), esinw_err=S.esinw.std(),
                      cota_inf_abs_ecosw=abs(ec) if sec else 0.0,
                      cota_inf_2sigma=max(0.0, abs(ec) - 2 * ec_sd) if sec else 0.0,
                      e_p2_5=pc(S.e, 2.5), cota_sup_e_p97_5=pc(S.e, 97.5),
                      prior_mediana=0.5, prior_p2_5=0.025, prior_p97_5=0.975,
                      contraccion=1 - S.e.std() / SD_PRIOR,
                      sec_detectado=sec))
E = pd.DataFrame(filas)
E["ecosw_signif"] = (E.ecosw.abs() / E.ecosw_err)
E.round(4).to_csv(AQUI / "excentricidad_cotas_por_sistema.csv", index=False)
pd.set_option("display.width", 260)
print(E.drop(columns=["prior_mediana", "prior_p2_5", "prior_p97_5"]).round(4).to_string(index=False))
