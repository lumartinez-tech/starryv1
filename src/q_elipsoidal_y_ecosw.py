"""(1) q desde la variación elipsoidal en los sistemas con armónico a2 < 0 estable (3, 4 y 11; el 19 tiene a2 > 0 y se
excluye). Modelo de dos términos (Morris 1985), pesado por la fracción de luz de cada estrella:
    A2 = sin²i [ α_A q (R_A/a)³ ℓ_A + α_B (1/q) (R_B/a)³ ℓ_B ],   α ≈ 0.15 (15+u)(1+τ)/(3−u) ∈ [1.0, 1.6]
Se resuelve la cuadrática en q con muestras del posterior de starry (R/a, i, amplitudes) y A2 ~ N(medido, err) por
sector, con α_A, α_B ~ U(1.0, 1.6). Se reportan las dos raíces (q_alta: domina la deformación de A).
(2) e cos ω: starry frente a la fase del secundario del cuaderno 1b (ecosw_1b ± bootstrap).
"""
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion")
AQUI = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)
T = pd.read_csv(RAIZ / "diagnostico_resultados/tabla_maestra.csv")
H = pd.read_csv(RAIZ / "diagnostico_resultados/armonicos_prot_porb.csv")

filas = []
for _, h in H[H.a2_ppt < 0].iterrows():
    r = T[(T.TIC == h.TIC) & (T.Sector == h.sector)].iloc[0]
    P = pd.read_csv(Path(r.dir) / f"TIC_{r.TIC}_S{r.Sector}_Posteriors.csv").sample(4000, random_state=2, replace=True)
    a = P.a_AU * 215.032
    rA, rB = P.A_r / a, P.B_r / a
    s2 = np.sin(np.radians(P.B_inc)) ** 2
    lA = P.A_amp / (P.A_amp + P.B_amp); lB = 1 - lA
    A2 = (-h.a2_ppt + h.err_ppt * RNG.standard_normal(len(P))) * 1e-3
    aA, aB = RNG.uniform(1.0, 1.6, len(P)), RNG.uniform(1.0, 1.6, len(P))
    c2, c1, c0 = aA * rA**3 * lA, -A2 / s2, aB * rB**3 * lB
    disc = c1**2 - 4 * c2 * c0
    ok = disc > 0
    qa = np.where(ok, (-c1 + np.sqrt(np.clip(disc, 0, None))) / (2 * c2), np.nan)
    qb = np.where(ok, (-c1 - np.sqrt(np.clip(disc, 0, None))) / (2 * c2), np.nan)
    pc = lambda v: np.nanpercentile(v, [16, 50, 84])
    filas.append(dict(sistema=h.sistema, TIC=h.TIC, Sector=h.sector, A2_ppt=-h.a2_ppt, A2_err_ppt=h.err_ppt,
                      rA_a=rA.median(), inc=P.B_inc.median(), lB=lB.median(), frac_sol=ok.mean(),
                      **{f"q_alta_{n}": v for n, v in zip(("p16", "p50", "p84"), pc(qa))},
                      **{f"q_baja_{n}": v for n, v in zip(("p16", "p50", "p84"), pc(qb))},
                      q_actual=r.q))
E = pd.DataFrame(filas)
E.round(4).to_csv(AQUI / "q_elipsoidal_por_sector.csv", index=False)

C = T[["sistema", "TIC", "Sector", "ecosw", "ecosw_sd", "ecosw_1b", "ecosw_1b_err", "sec_det"]].copy()
C["dif_sigma"] = (C.ecosw - C.ecosw_1b) / np.hypot(C.ecosw_sd, C.ecosw_1b_err)
C.round(5).to_csv(AQUI / "ecosw_starry_vs_fase.csv", index=False)
pd.set_option("display.width", 250)
print(E.round(3).to_string(index=False)); print()
print(C.round(4).to_string(index=False))
