"""Análisis del paso 2: TLS frente a BLS (mismas inyecciones y mismos datos), umbral de TLS y candidatos de las curvas reales con vetos adicionales.

Vetos adicionales sobre los candidatos reales (sin ellos pasan señales que cumplen los criterios de errores pero no son tránsitos):
  - rotación: el periodo coincide (2 %) con k x el pico de Lomb-Scargle de la curva sin aplanar (k = 1/2, 1, 2);
  - duración: entre 0.3 y 2.6 veces la de un tránsito central con la densidad de la estrella (TIC);
  - recurrencia: el periodo se confirma en otros sectores de la misma estrella.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

AQUI = Path("/home/luciam/Documents/starry/Resultados_Pleyades/completitud"); RES = AQUI / "resultados"; FIG = AQUI / "figuras"
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ci", AQUI / "completitud_inyecciones.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
real = pd.read_csv(RES / "tls_reales.csv"); iny = pd.read_csv(RES / "tls_vs_bls_inyecciones.csv"); nul = pd.read_csv(RES / "tls_nulos.csv")


def valido(df, pref, k, thr, dur_min=False):
    g = lambda c, d: df.get(f"{pref}_{c}_{k}", pd.Series(d, index=df.index))
    v = ((g("ok", False).fillna(False).astype(bool) | (g("motivo", "").fillna("") == "ruido_rojo")) & (g("SDE", 0.0).fillna(0.0) >= thr) & g("dur_ok", False).fillna(False).astype(bool)).to_numpy()
    return v


def hay(df, pref, thr):
    v = np.zeros(len(df), bool)
    for k in range(int(df[f"{pref}_n_picos"].max())):
        v |= valido(df, pref, k, thr)
    return v


def recup(df, pref, thr):
    v = np.zeros(len(df), bool)
    for k in range(int(df[f"{pref}_n_picos"].max())):
        P = df.get(f"{pref}_P_{k}", pd.Series(np.nan, index=df.index)).to_numpy(float)
        v |= valido(df, pref, k, thr) & (np.abs(P / df.P_inj.to_numpy() - 1) < 0.01)
    return v


fap = {t: hay(nul, "tls", t).mean() for t in (7, 8, 9, 10, 11, 12, 13, 14)}
print(f"FAP de TLS en {len(nul)} nulos:", {k: round(v, 3) for k, v in fap.items()})

iny["tls"] = recup(iny, "tls", 9.0); iny["bls_13"] = recup(iny, "bls", 13.5); iny["bls_9"] = recup(iny, "bls", 9.0)
iny["bin_depth"] = pd.cut(iny.depth_inj, [0.015, 0.03, 0.05, 0.075, 0.10001]); iny["bin_P"] = pd.cut(iny.P_inj, [0.5, 1, 2, 4, 8, 12.0001])
tabla = pd.concat({"por profundidad": iny.groupby("bin_depth")[["tls", "bls_9", "bls_13"]].mean(), "por periodo": iny.groupby("bin_P")[["tls", "bls_9", "bls_13"]].mean()})
tabla.round(3).to_csv(RES / "tls_vs_bls_recuperacion.csv")
print("\nRecuperación con las mismas inyecciones y datos (TLS SDE>=9; BLS SDE>=9 y >=13.5):")
print(tabla.round(2).to_string()); print("global:", iny[["tls", "bls_9", "bls_13"]].mean().round(3).to_dict())

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, (col, tit) in zip(ax, (("bin_depth", "Profundidad inyectada"), ("bin_P", "Periodo inyectado [d]"))):
    g = iny.groupby(col)[["tls", "bls_9", "bls_13"]].mean(); x = np.arange(len(g))
    a.bar(x - 0.27, g.tls, 0.27, color="#eb6834", label="TLS (SDE ≥ 9)"); a.bar(x, g.bls_9, 0.27, color="#9ec5f4", label="BLS (SDE ≥ 9)"); a.bar(x + 0.27, g.bls_13, 0.27, color="#2a78d6", label="BLS (SDE ≥ 13.5)")
    a.set_xticks(x); a.set_xticklabels([f"{i.left:g}–{min(i.right, 12 if col == 'bin_P' else 0.1):g}" for i in g.index], fontsize=9); a.set_xlabel(tit); a.grid(axis="y", alpha=0.3)
ax[0].set_ylabel("Fracción recuperada"); ax[0].legend(fontsize=8); fig.suptitle(f"TLS frente a BLS con las mismas {len(iny)} inyecciones (sector 43, datos en bins de 10 min; todas pasan los criterios de consistencia)", x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "tls_vs_bls.png", dpi=140); fig.savefig(FIG / "tls_vs_bls.svg", format="svg"); plt.close(fig)

# ---------------------------------------------------------------- candidatos reales con vetos adicionales
CUR = {(c["tic"], c["sector"]): c for c in m.CURVAS}
cand = []
for _, r in real.iterrows():
    for k in range(int(real.tls_n_picos.max())):
        if valido(real.loc[[r.name]], "tls", k, 7.0)[0]:
            tic, sec = int(r.tic), int(r.sector); c = CUR[(tic, sec)]
            fr = np.linspace(1 / 15, 1 / 0.2, 8000); pw = LombScargle(c["t"], c["y"]).power(fr)
            loc = sorted([j for j in range(1, len(pw) - 1) if pw[j] > pw[j - 1] and pw[j] >= pw[j + 1]], key=lambda j: -pw[j])[:3]
            P = r[f"tls_P_{k}"]; dur = r[f"tls_dur_{k}"]; rho = m.PARAM.loc[tic, "rho"]
            T0 = 13.0 / 24 * (P / 365.25) ** (1 / 3) * rho ** (-1 / 3)
            rot = any(abs(P / (kk / fr[j]) - 1) < 0.02 for j in loc for kk in (0.5, 1, 2))
            cand.append(dict(tic=tic, sector=sec, P=P, SDE=r[f"tls_SDE_{k}"], snr=r[f"tls_snr_{k}"], depth=r[f"tls_depth_tls_{k}"] if f"tls_depth_tls_{k}" in r else np.nan, n_tr=r[f"tls_n_tr_{k}"],
                             dur=dur, dur_sobre_esperada=dur / T0, coincide_rotacion=rot, P_LS=";".join(f"{1 / fr[j]:.3f}" for j in loc)))
            break
cand = pd.DataFrame(cand)
if len(cand):
    cand["recurre_en_otro_sector"] = [any(((cand.tic == r.tic) & (cand.sector != r.sector) & (np.abs(cand.P / r.P - 1) < 0.01))) for r in cand.itertuples()]
    cand["veredicto"] = np.where(cand.coincide_rotacion, "rotación (veto)", np.where((cand.dur_sobre_esperada < 0.3) | (cand.dur_sobre_esperada > 2.6), "duración no física", np.where(cand.recurre_en_otro_sector, "recurre en otros sectores: revisar", "un solo sector: no confirmado")))
cand.to_csv(RES / "tls_candidatos.csv", index=False)
print("\nCandidatos reales de TLS (SDE >= 7, consistentes) y su veredicto tras los vetos:")
print(cand.round(3).to_string(index=False) if len(cand) else "  ninguno")
print("curvas con candidato que sobrevive a los vetos:", int(cand.veredicto.str.contains("revisar|no confirmado").sum()) if len(cand) else 0, "de", len(real))
