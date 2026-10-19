"""Análisis de la búsqueda de tránsitos en los residuos de las 4 EB: umbral de SDE, completitud por radio y periodo, límites superiores y candidatos."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades/completitud"); RES = BASE / "resultados"; FIG = BASE / "figuras"
nul = pd.read_csv(RES / "eb_nulos.csv"); iny = pd.read_csv(RES / "eb_inyecciones.csv"); real = pd.read_csv(RES / "eb_reales.csv")
P_EDG = np.array([0.5, 1, 2, 4, 8, 12.0001]); R_EDG = np.array([0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 3.0001])
NOM = {26017840: "TIC 26017840 (EB, M enana)", 35159657: "TIC 35159657 (EB, K)", 125754991: "TIC 125754991 (EB, A0)", 346541355: "TIC 346541355 (EB, A1)"}
FAP_OBJ = 0.015


def picos(df):
    return range(int(df.n_picos.max()))


def valido(df, k, thr):
    g = lambda c, d: df.get(f"{c}_{k}", pd.Series(d, index=df.index))
    # beta es diagnóstico, no veto: el ruido rojo ya está en el error por época (empírico); "ruido_rojo" como único motivo cuenta como válido
    ok_sin_beta = g("ok", False).fillna(False).astype(bool) | (g("motivo", "").fillna("") == "ruido_rojo")
    return (ok_sin_beta & (g("SDE", 0.0).fillna(0.0) >= thr) & g("durok", False).fillna(False).astype(bool) & ~g("arm", True).fillna(True).astype(bool)).to_numpy()


def hay(df, thr):
    v = np.zeros(len(df), bool)
    for k in picos(df):
        v |= valido(df, k, thr)
    return v


def recuperada(df, thr):
    v = np.zeros(len(df), bool)
    for k in picos(df):
        co = df.get(f"coincide_{k}", pd.Series("no", index=df.index)).fillna("no")
        v |= valido(df, k, thr) & (co == "exacto").to_numpy()
    return v


fap = {thr: hay(nul, thr).mean() for thr in np.arange(4, 16.1, 0.5)}
THR = next((thr for thr, f in fap.items() if f <= FAP_OBJ), max(fap))
print(f"FAP vs umbral (nulos, n = {len(nul)}):", {k: round(v, 3) for k, v in fap.items() if k <= 14})
print(f"-> umbral de SDE global = {THR} (FAP = {fap[THR]:.3f})")
THR_TIC = {}
for tic_, n_ in nul.groupby("tic"):
    THR_TIC[tic_] = next((t for t in np.arange(5, 16.1, 0.5) if hay(n_, t).mean() <= FAP_OBJ), 16.0)
print("umbral de SDE por estrella (FAP <= %.1f %% en sus nulos):" % (100 * FAP_OBJ), THR_TIC)


def por_tic(df, funcion):
    out = np.zeros(len(df), bool)
    for tic_, thr_ in THR_TIC.items():
        m_ = (df.tic == tic_).to_numpy()
        if m_.any():
            out[m_] = funcion(df[m_], thr_)
    return out

iny["arm_inj"] = iny.arm_inj.astype(bool)
print(f"inyecciones: {len(iny)}; en armónico de P_EB (no recuperables por el veto, se excluyen): {int(iny.arm_inj.sum())}")
iny = iny[~iny.arm_inj].copy()
iny["rec"] = por_tic(iny, recuperada)
iny.to_csv(RES / "eb_inyecciones_con_recuperacion.csv", index=False)
print("completitud global:", round(iny.rec.mean(), 3))

real["detectada"] = por_tic(real, hay)
print(f"\nCurvas de EB con algún pico válido (umbral por estrella): {int(real.detectada.sum())} de {len(real)}")
real["detectada_laxo"] = hay(real, 9.0)
cand = []
for _, r in real.iterrows():
    for k in picos(real):
        if valido(real.loc[[r.name]], k, 9.0)[0]:
            cand.append(dict(tic=int(r.tic), sector=int(r.sector), P=r[f"P_{k}"], SDE=r[f"SDE_{k}"], depth=r[f"depth_{k}"], snr=r[f"snr_{k}"], n_tr=r[f"n_tr_{k}"], p_cons=r[f"p_cons_{k}"], oe=r[f"oe_{k}"], beta=r[f"beta_{k}"], estricto=bool(r[f"SDE_{k}"] >= THR))); break
cand = pd.DataFrame(cand); cand.to_csv(RES / "eb_candidatos.csv", index=False)
print("Candidatos con SDE >= 9 (5 % de falsas alarmas aprox.):"); print(cand.round(3).to_string(index=False) if len(cand) else "  ninguno")

# ---------------------------------------------------------------- mapas de completitud por Rp y periodo
AZ = LinearSegmentedColormap.from_list("c", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
tics = sorted(iny.tic.unique())
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
filas = []
for ax, tic in zip(axs, tics):
    g = iny[iny.tic == tic]
    M = np.full((len(R_EDG) - 1, len(P_EDG) - 1), np.nan)
    for i in range(len(R_EDG) - 1):
        for j in range(len(P_EDG) - 1):
            s = g[(g.Rp_RJ >= R_EDG[i]) & (g.Rp_RJ < R_EDG[i + 1]) & (g.P_inj >= P_EDG[j]) & (g.P_inj < P_EDG[j + 1])]
            if len(s) >= 4:
                M[i, j] = s.rec.mean()
    im = ax.pcolormesh(P_EDG, R_EDG, M, cmap=AZ, vmin=0, vmax=1, shading="flat")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(np.sqrt(P_EDG[j] * P_EDG[j + 1]), np.sqrt(R_EDG[i] * R_EDG[i + 1]), f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.55 else "#52514e")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8, 12]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "12"]); ax.minorticks_off()
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
    ax.set_title(NOM[tic] + f"\nR$_A$ = {g.R_A.iloc[0]:.2f} R$_☉$", fontsize=9.5, loc="left"); ax.set_xlabel("Periodo [d]")
    for j in range(len(P_EDG) - 1):
        col = M[:, j]; yc = np.sqrt(R_EDG[:-1] * R_EDG[1:]); fil = dict(tic=tic, P_bin=f"{P_EDG[j]:g}-{min(P_EDG[j + 1], 12):g}")
        for nivel in (0.5, 0.9):
            idx = np.where(np.isfinite(col) & (col >= nivel))[0]
            if len(idx) and idx[0] > 0 and np.isfinite(col[idx[0] - 1]):
                i = idx[0]; f = (nivel - col[i - 1]) / (col[i] - col[i - 1]); d = np.exp(np.log(yc[i - 1]) + f * (np.log(yc[i]) - np.log(yc[i - 1])))
            else:
                d = yc[idx[0]] if len(idx) else np.nan
            fil[f"Rp_RJ_{int(100 * nivel)}"] = d
        filas.append(fil)
axs[0].set_ylabel("Radio del planeta [R$_{Jup}$]")
fig.colorbar(im, ax=axs, shrink=0.85, label="Fracción recuperada (umbral de SDE por estrella, FAP ≈ 1 %)")
fig.suptitle("Residuos de las EB (eclipses enmascarados): completitud de la búsqueda de tránsitos (números en %)", x=0.01, y=1.07, ha="left", fontsize=11.5)
fig.savefig(FIG / "completitud_residuos_EB.png", dpi=140, bbox_inches="tight"); fig.savefig(FIG / "completitud_residuos_EB.svg", format="svg", bbox_inches="tight"); plt.close(fig)
L = pd.DataFrame(filas); L.to_csv(RES / "eb_limites_superiores.csv", index=False)
print("\nRadio mínimo (R_Jup) con 50 % y 90 % de completitud por bin de periodo:")
print(L.pivot(index="tic", columns="P_bin", values="Rp_RJ_50").round(2).to_string()); print(L.pivot(index="tic", columns="P_bin", values="Rp_RJ_90").round(2).to_string())
print("figuras en", FIG)
