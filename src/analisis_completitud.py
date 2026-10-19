"""Análisis de la inyección-recuperación: umbral de SDE, mapas de completitud y límites superiores de radio.

1. Umbral de SDE: se elige el menor valor para el cual <= 1 % de las curvas nulas (flujo invertido, bloques barajados) tiene algún pico que pase
   los criterios de consistencia (S/N con ruido rojo, tránsitos consistentes, par-impar, beta).
2. Una inyección está recuperada si en alguna de las 3 iteraciones el pico tiene el periodo inyectado (1 %), pasa la consistencia y SDE >= umbral.
   Los picos a P/2 o 2P se cuentan aparte (recuperada con armónico).
3. Completitud por TIC (todos los sectores juntos; cada inyección se busca en un solo sector) en rejilla de periodo y profundidad.
4. Límites superiores: profundidad con 50 % y 90 % de completitud por bin de periodo -> Rp = sqrt(profundidad) * R* (TIC).
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades/completitud"); RES = BASE / "resultados"; FIG = BASE / "figuras"; FIG.mkdir(exist_ok=True)
PARAM = pd.read_csv("/home/luciam/Documents/massage_workshop/massage.py/massageXtesss_pleyades.csv").drop_duplicates("TIC").set_index("TIC")
RJ = 9.731
P_EDG = np.array([0.5, 1, 2, 4, 8, 12.0001]); D_EDG = np.array([5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.10001])
FAP_OBJ = 0.02

nul = pd.read_csv(RES / "nulos.csv"); iny = pd.read_csv(RES / "inyecciones.csv"); real = pd.read_csv(RES / "detecciones_reales.csv")


def picos(df):
    return range(int(df.n_picos.max()))


FAC_DUR = 2.6    # tolerancia: 1.3 (k hasta 0.3) x 2 (la rejilla de duraciones del BLS es gruesa)


def dur_ok(df, k):
    """La duración del pico no puede superar la máxima física de un tránsito a ese periodo y con la densidad de la estrella (TIC)."""
    rho = df.tic.map(PARAM.rho).to_numpy(float)
    P = df.get(f"P_{k}", pd.Series(np.nan, index=df.index)).to_numpy(float)
    T0 = 13.0 / 24 * (P / 365.25) ** (1 / 3) * rho ** (-1 / 3)
    d = df.get(f"dur_{k}", pd.Series(np.nan, index=df.index)).to_numpy(float)
    return np.isfinite(d) & (d <= FAC_DUR * T0)


def hay(df, thr):
    """Por fila: algún pico que pasa la consistencia con SDE >= thr."""
    v = np.zeros(len(df), bool)
    for k in picos(df):
        ok = df.get(f"ok_{k}", pd.Series(False, index=df.index)).fillna(False).astype(bool) | (df.get(f"motivo_{k}", pd.Series("", index=df.index)).fillna("") == "ruido_rojo")   # beta: diagnóstico, no veto
        sde = df.get(f"SDE_{k}", pd.Series(0.0, index=df.index)).fillna(0.0)
        v |= (ok & (sde >= thr)).to_numpy() & dur_ok(df, k)
    return v


fap = {thr: hay(nul, thr).mean() for thr in np.arange(4, 14.1, 0.5)}
THR = next((thr for thr, f in fap.items() if f <= FAP_OBJ), max(fap))
print("FAP vs umbral de SDE (nulos, n = %d):" % len(nul), {k: round(v, 3) for k, v in fap.items()})
print(f"-> umbral de SDE global = {THR} (FAP = {fap[THR]:.3f})")
THR_TIC = {tic_: next((t for t in np.arange(5, 16.1, 0.5) if hay(n_, t).mean() <= FAP_OBJ), 16.0) for tic_, n_ in nul.groupby("tic")}
print("umbral de SDE por estrella (FAP <= %.0f %% en sus %d nulos):" % (100 * FAP_OBJ, len(nul) // len(THR_TIC)), THR_TIC)


def por_tic(df, funcion):
    out = np.zeros(len(df), bool)
    for tic_, thr_ in THR_TIC.items():
        m_ = (df.tic == tic_).to_numpy()
        if m_.any():
            out[m_] = funcion(df[m_], thr_)
    return out

# ------------------------------------------------------------ recuperación
def recuperada(df, thr, tipo):
    v = np.zeros(len(df), bool)
    for k in picos(df):
        ok = df.get(f"ok_{k}", pd.Series(False, index=df.index)).fillna(False).astype(bool) | (df.get(f"motivo_{k}", pd.Series("", index=df.index)).fillna("") == "ruido_rojo")
        sde = df.get(f"SDE_{k}", pd.Series(0.0, index=df.index)).fillna(0.0)
        co = df.get(f"coincide_{k}", pd.Series("no", index=df.index)).fillna("no")
        v |= (ok & (sde >= thr) & (co == tipo)).to_numpy() & dur_ok(df, k)
    return v


iny["rec"] = por_tic(iny, lambda d, t: recuperada(d, t, "exacto"))
iny["rec_arm"] = por_tic(iny, lambda d, t: recuperada(d, t, "mitad") | recuperada(d, t, "doble"))
iny["rec_cualquiera"] = iny.rec | iny.rec_arm
iny["bin_P"] = pd.cut(iny.P_inj, P_EDG); iny["bin_D"] = pd.cut(iny.depth_inj, D_EDG)
iny.to_csv(RES / "inyecciones_con_recuperacion.csv", index=False)
print("completitud global (exacta):", round(iny.rec.mean(), 3), "| con armónicos:", round(iny.rec_cualquiera.mean(), 3))

# ------------------------------------------------------------ detecciones reales con el mismo criterio
real["detectada"] = por_tic(real, hay)
cols = [c for c in real.columns if c.endswith("_0") or c in ("tic", "sector", "detectada")]
rd = real[real.detectada][["tic", "sector"] + [f"{n}_{k}" for k in picos(real) for n in ("P", "SDE", "snr", "depth", "ok") if f"{n}_{k}" in real.columns]]
print(f"\nDetecciones reales que pasan los criterios (SDE >= {THR}): {int(real.detectada.sum())} de {len(real)} curvas")
if len(rd):
    print(rd.round(3).to_string(index=False))
real.to_csv(RES / "detecciones_reales_con_criterio.csv", index=False)

# ------------------------------------------------------------ mapas de completitud
AZ = LinearSegmentedColormap.from_list("c", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
tics = sorted(iny.tic.unique())
fig, axs = plt.subplots(3, 3, figsize=(13, 10.5), sharex=True, sharey=True)
filas = []
for ax, tic in zip(axs.ravel(), tics):
    g = iny[iny.tic == tic]
    M = np.full((len(D_EDG) - 1, len(P_EDG) - 1), np.nan)
    for i in range(len(D_EDG) - 1):
        for j in range(len(P_EDG) - 1):
            s = g[(g.depth_inj >= D_EDG[i]) & (g.depth_inj < D_EDG[i + 1]) & (g.P_inj >= P_EDG[j]) & (g.P_inj < P_EDG[j + 1])]
            if len(s) >= 3:
                M[i, j] = s.rec.mean()
    im = ax.pcolormesh(P_EDG, D_EDG, M, cmap=AZ, vmin=0, vmax=1, shading="flat")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(np.sqrt(P_EDG[j] * P_EDG[j + 1]), np.sqrt(D_EDG[i] * D_EDG[i + 1]), f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.55 else "#52514e")
    R = PARAM.loc[tic, "Rad"]; Teff = PARAM.loc[tic, "Teff"]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8, 12]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "12"]); ax.minorticks_off()
    ax.set_title(f"TIC {tic}  (R* = {R:.2f} R☉, Teff = {Teff:.0f} K)", fontsize=9, loc="left")
    # profundidad con 50 % y 90 % por bin de periodo
    for j in range(len(P_EDG) - 1):
        col = M[:, j]; yc = np.sqrt(D_EDG[:-1] * D_EDG[1:])
        fil = dict(tic=tic, P_bin=f"{P_EDG[j]:g}-{min(P_EDG[j + 1], 12):g}")
        for nivel in (0.5, 0.9):
            ok = np.isfinite(col)
            idx = np.where(ok & (col >= nivel))[0]
            if len(idx) and idx[0] > 0 and np.isfinite(col[idx[0] - 1]):
                i = idx[0]; f = (nivel - col[i - 1]) / (col[i] - col[i - 1])
                d = np.exp(np.log(yc[i - 1]) + f * (np.log(yc[i]) - np.log(yc[i - 1])))
            elif len(idx):
                d = yc[idx[0]]
            else:
                d = np.nan
            fil[f"depth_{int(100 * nivel)}"] = d; fil[f"Rp_RJ_{int(100 * nivel)}"] = np.sqrt(d) * R * RJ if np.isfinite(d) else np.nan
        filas.append(fil)
for ax in axs[:, 0]:
    ax.set_ylabel("Profundidad del tránsito")
for ax in axs[-1, :]:
    ax.set_xlabel("Periodo [d]")
fig.colorbar(im, ax=axs, shrink=0.6, label="Fracción recuperada (umbral de SDE por estrella)")
fig.suptitle(f"Completitud de la búsqueda de tránsitos por inyección y recuperación (umbral de SDE por estrella; números en %)", x=0.01, ha="left", fontsize=12)
fig.savefig(FIG / "completitud_por_tic.png", dpi=140, bbox_inches="tight"); fig.savefig(FIG / "completitud_por_tic.svg", format="svg", bbox_inches="tight"); plt.close(fig)

L = pd.DataFrame(filas); L.to_csv(RES / "limites_superiores.csv", index=False)
print("\nLímites superiores (profundidad y Rp con 90 % de completitud):")
print(L.pivot(index="tic", columns="P_bin", values="Rp_RJ_90").round(2).to_string())

# figura: Rp con 90 % de completitud vs periodo
fig, ax = plt.subplots(figsize=(7.2, 5))
xc = np.sqrt(P_EDG[:-1] * P_EDG[1:])
for tic in tics:
    g = L[L.tic == tic]
    ax.plot(xc, g.Rp_RJ_90.to_numpy(), "o-", ms=5, lw=1.2, label=f"TIC {tic}")
ax.set_xscale("log"); ax.set_xticks([0.7, 1.4, 2.8, 5.6, 9.8]); ax.set_xticklabels(["0.5–1", "1–2", "2–4", "4–8", "8–12"])
ax.set_xlabel("Periodo [d]"); ax.set_ylabel("Radio mínimo detectable con 90 % [R$_{Jup}$]")
ax.axhline(1.0, color="#8a8a85", ls=":", lw=1); ax.text(0.72, 1.03, "1 R$_{Jup}$", fontsize=8, color="#52514e")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "radio_minimo_detectable.png", dpi=140); fig.savefig(FIG / "radio_minimo_detectable.svg", format="svg"); plt.close(fig)

# ------------------------------------------------------------ completitud para planetas de 0.8-1.4 R_Jup y 0.3-0.6 R_Jup, y candidatos con umbral laxo
PBINS = [0.5, 1, 2, 4, 8, 12.0001]


def tabla_rp(thr, lo, hi):
    out = []
    for tic, g in iny.groupby("tic"):
        R = PARAM.loc[tic, "Rad"]; rp = np.sqrt(g.depth_inj) * R * RJ
        gg = g[(rp >= lo) & (rp < hi)]; r = recuperada(gg, THR_TIC[tic] if thr == THR else thr, "exacto")
        row = {"tic": tic, "R*": R, "n_inyecciones": len(gg), "SDE_umbral": ("por estrella" if thr == THR else thr), "Rp_RJ": f"{lo}-{hi}"}
        for a, b in zip(PBINS[:-1], PBINS[1:]):
            m = ((gg.P_inj >= a) & (gg.P_inj < b)).to_numpy()
            row[f"{a:g}-{min(b, 12):g} d"] = round(float(r[m].mean()), 3) if m.sum() >= 5 else np.nan
        out.append(row)
    return pd.DataFrame(out)


THR_LAXO = 9.0
tablas = pd.concat([tabla_rp(t, lo, hi) for t in (THR, THR_LAXO) for lo, hi in ((0.8, 1.4), (0.3, 0.6))])
tablas.to_csv(RES / "completitud_por_Rp.csv", index=False)
print("\nCompletitud por Rp y umbral (guardada en completitud_por_Rp.csv)")
print(tablas[(tablas.SDE_umbral == THR) & (tablas.Rp_RJ == "0.8-1.4")].to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, (lo, hi), tit in zip(axs, ((0.8, 1.4), (0.3, 0.6)), ("Planeta de ~1 R$_{Jup}$ (0.8–1.4)", "Planeta de ~0.45 R$_{Jup}$ (0.3–0.6)")):
    t = tabla_rp(THR, lo, hi).set_index("tic")[[c for c in tabla_rp(THR, lo, hi).columns if c.endswith(" d")]]
    im = ax.imshow(t.to_numpy(float), cmap=AZ, vmin=0, vmax=1, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 0.55 else "#52514e")
    ax.set_xticks(range(t.shape[1])); ax.set_xticklabels(t.columns, fontsize=9); ax.set_yticks(range(t.shape[0])); ax.set_yticklabels([f"TIC {i}" for i in t.index], fontsize=9)
    ax.set_xlabel("Periodo"); ax.set_title(tit, loc="left", fontsize=10.5); ax.grid(False)
fig.colorbar(im, ax=axs, shrink=0.8, label="Fracción recuperada (umbral de SDE por estrella y criterios de consistencia)")
fig.savefig(FIG / "completitud_por_planeta.png", dpi=140, bbox_inches="tight"); fig.savefig(FIG / "completitud_por_planeta.svg", format="svg", bbox_inches="tight"); plt.close(fig)

real["detectada_laxo"] = hay(real, THR_LAXO)
print(f"\nCurvas reales con algún pico consistente y SDE >= {THR_LAXO} (5 % de falsas alarmas): {int(real.detectada_laxo.sum())} de {len(real)}")
cand = []
for _, r in real[real.detectada_laxo].iterrows():
    for k in picos(real):
        if r.get(f"ok_{k}") == True and r.get(f"SDE_{k}", 0) >= THR_LAXO and dur_ok(real.loc[[r.name]], k)[0]:
            cand.append(dict(tic=int(r.tic), sector=int(r.sector), P=r[f"P_{k}"], SDE=r[f"SDE_{k}"], depth=r[f"depth_{k}"], snr=r[f"snr_{k}"], n_tr=r[f"n_tr_{k}"], p_cons=r[f"p_cons_{k}"], oe=r[f"oe_{k}"], beta=r[f"beta_{k}"])); break
cand = pd.DataFrame(cand); cand.to_csv(RES / "candidatos_umbral_laxo.csv", index=False)
print(cand.round(3).to_string(index=False) if len(cand) else "ninguna")
print("figuras en", FIG)
