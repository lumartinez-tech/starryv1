"""Sistemas con Prot ≈ Porb/2 (3, 4, 11, 19): ¿variación elipsoidal, rotación sincrónica con manchas (armónico) o periodo orbital duplicado?

1. ¿Porb es el periodo correcto? Si lo fuera el doble de uno real, el primario y el «secundario» serían idénticos: se comparan sus profundidades (D2/D1).
2. Fuera de eclipse se ajusta una serie de Fourier en la fase orbital φ (primario en φ = 0): y = c + Σ_k [a_k cos 2πkφ + b_k sin 2πkφ], k = 1, 2, 3.
   - Elipsoidal: término k = 2 con a_2 < 0 y b_2 ≈ 0 (mínimos en los eclipses), fase estable entre sectores, amplitud ≈ α q (R_A/a)³ sin²i (α ≈ 1.5).
   - Manchas síncronas: fase del término k = 2 que cambia entre sectores y término k = 1 no despreciable.
3. Periodo del pico de Lomb-Scargle frente a Porb/2 exacto.
Salida: armonicos_prot_porb.csv y figuras/armonicos_prot_porb.png.
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from astropy.timeseries import LombScargle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RAIZ = Path("/home/luciam/Documents/starry/Resultados_Orion"); AQUI = RAIZ / "diagnostico_resultados"
T = pd.read_csv(AQUI / "tabla_maestra.csv"); PRI = pd.read_csv(RAIZ / "priors_v2.csv")
SIST = {3: 11412481, 4: 50897998, 11: 138829403, 19: 427347825}
ALFA = 1.5


def binear(t, f, minutos=30):
    k = np.floor((t - t.min()) / (minutos / 1440)).astype(int)
    g = pd.DataFrame(dict(k=k, t=t, f=f)).groupby("k").mean()
    return g.t.to_numpy(), g.f.to_numpy()


def cargar_propia(tic, s):
    f = RAIZ / "Light_Curve_Files" / f"tic {tic}_S_{s}_LC.txt"
    d = pd.read_csv(f, sep=r"\s+").rename(columns={"#Time": "t"}).dropna(subset=["t", "Flux_CBV_SS"])
    t, y = d.t.to_numpy(float), d.Flux_CBV_SS.to_numpy(float)
    return t, y / np.median(y)


def efemeride(r, tic, s):
    P_ = pd.read_csv(Path(r["dir"]) / f"TIC_{tic}_S{s}_Posteriors.csv", index_col=0)
    t0 = float(P_["B_t0"].mean())
    pr = PRI[(PRI.TIC == tic) & (PRI.Sector == s)]; dur = float(pr.Duration.iloc[0]) if len(pr) else 0.25
    f2 = r.get("fase_sec_ref", 0.5); f2 = 0.5 if not np.isfinite(f2) else float(f2)
    return float(r.P), t0, min(dur / r.P, 0.25), f2


filas, plegadas = [], {}
for sis, tic in SIST.items():
    sub = T[T.TIC == tic].sort_values("Sector")
    for _, r in sub.iterrows():
        s = int(r.Sector)
        try:
            t, y = cargar_propia(tic, s)
        except Exception:
            continue
        P, t0, W, f2 = efemeride(r, tic, s)
        ph = ((t - t0) / P) % 1.0
        d1, d2 = r.get("D1", np.nan), r.get("D2", np.nan)
        # fuera de eclipse; tendencia lenta con filtro de mediana de 3 Porb (no toca P ni P/2)
        fe = (np.abs(((ph + 0.5) % 1) - 0.5) > 0.8 * W) & (np.abs(((ph - f2 + 0.5) % 1) - 0.5) > 0.8 * W)
        tb, yb = binear(t[fe], y[fe]); k = max(int(3 * P / (30 / 1440)), 5)
        tr = pd.Series(yb).rolling(k, center=True, min_periods=k // 3).median().to_numpy()
        yd = yb - np.where(np.isfinite(tr), tr, np.nanmedian(yb))
        phb = ((tb - t0) / P) % 1.0
        A = np.column_stack([np.ones(len(tb))] + [f(2 * np.pi * kk * phb) for kk in (1, 2, 3) for f in (np.cos, np.sin)])
        c = np.linalg.lstsq(A, yd, rcond=None)[0]; resid = yd - A @ c; sig = np.std(resid)
        cov = sig ** 2 * np.linalg.inv(A.T @ A); err = np.sqrt(np.diag(cov))
        a1, b1, a2, b2 = c[1], c[2], c[3], c[4]
        A1, A2 = np.hypot(a1, b1), np.hypot(a2, b2); fase2 = (np.arctan2(b2, a2) / (4 * np.pi)) % 0.5      # fase del máximo de cos(4πφ) en ciclos orbitales
        # Lomb-Scargle
        fr = np.linspace(1 / (P * 1.5), 1 / (P / 3), 8000); pw = LombScargle(tb, yd).power(fr); Pl = 1 / fr[np.argmax(pw)]
        ell = ALFA * r.q * r.rA_a ** 3 * np.sin(np.radians(r.inc)) ** 2
        filas.append(dict(sistema=sis, TIC=tic, sector=s, P_orb=P, D2_sobre_D1=(d2 / d1 if d1 and np.isfinite(d1) and d1 > 0 else np.nan),
                          A1_ppt=1e3 * A1, A2_ppt=1e3 * A2, a2_ppt=1e3 * a2, b2_ppt=1e3 * b2, err_ppt=1e3 * err[3], fase_A2=fase2,
                          P_LS=Pl, P_LS_sobre_Porb=Pl / P, A_elipsoidal_pred_ppt=1e3 * ell, A2_sobre_pred=A2 / ell if ell > 0 else np.nan))
        plegadas[(sis, s)] = (phb, yd, c, P)
D = pd.DataFrame(filas); D.to_csv(AQUI / "armonicos_prot_porb.csv", index=False)
pd.set_option("display.width", 230)
print(D.round(3).to_string(index=False))

# coherencia de fase del término k=2 entre sectores (estadística circular de la fase del cos(4πφ), periodo 0.5 ciclos)
print("\nCoherencia de fase de A2 entre sectores (1 = misma fase en todos; fase = ángulo de 4πφ):")
for sis in SIST:
    g = D[D.sistema == sis]
    ang = np.arctan2(g.b2_ppt, g.a2_ppt); R = np.abs(np.mean(np.exp(1j * ang)))
    sign = "a2 < 0 (mínimos en los eclipses)" if (g.a2_ppt < 0).all() else ("a2 > 0" if (g.a2_ppt > 0).all() else "a2 de signo variable")
    print(f"  sistema {sis}: R = {R:.2f} ({len(g)} sectores) · {sign} · A2/A1 = {np.round((g.A2_ppt / g.A1_ppt).to_numpy(), 1)}")

# figura: plegado orbital fuera de eclipse por sistema y sector
fig, ax = plt.subplots(len(SIST), 3, figsize=(13, 2.6 * len(SIST)), squeeze=False)
for i, sis in enumerate(SIST):
    secs = sorted(s for (x, s) in plegadas if x == sis)
    for j in range(3):
        a = ax[i, j]
        if j >= len(secs):
            a.axis("off"); continue
        ph, yd, c, P = plegadas[(sis, secs[j])]
        a.plot(ph, 1e3 * yd, ".", ms=2.5, color="#9aa7b8", alpha=0.7)
        x = np.linspace(0, 1, 300); m = c[0] + sum(c[1 + 2 * (k - 1)] * np.cos(2 * np.pi * k * x) + c[2 + 2 * (k - 1)] * np.sin(2 * np.pi * k * x) for k in (1, 2, 3))
        a.plot(x, 1e3 * m, color="#eb6834", lw=1.8)
        for v in (0, 0.5, 1): a.axvline(v, color="0.6", ls=":", lw=0.8)
        a.set_title(f"sistema {sis} · S{secs[j]} · P={P:.3f} d", fontsize=9, loc="left"); a.set_xlabel("fase orbital"); a.set_ylabel("flujo − tendencia [ppt]")
fig.suptitle("Modulación fuera de eclipse plegada en el periodo orbital (naranja = ajuste con k = 1, 2, 3). Línea punteada: fases 0 y 0.5", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(); (AQUI / "figuras").mkdir(exist_ok=True); fig.savefig(AQUI / "figuras/armonicos_prot_porb.png", dpi=125, bbox_inches="tight")
