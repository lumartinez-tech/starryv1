"""Paso 2: TLS sobre las no detecciones y comparación TLS contra BLS con las mismas inyecciones.

A) Las 44 curvas reales (9 candidatas sin eclipses): wotan 0.5 d -> bins de 10 min -> TLS iterativo (3 picos) -> mismos criterios de consistencia
   que en completitud_inyecciones.py (S/N con ruido rojo, tránsitos consistentes, par-impar, beta) + duración física.
B) Inyecciones (profundidad 1.5-10 %, P 0.5-12 d) en el sector 43 de cada estrella: se busca con TLS y con BLS (el mismo BLS iterativo del paso 1)
   sobre exactamente los mismos datos, y se compara la recuperación.
C) Nulos (flujo invertido, bloques de 0.5 d barajados) para calibrar el umbral de SDE de TLS.
Entorno: transitos (TLS 2.0). Salida: resultados/tls_*.csv
"""
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
AQUI = Path("/home/luciam/Documents/starry/Resultados_Pleyades/completitud")
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ci", AQUI / "completitud_inyecciones.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from transitleastsquares import transitleastsquares, transit_mask
RES = AQUI / "resultados"
N_INY, N_NULOS = 25, 60
N_ITER, USAR_HILOS = 3, 7
FAC_DUR = 2.6


def binear(t, y, minutos=10.0):
    k = np.floor((t - t.min()) / (minutos / 1440)).astype(int)
    df = pd.DataFrame(dict(k=k, t=t, y=y)).groupby("k").mean()
    return df.t.to_numpy(), df.y.to_numpy()


def dur_ok(P, dur, rho):
    return dur <= FAC_DUR * 13.0 / 24 * (P / 365.25) ** (1 / 3) * rho ** (-1 / 3)


def tls_iterativo(t, plano, sig, par):
    """TLS con 3 pasadas; si el pico no pasa la consistencia se enmascaran sus tránsitos y se repite."""
    tm, pm = t.copy(), plano.copy(); picos = []
    for it in range(N_ITER):
        if len(tm) < 200:
            break
        mod = transitleastsquares(tm, pm, np.full_like(pm, sig))
        r = mod.power(period_min=m.PMIN, period_max=m.PMAX, R_star=par.Rad, M_star=par.Mass, R_star_min=0.1, R_star_max=3.0, M_star_min=0.1, M_star_max=3.0,
                      u=[0.3, 0.3], oversampling_factor=2, duration_grid_step=1.15, use_threads=USAR_HILOS, show_progress_bar=False)
        if not np.isfinite(r.period) or r.period <= 0:
            break
        v = m.vetar(t, plano, float(r.period), float(r.T0), float(r.duration), sig)
        picos.append(dict(P=float(r.period), t0=float(r.T0), dur=float(r.duration), SDE=float(r.SDE), snr_tls=float(r.snr), depth_tls=float(1 - r.depth),
                          n_tr_tls=int(r.transit_count), **v, dur_ok=bool(dur_ok(float(r.period), float(r.duration), par.rho))))
        if v["ok"] and picos[-1]["dur_ok"]:
            break
        tm, pm = tm[~transit_mask(tm, float(r.period), 2 * float(r.duration), float(r.T0))], pm[~transit_mask(tm, float(r.period), 2 * float(r.duration), float(r.T0))]
    return picos


def bls_iterativo_binneado(t, plano, sig, par):
    tm, pm = t.copy(), plano.copy(); picos = []
    for it in range(m.N_ITER):
        if len(tm) < 200:
            break
        P, t0, dur, sde = m.bls_pico(tm, pm, sig)
        v = m.vetar(t, plano, P, t0, dur, sig)
        picos.append(dict(P=P, t0=t0, dur=dur, SDE=sde, **v, dur_ok=bool(dur_ok(P, dur, par.rho))))
        if v["ok"] and picos[-1]["dur_ok"]:
            break
        ph = ((tm - t0 + 0.5 * P) % P) - 0.5 * P; k = np.abs(ph) > 1.0 * dur
        tm, pm = tm[k], pm[k]
    return picos


def preparar(t, y):
    tt, plano = m.aplanar(t, y)
    tb, pb = binear(tt, plano)
    sig = 1.4826 * np.nanmedian(np.abs(pb - np.nanmedian(pb)))
    return tb, pb, sig


def aplanar_fila(prefijo, picos, fila):
    fila[f"{prefijo}_n_picos"] = len(picos)
    for k, pk in enumerate(picos):
        for n, val in pk.items():
            fila[f"{prefijo}_{n}_{k}"] = val
    return fila


if __name__ == "__main__":
    t_ini = time.time()
    # ---------------- A) curvas reales
    filas = []
    for ic, c in enumerate(m.CURVAS):
        par = m.PARAM.loc[c["tic"]]
        tb, pb, sig = preparar(c["t"], c["y"])
        picos = tls_iterativo(tb, pb, sig, par)
        filas.append(aplanar_fila("tls", picos, dict(tic=c["tic"], sector=c["sector"], sigma=sig)))
        print(f"[A {ic + 1}/{len(m.CURVAS)}] TIC {c['tic']} S{c['sector']}: {[(round(p['P'], 3), round(p['SDE'], 1), p['ok'] and p['dur_ok']) for p in picos]}  ({(time.time() - t_ini) / 60:.1f} min)", flush=True)
        pd.DataFrame(filas).to_csv(RES / "tls_reales.csv", index=False)
    # ---------------- B) inyecciones: TLS frente a BLS en los mismos datos (sector 43 de cada estrella)
    filas = []
    for tic in m.TICS:
        ic = next(i for i, c in enumerate(m.CURVAS) if c["tic"] == tic and c["sector"] == 43)
        c = m.CURVAS[ic]; par = m.PARAM.loc[tic]
        for j in range(N_INY):
            rng = np.random.default_rng(777000 + 1000 * ic + j)
            P = float(np.exp(rng.uniform(np.log(m.PMIN), np.log(m.PMAX)))); dep = float(np.exp(rng.uniform(np.log(0.015), np.log(0.10))))
            k = np.sqrt(dep); b = float(rng.uniform(0, 1 - k))
            T = 13.0 / 24 * (P / 365.25) ** (1 / 3) * par.rho ** (-1 / 3) * np.sqrt(max((1 + k) ** 2 - b ** 2, 1e-3))
            tau = min(0.5 * T, T * k / ((1 + k) * max(1 - b ** 2, 0.05))); t0i = float(c["t"].min() + rng.uniform(0, P))
            y = c["y"] * m.trapecio(c["t"], t0i, P, dep, T, tau)
            tb, pb, sig = preparar(c["t"], y)
            fila = dict(tic=tic, sector=43, P_inj=P, depth_inj=dep, T_inj=T)
            aplanar_fila("tls", tls_iterativo(tb, pb, sig, par), fila)
            aplanar_fila("bls", bls_iterativo_binneado(tb, pb, sig, par), fila)
            filas.append(fila)
        pd.DataFrame(filas).to_csv(RES / "tls_vs_bls_inyecciones.csv", index=False)
        print(f"[B] TIC {tic} listo ({(time.time() - t_ini) / 60:.1f} min)", flush=True)
    # ---------------- C) nulos para el umbral de SDE de TLS
    filas = []
    rng = np.random.default_rng(31415)
    for j in range(N_NULOS):
        tic = m.TICS[j % len(m.TICS)]
        ic = next(i for i, c in enumerate(m.CURVAS) if c["tic"] == tic and c["sector"] == 43)
        c = m.CURVAS[ic]; par = m.PARAM.loc[tic]
        yi = 2 - c["y"]; kk = np.floor((c["t"] - c["t"].min()) / 0.5).astype(int)
        bloques = [yi[kk == u] for u in np.unique(kk)]; orden = rng.permutation(len(bloques))
        y = np.concatenate([bloques[q] for q in orden])[: len(c["t"])]
        if len(y) < len(c["t"]):
            y = np.concatenate([y, np.full(len(c["t"]) - len(y), 1.0)])
        tb, pb, sig = preparar(c["t"], y)
        filas.append(aplanar_fila("tls", tls_iterativo(tb, pb, sig, par), dict(tic=tic, sector=43, modo="nulo")))
        pd.DataFrame(filas).to_csv(RES / "tls_nulos.csv", index=False)
        print(f"[C {j + 1}/{N_NULOS}] ({(time.time() - t_ini) / 60:.1f} min)", flush=True)
    print("FIN", flush=True)
