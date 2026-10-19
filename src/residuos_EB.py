"""Paso 3: búsqueda de tránsitos en los residuos de las 4 EB (26017840, 35159657, 125754991, 346541355).

Se enmascaran los eclipses primario y secundario con la efeméride v2 (pipeline/interpolador/profundidades_v2_pleyades.csv; el secundario en la fase
mediana del sistema, p. ej. 0.355 en 346541355), se aplana con wotan y se busca con el BLS iterativo del paso 1 con los mismos criterios de consistencia
de errores (S/N con ruido rojo, tránsitos consistentes, par-impar, beta) + duración física para la densidad de la primaria. Los picos en armónicos del periodo
de la EB (k P_EB, P_EB/k) se vetan: son residuos de los eclipses.
Inyección-recuperación: tránsitos trapezoidales de radio Rp (0.1 a min(3 R_Jup, 0.3 R_A)) alrededor de la primaria, inyectados ANTES de enmascarar (los que
caen en un eclipse se pierden, como en la realidad). Profundidad observada = amp_A (Rp/R_A)^2, con amp_A la fracción de luz de la primaria.
Uso: python residuos_EB.py [N_inyecciones_por_curva]
"""
import importlib.util, sys, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
AQUI = Path("/home/luciam/Documents/starry/Resultados_Pleyades/completitud"); BASE = AQUI.parent
N_INY = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ci", AQUI / "completitud_inyecciones.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
RES = AQUI / "resultados"
m.N_ITER = 5
from astropy.timeseries import BoxLeastSquares

EDGES = [None, 1.0, 2.0, 4.0, None]                # tramos de periodo; la duración permitida en cada tramo es <= 0.2 x el periodo mínimo del tramo


def bls_pico_seg(t, plano, sig):
    """BLS por tramos de periodo con duración <= 20 % del periodo (evita picos con duración ~ periodo)."""
    base = t.max() - t.min()
    bordes = [m.PMIN, 1.0, 2.0, 4.0, m.PMAX]
    bordes = [b for b in bordes if m.PMIN <= b <= m.PMAX]
    per_all, pw_all, tt_all, du_all = [], [], [], []
    for a, b in zip(bordes[:-1], bordes[1:]):
        durs = m.DURS[m.DURS <= 0.2 * a]
        if len(durs) == 0:
            continue
        frec = np.arange(1 / b, 1 / a, 0.01 / base); per = np.sort(1 / frec)
        r = BoxLeastSquares(t, plano, dy=np.full_like(plano, sig)).power(per, durs)
        per_all.append(r.period); pw_all.append(r.power); tt_all.append(r.transit_time); du_all.append(r.duration)
    per, pw, tt, du = map(np.concatenate, (per_all, pw_all, tt_all, du_all))
    i = int(np.argmax(pw))
    return float(per[i]), float(tt[i]), float(du[i]), float((pw[i] - np.mean(pw)) / np.std(pw))


m.bls_pico = bls_pico_seg


def _depth_epochs(t, plano, tc_list, dur):
    """Profundidad de cada época: mediana de la línea base local (1.2-3.5 duraciones a ambos lados) menos media en 0.8 duraciones centrales."""
    out, nn = [], []
    for tc in tc_list:
        i0, i1 = np.searchsorted(t, [tc - 3.5 * dur, tc + 3.5 * dur]); tt = t[i0:i1]; yy = plano[i0:i1]
        if len(tt) < 8:
            continue
        x = np.abs(tt - tc); dentro = x < 0.4 * dur; loc = (x > 1.2 * dur)
        if dentro.sum() >= 3 and loc.sum() >= 5:
            out.append(np.median(yy[loc]) - np.mean(yy[dentro])); nn.append(dentro.sum())
    return np.array(out), np.array(nn)


def ruido_empirico(t, plano, P, t0, dur, P_eb_mask=None):
    """Error por época medido en el propio dato: el mismo estadístico en 12 fases desplazadas del mismo periodo (excluye las cercanas al tránsito candidato)."""
    ph0 = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    e_all = np.arange(np.floor((t.min() - t0) / P) - 1, np.ceil((t.max() - t0) / P) + 2)
    pool = []
    for j in range(12):
        off = (j + 0.5) / 12 * P
        if min(off, P - off) < 3 * dur:
            continue
        d, _ = _depth_epochs(t, plano, t0 + off + e_all * P, dur)
        pool.append(d)
    pool = np.concatenate(pool) if pool else np.array([])
    if len(pool) < 20:
        return np.nan
    return float(1.4826 * np.median(np.abs(pool - np.median(pool))))


def vetar_local(t, plano, P, t0, dur, sig):
    """Igual que vetar() del paso 1, pero cada tránsito se mide contra su línea base LOCAL (ventanas de 1.2 a 3.5 duraciones a ambos lados):
    con variabilidad lenta (estrellas brillantes) la línea base global sobreestima la dispersión entre tránsitos."""
    ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    ep = np.round((t - t0) / P).astype(int)
    dentro = np.abs(ph) < 0.4 * dur
    k = np.floor((t - t.min()) / dur).astype(int)
    exc = np.abs(ph) < 1.5 * dur
    dfb = pd.DataFrame(dict(k=k[~exc], y=plano[~exc])).groupby("k").agg(m=("y", "mean"), n=("y", "size"))
    dfb = dfb[dfb.n >= 3]
    if len(dfb) < 10:
        return dict(ok=False, motivo="pocas_ventanas", snr=np.nan, p_cons=np.nan, oe=np.nan, beta=np.nan, n_tr=0, depth=np.nan)
    sT = 1.4826 * np.median(np.abs(dfb.m - np.median(dfb.m)))
    cad = np.median(np.diff(t)); m_ = max(dur / cad, 1.0)
    beta = float(sT / (sig / np.sqrt(m_)))
    d_i, e_i, n_i = [], [], []
    for e in np.unique(ep[dentro]):
        s = dentro & (ep == e)
        loc = (ep == e) & (np.abs(ph) > 1.2 * dur) & (np.abs(ph) < 3.5 * dur)
        if s.sum() >= 3 and loc.sum() >= 5:
            d_i.append(np.median(plano[loc]) - np.mean(plano[s])); e_i.append(e); n_i.append(s.sum())
    d_i, e_i, n_i = np.array(d_i), np.array(e_i), np.array(n_i)
    n = len(d_i)
    if n < 2:
        return dict(ok=False, motivo="menos_de_2_transitos", snr=np.nan, p_cons=np.nan, oe=np.nan, beta=beta, n_tr=n, depth=np.nanmean(d_i) if n else np.nan)
    err_formal = sT * np.sqrt(np.maximum(1.0, m_ / n_i)) * np.sqrt(1 + 1.0 / 3.0)
    s_emp = ruido_empirico(t, plano, P, t0, dur)
    if not np.isfinite(s_emp):
        return dict(ok=False, motivo="pocas_ventanas", snr=np.nan, p_cons=np.nan, oe=np.nan, beta=beta, n_tr=n, depth=np.nanmean(d_i))
    err_i = np.maximum(err_formal, s_emp)                                           # error por época medido en el dato (incluye ruido rojo)
    z = np.abs(d_i - np.median(d_i)) / err_i; bueno = z < 4.0; frac_malas = 1 - bueno.mean()
    d_i, e_i, err_i = d_i[bueno], e_i[bueno], err_i[bueno]
    if len(d_i) < 2:
        return dict(ok=False, motivo="menos_de_2_transitos", snr=np.nan, p_cons=np.nan, oe=np.nan, beta=beta, n_tr=len(d_i), depth=np.nan)
    w = 1 / err_i ** 2; dep = float(np.sum(w * d_i) / np.sum(w)); sdep = float(np.sum(w) ** -0.5); snr = dep / sdep
    chi = float(np.sum(((d_i - dep) / err_i) ** 2)); p = float(m.chi2d.sf(chi, len(d_i) - 1))
    par, imp = d_i[e_i % 2 == 0], d_i[e_i % 2 == 1]; ep_, ei_ = err_i[e_i % 2 == 0], err_i[e_i % 2 == 1]
    if len(par) and len(imp):
        oe = abs(np.average(par, weights=1 / ep_ ** 2) - np.average(imp, weights=1 / ei_ ** 2)) / np.sqrt(np.sum(ep_ ** -2) ** -1 + np.sum(ei_ ** -2) ** -1)
    else:
        oe = 0.0
    incons = (p <= m.P_CONS) or (frac_malas > 0.2)
    ok = (snr >= m.SNR_MIN) and (not incons) and (oe < m.OE_MAX) and (beta <= m.BETA_MAX)
    motivo = "ok" if ok else ",".join(k_ for k_, c in (("snr_bajo", snr < m.SNR_MIN), ("inconsistente", incons), ("par_impar", oe >= m.OE_MAX), ("ruido_rojo", beta > m.BETA_MAX)) if c)
    return dict(ok=bool(ok), motivo=motivo, snr=float(snr), p_cons=p, oe=float(oe), beta=beta, n_tr=len(d_i), depth=dep, f_malas=float(frac_malas))


m.vetar = vetar_local

RJ = 9.731
EB = [26017840, 35159657, 125754991, 346541355]
LC_DIRS = {346541355: BASE / "tessextractor_local/tess_extractor_ 346541355/Light_Curve_Files"}
med = pd.read_csv(BASE / "pipeline/interpolador/profundidades_v2_pleyades.csv")
prim = pd.read_csv(BASE / "pipeline/interpolador/primarias_pleyades.csv").set_index("TIC")
fase2 = med[med.secundario_detectado].groupby("TIC").fase_secundario.median()
d2 = med.groupby("TIC").D_secundario.median()
detectado = med.groupby("TIC").secundario_detectado.any()


def cargar_eb(tic, sector):
    f = LC_DIRS.get(tic, m.LCDIR) / f"tic {tic}_S_{sector}_LC.txt"
    d = pd.read_csv(f, sep=r"\s+").rename(columns={"#Time": "t"}).dropna(subset=["t", "Flux_CBV_SS"]).sort_values("t").drop_duplicates("t")
    t, y = d.t.to_numpy(float), d.Flux_CBV_SS.to_numpy(float)
    return t, y / np.nanmedian(y)


CURVAS = []
for r in med.itertuples():
    if r.TIC not in EB or r.Calidad_depth != "ok":
        continue
    t, y = cargar_eb(r.TIC, r.Sector)
    P = r.P; t0 = r.t0 + (r.centro_primario if np.isfinite(r.centro_primario) else 0.0) * P
    f2 = fase2[r.TIC] if detectado[r.TIC] else 0.5
    ph1 = ((t - t0 + 0.5 * P) % P) - 0.5 * P; ph2 = ((t - t0 - f2 * P + 0.5 * P) % P) - 0.5 * P
    half = 1.6 * r.Duration
    mask = (np.abs(ph1) < half) | (np.abs(ph2) < half)
    a = prim.loc[r.TIC]
    mA = a.M1_MIST if np.isfinite(a.get("M1_MIST", np.nan)) else (a.M1_PARSEC if np.isfinite(a.get("M1_PARSEC", np.nan)) else a.M1_massage)
    CURVAS.append(dict(tic=r.TIC, sector=int(r.Sector), t=t, y=y, mask=mask, P_eb=P, R_A=a.R1, rho_A=mA / a.R1 ** 3,
                       amp_A=1 - (d2[r.TIC] if detectado[r.TIC] else 0.0)))


def armonico_eb(P, P_eb):
    for k in (1, 2, 3, 4, 5, 6):
        if abs(P / (k * P_eb) - 1) < 0.015 or abs(P / (P_eb / k) - 1) < 0.015:
            return True
    return False


def dur_ok(P, dur, rho):
    return dur <= 2.6 * 13.0 / 24 * (P / 365.25) ** (1 / 3) * rho ** (-1 / 3)


DURS_FRIA = np.array([0.03, 0.05, 0.08, 0.12, 0.18, 0.25]); DURS_CALIENTE = np.array([0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35])


def tarea(a):
    ic, seed, modo = a
    c = CURVAS[ic]; rng = np.random.default_rng(seed)
    # primarias calientes (poco densas): tránsitos más largos y periodo mínimo mayor para que duración < periodo; frías: como en el paso 1
    if c["rho_A"] < 1.0:
        m.DURS, m.PMIN = DURS_CALIENTE, 0.8
    else:
        m.DURS, m.PMIN = DURS_FRIA, 0.5
    t, y = c["t"], c["y"].copy()
    fila = dict(tic=c["tic"], sector=c["sector"], modo=modo, seed=seed, P_eb=c["P_eb"], R_A=c["R_A"], amp_A=c["amp_A"], eclipses_enmascarados=float(c["mask"].mean()))
    if modo == "iny":
        k_lo, k_hi = 0.1 / (c["R_A"] * RJ), min(3.0, 0.3 * c["R_A"] * RJ) / (c["R_A"] * RJ)
        k = float(np.exp(rng.uniform(np.log(k_lo), np.log(k_hi)))); dep = c["amp_A"] * k ** 2
        P = float(np.exp(rng.uniform(np.log(m.PMIN), np.log(m.PMAX)))); b = float(rng.uniform(0, 1 - k))
        T = 13.0 / 24 * (P / 365.25) ** (1 / 3) * c["rho_A"] ** (-1 / 3) * np.sqrt(max((1 + k) ** 2 - b ** 2, 1e-3))
        tau = min(0.5 * T, T * k / ((1 + k) * max(1 - b ** 2, 0.05))); t0 = float(t.min() + rng.uniform(0, P))
        y = y * m.trapecio(t, t0, P, dep, T, tau)
        fila.update(P_inj=P, depth_inj=dep, T_inj=T, k_inj=k, Rp_RJ=k * c["R_A"] * RJ, arm_inj=armonico_eb(P, c["P_eb"]))
    elif modo == "nulo":
        yi = 2 - y; kk = np.floor((t - t.min()) / 0.5).astype(int)
        bloques = [yi[kk == u] for u in np.unique(kk)]; orden = rng.permutation(len(bloques))
        y = np.concatenate([bloques[q] for q in orden])[: len(t)]
        if len(y) < len(t):
            y = np.concatenate([y, np.full(len(t) - len(y), 1.0)])
    try:
        picos, sig = m.buscar_iterativo(t[~c["mask"]], y[~c["mask"]])
    except Exception as e:                      # noqa
        fila.update(error=str(e)[:60]); return fila
    fila["sigma"] = sig; fila["n_picos"] = len(picos)
    for k_, pk in enumerate(picos):
        for nombre, val in pk.items():
            fila[f"{nombre}_{k_}"] = val
        fila[f"arm_{k_}"] = armonico_eb(pk["P"], c["P_eb"]); fila[f"durok_{k_}"] = bool(dur_ok(pk["P"], pk["dur"], c["rho_A"]))
        if modo == "iny":
            fila[f"coincide_{k_}"] = m.coincide(pk["P"], fila["P_inj"])
    return fila


if __name__ == "__main__":
    print(f"{len(CURVAS)} curvas de EB; {N_INY} inyecciones por curva", flush=True)
    for c in CURVAS:
        print(f"  TIC {c['tic']} S{c['sector']}: {100 * c['mask'].mean():.1f} % de los puntos enmascarados; R_A = {c['R_A']:.2f}, rho_A = {c['rho_A']:.2f}, amp_A = {c['amp_A']:.3f}", flush=True)
    with Pool(6) as pool:
        real = pool.map(tarea, [(i, 0, "real") for i in range(len(CURVAS))])
        pd.DataFrame(real).to_csv(RES / "eb_reales.csv", index=False); print("reales listas", flush=True)
        nul = pool.map(tarea, [(i, 2000 + 31 * j + i, "nulo") for i in range(len(CURVAS)) for j in range(15)], chunksize=4)
        pd.DataFrame(nul).to_csv(RES / "eb_nulos.csv", index=False); print("nulos listos", flush=True)
        iny = pool.map(tarea, [(i, 90000 + 100000 * i + j, "iny") for i in range(len(CURVAS)) for j in range(N_INY)], chunksize=8)
        pd.DataFrame(iny).to_csv(RES / "eb_inyecciones.csv", index=False); print("inyecciones listas", flush=True)
