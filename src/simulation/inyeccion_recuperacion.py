"""Prueba de inyección-recuperación del cotrending.

Se inyectan eclipses sintéticos (trapecio) en las curvas ORIGINALES de estrellas elegidas al azar (sin binarias conocidas),
se les aplica exactamente el mismo cotrending (misma base, misma elección de K por validación cruzada) y se mide:
  - razón de profundidad recuperada: a_corr / D inyectada, y a_orig / D sin cotrending (referencia),
    con a = ajuste lineal de la plantilla conocida (P, t0, forma) sobre (flujo - 1);
  - ganancia de S/N de la plantilla: (a / error) después / antes;
  - K elegido.
Uso: python inyeccion_recuperacion.py <sector> [n_por_grupo] [profundidades_ppt ...]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cotrending_pleyades as C

AQUI = C.SALIDA
CONOCIDAS = set(pd.read_csv(C.BASE / "membresia/binarias_eclipsantes_conocidas.csv").dropna(subset=["TIC"]).TIC.astype(int))


def plantilla(t, P, t0, dur, ingreso=0.25):
    """Trapecio de profundidad 1 (duración total dur, ingreso/egreso = ingreso*dur)."""
    x = np.abs(((t - t0 + P / 2) % P) - P / 2)
    semi, rampa = dur / 2, ingreso * dur
    return np.clip((semi - x) / rampa, 0, 1)


def coef_plantilla(y, m):
    ok = np.isfinite(y)
    a = np.sum((y[ok] - np.median(y[ok])) * m[ok]) / np.sum(m[ok] ** 2)
    r = y[ok] - np.median(y[ok]) - a * m[ok]
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    err = s / np.sqrt(np.sum(m[ok] ** 2))
    return a, err


def correr(sector, n_por_grupo=25, profs=(0.010, 0.020, 0.040), seed=1):
    rng = np.random.default_rng(seed)
    tics, t, F, E, M0 = C.cargar_sector(sector)
    tmag = C.MIEMBROS.reindex(tics).Tmag.to_numpy()
    filas = []
    for lo, hi in C.GRUPOS:
        ruta = AQUI / f"base_S{sector}_{lo}_{hi}.npz"
        if not ruta.exists():
            print("sin base:", ruta.name); continue
        V = np.load(ruta)["V"]
        sel = [i for i in np.flatnonzero((tmag > lo) & (tmag <= hi)) if tics[i] not in CONOCIDAS and np.isfinite(F[i]).mean() > 0.9]
        for i in rng.choice(sel, size=min(n_por_grupo, len(sel)), replace=False):
            for D in profs:
                P = rng.uniform(1.0, 6.0)
                t0 = t[0] + rng.uniform(0, P)
                dur = 0.035 * P + 0.05                       # 0.08-0.26 d
                m = plantilla(t, P, t0, dur)
                f = F[i] * (1 - D * m)                       # eclipse inyectado (flujo relativo)
                y = f - 1
                K, cv = C.elegir_K(y, V)
                A = np.column_stack([np.ones(len(t))] + [V[k] for k in range(K)]) if K else np.ones((len(t), 1))
                coef, _ = C.ajuste_robusto(y, A)
                sist = A[:, 1:] @ coef[1:] if K else np.zeros(len(t))
                fc = f - sist; fc = fc / np.nanmedian(fc)
                a0, e0 = coef_plantilla(f, m)
                a1, e1 = coef_plantilla(fc, m)
                filas.append(dict(TIC=tics[i], Sector=sector, Tmag=tmag[i], grupo=f"({lo},{hi}]", P=P, dur_h=24 * dur,
                                  D_ppt=1e3 * D, K=K, rec_antes=-a0 / D, rec_despues=-a1 / D,
                                  snr_antes=-a0 / e0, snr_despues=-a1 / e1))
    return pd.DataFrame(filas)


if __name__ == "__main__":
    s = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    profs = tuple(float(x) / 1e3 for x in sys.argv[3:]) or (0.010, 0.020, 0.040)
    d = correr(s, n, profs)
    d.to_csv(AQUI / f"inyeccion_S{s}.csv", index=False)
    d["ganancia_snr"] = d.snr_despues / d.snr_antes
    print(d.groupby(["grupo", "D_ppt"]).agg(n=("TIC", "size"), rec_antes=("rec_antes", "median"), rec_despues=("rec_despues", "median"),
                                           rec_desp_p16=("rec_despues", lambda x: x.quantile(.16)), rec_desp_p84=("rec_despues", lambda x: x.quantile(.84)),
                                           ganancia_snr=("ganancia_snr", "median")).round(2))
