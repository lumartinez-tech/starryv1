"""Triage de las Pléyades sobre las curvas con cotrending (sin figuras).

Reutiliza la detección de 3_fichas_inspeccion.ipynb (LOCoR para rotadores rápidos, filtro notch, TLS por campaña,
plegado y profundidades, eventos comunes) pero leyendo cotrending/dat_cotrended y escribiendo todo en cotrending/triage/.
Por estrella guarda triage/estrellas/TIC_x.npz (curvas plegadas) y al final triage/triage_tabla.csv.
Uso: python triage_cotrending.py [n_procesos]
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")

INSP = Path("/home/luciam/Documents/starry/Resultados_Pleyades/inspeccion")
nb = json.load(open(INSP / "3_fichas_inspeccion.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
exec(src.split("INV = inventario()")[0])

# redirigir todo a las curvas con cotrending (no se toca inspeccion/)
COT = BASE / "cotrending"
DIR_DAT = COT / "dat_cotrended"
DIR_SALIDA = COT / "triage"
DIR_FICHAS = DIR_SALIDA / "fichas"
DIR_CURVAS = DIR_SALIDA / "curvas_normalizadas"
DIR_CACHE = DIR_SALIDA / "cache"
DIR_EST = DIR_SALIDA / "estrellas"
for d in (DIR_SALIDA, DIR_FICHAS, DIR_CURVAS, DIR_CACHE, DIR_EST):
    d.mkdir(parents=True, exist_ok=True)
INV = inventario()


def _uno(tic):
    try:
        res, d = analizar(int(tic), INV)
        for j, b in enumerate(d["busq"]):
            res[f"P_camp{j + 1}"] = b["P"] if b is not None else np.nan
            res[f"SDE_camp{j + 1}"] = b["SDE"] if b is not None else np.nan
        res["n_campanas"] = len(d["camp"])
        np.savez_compressed(DIR_EST / f"TIC_{int(tic)}.npz", ph=d["ph"], ph_otro=d["ph_otro"], ph_ip=d["ph_ip"],
                            r=d["r"], s=d["s"], P=res["P"], t0s=np.array(d["t0s"], float))
        return res
    except Exception as e:
        return dict(TIC=int(tic), error=str(e))


if __name__ == "__main__":
    t0 = time.time()
    for s in sorted(INV["Sector"].unique()):          # eventos comunes una vez por sector (caché), con las curvas ya corregidas
        eventos_comunes(s, INV)
        print(f"eventos comunes S{s} listos", flush=True)
    tics = sorted(INV["TIC"].unique(), key=lambda t: MIEMBROS["Tmag"].get(t, 99))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    filas = []
    with ProcessPoolExecutor(n) as ex:
        for i, r in enumerate(ex.map(_uno, tics, chunksize=2), 1):
            filas.append(r)
            if i % 50 == 0:
                print(f"{i}/{len(tics)}  {time.time() - t0:.0f} s", flush=True)
                pd.DataFrame(filas).to_csv(DIR_SALIDA / "triage_tabla_parcial.csv", index=False)
    d = pd.DataFrame(filas)
    d["Tmag"] = d["TIC"].map(MIEMBROS["Tmag"])
    d["Rcont"] = d["TIC"].map(MIEMBROS["Rcont"]) if "Rcont" in MIEMBROS.columns else np.nan
    d.to_csv(DIR_SALIDA / "triage_tabla.csv", index=False)
    print(f"{len(d)} estrellas; errores: {d['error'].notna().sum() if 'error' in d else 0}; {time.time() - t0:.0f} s", flush=True)
