"""¿Recupera la detección (LOCoR + notch + TLS) el periodo de las binarias eclipsantes conocidas? Compara curvas originales y con
cotrending, y dos rangos de búsqueda de periodo. Salida: prueba_conocidas.csv. Uso: python prueba_conocidas.py [n_procesos]"""
import json, sys, itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")

INSP = Path("/home/luciam/Documents/starry/Resultados_Pleyades/inspeccion")
nb = json.load(open(INSP / "3_fichas_inspeccion.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
exec(src.split("INV = inventario()")[0])
COT = BASE / "cotrending"
SAL = COT / "prueba_conocidas"; SAL.mkdir(exist_ok=True)

cat = pd.read_csv(BASE / "membresia/binarias_eclipsantes_conocidas.csv").dropna(subset=["TIC"])
cat["P_cat"] = cat[["P_VSX", "P_TESS_EB", "P_Gaia_EB"]].bfill(axis=1).iloc[:, 0]
cat["TIC"] = cat["TIC"].astype(int)


def configurar(curvas, pmax):
    global DIR_DAT, DIR_CACHE, DIR_CURVAS, DIR_SALIDA, DIR_FICHAS, PERIODOS, INV
    if curvas == "orig":
        DIR_DAT, DIR_CACHE = BASE / "tessextractor_local/resultados/dat", INSP / "cache"
    else:
        DIR_DAT, DIR_CACHE = COT / "dat_cotrended", COT / "triage/cache"
    DIR_SALIDA = SAL; DIR_FICHAS = SAL / "fichas"; DIR_CURVAS = SAL / f"curvas_{curvas}"; DIR_CURVAS.mkdir(parents=True, exist_ok=True)
    PERIODOS = (0.25, pmax)
    INV = inventario()


def tarea(args):
    tic, curvas, pmax = args
    try:
        configurar(curvas, pmax)
        res, _ = analizar(int(tic), INV)
        return dict(TIC=tic, curvas=curvas, pmax=pmax, P=res["P"], SDE=res["SDE_TLS"], snr=res["snr_prim"], sig_sec=res["sig_sec"], dur_h=res["dur_h"])
    except Exception as e:
        return dict(TIC=tic, curvas=curvas, pmax=pmax, error=str(e)[:120])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tics = [t for t in cat["TIC"] if (COT / "dat_cotrended").glob(f"TIC_{t}_S_*_LC.dat") and list((COT / "dat_cotrended").glob(f"TIC_{t}_S_*_LC.dat"))]
    tareas = list(itertools.product(tics, ["orig", "cot"], [15.0, 8.0]))
    with ProcessPoolExecutor(n) as ex:
        filas = list(ex.map(tarea, tareas, chunksize=1))
    d = pd.DataFrame(filas).merge(cat[["TIC", "P_cat", "Tmag"]], on="TIC")
    d["P_ok"] = [any(abs(p / pc - x) < 0.03 * x for x in (1, 2, 0.5)) if np.isfinite(p) else False for p, pc in zip(d["P"], d["P_cat"])]
    d.to_csv(COT / "prueba_conocidas.csv", index=False)
    print(d.groupby(["curvas", "pmax"]).P_ok.agg(["sum", "count"]))
