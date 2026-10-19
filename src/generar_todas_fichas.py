"""Genera la ficha (PNG) y el resumen BLS de todas las estrellas con curva, en paralelo.
Usa las funciones de 3_fichas_inspeccion.ipynb. Uso: python generar_todas_fichas.py [n_procesos]
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use("Agg")
nb = json.load(open("/home/luciam/Documents/starry/Resultados_Pleyades/inspeccion/3_fichas_inspeccion.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
exec(src.split("INV = inventario()")[0])
INV = inventario()


def _uno(tic):
    try:
        return ficha(int(tic), mostrar=False, inv=INV)
    except Exception as e:
        return dict(TIC=int(tic), error=str(e))


if __name__ == "__main__":
    for s in sorted(INV["Sector"].unique()):          # eventos comunes una vez por sector (caché)
        eventos_comunes(s, INV)
        print(f"eventos comunes S{s} listos", flush=True)
    tics = sorted(INV["TIC"].unique(), key=lambda t: MIEMBROS["Tmag"].get(t, 99))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    with ProcessPoolExecutor(n) as ex:
        filas = list(ex.map(_uno, tics, chunksize=2))
    r = resumen_ordenado(filas)
    r.to_csv(DIR_SALIDA / "resumen_tls.csv", index=False)
    print(f"{len(r)} estrellas; errores: {r['error'].notna().sum() if 'error' in r else 0}", flush=True)
