"""MassAge (50 iteraciones) para los grupos de los sistemas 7, 11 y 20, con las mismas funciones del notebook."""
import json
from pathlib import Path
nb = json.load(open("/home/luciam/Documents/starry/Resultados_Orion/edades_vecindad/edades_por_vecindad.ipynb"))
src = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
exec(src[0]); exec(src[1])          # configuración y funciones de MassAge
import pandas as pd, time
MASSAGE_NPROC = 2
for tic in (264483228, 138829403, 264459857):
    g = pd.read_csv(AQUI / f"TIC_{tic}" / "grupo_TIC82.csv")
    t0 = time.time()
    r = correr_massage(entrada_massage(g), "massage_grupo", AQUI / f"TIC_{tic}", nmc=NMC_DEFECTO)
    print(f"TIC {tic}: {len(r)} estrellas en {(time.time() - t0) / 60:.1f} min", flush=True)
