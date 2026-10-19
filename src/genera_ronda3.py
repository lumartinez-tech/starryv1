"""Ronda 3 (opción A): más muestreo para las corridas que ajustan bien pero tienen R̂/ESS marginales.

Copia el notebook que produjo cada corrida vigente y cambia SOLO el muestreo (modelo y priors idénticos):
  TARGET_ACCEPT 0.95, TUNE 3000, DRAWS 3000, 4 cadenas con 4 núcleos, CARGAR_TRACE_EXISTENTE = False.
Escribe en ronda3/TIC_x/Sn (no toca las carpetas originales). Uso: python genera_ronda3.py
"""
import json, re
from pathlib import Path
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
CASOS = [  # (TIC, sector, notebook fuente que produjo la corrida vigente, problema)
    (11300044, 6, "starry_por_tic/TIC_11300044/full_11300044_s6_ecc.ipynb", "R̂ 1.02, ESS 232"),
    (11300044, 98, "starry_por_tic/TIC_11300044/full_11300044_s98_ecc.ipynb", "R̂ 1.05, ESS 79"),
    (199780530, 6, "starry_por_tic/TIC_199780530/full_199780530_s6.ipynb", "R̂ 1.03, ESS 106"),
    (249032831, 98, "starry_por_tic/TIC_249032831/full_249032831_s98.ipynb", "R̂ 1.05, ESS 91"),
    (427347964, 6, "starry_por_tic/TIC_427347964/full_427347964_s6.ipynb", "R̂ 1.02, ESS 223"),
    (427347964, 32, "starry_por_tic/TIC_427347964/full_427347964_s32.ipynb", "R̂ 1.05, ESS 74"),
    (176775425, 98, "starry_correcciones/TIC_176775425/full_176775425_s98_corr.ipynb", "R̂ 1.04, ESS 137"),
    (264483228, 98, "starry_edad_vecindad/TIC_264483228/full_264483228_s98_edadvec.ipynb", "R̂ 1.03, ESS 155"),
]
CAMBIOS = [  # (patrón, reemplazo, obligatorio)
    (r"TARGET_ACCEPT = [\d.]+\n", "TARGET_ACCEPT = 0.95\n", True),
    (r"TUNE, DRAWS = \d+, \d+", "TUNE, DRAWS = 3000, 3000", True),
    (r"CHAINS, CORES = \d+, \d+", "CHAINS, CORES = 4, 4", True),
    (r"CARGAR_TRACE_EXISTENTE = \w+", "CARGAR_TRACE_EXISTENTE = False", True),
]

filas = []
for tic, sec, fuente, problema in CASOS:
    nb = json.loads((RAIZ / fuente).read_text())
    out = AQUI / f"TIC_{tic}" / f"S{sec}"
    out.mkdir(parents=True, exist_ok=True)
    hecho = False
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for pat, rep, _ in CAMBIOS:
            src = re.sub(pat, rep, src)
        if "OUT_DIR = Path(" in src:
            src, n = re.subn(r'OUT_DIR = Path\("[^"]+"\)', f'OUT_DIR = Path("{out}")', src)
            hecho = hecho or n > 0
        c["source"] = src.splitlines(True)
        c["outputs"], c["execution_count"] = [], None          # sin salidas viejas
    assert hecho, f"no encontré OUT_DIR en {fuente}"
    codigo = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for pat, rep, oblig in CAMBIOS:
        assert re.search(re.escape(rep.strip()), codigo), (fuente, rep)
    nb["cells"].insert(0, dict(cell_type="markdown", metadata={}, source=[
        f"> **Ronda 3 (solo muestreo).** Copia de `{fuente}` con el mismo modelo y los mismos priors. "
        f"Cambia: `TARGET_ACCEPT = 0.95`, `TUNE, DRAWS = 3000, 3000`, `CARGAR_TRACE_EXISTENTE = False`. "
        f"Motivo: {problema} en la corrida vigente, con buen ajuste. Salidas en `{out}`.\n"]))
    destino = AQUI / f"TIC_{tic}" / f"full_{tic}_s{sec}_r3.ipynb"
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    filas.append(dict(TIC=tic, Sector=sec, notebook=str(destino.relative_to(AQUI)), fuente=fuente, problema_previo=problema))
pd.DataFrame(filas).to_csv(AQUI / "indice_ronda3.csv", index=False)
print(len(filas), "notebooks generados")
