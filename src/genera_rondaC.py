"""Ronda C: prior sobre e cos w con la fase del secundario medida en 1b (supuesto 3: información de la fotometría de TESS)
y cota suave e < E_MAX, para las dos corridas cuyo secundario del modelo cae en una fase equivocada.

Mismo modelo y mismos priors que la corrida vigente; solo se añaden dos potenciales a la excentricidad:
  ecosw_1b : Normal(ECOSW_OBS, SD_ECOSW) sobre e cos w = h * sqrt(h^2 + k^2), con (h, k) = sqrt(e) (cos w, sin w)
  e_max    : penalización cuadrática suave (ancho 0.02) para e > E_MAX
SD_ECOSW = max(ecosw_err de 1b, 0.005). La relación e cos w ~ (pi/2)(fase - 0.5) de 1b vale para e pequeña.
El muestreo pasa a TARGET_ACCEPT 0.95, TUNE/DRAWS 3000/3000 (igual que la ronda 3). Salidas en rondaC/TIC_x/Sn.
"""
import json, re
from pathlib import Path
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
PRIORS = pd.read_csv(RAIZ / "priors_v2.csv")
E_MAX = 0.6
CASOS = [
    (427378979, 32, "starry_correcciones/ronda2/TIC_427378979/full_427378979_s32_r2.ipynb",
     "secundario del modelo en fase 0.338 (observada 0.504), e = 0.985, R̂ = 2.2"),
    (138829403, 98, "starry_edad_vecindad/TIC_138829403/full_138829403_s98_edadvec.ipynb",
     "secundario del modelo en fase 0.654 (observada 0.513), e = 0.27"),
]
MUESTREO = [(r"TARGET_ACCEPT = [\d.]+\n", "TARGET_ACCEPT = 0.95\n"), (r"TUNE, DRAWS = \d+, \d+", "TUNE, DRAWS = 3000, 3000"),
            (r"CHAINS, CORES = \d+, \d+", "CHAINS, CORES = 4, 4"), (r"CARGAR_TRACE_EXISTENTE = \w+", "CARGAR_TRACE_EXISTENTE = False")]
ANCLA = re.compile(r'(    B_w = pm\.Deterministic\("B_w".*\n)')

filas = []
for tic, sec, fuente, motivo in CASOS:
    r = PRIORS[(PRIORS.TIC == tic) & (PRIORS.Sector == sec)].iloc[0]
    ecosw, sd = float(r.ecosw), max(float(r.ecosw_err), 0.005)
    bloque = (f"    # --- Ronda C: e cos w a partir de la fase del secundario (1b) y cota suave de e ---\n"
              f"    ECOSW_OBS, SD_ECOSW, E_MAX = {ecosw:.5f}, {sd:.5f}, {E_MAX}\n"
              f"    B_ecosw_t = B_hk[0] * tt.sqrt(tt.sum(B_hk ** 2) + 1e-12)\n"
              f"    pm.Potential(\"ecosw_1b\", -0.5 * ((B_ecosw_t - ECOSW_OBS) / SD_ECOSW) ** 2)\n"
              f"    pm.Potential(\"e_max\", -0.5 * (tt.maximum(B_ecc - E_MAX, 0.0) / 0.02) ** 2)\n")
    nb = json.loads((RAIZ / fuente).read_text())
    out = AQUI / f"TIC_{tic}" / f"S{sec}"
    out.mkdir(parents=True, exist_ok=True)
    n_anclas = n_out = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for pat, rep in MUESTREO:
            src = re.sub(pat, rep, src)
        src, k = re.subn(r'OUT_DIR = Path\("[^"]+"\)', f'OUT_DIR = Path("{out}")', src)
        n_out += k
        if ANCLA.search(src):
            src, k = ANCLA.subn(lambda m: m.group(1) + bloque, src, count=1)
            n_anclas += k
        c["source"] = src.splitlines(True)
        c["outputs"], c["execution_count"] = [], None
    assert n_anclas == 1 and n_out >= 1, (fuente, n_anclas, n_out)
    nb["cells"].insert(0, dict(cell_type="markdown", metadata={}, source=[
        f"> **Ronda C.** Copia de `{fuente}`: mismo modelo y priors, más un prior sobre e cos w = {ecosw:.4f} ± {sd:.4f} "
        f"(fase del secundario medida en 1b) y una cota suave e < {E_MAX}. Muestreo: target_accept 0.95, tune/draws 3000/3000. "
        f"Motivo: {motivo}. Salidas en `{out}`.\n"]))
    destino = AQUI / f"TIC_{tic}" / f"full_{tic}_s{sec}_rC.ipynb"
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    filas.append(dict(TIC=tic, Sector=sec, notebook=str(destino.relative_to(AQUI)), fuente=fuente,
                      ecosw_obs=ecosw, sd_ecosw=sd, e_max=E_MAX, motivo=motivo))
pd.DataFrame(filas).to_csv(AQUI / "indice_rondaC.csv", index=False)
print(len(filas), "notebooks generados")
