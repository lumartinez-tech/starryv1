"""Segunda ronda de correcciones de starry (3 casos que no quedaron bien en la primera ronda).

Reutiliza notebook_corregido() de ../genera_correcciones.py y cambia, por caso:
  - 427378979 S32 y 427395536 S6: amplitud de la secundaria acotada a amp_B ≥ D2 − 2σ (una estrella no puede
    perder en el eclipse más luz de la que emite); evita el modo de "secundaria oscura" (B_amp → 0).
  - 50897998 S6: priors de masa nuevos con el interpolador de isócronas (MIST), invirtiendo R(M) con los radios
    de S32/S98 y edad 8–20 Myr (priors_50897998/priors_masa.py).
Salidas en starry_correcciones/ronda2/; la primera ronda y starry_por_tic/ no se tocan.
Uso: python genera_ronda2.py
"""
from pathlib import Path

import pandas as pd

RONDA2 = Path(__file__).resolve().parent
GEN1 = RONDA2.parent / "genera_correcciones.py"
fuente = GEN1.read_text()
__file__ = str(GEN1)               # las rutas del generador de la primera ronda son relativas a él
exec(compile(fuente[:fuente.index("\nfilas = []\nfor (tic, sector), caso in sorted(CASOS.items())")], str(GEN1), "exec"))
AQUI = RONDA2                                 # notebook_corregido() escribe las salidas aquí

LINEA_BAMP = 'B_amp = PositiveNormal("B_amp", mu=amp_B, sd=sd_amp_B, testval=amp_B)'
pm_ = pd.read_csv(AQUI.parent / "priors_50897998" / "prior_final_masas.csv").set_index("componente")
D = pd.read_csv(RUTA_ORION / "priors_v2.csv").set_index(["TIC", "Sector"])


def cota_amp_B(tic, s):
    d = D.loc[(tic, s)]
    return round(float(d["Depth_secondary"] - 2 * d["Depth_secondary_err"]), 5), float(d["Depth_secondary"])


R2 = {
    (427378979, 32): dict(
        base=CASOS[(427378979, 32)], amp_B_min=cota_amp_B(427378979, 32),
        motivo="Primera ronda: R̂ = 1.99, ESS = 5. Tres de las cuatro cadenas llevaron B_amp a ~0.001 (secundaria "
               "sin luz); sin eclipse secundario, la excentricidad quedó libre (e 0.26–0.68 según la cadena)."),
    (427395536, 6): dict(
        base=CASOS[(427395536, 6)], amp_B_min=cota_amp_B(427395536, 6),
        motivo="Primera ronda: convergió (R̂ 1.01) pero con B_amp = 0.036 < D2 = 0.046: el modelo no puede alcanzar la "
               "profundidad del secundario (residuo −4.4σ) y lo compensa con e sin ω = −0.18."),
    (50897998, 6): dict(
        base=CASOS[(50897998, 6)], extra_opciones=["tune_largo"],
        v=dict(m_A=float(pm_.loc["A", "m"]), sd_m_A=float(pm_.loc["A", "sd"]),
               m_B=float(pm_.loc["B", "m"]), sd_m_B=float(pm_.loc["B", "sd"])),
        motivo="Primera ronda: R̂ = 1.14 en A_m, A_r, B_r (ESS ≈ 20) con la geometría bien determinada; las cadenas "
               "recorren la degeneración masa–radio (a ∝ M^1/3) y las divergencias se concentran en A_m baja. "
               "Prior original m_A = 10.5 ± 3.1 M☉."),
}

filas = []
for (tic, s), r in R2.items():
    caso = dict(r["base"])
    caso["opciones"] = list(caso["opciones"]) + r.get("extra_opciones", [])
    caso["v"] = {**caso["v"], **r.get("v", {})}
    if "amp_B_min" in r:
        lo, d2 = r["amp_B_min"]
        cambios = (f"amp_B acotada a ≥ D2 − 2σ = {lo:.4f} (D2 = {d2:.4f}); se mantienen las correcciones de la primera "
                   f"ronda: {caso['cambios']}")
    else:
        cambios = (f"priors de masa del interpolador de isócronas (MIST, R→M con los radios de S32/S98, edad 8–20 Myr): "
                   f"m_A = {caso['v']['m_A']:.2f} ± {caso['v']['sd_m_A']:.2f} M☉ (antes 10.48 ± 3.15), "
                   f"m_B = {caso['v']['m_B']:.2f} ± {caso['v']['sd_m_B']:.2f} M☉ (antes 3.96 ± 1.29); tune 3000; "
                   f"resto como en la primera ronda: {caso['cambios']}")
    caso.update(motivo=r["motivo"], cambios=cambios)
    nb, opciones = notebook_corregido(tic, s, caso)
    if "amp_B_min" in r:
        n = 0
        for c in nb.cells:
            if LINEA_BAMP in c.source:
                c.source = c.source.replace(LINEA_BAMP,
                    f'B_amp = pm.Bound(pm.Normal, lower={lo})("B_amp", mu=amp_B, sd=sd_amp_B, '
                    f'testval=max(amp_B, 1.05 * {lo}))   # amp_B ≥ D2 − 2σ (ronda 2)'); n += 1
        assert n == 1
    nb.cells[1].source = nb.cells[1].source.replace("## Corrida corregida", "## Corrida corregida, ronda 2") \
        .replace(f"starry_correcciones/TIC_{tic}", f"starry_correcciones/ronda2/TIC_{tic}")
    ruta = AQUI / f"TIC_{tic}" / f"full_{tic}_s{s}_r2.ipynb"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, ruta)
    filas.append(dict(TIC=tic, Sector=s, notebook=str(ruta.relative_to(AQUI)), cambios=cambios))
pd.DataFrame(filas).to_csv(AQUI / "indice_ronda2.csv", index=False)
print(pd.DataFrame(filas)[["TIC", "Sector", "notebook"]].to_string(index=False))
