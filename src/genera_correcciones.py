"""Corridas de starry corregidas, caso por caso, para los TIC/sectores que no convergieron o dieron
resultados inconclusos en starry_por_tic/ (ver starry_por_tic/revision_corridas.csv).

Usa la plantilla y los priors de ../starry_por_tic/genera_starry.py (sin ejecutar su bucle principal) y
aplica encima las correcciones de cada caso. No toca nada de starry_por_tic/.
Uso: python genera_correcciones.py
"""
import re
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
GEN = AQUI.parent / "starry_por_tic" / "genera_starry.py"

# definiciones del generador original (priors_de, CELDAS, crear, ...) sin su bucle principal
fuente_gen = GEN.read_text()
exec(compile(fuente_gen[:fuente_gen.index("filas_indice = []")], str(GEN), "exec"))

# ---------------------------------------------------------------------------
# correcciones generales (se aplican a todos los casos)
# ---------------------------------------------------------------------------
GENERALES = [
    ("amplitudes positivas",
     'A_amp = pm.Normal("A_amp", mu=amp_A, sd=sd_amp_A, testval=amp_A)',
     'A_amp = PositiveNormal("A_amp", mu=amp_A, sd=sd_amp_A, testval=amp_A)'),
    ("amplitudes positivas",
     'B_amp = pm.Normal("B_amp", mu=amp_B, sd=sd_amp_B, testval=amp_B)',
     'B_amp = PositiveNormal("B_amp", mu=amp_B, sd=sd_amp_B, testval=amp_B)'),
    ("4 núcleos por corrida (2 corridas en paralelo)", "CHAINS, CORES = 4, 5", "CHAINS, CORES = 4, 4"),
    ("muestreo nuevo (no cargar traces viejos)", "CARGAR_TRACE_EXISTENTE = True", "CARGAR_TRACE_EXISTENTE = False"),
]

# correcciones opcionales: nombre -> lista de (buscar, reemplazar)
OPCIONES = {
    "target_accept": [("TARGET_ACCEPT = 0.9\n", "TARGET_ACCEPT = 0.95\n")],
    "tune_largo": [("TUNE, DRAWS = 1500, 1500", "TUNE, DRAWS = 3000, 1500")],
    "inc_hasta_90": [('B_inc = pm.Normal("B_inc", mu=inc_B, sd=sd_inc_B, testval=inc_B)',
                      'B_inc = pm.Bound(pm.Normal, upper=90.0)("B_inc", mu=inc_B, sd=sd_inc_B, testval=min(inc_B, 89.5))')],
    "u_B_fijo": [('B_u = xo.distributions.QuadLimbDark("u_B", testval=np.array(B["u"]))',
                  'B_u = pm.Deterministic("u_B", tt.as_tensor_variable(np.array(B["u"], dtype=float)))  # fijo')],
}

# ---------------------------------------------------------------------------
# casos: (TIC, sector) -> correcciones particulares
#   v:        valores que reemplazan a los de priors_de (priors y puntos de partida)
#   opciones: claves de OPCIONES
#   u_B:      coeficientes de limb darkening fijos de la secundaria (con "u_B_fijo")
#   motivo / cambios: texto para el encabezado del notebook
# ---------------------------------------------------------------------------
AMP_ANCHO = "amp_B_ancho"   # sd_amp_B = max(0.5 amp_B, 0.01): el secundario es parcial, amp_B > D2

CASOS = {
    # --- modo excéntrico equivocado: B_amp < 0 borraba el secundario ---
    (427353034, 6): dict(motivo="Modo excéntrico equivocado: el modelo puso el secundario en fase 0.70 (datos: 0.50) "
                                "usando una amplitud de B negativa (B_amp = −0.007, 11 sd por debajo de su prior).",
                         v=dict(r_A=1.08, r_B=0.621, ecc=0.001, w=90.0),
                         opciones=["target_accept", AMP_ANCHO],
                         cambios="amplitudes > 0; prior de amp_B ancho; arranque desde la solución de S32 "
                                 "(R_A = 1.08, R_B = 0.62, órbita circular)."),
    (427353034, 98): dict(motivo="Modo excéntrico equivocado: secundario del modelo en fase 0.43 (datos: 0.50), "
                                 "B_amp = −0.008.",
                          v=dict(r_A=1.08, r_B=0.621, ecc=0.001, w=90.0),
                          opciones=["target_accept", AMP_ANCHO],
                          cambios="amplitudes > 0; prior de amp_B ancho; arranque desde la solución de S32."),
    # --- 50897998 ---
    (50897998, 6): dict(motivo="e = 0.14 e i = 86.9° frente a e ≈ 0 e i ≈ 78° en S32/S98; corrida con una versión "
                               "anterior del generador.",
                        v=dict(r_A=4.98, r_B=2.33, ecc=0.001, w=90.0),
                        opciones=["target_accept"],
                        cambios="versión actual del generador (detrending orbital); amplitudes > 0; arranque desde "
                                "la solución de S32 (circular)."),
    (50897998, 98): dict(motivo="No convergió: R̂ = 1.17, ESS = 17, 1143 divergencias.",
                         v=dict(r_A=4.98, r_B=2.33, ecc=0.001, w=90.0),
                         opciones=["target_accept", "tune_largo"],
                         cambios="target_accept 0.95; tune 3000; amplitudes > 0; arranque desde S32."),
    # --- 276662451: secundario parcial, amp_B tirado +3-4 sd ---
    (276662451, 6): dict(motivo="El secundario del modelo queda más somero que los datos; B_amp se va 2.7 sd por "
                                "encima de su prior (secundario parcial: amp_B > D2).",
                         v=dict(r_A=1.68, r_B=0.947, ecc=0.247, w=66.0),
                         opciones=["target_accept", AMP_ANCHO],
                         cambios="prior de amp_B ancho; amplitudes > 0; arranque desde la solución excéntrica."),
    (276662451, 32): dict(motivo="Secundario del modelo 13 ppt más somero que los datos (−17σ); B_amp +2.6 sd.",
                          v=dict(r_A=1.68, r_B=0.947, ecc=0.257, w=67.0),
                          opciones=["target_accept", AMP_ANCHO],
                          cambios="prior de amp_B ancho; amplitudes > 0; arranque desde la solución excéntrica."),
    (276662451, 98): dict(motivo="No convergió: R̂ = 1.76, ESS = 6, 1770 divergencias; B_amp +4.1 sd.",
                          v=dict(r_A=1.68, r_B=0.947, ecc=0.257, w=67.0),
                          opciones=["target_accept", "tune_largo", AMP_ANCHO],
                          cambios="target_accept 0.95; tune 3000; prior de amp_B ancho; arranque desde S6/S32."),
    # --- 427347964 S98: una cadena en i > 90 ---
    (427347964, 98): dict(motivo="No convergió: R̂ = 1.54, ESS = 7; una cadena quedó en el modo espejo i > 90° "
                                 "(B_inc = 97°).",
                          v=dict(r_A=1.158, r_B=0.373, ecc=0.001, w=90.0),
                          opciones=["target_accept", "tune_largo", "inc_hasta_90"],
                          cambios="inclinación acotada a ≤ 90°; target_accept 0.95; tune 3000; arranque desde S32."),
    # --- 176775425 S98: u_B sin restringir ---
    (176775425, 98): dict(motivo="No convergió: R̂ = 1.26 en u_B[0], ESS = 12, 787 divergencias (limb darkening de "
                                 "la secundaria sin restringir).",
                          v=dict(r_A=0.555, r_B=0.282),
                          opciones=["target_accept", "u_B_fijo"], u_B=[0.246, 0.127],
                          cambios="u_B fijo en el posterior de S87 (0.246, 0.127); target_accept 0.95."),
    # --- 427347825: eclipses de 1-2 %, no converge e inconsistente ---
    **{(427347825, s): dict(motivo="No convergió (S6: R̂ 1.17; S32: R̂ 1.49, 1698 divergencias) e inconsistente entre "
                                   "sectores; eclipses de 1–2 % y B_amp < 0 en S98.",
                            v=dict(),
                            opciones=["target_accept", "tune_largo", "inc_hasta_90", "u_B_fijo", AMP_ANCHO],
                            u_B=[0.3, 0.2],
                            cambios="amplitudes > 0; prior de amp_B ancho; u_B fijo (0.3, 0.2): con un secundario "
                                    "de 0.3 % no se puede medir; inclinación ≤ 90°; target_accept 0.95; tune 3000.")
       for s in (6, 32, 98)},
    # --- 427378979: inconsistente entre sectores, residuo −12σ en el secundario ---
    **{(427378979, s): dict(motivo="k y e muy distintos entre sectores (k 0.42/0.27/0.23; e 0.27/0.10/0.01); "
                                   "residuos de −9 a −12σ; corridas con una versión anterior del generador.",
                            v=dict(ecc=0.001, w=90.0),
                            opciones=["target_accept", AMP_ANCHO],
                            cambios="versión actual del generador; amplitudes > 0; prior de amp_B ancho; arranque "
                                    "circular (el secundario está en fase 0.50 en los tres sectores).")
       for s in (6, 32, 98)},
    # --- 427395536 ---
    (427395536, 6): dict(motivo="k = 0.20 en S6 frente a 0.39 en S98; B_amp +6.7 sd por encima del prior.",
                         v=dict(), opciones=["target_accept", "inc_hasta_90", AMP_ANCHO],
                         cambios="prior de amp_B ancho; amplitudes > 0; inclinación ≤ 90°."),
    (427395536, 98): dict(motivo="Inclinación en el modo espejo (B_inc = 101°); B_amp +3.7 sd.",
                          v=dict(), opciones=["target_accept", "inc_hasta_90", AMP_ANCHO],
                          cambios="inclinación ≤ 90°; prior de amp_B ancho; amplitudes > 0."),
    (427395536, 32): dict(motivo="No convergió (R̂ 3.24) y el periodo del sector era erróneo (BLS encontró 0.734 d; "
                                 "el sistema tiene 4.674 d).",
                          priors_de_sector=98, periodo=4.673937,
                          v=dict(), opciones=["target_accept", "inc_hasta_90", AMP_ANCHO],
                          cambios="priors de S98 con P = 4.673937 d y t0 propagado desde S98; inclinación ≤ 90°; "
                                  "prior de amp_B ancho; amplitudes > 0."),
    # --- 249074181: inconsistente; S32 sin terminar ---
    **{(249074181, s): dict(motivo="k y e distintos entre sectores (k 0.35/0.29/0.42; e 0.04/0.01/0.17); S32 sin "
                                   "terminar.",
                            v=dict(ecc=0.001, w=90.0),
                            opciones=["target_accept", AMP_ANCHO],
                            cambios="amplitudes > 0; prior de amp_B ancho; arranque circular (secundario en fase 0.50).")
       for s in (5, 32, 98)},
}


def notebook_corregido(tic, sector, caso):
    s_pri = caso.get("priors_de_sector", sector)
    v, avisos, fuente = priors_de(tic, s_pri)
    p = priors.loc[(priors["TIC"] == tic) & (priors["Sector"] == s_pri)].iloc[0]
    v["prot"] = p["P_rot"]
    file_lc = RUTA_ORION / "Light_Curve_Files" / f"tic {tic}_S_{sector}_LC.txt"
    if "periodo" in caso:
        # efeméride de otro sector propagada a este: t0 = t0_ref + n P, con el eclipse más cercano al sector
        t = pd.read_csv(file_lc, sep=r"\s+", usecols=["#Time"])["#Time"].dropna().to_numpy()
        n = np.round((np.median(t) - v["t0"]) / caso["periodo"])
        v["porb"], v["t0"] = caso["periodo"], v["t0"] + n * caso["periodo"]
        avisos = [a for a in avisos if not a.startswith("P_orb de este sector")]
    v.update(caso["v"])
    opciones = list(caso["opciones"])
    if AMP_ANCHO in opciones:
        v["sd_amp_B"] = max(0.5 * v["amp_B"], 0.01)
        v["sd_amp_A"] = v["sd_amp_B"]
        fuente["amp"] += f". **Corrección:** sd_amp_B = max(0.5 amp_B, 0.01) = {v['sd_amp_B']:.4f}"
    t = pd.read_csv(file_lc, sep=r"\s+", usecols=["#Time"], nrows=50)["#Time"].to_numpy()
    cadencia = np.median(np.diff(t)) * 1440
    out_dir = AQUI / f"TIC_{tic}" / f"S{sector}"
    nb = crear(tic, sector, v, avisos, fuente, file_lc, n_puntos(file_lc), cadencia, out_dir)

    reemplazos = [(n, a, b) for n, a, b in GENERALES]
    for o in opciones:
        reemplazos += [(o, a, b) for a, b in OPCIONES.get(o, [])]
    if "u_B_fijo" in opciones:
        reemplazos.append(("u_B fijo", "u_B = [0.3, 0.2]", f"u_B = {list(caso['u_B'])}   # fijo (corrección)"))
    hechos = set()
    for c in nb.cells:
        for nombre, a, b in reemplazos:
            if a in c.source:
                c.source = c.source.replace(a, b)
                hechos.add((nombre, a))
    faltan = [(n, a) for n, a, _ in reemplazos if (n, a) not in hechos]
    assert not faltan, f"TIC {tic} S{sector}: no se encontró {faltan}"

    encabezado = (f"## Corrida corregida\n\n"
                  f"**Problema en `starry_por_tic/`:** {caso['motivo']}\n\n"
                  f"**Correcciones:** {caso['cambios']}\n\n"
                  f"Correcciones aplicadas al código: "
                  + ", ".join(sorted({n for n, _ in hechos})) + ".\n\n"
                  f"Las salidas van a `starry_correcciones/TIC_{tic}/S{sector}/`; las de la corrida original "
                  f"quedan intactas en `starry_por_tic/`.")
    nb.cells.insert(1, nbf.v4.new_markdown_cell(encabezado))
    return nb, opciones


filas = []
for (tic, sector), caso in sorted(CASOS.items()):
    nb, opciones = notebook_corregido(tic, sector, caso)
    ruta = AQUI / f"TIC_{tic}" / f"full_{tic}_s{sector}_corr.ipynb"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, ruta)
    filas.append(dict(TIC=tic, Sector=sector, notebook=str(ruta.relative_to(AQUI)), motivo=caso["motivo"],
                      correcciones=caso["cambios"], opciones=", ".join(opciones)))
ind = pd.DataFrame(filas)
ind.to_csv(AQUI / "indice_correcciones.csv", index=False)
print(f"{len(ind)} notebooks en {AQUI}")
print(ind[["TIC", "Sector", "opciones"]].to_string(index=False))
