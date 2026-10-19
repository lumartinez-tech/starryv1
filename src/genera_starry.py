"""Genera un notebook de starry por (TIC, sector) a partir de los resultados de
4_pipeline_completo.ipynb. Uso: python genera_starry.py [--test]"""
import re
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

RUTA_ORION = Path("/home/luciam/Documents/starry/Resultados_Orion")
RES = RUTA_ORION / "resultados_pipeline"
DESTINO = RUTA_ORION / "starry_por_tic"
TEST = "--test" in sys.argv

priors = pd.read_csv(RUTA_ORION / "priors_v2.csv")   # profundidades v2 (1b_profundidades_eclipse_v2.ipynb)
cat = pd.read_csv(RUTA_ORION / "catalogo_completo_mist_sommers.csv")
resumen = pd.read_csv(RES / "resumen_por_tic_sector.csv")
largo = pd.read_csv(RES / "secundarias_por_sector_largo.csv")
comb = pd.read_csv(RES / "resumen_por_tic_combinado.csv")

# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
H, C, KB = 6.62607e-34, 2.99792e8, 1.380649e-23
LAMBDA_TESS = 786.5e-9  # m, longitud de onda efectiva de TESS


def planck(T):
    return 1.0 / np.expm1(H * C / (LAMBDA_TESS * KB * T))


def fmt(x, n=6):
    return f"{float(x):.{n}g}"


def n_puntos(path):
    with open(path) as fh:
        return sum(1 for _ in fh) - 1


def priors_de(tic, sector):
    """Valores de los priors para starry y lista de avisos/procedencia."""
    avisos, fuente = [], {}
    p = priors.loc[(priors["TIC"] == tic) & (priors["Sector"] == sector)].iloc[0]
    r = resumen.loc[(resumen["TIC"] == tic) & (resumen["Sector"] == sector)].iloc[0]
    c = cat.loc[cat["TIC"] == tic].iloc[0]
    rama = r["modelo_A"]
    suf = {"MIST": "1", "Somers17": "2"}[rama]

    # -------- primaria --------
    r_A, sd_r_A = r["R_A"], r["R_A_err"]
    m_A, sd_m_A = c[f"mass_median_{suf}"], c[f"mass_std_{suf}"]
    sd_m_A = max(sd_m_A, 0.1 * m_A) if np.isfinite(sd_m_A) else 0.2 * m_A
    fuente["A"] = (f"rama {rama} del ajuste massage: R_A = Stefan-Boltzmann(T_median_{suf}, lgL_median_{suf}), "
                   f"M_A = mass_median_{suf} (sd = mass_std_{suf}, mínimo 10 %)")
    if sd_r_A / r_A > 0.5:
        avisos.append(f"sd_r_A es {sd_r_A / r_A:.0%} de R_A (Teff/L de la primaria mal acotados); el prior de R_A es débil.")

    # -------- secundaria: radio --------
    D, sD = p["Depth_primary"], p["Depth_primary_err"]
    r_B = r_A * np.sqrt(D)
    sd_r_B = r_B * np.hypot(sd_r_A / r_A, 0.5 * sD / D)
    k, sd_k = np.sqrt(D), 0.5 * sD / np.sqrt(D)

    # -------- secundaria: masa (BHAC15 -> malla propia -> M~R) --------
    m_B, sd_m_B = r.get("M_B", np.nan), r.get("M_B_err", np.nan)
    if np.isfinite(m_B):
        fuente["m_B"] = f"BHAC15 a la edad de {r['modelo_B']} (M2 que reproduce R2 en la isocrona)"
    else:
        s = largo.loc[(largo["TIC"] == tic) & (largo["Sector"] == sector) & (largo["rama"] == rama)
                      & (largo["malla"] == rama)]
        if len(s) and np.isfinite(s["M2"].iloc[0]):
            m_B, sd_m_B = s["M2"].iloc[0], s["M2_err"].iloc[0]
            fuente["m_B"] = f"malla {rama} (R2 fuera de BHAC15)"
            avisos.append(f"M_B no sale de BHAC15; se usa la malla {rama}.")
        else:
            m_B, sd_m_B = r_B, 0.5 * r_B
            fuente["m_B"] = "aproximación M ~ R (R2 fuera de todas las mallas), sd 50 %"
            avisos.append("R_B cae fuera de todas las mallas: m_B = R_B (aprox. M~R) con sd 50 %. Revisar a mano.")
    sd_m_B = max(sd_m_B, 0.1 * m_B) if np.isfinite(sd_m_B) else 0.3 * m_B

    # -------- amplitudes (fracción de luz en banda TESS) --------
    D2, sD2 = p["Depth_secondary"], p["Depth_secondary_err"]
    T1 = r["Teff_A"]
    T2 = r.get("Teff_B", np.nan)
    amp_bb = np.nan
    if np.isfinite(T2):
        fB = r_B ** 2 * planck(T2)
        amp_bb = fB / (fB + r_A ** 2 * planck(T1))
    if bool(p["secundario_detectado"]):
        amp_B, sd_amp_B = D2, max(sD2, 0.1 * D2, 0.001)
        fuente["amp"] = ("amp_B = Depth_secondary v2 (eclipse secundario detectado a "
                         f"{p['signif_emp_secundario']:.0f} sigma; en un eclipse secundario total la profundidad es "
                         "la fracción de luz de B), sd = max(error, 10 %)")
        if np.isfinite(amp_bb):
            fuente["amp"] += f"; estimación de cuerpo negro en banda TESS: {amp_bb:.4f}"
    elif np.isfinite(amp_bb):
        amp_B, sd_amp_B = amp_bb, max(0.5 * amp_bb, 0.001)
        fuente["amp"] = ("secundario no detectado: amp_B de cuerpo negro a 786 nm con "
                         "(R_A, Teff_A, R_B, Teff_B), sd 50 %")
    else:
        amp_B, sd_amp_B = 0.002, 0.001
        fuente["amp"] = "sin Depth_secondary significativo ni Teff_B: amp_B = 0.002 (valor del ejemplo)"
        avisos.append("amp_B sin información (ni eclipse secundario ni Teff_B): se usa 0.002 ± 0.001.")
    amp_B = float(np.clip(amp_B, 1e-4, 0.5))
    amp_A, sd_amp_A = 1 - amp_B, sd_amp_B

    # -------- excentricidad: e cos w del desfase del secundario (1b, sección 2.5) --------
    ecosw = p["ecosw"]
    if np.isfinite(ecosw) and abs(ecosw) > 0.03:
        ecc, w = abs(ecosw), (0.0 if ecosw > 0 else 180.0)
        fuente["ecc"] = (f"prior no informativo (uniforme en e y en w). Solo el valor inicial sale del "
                         f"secundario en fase {p['fase_secundario']:.4f}: e cos w = (pi/2)(fase - 0.5) = "
                         f"{ecosw:.3f} -> e0 = |e cos w|, w0 = {w:.0f} deg (e sin w = 0 al inicio)")
        avisos.append(f"Órbita excéntrica: secundario en fase {p['fase_secundario']:.3f} (e cos w = {ecosw:.3f}).")
    else:
        ecc, w = 0.001, 90.0
        fuente["ecc"] = ("prior no informativo (uniforme en e y en w). Secundario en fase ~0.5 "
                         "(|e cos w| < 0.03) o no detectado: se empieza casi circular (e0 = 0.001), "
                         "w poco restringido")

    # -------- calidad de la profundidad (1b) --------
    cal = str(p["Calidad_depth"])
    if "eclipse_ancho" in cal:
        avisos.append("BLS no encontró un eclipse separado (ancho > 0.25 en fase): posible sistema de contacto, "
                      "variable por manchas o falso positivo. **No ajustar con starry** sin confirmar antes que es una "
                      "binaria eclipsante: el modelo de dos esferas no describe una modulación continua.")
    if "P_corregido" in cal:
        avisos.append(f"P_orb corregido a la mitad ({p['P_orb_v1']:.6f} -> {p['P_orb']:.6f} d): "
                      "los eclipses en fase 0 y 0.5 tenían la misma profundidad.")
    if "texp_largo" in cal:
        avisos.append(f"Cadencia larga frente a la duración (t_exp/duración = {p['texp_sobre_duracion']:.2f}): "
                      "TEXP está activado para integrar el modelo durante la exposición.")

    # -------- órbita --------
    porb, t0, dur = p["P_orb"], p["t0"], p["Duration"]
    # duración robusta: si la de BLS en este sector se aleja >40 % de la mediana del sistema, se usa la mediana
    g_tic = priors.loc[(priors["TIC"] == tic) & ~priors["Calidad_depth"].fillna("").str.contains("eclipse_ancho")]
    dur_med = g_tic["Duration"].median() if len(g_tic) else dur
    if np.isfinite(dur_med) and abs(dur / dur_med - 1) > 0.4:
        avisos.append(f"La duración de BLS en este sector ({dur:.3f} d) difiere de la mediana del sistema "
                      f"({dur_med:.3f} d): se usa la mediana para la máscara de eclipses y el prior de t0.")
        dur = dur_med
    modo = "orbital" if (np.isfinite(T1) and T1 > 6500) else "mascara"
    if modo == "orbital":
        avisos.append(f"Primaria caliente (Teff = {T1:.0f} K, envolvente radiativa): la modulación fuera de eclipse "
                      "es casi seguro elipsoidal/reflexión, no manchas. Se usa DETRENDING = 'orbital'.")
    otros = priors.loc[(priors["TIC"] == tic) & (priors["Sector"] != sector), ["Sector", "P_orb"]]
    distintos = otros.loc[(otros["P_orb"] / porb - 1).abs() > 0.02]
    if len(distintos):
        avisos.append("P_orb de este sector no coincide con otros sectores: "
                      + ", ".join(f"S{int(s)}: {q:.5f} d" for s, q in distintos.values)
                      + f" (aquí {porb:.5f} d). Revisar si es P, 2P o P/2.")
    fila_c = comb.loc[comb["TIC"] == tic]
    if len(fila_c) and bool(fila_c["inconsistente"].iloc[0]):
        otros_d = priors.loc[priors["TIC"] == tic, ["Sector", "Depth_primary"]].dropna()
        avisos.append("La profundidad primaria cambia entre sectores más de lo que explican sus errores "
                      f"(chi2_red = {fila_c['chi2_red'].iloc[0]:.0f}: "
                      + ", ".join(f"S{int(a)}: {b:.3f}" for a, b in otros_d.values)
                      + "). Suele ser dilución/contaminación distinta por sector o suavizado por cadencia.")

    v = dict(
        r_A=r_A, sd_r_A=sd_r_A, m_A=m_A, sd_m_A=sd_m_A, amp_A=amp_A, sd_amp_A=sd_amp_A,
        r_B=r_B, sd_r_B=sd_r_B, k=k, sd_k=sd_k, m_B=m_B, sd_m_B=sd_m_B, amp_B=amp_B, sd_amp_B=sd_amp_B,
        porb=porb, t0=t0, dur=dur, D=D, sD=sD, D2=D2, T1=T1, T2=T2, rama=rama, modo=modo,
        ecc=ecc, w=w, D_v1=p["Depth_primary_v1"],
        sesgo=p["sesgo_max_primario"], texp_ratio=p["texp_sobre_duracion"],
        fase2=p["fase_secundario"] if bool(p["secundario_detectado"]) else 0.5,
        ecosw_1b=p["ecosw"], sec_det=bool(p["secundario_detectado"]),
        edad=r["edad_A_Myr"],
    )
    return v, avisos, fuente


# ---------------------------------------------------------------------------
# plantilla del notebook (tokens {{KEY}})
# ---------------------------------------------------------------------------
CELDAS = []
md = lambda s: CELDAS.append(("markdown", s.strip("\n")))
code = lambda s: CELDAS.append(("code", s.strip("\n")))

md(r"""
# starry: TIC {{TIC}}, sector {{SECTOR}}

Ajuste de binaria eclipsante con starry + PyMC3. Los priors salen de
`4_pipeline_completo.ipynb` (carpeta `resultados_pipeline/`) y de `priors_v2.csv`
(profundidades medidas en el fondo del eclipse, ver `1b_profundidades_eclipse_v2.ipynb`). Generado automáticamente; los valores de los priors están
escritos en la sección **Priors** y se pueden editar a mano.

- Archivo: `{{ARCHIVO}}` ({{NPUNTOS}} puntos, cadencia ~{{CADENCIA}} min)
- Primaria: rama **{{RAMA}}**, edad {{EDAD}} Myr

{{AVISOS}}

Todas las salidas se guardan en `{{OUT_REL}}/`. El trace se guarda en `.nc`, así que
para rehacer figuras sin volver a muestrear basta con `CARGAR_TRACE_EXISTENTE = True`.
""")

code(r"""
# =========================
# Básicas
# =========================
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from IPython.display import HTML, display

# =========================
# Light curves
# =========================
from wotan import flatten
from astropy.timeseries import LombScargle

# =========================
# Modelado
# =========================
import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as xo
import starry
import theano
import theano.tensor as tt
import arviz as az

# =========================
# Utilidades
# =========================
from scipy.stats import median_abs_deviation
import corner
""")

code(r"""
# =========================
# Configuración
# =========================
TIC, SECTOR = {{TIC}}, {{SECTOR}}
Label = f"TIC_{TIC}_S{SECTOR}"

FILE_LC = Path("{{FILE_LC}}")
OUT_DIR = Path("{{OUT_DIR}}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "Figures"
FIG_DIR.mkdir(exist_ok=True)

# Grado de los mapas. Con 0 las estrellas son discos uniformes con limb darkening: es el mismo
# modelo que ydeg=5 con todos los coeficientes y en cero (como en el ejemplo, donde nunca se
# ajustan), pero ~70 veces más rápido. Usar > 0 solo si se van a muestrear los coeficientes y (manchas).
YDEG = 0

# Detrending (ver sección 1):
#   "orbital": tendencia = modulación orbital (serie de Fourier en la fase) x tendencia lenta (biweight
#              de WINDOW_LENTA días). Para modulación ligada a la órbita: elipsoidal, reflexión o manchas
#              en una estrella sincronizada. Recomendado para primarias calientes (sin manchas).
#   "mascara": biweight de WINDOW_WOTAN días calculado solo fuera de eclipses e interpolado sobre ellos.
#   "simple":  biweight sobre toda la curva (como el notebook de ejemplo).
DETRENDING = "{{MODO}}"
WINDOW_WOTAN = 0.55     # días, ventana de wotan para "mascara" y "simple" (igual que el ejemplo)
N_ARMONICOS = 2         # armónicos de la serie de Fourier orbital (2 = reflexión + elipsoidal)
WINDOW_LENTA = 2.0      # días, ventana de la tendencia lenta en "orbital"
REFINAR_EFEMERIDE = True   # refina P_orb y t0 con BLS en ±0.5 % del periodo del catálogo
BIN_MIN = {{BIN_MIN}}          # None = sin binear; p. ej. 10 -> bins de 10 min (acelera mucho S87/S98)

# muestreo
TUNE, DRAWS = {{TUNE}}, {{DRAWS}}
CHAINS, CORES = {{CHAINS}}, {{CORES}}
TARGET_ACCEPT = 0.9
CARGAR_TRACE_EXISTENTE = True   # si ya existe el .nc de este TIC/sector, se carga en vez de muestrear
N_CURVAS = 300                  # muestras del posterior para las bandas 16-84 % del modelo

# ¿B_r libre (como el ejemplo) o B_r = A_r * k con k ~ Normal(sqrt(Depth_primary), sigma)?
# La segunda opción usa directamente Depth_primary = (R_B/R_A)^2 y no cuenta dos veces el error de R_A.
RADIO_B_COMO_RAZON = False

# Prior de los radios:
#   "geometria" (recomendado): log-uniforme en [0.05, 20] Rsun para R_A y [0.01, 20] para R_B. Los radios
#       quedan fijados por la geometría de los eclipses (R/a) y la escala a de Kepler con las masas, sin
#       arrastrar errores sistemáticos de Teff/L (massage) hacia la inclinación ni la excentricidad.
#   "stefan_boltzmann": normal centrada en R de Teff/L (como antes).
PRIOR_RADIOS = "geometria"

# Tiempo de integración de cada punto (días). Con cadencia >= 10 min los eclipses se "estiran" y eso
# sesga las duraciones, que es de donde sale e sin w. starry integra el modelo durante la exposición.
# Cadencia de este sector: {{CADENCIA}} min.
TEXP = {{TEXP_DEFAULT}}

HACER_ANIMACION = True   # requiere ffmpeg
TRACE_PATH = OUT_DIR / f"{Label}_trace.nc"

np.random.seed(12)
starry.config.lazy = False
starry.config.quiet = True
""")

md(r"""
## 1) Datos y detrending

**Efeméride.** El $P_{\rm orb}$ del catálogo sale de una búsqueda BLS gruesa y puede tener un error de
~$10^{-3}$ d. En un sector de 27 días eso corre los eclipses ~0.05 en fase, lo suficiente para que la
máscara de eclipses deje escapar sus bordes. Con `REFINAR_EFEMERIDE`, BLS se repite en
$P_{\rm cat}(1\pm0.005)$ con 4000 periodos y la duración del catálogo, y el $t_0$ se toma del mismo
eclipse primario. $P_{\rm REF}$ y $t_0$ refinados son también los centros de los priors.

**Detrending.** Hay dos tipos de variación fuera de eclipse:

- **Ligada a la órbita:** la **variación elipsoidal** (la estrella deformada por la marea muestra más
  área en las cuadraturas, $\propto\cos4\pi\phi$, con mínimos en los eclipses), la **reflexión**
  ($\propto\cos2\pi\phi$) y las manchas de una estrella sincronizada. Se repite exactamente con
  $P_{\rm orb}$.
- **No ligada:** sistemáticos del instrumento y manchas que evolucionan. Es lenta, de días.

Con `DETRENDING = "orbital"` la tendencia se modela como producto de las dos:

$$T(t)=S(t)\,H(\phi),\qquad H(\phi)=c_0+\sum_{k=1}^{N}\left[a_k\cos2\pi k\phi+b_k\sin2\pi k\phi\right],$$

con $H$ ajustada por mínimos cuadrados a los puntos fuera de eclipse ($|\phi|,|\phi-\phi_2|>0.6\,W$), y $S$
un biweight de `WINDOW_LENTA` = 2 d sobre $f/H$. Las dos se alternan 3 veces. Como $H$ es una función
periódica suave conocida en **todas** las fases, se evalúa bien *dentro* de los eclipses; una
interpolación, en cambio, solo une los puntos de los bordes.

Con `"mascara"`, un biweight de 0.55 d no puede seguir una señal de periodo $P_{\rm orb}/2$ (0.74 d en
TIC 50897998). Además, con eclipses anchos la máscara quita justo los mínimos de la variación elipsoidal
(59–68 % de la órbita en ese sistema). Por eso, en primarias calientes ($T_{\rm eff}>6500$ K,
envolvente radiativa, sin manchas) se usa `"orbital"` por defecto.

⚠️ Dividir entre $H$ también dentro del eclipse es una aproximación de primer orden: el modelo de starry
son esferas y no incluye la deformación. Por eso las profundidades aplanadas pueden diferir unos % de
las de `1b`.
""")

code(r"""
data = pd.read_csv(FILE_LC, sep=r"\s+").rename(columns={"#Time": "Time"})
data = data.dropna(subset=["Time", "Flux_CBV_SS"]).sort_values("Time").drop_duplicates("Time")

time = data["Time"].to_numpy(float)
flux = data["Flux_CBV_SS"].to_numpy(float)

P_CAT, T0_CAT, DUR_MASK, FASE2_MASK = {{porb}}, {{t0}}, {{dur}}, {{FASE2}}   # del catálogo (priors_v2)

# -------- 1a) efeméride: P_orb y t0 refinados con BLS en ±0.5 % del periodo del catálogo --------
P_REF, T0_REF = P_CAT, T0_CAT
if REFINAR_EFEMERIDE:
    from astropy.timeseries import BoxLeastSquares
    periodos = np.linspace(P_CAT * 0.995, P_CAT * 1.005, 4000)
    duraciones = DUR_MASK * np.array([0.6, 0.8, 1.0, 1.2])
    res_bls = BoxLeastSquares(time, flux / np.nanmedian(flux)).power(periodos, duraciones)
    j = int(np.argmax(res_bls.power))
    P_REF = float(res_bls.period[j])
    # t0 refinado: el eclipse de este BLS más cercano al t0 del catálogo (mismo eclipse primario)
    n = np.round((T0_CAT - res_bls.transit_time[j]) / P_REF)
    T0_REF = float(res_bls.transit_time[j] + n * P_REF)
    if abs(((T0_REF - T0_CAT) / P_REF + 0.5) % 1 - 0.5) > 0.25:   # BLS se enganchó al secundario
        T0_REF += 0.5 * P_REF
    print(f"P_orb: catálogo {P_CAT:.6f} -> refinado {P_REF:.6f} d;  t0: {T0_CAT:.5f} -> {T0_REF:.5f}")

# -------- 1b) detrending --------
W = min(DUR_MASK / P_REF, 0.4)
ph_mask = ((time - T0_REF) / P_REF) % 1
dist = lambda c: np.abs((ph_mask - c + 0.5) % 1 - 0.5)


def serie_fourier(ph, n):
    cols = [np.ones_like(ph)]
    for k in range(1, n + 1):
        cols += [np.cos(2 * np.pi * k * ph), np.sin(2 * np.pi * k * ph)]
    return np.column_stack(cols)


# protección: si la máscara deja < 20 % de los puntos (eclipse "ancho", W > 0.25 en fase), no hay con qué
# calcular la tendencia fuera de eclipse; se usa el biweight sobre toda la curva y se avisa
semiancho = {"orbital": 0.6 * W, "mascara": W}.get(DETRENDING, 0.0)
if DETRENDING != "simple" and ((dist(0.0) > semiancho) & (dist(FASE2_MASK) > semiancho)).mean() < 0.2:
    print(f"[!] Con W = {W:.2f} la máscara '{DETRENDING}' cubre casi toda la órbita: se usa DETRENDING = 'simple'. "
          "Revisa si este objeto es realmente una binaria eclipsante (aviso del encabezado).")
    DETRENDING = "simple"

if DETRENDING == "orbital":
    # máscara de semiancho 0.6 W: el eclipse (semiancho W/2) más un margen del 20 %
    fuera = (dist(0.0) > 0.6 * W) & (dist(FASE2_MASK) > 0.6 * W)
    lenta = np.ones_like(flux)
    for _ in range(3):   # iteración: modulación orbital <-> tendencia lenta
        c, *_ = np.linalg.lstsq(serie_fourier(ph_mask[fuera], N_ARMONICOS), (flux / lenta)[fuera], rcond=None)
        orbital = serie_fourier(ph_mask, N_ARMONICOS) @ c
        _, tr = flatten(time[fuera], (flux / orbital)[fuera], window_length=WINDOW_LENTA, break_tolerance=1.5,
                        method="biweight", return_trend=True)
        ok_tr = np.isfinite(tr)
        lenta = np.interp(time, time[fuera][ok_tr], tr[ok_tr])
    trend_flux = lenta * orbital
    amp = {k: c[2 * k - 1] / c[0] for k in range(1, N_ARMONICOS + 1)}
    print("Modulación orbital (amplitud relativa de cos 2πkφ): "
          + ", ".join(f"k={k}: {a:+.4f}" for k, a in amp.items())
          + "   (k=2 negativo = elipsoidal: mínimos en los eclipses)")
elif DETRENDING == "mascara":
    fuera = (dist(0.0) > W) & (dist(FASE2_MASK) > W)
    _, tr = flatten(time[fuera], flux[fuera], window_length=WINDOW_WOTAN, break_tolerance=1.5,
                    method="biweight", return_trend=True)
    ok_tr = np.isfinite(tr)
    trend_flux = np.interp(time, time[fuera][ok_tr], tr[ok_tr])
else:
    _, trend_flux = flatten(time, flux, window_length=WINDOW_WOTAN, break_tolerance=1.5,
                            method="biweight", return_trend=True)
flat_flux = flux / trend_flux
print(f"Detrending '{DETRENDING}': {100 * (1 - fuera.mean()) if DETRENDING != 'simple' else 0:.0f} % de los puntos enmascarados")
ok = np.isfinite(flat_flux)
time, flux, flat_flux, trend_flux = time[ok], flux[ok], flat_flux[ok], trend_flux[ok]

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax[0].scatter(time, flux, s=5, color="k", label="Original")
ax[0].plot(time, trend_flux, color="r", lw=2, label="Trend")
ax[0].set_ylabel("Flux"); ax[0].legend(); ax[0].set_title("Wotan detrending")
ax[1].scatter(time, flat_flux, s=4.5, color="b")
ax[1].set_xlabel("Time [days]"); ax[1].set_ylabel("Normalised flux")
plt.tight_layout(); plt.show()

pd.DataFrame({"time": time, "flat_flux": flat_flux, "trend_flux": trend_flux}).to_csv(
    OUT_DIR / f"{Label}_trending.csv", index=False)
""")

code(r"""
# =========================
# Datos para el ajuste (opcionalmente bineados)
# =========================
t_real, flux_real = time, flat_flux
if BIN_MIN:
    idx = np.floor((t_real - t_real[0]) / (BIN_MIN / 1440)).astype(int)
    b = pd.DataFrame({"t": t_real, "f": flux_real, "i": idx}).groupby("i").mean()
    t_real, flux_real = b["t"].to_numpy(), b["f"].to_numpy()

sigma_sd = 1.4826 * median_abs_deviation(flux_real)
print(f"{len(t_real)} puntos para el ajuste (de {len(time)}), sigma_sd = {sigma_sd:.2e}")

fig, ax = plt.subplots(1, figsize=(12, 5))
ax.plot(t_real, flux_real, ".k", alpha=0.6, ms=4)
ax.set_xlabel("time [days]", fontsize=16); ax.set_ylabel("normalized flux", fontsize=16);
""")

md(r"""
## 2) Priors

{{FUENTES}}
""")

code(r"""
# =========================
# Priors (generados desde resultados_pipeline/; editar aquí si hace falta)
# =========================
# Primaria
inc_A, sd_inc_A = 88.0, 1
amp_A, sd_amp_A = {{amp_A}}, {{sd_amp_A}}
r_A, sd_r_A = {{r_A}}, {{sd_r_A}}
m_A, sd_m_A = {{m_A}}, {{sd_m_A}}
u_A = [0.3, 0.2]

# Secundaria
inc_B, sd_inc_B = 88.0, 1
amp_B, sd_amp_B = {{amp_B}}, {{sd_amp_B}}
r_B, sd_r_B = {{r_B}}, {{sd_r_B}}        # R_A * sqrt(Depth_primary), Depth_primary (v2) = {{D}} ± {{sD}}
k_B, sd_k_B = {{k}}, {{sd_k}}            # R_B/R_A = sqrt(Depth_primary) (solo si RADIO_B_COMO_RAZON)
m_B, sd_m_B = {{m_B}}, {{sd_m_B}}
porb_B, sd_porb_B = P_REF, 0.01          # refinado en la sección 1 (catálogo: {{porb}} d)
t0_B, sd_t0_B = T0_REF, {{dur}}          # sd = duración del eclipse (catálogo t0: {{t0}})
# Excentricidad: SIN prior informativo. Se muestrea (sqrt(e) cos w, sqrt(e) sin w) uniforme en el
# disco unitario (sección 4). Estos dos números son solo el punto de partida (testval).
ecc_B, w_B = {{ecc}}, {{w}}
u_B = [0.3, 0.2]

prot_s = 0
""")

md("## 3) Modelo inicial")

code(r"""
A_y = np.zeros((YDEG + 1) ** 2 - 1)   # coeficientes y (sin y00): mapa uniforme
B_y = np.zeros((YDEG + 1) ** 2 - 1)

A = dict(ydeg=YDEG, udeg=2, inc=inc_A, amp=amp_A, r=r_A, m=m_A, prot=prot_s, u=u_A, y=A_y)
B = dict(ydeg=YDEG, udeg=2, inc=inc_B, amp=amp_B, r=r_B, m=m_B, prot=prot_s, porb=porb_B, t0=t0_B,
         ecc=ecc_B, w=w_B, u=u_B, y=B_y)

pri = starry.Primary(starry.Map(ydeg=A["ydeg"], udeg=A["udeg"], inc=A["inc"], amp=A["amp"]),
                     r=A["r"], m=A["m"], prot=A["prot"])
pri.map[1:] = A["u"]
if YDEG > 0:
    pri.map[1:, :] = A["y"]

sec = starry.Secondary(starry.Map(ydeg=B["ydeg"], udeg=B["udeg"], inc=B["inc"], amp=B["amp"]),
                       r=B["r"], m=B["m"], porb=B["porb"], prot=B["prot"], t0=B["t0"],
                       inc=B["inc"], ecc=B["ecc"], w=B["w"])
sec.map[1:] = B["u"]
if YDEG > 0:
    sec.map[1:, :] = B["y"]

sys = starry.System(pri, sec, texp=TEXP)
flux_true = sys.flux(t_real)

fig, ax = plt.subplots(1, figsize=(16, 7))
ax.plot(t_real, flux_real, "k.", alpha=0.5, ms=6, label="datos")
ax.plot(t_real, flux_true, lw=2, label="modelo con los priors")
ax.set_xlabel("time [days]", fontsize=16); ax.set_ylabel("normalized flux", fontsize=16); ax.legend();
""")

md(r"""
## 4) Modelo PyMC3

Diferencias con el notebook de ejemplo:

- Los mapas usan `ydeg = YDEG` (0 por defecto) en lugar de 5. Como los coeficientes y nunca se
  ajustan (están en cero), el modelo es el mismo (se verificó que el flujo coincide hasta
  ~10⁻¹⁵), pero cada evaluación de logp + gradiente es ~70 veces más rápida.
- Los coeficientes de limb darkening `u_A`, `u_B` (`QuadLimbDark`) ahora **sí** entran en el
  mapa.
- **Excentricidad.** No se muestrean $e$ y $\omega$ directamente, sino

  $$h=\sqrt e\cos\omega,\qquad k=\sqrt e\sin\omega,\qquad (h,k)\sim\mathcal U(\text{disco unitario}),$$

  y $e=h^2+k^2$, $\omega=\operatorname{atan2}(k,h)$ se guardan como `Deterministic`
  (Eastman, Gaudi & Agol 2013).

  *Prior implícito.* En polares $(r,\omega)$ con $r=\sqrt e$, el elemento de área es
  $dh\,dk=r\,dr\,d\omega$, y como $dr=de/(2\sqrt e)$ queda $dh\,dk=\tfrac12\,de\,d\omega$. Así que un
  prior uniforme en el disco equivale a $e\sim\mathcal U(0,1)$ y $\omega\sim\mathcal U(-\pi,\pi)$,
  independientes: **no favorece ninguna excentricidad**.

  *Por qué ayuda al muestreo.*
  1. En $(e,\omega)$, el punto $e=0$ es un borde donde $\omega$ no está definido. Las cadenas se
     amontonan contra él y aparecen divergencias (en las corridas anteriores: 99–725). En
     $(h,k)$, $e=0$ es el centro del disco, un punto interior sin nada especial.
  2. Los datos fijan muy bien $e\cos\omega$ (por el momento del secundario) y mal $e\sin\omega$ (por
     las duraciones). En $(e,\omega)$ eso es una curva angosta con correlación ≈ 1 entre $e$ y
     $\omega$. En $(h,k)$ es una franja casi recta, que la matriz de masa densa de NUTS sí puede
     adaptar.

  *De dónde sale la información.* $e\cos\omega$ viene del **momento** del secundario,
  $\Delta\phi\approx\tfrac{2}{\pi}e\cos\omega$, y queda muy preciso. $e\sin\omega$ viene del
  **cociente de duraciones**, $d_2/d_1\approx(1+e\sin\omega)/(1-e\sin\omega)$ (más un término por
  los parámetros de impacto en eclipses rasantes), y es menos preciso. Por eso importan dos
  cosas más:

  - **Radios con prior geométrico** (`PRIOR_RADIOS = "geometria"`). Con priors informativos de
    Stefan-Boltzmann, cualquier error sistemático en Teff/L (por ejemplo, la luz de la secundaria
    sumada a L en el ajuste massage) cambia la densidad $\propto M/R^3$ que el modelo "espera".
    Ese error puede compensarse con $e\sin\omega$ (efecto foto-excéntrico, Dawson & Johnson
    2012), porque $e\sin\omega$ también cambia la duración de los eclipses. Con radios
    log-uniformes, la escala del sistema queda fijada solo por la geometría de los eclipses
    más Kepler con las masas, y $e\sin\omega$ solo por lo que sí distingue a los dos eclipses.
    En la prueba con TIC 427395536 S6 el ESS de $e$ pasó de 553 a 4041. Esto mejora el muestreo
    y quita un sesgo, pero **no** vuelve robusto a $e\sin\omega$ (ver sección 12).
  - **Tiempo de integración** (`TEXP`). Con cadencia de 30 min cada punto promedia media hora de
    eclipse. Si no se modela, las dos duraciones se alargan casi lo mismo y el cociente se acerca
    a 1, lo que sesga $e\sin\omega$ hacia 0.

  *Sin doble conteo.* El $e\cos\omega$ medido en `1b` sale de **esta misma curva de luz**, así que
  ya está en la verosimilitud. Usarlo también como prior (como antes: $e\sim\mathcal N(|e\cos\omega|,\,\cdot)$,
  $\omega\sim\mathcal N(0°,60°)$) lo contaba dos veces y encima jalaba $\omega$ hacia 0°. Ahora solo
  se usa como **punto de partida** del optimizador y del muestreo. En el ejemplo se muestreaban pero el
mapa usaba los valores fijos `[0.3, 0.2]`, así que su posterior era solo el prior.
""")

code(r"""
starry.config.lazy = True
starry.config.quiet = True

with pm.Model() as model:
    PositiveNormal = pm.Bound(pm.Normal, lower=0.0)

    # Primaria
    A_inc = pm.Normal("A_inc", mu=inc_A, sd=sd_inc_A, testval=inc_A)
    A_amp = pm.Normal("A_amp", mu=amp_A, sd=sd_amp_A, testval=amp_A)
    if PRIOR_RADIOS == "geometria":
        A_logr = pm.Uniform("A_logr", np.log(0.05), np.log(20.0), testval=np.log(r_A))
        A_r = pm.Deterministic("A_r", tt.exp(A_logr))
    else:
        A_r = PositiveNormal("A_r", mu=r_A, sd=sd_r_A, testval=r_A)
    A_m = PositiveNormal("A_m", mu=m_A, sd=sd_m_A, testval=m_A)
    A_u = xo.distributions.QuadLimbDark("u_A", testval=np.array(A["u"]))
    pri = starry.Primary(starry.Map(ydeg=A["ydeg"], udeg=A["udeg"], inc=A_inc, amp=A_amp),
                         r=A_r, m=A_m, prot=prot_s)
    pri.map[1] = A_u[0]
    pri.map[2] = A_u[1]

    # Secundaria
    B_inc = pm.Normal("B_inc", mu=inc_B, sd=sd_inc_B, testval=inc_B)
    B_amp = pm.Normal("B_amp", mu=amp_B, sd=sd_amp_B, testval=amp_B)
    if RADIO_B_COMO_RAZON:
        B_k = PositiveNormal("B_k", mu=k_B, sd=sd_k_B, testval=k_B)
        B_r = pm.Deterministic("B_r", A_r * B_k)
    else:
        if PRIOR_RADIOS == "geometria":
            B_logr = pm.Uniform("B_logr", np.log(0.01), np.log(20.0), testval=np.log(r_B))
            B_r = pm.Deterministic("B_r", tt.exp(B_logr))
        else:
            B_r = PositiveNormal("B_r", mu=r_B, sd=sd_r_B, testval=r_B)
    B_m = PositiveNormal("B_m", mu=m_B, sd=sd_m_B, testval=m_B)
    B_porb = PositiveNormal("B_porb", mu=porb_B, sd=sd_porb_B, testval=porb_B)
    B_t0 = pm.Normal("B_t0", mu=t0_B, sd=sd_t0_B, testval=t0_B)
    B_u = xo.distributions.QuadLimbDark("u_B", testval=np.array(B["u"]))
    # (h, k) = (sqrt(e) cos w, sqrt(e) sin w), uniforme en el disco unitario -> e ~ U(0,1), w ~ U(-180,180)
    hk0 = np.sqrt(ecc_B) * np.array([np.cos(np.deg2rad(w_B)), np.sin(np.deg2rad(w_B))])
    B_hk = pmx.UnitDisk("B_hk", testval=hk0)
    B_ecc = pm.Deterministic("B_ecc", tt.sum(B_hk ** 2))
    B_w = pm.Deterministic("B_w", tt.arctan2(B_hk[1], B_hk[0]) * 180.0 / np.pi)
    sec = starry.Secondary(starry.Map(ydeg=B["ydeg"], udeg=B["udeg"], inc=B_inc, amp=B_amp),
                           r=B_r, m=B_m, porb=B_porb, prot=prot_s, t0=B_t0, inc=B_inc, ecc=B_ecc, w=B_w)
    sec.map[1] = B_u[0]
    sec.map[2] = B_u[1]

    sys = starry.System(pri, sec, texp=TEXP)
    flux_model = sys.flux(t=t_real)
    flux_model_guess = pmx.eval_in_model(flux_model)

    pm.Normal("obs", mu=flux_model, sd=sigma_sd, observed=flux_real)

var_names_A = ["A_m", "A_r", "A_inc", "u_A", "A_amp"]
var_names_B = ["B_m", "B_r", "B_porb", "B_inc", "B_t0", "B_hk", "B_ecc", "B_w", "u_B", "B_amp"]
""")

code(r"""
with model:
    map_soln = pmx.optimize(start=model.test_point)

plt.figure(figsize=(12, 5))
plt.plot(t_real, flux_real, "k.", alpha=0.3, ms=2, label="data")
plt.plot(t_real, flux_model_guess, "C1--", lw=1, alpha=0.5, label="Initial")
plt.plot(t_real, pmx.eval_in_model(flux_model, map_soln, model=model), "C1-", label="MAP", lw=1)
plt.legend(fontsize=10, numpoints=5)
plt.xlabel("time [days]", fontsize=16); plt.ylabel("relative flux", fontsize=16);
{k: np.round(v, 5) for k, v in map_soln.items() if not k.endswith("__")}
""")

md(r"""
## 5) Muestreo

`flux_model` no se guarda como Deterministic (en S98 serían ~0.8 GB). Las curvas del
modelo se recalculan después a partir de `N_CURVAS` muestras del posterior.
""")

code(r"""
if CARGAR_TRACE_EXISTENTE and TRACE_PATH.exists():
    trace = az.from_netcdf(TRACE_PATH)
    print(f"Trace cargado de {TRACE_PATH}")
else:
    with model:
        mtrace = pmx.sample(tune=TUNE, draws=DRAWS, start=map_soln, cores=CORES, chains=CHAINS,
                            target_accept=TARGET_ACCEPT, return_inferencedata=False)
        trace = az.from_pymc3(mtrace)
    trace.to_netcdf(TRACE_PATH)
    print(f"Trace guardado en {TRACE_PATH}")

summary = az.summary(trace, var_names=var_names_A + var_names_B)
summary.to_csv(OUT_DIR / f"{Label}_Summary.csv")
display(summary)
""")

md("## 6) Posteriors")

code(r"""
post = trace.posterior
inf_pos = {n: post[n].values.reshape(-1) for n in ["A_m", "A_r", "A_inc", "A_amp"]}
inf_pos["A_u0"] = post["u_A"].values[..., 0].reshape(-1)
inf_pos["A_u1"] = post["u_A"].values[..., 1].reshape(-1)
inf_pos.update({n: post[n].values.reshape(-1) for n in ["B_m", "B_r", "B_porb", "B_inc", "B_t0", "B_ecc", "B_w", "B_amp"]})
# w es angular: se recentra alrededor de su media circular para que no se parta en ±180°
w_rad = np.deg2rad(inf_pos["B_w"])
w_c = np.rad2deg(np.arctan2(np.mean(np.sin(w_rad)), np.mean(np.cos(w_rad))))
inf_pos["B_w"] = (inf_pos["B_w"] - w_c + 180.0) % 360.0 + w_c - 180.0
inf_pos["B_h"] = post["B_hk"].values[..., 0].reshape(-1)
inf_pos["B_k_disk"] = post["B_hk"].values[..., 1].reshape(-1)
inf_pos["B_ecosw"] = inf_pos["B_ecc"] * np.cos(np.deg2rad(inf_pos["B_w"]))
inf_pos["B_esinw"] = inf_pos["B_ecc"] * np.sin(np.deg2rad(inf_pos["B_w"]))
inf_pos["B_u0"] = post["u_B"].values[..., 0].reshape(-1)
inf_pos["B_u1"] = post["u_B"].values[..., 1].reshape(-1)

# semieje mayor [AU] por Kepler
G, M_sun, sec_day, AU = 6.6742e-11, 1.99e30, 86400, 1.496e11
P_s = inf_pos["B_porb"] * sec_day
inf_pos["a_AU"] = (G * (inf_pos["A_m"] + inf_pos["B_m"]) * M_sun * P_s ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / AU

Pos = pd.DataFrame(inf_pos)
Pos.to_csv(OUT_DIR / f"{Label}_Posteriors.csv")
Pos.describe().T
""")

code(r"""
CORNER_KWARGS = dict(
    smooth=0.7, label_kwargs={"fontsize": 14, "labelpad": 8}, title_kwargs={"fontsize": 11, "pad": 10},
    quantiles=[0.16, 0.50, 0.84], plot_density=True, plot_datapoints=True, fill_contours=True,
    show_titles=True, max_n_ticks=4, title_fmt=".3f", use_math_text=True, color="#4C6A4A",
    hist_kwargs={"density": True, "alpha": 0.75, "linewidth": 1.2},
    data_kwargs={"ms": 2.0, "alpha": 0.20}, contour_kwargs={"linewidths": 1.2},
)


def corner_bonito(cols, labels, size, nombre):
    fig = corner.corner(Pos[cols].values, labels=labels, fig=plt.figure(figsize=(size, size)), **CORNER_KWARGS)
    for ax in fig.get_axes():
        ax.tick_params(axis="both", which="major", labelsize=9, length=4, width=0.8, pad=3)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        for s in ax.spines.values():
            s.set_linewidth(0.8)
    fig.suptitle(f"TIC {TIC}  S{SECTOR}", fontsize=16, fontweight="bold", y=1.0)
    fig.savefig(FIG_DIR / nombre, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


corner_bonito(["A_m", "A_r", "A_inc", "A_amp", "A_u0", "A_u1"],
              [r"$M_A\ [M_\odot]$", r"$R_A\ [R_\odot]$", r"$i_A\ [^\circ]$", r"amp$_A$", r"$u_{0,A}$", r"$u_{1,A}$"],
              10, f"Corner_{Label}_A.png")
corner_bonito(["B_ecosw", "B_esinw", "B_ecc", "B_w", "B_inc", "B_r"],
              [r"$e\cos\omega$", r"$e\sin\omega$", r"$e$", r"$\omega\ [^\circ]$", r"$i_B\ [^\circ]$", r"$R_B\ [R_\odot]$"],
              11.5, f"Corner_{Label}_excentricidad.png")
corner_bonito(["B_m", "B_r", "B_inc", "a_AU", "B_ecc", "B_w"],
              [r"$M_B\ [M_\odot]$", r"$R_B\ [R_\odot]$", r"$i_B\ [^\circ]$", r"$a\ [AU]$", r"$e_B$", r"$\omega_B\ [^\circ]$"],
              11.5, f"Corner_{Label}_B.png")
""")

md("## 7) Curva del modelo (percentiles 16/50/84)")

code(r"""
# función del flujo en términos de las variables del posterior (sin reconstruir el modelo)
nombres_in = ["A_inc", "A_amp", "A_r", "A_m", "u_A", "B_inc", "B_amp",
              "B_k" if RADIO_B_COMO_RAZON else "B_r", "B_m", "B_porb", "B_t0", "u_B", "B_ecc", "B_w"]
f_flux = theano.function([model[n] for n in nombres_in], flux_model, on_unused_input="ignore")

n_tot = post.sizes["chain"] * post.sizes["draw"]
sel = np.random.default_rng(1).choice(n_tot, size=min(N_CURVAS, n_tot), replace=False)
planos = {n: post[n].values.reshape((n_tot,) + post[n].shape[2:]) for n in nombres_in}
curvas = np.array([f_flux(*[planos[n][i] for n in nombres_in]) for i in sel])
q16, q50, q84 = np.percentile(curvas, [16, 50, 84], axis=0)

for q, nombre in [(q16, "q16"), (q50, "q50"), (q84, "q84")]:
    pd.DataFrame({"time": t_real, "flux": q}).to_csv(OUT_DIR / f"{Label}_{nombre}.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(t_real, q16, q84, alpha=0.3, label="16--84 percentile")
ax.plot(t_real, q50, lw=2, label="Median model")
ax.scatter(t_real, flux_real, s=8, alpha=0.5, label="Observed")
ax.set_title(Label, fontsize=16, fontweight="bold")
ax.set_xlabel("Time [d]"); ax.set_ylabel("Normalized flux"); ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIG_DIR / f"{Label}_model.png", dpi=300, bbox_inches="tight")
plt.show()
""")

md("## 8) Tabla de parámetros")

code(r"""
pc = {c: np.percentile(Pos[c], [16, 50, 84]) for c in Pos.columns}


def format_parameter(c, decimals=3):
    lo, med, hi = pc[c]
    return rf"${med:.{decimals}f}^{{+{hi - med:.{decimals}f}}}_{{-{med - lo:.{decimals}f}}}$"


filas = [
    (r"$M_A$", "A_m", 3, r"$M_\odot$"), (r"$R_A$", "A_r", 3, r"$R_\odot$"),
    (r"$u_{0,A}$", "A_u0", 3, ""), (r"$u_{1,A}$", "A_u1", 3, ""),
    (r"$M_B$", "B_m", 3, r"$M_\odot$"), (r"$R_B$", "B_r", 3, r"$R_\odot$"),
    (r"$i$", "B_inc", 3, r"$\mathrm{deg}$"), (r"$P_{\mathrm{orb}}$", "B_porb", 5, r"$\mathrm{d}$"),
    (r"$a$", "a_AU", 4, r"$\mathrm{AU}$"), (r"$t_0$", "B_t0", 5, r"$\mathrm{d}$"),
    (r"$e$", "B_ecc", 4, ""), (r"$\omega$", "B_w", 3, r"$\mathrm{deg}$"),
    (r"$e\cos\omega$", "B_ecosw", 5, ""), (r"$e\sin\omega$", "B_esinw", 4, ""),
    (r"$u_{0,B}$", "B_u0", 3, ""), (r"$u_{1,B}$", "B_u1", 3, ""),
]
Final_info = pd.DataFrame({"Parameter": [f[0] for f in filas],
                           "Value": [format_parameter(f[1], f[2]) for f in filas],
                           "Unit": [f[3] for f in filas]})
display(Final_info)

tabla_png = OUT_DIR / "Tables" / f"{Label}_parameters.png"
tabla_png.parent.mkdir(exist_ok=True)
try:
    import dataframe_image as dfi
    dfi.export(Final_info, str(tabla_png), table_conversion="matplotlib", dpi=300)
    print(f"Tabla guardada en {tabla_png}")
except Exception as e:
    print(f"No se pudo exportar la tabla como imagen: {e}")

# percentiles como variables sueltas (mismos nombres que el ejemplo)
M_a_16, M_a_50, M_a_84 = pc["A_m"]; R_a_16, R_a_50, R_a_84 = pc["A_r"]
amp_a_16, amp_a_50, amp_a_84 = pc["A_amp"]; u0_a_50, u1_a_50 = pc["A_u0"][1], pc["A_u1"][1]
M_b_16, M_b_50, M_b_84 = pc["B_m"]; R_b_16, R_b_50, R_b_84 = pc["B_r"]
i_b_50, p_b_50, t0_b_50 = pc["B_inc"][1], pc["B_porb"][1], pc["B_t0"][1]
e_b_50, w_b_50, amp_b_50 = pc["B_ecc"][1], pc["B_w"][1], pc["B_amp"][1]
u0_b_50, u1_b_50 = pc["B_u0"][1], pc["B_u1"][1]
""")

md("## 9) Figura final: detrending, curva aplanada y fase")

code(r"""
def phase_fold(t, y, period, t0):
    ph = np.remainder(t - t0, period) / period
    return np.concatenate((ph, ph + 1)), np.concatenate((y, y))


ph_d, f_d = phase_fold(t_real, flux_real, p_b_50, t0_b_50)
ph_m, m50 = phase_fold(t_real, q50, p_b_50, t0_b_50)
_, m16 = phase_fold(t_real, q16, p_b_50, t0_b_50)
_, m84 = phase_fold(t_real, q84, p_b_50, t0_b_50)
fase_d = ph_d - np.floor(ph_d + 0.5)
fase_m = ph_m - np.floor(ph_m + 0.5)
o = np.argsort(fase_m)

plt.rcParams.update({"font.size": 14})
fig = plt.figure(figsize=(8, 12), constrained_layout=True)
gs = GridSpec(3, 1, figure=fig)

ax1 = fig.add_subplot(gs[0])
ax1.scatter(time, flux, s=10, c="black", label="Original")
ax1.plot(time, trend_flux, c="r", lw=2, ls="--", label="Trend")
ax1.set_ylabel("Flux"); ax1.legend(loc="upper left")

ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.scatter(time, flat_flux, c="mediumblue", s=5, label="Flattened")
ax2.set_ylabel("Normalized Flux"); ax2.set_xlabel("Time [days]"); ax2.legend(loc="upper left")

ax3 = fig.add_subplot(gs[2])
ax3.scatter(fase_d, f_d, s=20, color="C1", alpha=0.8, edgecolors="black", lw=0.4, label="Data")
ax3.scatter(fase_m[o], m50[o], color="darkblue", s=5, label="Best Fit")
ax3.fill_between(fase_m[o], m16[o], m84[o], alpha=0.3, color="darkblue", label="Posterior")
ax3.set_xlabel("Phase"); ax3.set_ylabel("Normalized Flux"); ax3.set_xlim(-0.5, 0.5)
ax3.legend(loc="lower right")
ax3.text(-0.48, np.min(m50), f"$P$ = {p_b_50:.4f} days", fontsize=14)

plt.savefig(FIG_DIR / f"Full_{Label}.png", dpi=300)
plt.show()
plt.rcParams.update({"font.size": 10})
""")

md("## 10) Mapas y animación del sistema (mediana del posterior)")

code(r"""
starry.config.lazy = False
starry.config.quiet = True

A_med = starry.Primary(starry.Map(ydeg=3, udeg=2, amp=amp_a_50), r=R_a_50, m=M_a_50)
A_med.map[1:] = [u0_a_50, u1_a_50]
b_med = starry.Secondary(starry.Map(ydeg=3, udeg=2, inc=i_b_50, amp=amp_b_50), r=R_b_50, m=M_b_50,
                         porb=p_b_50, prot=1.0, t0=t0_b_50, inc=i_b_50, ecc=e_b_50, w=w_b_50)
b_med.map[1:] = [u0_b_50, u1_b_50]
sys_med = starry.System(A_med, b_med)
A_med.map.show()
b_med.map.show()
""")

code(r"""
if HACER_ANIMACION:
    BG, GREEN, GREEN2, ORANGE = "#F7F4EE", "#273A2F", "#4C6A4A", "#B07D4E"
    npts = 200
    t_an = np.linspace(t0_b_50 - p_b_50 / 2, t0_b_50 + p_b_50 / 2, npts)
    x, y, z = sys_med.position(t_an)
    x, y, z = x - x[0], y - y[0], z - z[0]
    theta_sec = [360.0 / s.prot * (t_an - s.t0) - s.theta0 for s in sys_med.secondaries]
    img = np.array([np.tile(sys_med.primary.map.render(res=300), (npts, 1, 1))]
                   + [s.map.render(theta=theta_sec[i], res=300) for i, s in enumerate(sys_med.secondaries)])
    flux_an = sys_med.flux(t_an)
    t_fold = (t_an - t0_b_50) / p_b_50

    fig = plt.figure(figsize=(7.0, 7.8), facecolor=BG)
    ax = fig.add_axes([0.08, 0.31, 0.84, 0.52])
    ax_xz = fig.add_axes([0.24, 0.83, 0.20, 0.15])
    ax_zy = fig.add_axes([0.56, 0.83, 0.20, 0.15])
    ax_lc = fig.add_axes([0.12, 0.07, 0.76, 0.17])
    for axis, arrs in zip([ax, ax_xz, ax_zy], [(x, y), (x, z), (z, y)]):
        axis.set_facecolor(BG); axis.axis("off")
        R = 1.2 * max(-np.min(arrs), np.max(arrs))
        axis.set_xlim(-R, R); axis.set_ylim(-R, R); axis.set_aspect("equal")
    ax_xz.text(0.02, 0.95, "Side", transform=ax_xz.transAxes, fontsize=11, fontweight="bold", color=GREEN, va="top")
    ax_zy.text(0.02, 0.95, "Orbit", transform=ax_zy.transAxes, fontsize=11, fontweight="bold", color=GREEN, va="top")

    ax_lc.set_facecolor(BG)
    ax_lc.plot(t_fold, flux_an, color=GREEN, lw=1.4)
    (lc,) = ax_lc.plot([t_fold[0]], [flux_an[0]], "o", ms=6, mfc=ORANGE, mec=GREEN)
    phase_line = ax_lc.axvline(t_fold[0], color=ORANGE, lw=1.2, ls="--", alpha=0.75)
    ax_lc.set_xlim(t_fold.min(), t_fold.max())
    ax_lc.set_xlabel("Folded Phase", color=GREEN); ax_lc.set_ylabel("Normalised Flux", color=GREEN)
    ax_lc.grid(True, alpha=0.18)

    n_c = 1 + len(sys_med.secondaries)
    radii = np.array([sys_med.primary.r] + [s.r for s in sys_med.secondaries])
    colores = [ORANGE] + [GREEN2] * len(sys_med.secondaries)
    xz, xy, zy, circ = [None] * n_c, [None] * n_c, [None] * n_c, [None] * n_c
    for i in range(n_c):
        ax_xz.plot(x[i], z[i], color=GREEN2, lw=1, alpha=0.35)
        ax_zy.plot(z[i], y[i], color=GREEN2, lw=1, alpha=0.35)
        xz[i] = ax_xz.scatter(x[i][0], z[i][0], s=35 if i == 0 else 22, color=colores[i], edgecolor=GREEN, zorder=10)
        zy[i] = ax_zy.scatter(z[i][0], y[i][0], s=35 if i == 0 else 22, color=colores[i], edgecolor=GREEN, zorder=10)
        ext = np.array([x[i][0], x[i][0], y[i][0], y[i][0]]) + np.array([-1, 1, -1, 1]) * radii[i]
        xy[i] = ax.imshow(img[i, 0], origin="lower", cmap="magma", extent=ext, interpolation="bilinear",
                          clip_on=False, zorder=z[i][0])
        circ[i] = plt.Circle((x[i][0], y[i][0]), radii[i], fill=False, color=colores[i], lw=1.8, zorder=z[i][0] + 1e-3)
        ax.add_artist(circ[i])
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", mfc=ORANGE, mec=GREEN, ms=7, label="Primary"),
                       Line2D([0], [0], marker="o", color="none", mfc=GREEN2, mec=GREEN, ms=6, label="Secondary")],
              loc="upper right", frameon=False, fontsize=9, labelcolor=GREEN)
    fig.text(0.5, 0.965, f"Eclipsing Binary Model -- TIC {TIC}", ha="center", fontsize=15, fontweight="bold", color=GREEN)

    def updatefig(k):
        for i in range(n_c):
            xz[i].set_offsets((x[i][k], z[i][k])); zy[i].set_offsets((z[i][k], y[i][k]))
            xy[i].set_extent(np.array([x[i][k], x[i][k], y[i][k], y[i][k]]) + np.array([-1, 1, -1, 1]) * radii[i])
            xy[i].set_zorder(z[i][k]); xy[i].set_data(img[i, k])
            circ[i].center = (x[i][k], y[i][k]); circ[i].set_zorder(z[i][k] + 1e-3)
        lc.set_xdata([t_fold[k]]); lc.set_ydata([flux_an[k]])
        phase_line.set_xdata([t_fold[k], t_fold[k]])
        return xz + xy + zy + circ + [lc, phase_line]

    ani = animation.FuncAnimation(fig, updatefig, interval=30, frames=npts, blit=False)
    plt.close()
    display(HTML(ani.to_html5_video()))
    ani.save(OUT_DIR / f"{Label}_system.mp4", writer=animation.FFMpegWriter(fps=20), dpi=200)
""")

md(r"""
## 11) Rotación (fuera de eclipses)

Lomb-Scargle sobre `flux/trend` quitando las fases de eclipse, con P_orb y t0 del
posterior. Prior de la curva de luz: P_rot = {{PROT}} d.
""")

code(r"""
phase_orb = ((time - t0_b_50) / p_b_50) % 1 - 0.5
mask = (np.abs(phase_orb) > 0.15) & (np.abs(np.abs(phase_orb) - 0.5) > 0.1)
time_clean, flux_clean = time[mask], (flux / trend_flux)[mask]

flux_ls = flux_clean / np.median(flux_clean) - 1
frequency, power = LombScargle(time_clean, flux_ls).autopower(minimum_frequency=0.1, maximum_frequency=5.0)
period = 1 / frequency
best_period = period[np.argmax(power)]
print(f"Periodo dominante: {best_period:.4f} d  (2x = {2 * best_period:.4f} d)")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(period, power, "k-")
ax[0].axvline(best_period, color="r", ls="--", label=f"{best_period:.3f} d")
ax[0].axvline(2 * best_period, color="b", ls="--", label=f"{2 * best_period:.3f} d")
ax[0].set(xlabel="Period [days]", ylabel="Power", xlim=(0, 5), title="Lomb-Scargle (sin eclipses)")
ax[0].legend()
fase = ((time_clean - t0_b_50) / best_period) % 1
ax[1].scatter(fase, flux_clean, s=8, color="red", alpha=0.4)
ax[1].set(xlabel="Phase", ylabel="Flux corrected", title=f"Fase con P = {best_period:.4f} d")
plt.tight_layout(); plt.show()
""")

md(r"""
## 12) Excentricidad, circularización y sincronización

**Qué es robusto y qué no.** La fotometría mide $e\cos\omega$ con muchísima precisión, por el
momento del eclipse secundario: $\Delta\phi\approx\tfrac{2}{\pi}e\cos\omega$ (Sterne 1939). En
cambio, $e\sin\omega$ sale del cociente de duraciones de los eclipses y es frágil en eclipses
parciales: se mezcla con la inclinación, los radios, el limb darkening y el suavizado por
cadencia. En las pruebas de este proyecto, cambiar solo `TEXP` y el prior de radios movió
$e\sin\omega$ en ~0.05 mientras $e\cos\omega$ no cambió.

Por eso, en estudios de circularización con fotometría se usa **$|e\cos\omega|$ como
diagnóstico principal** (p. ej. Van Eylen et al. 2016; Justesen & Albrecht 2021):

- es una **cota inferior** de la excentricidad, $e\ge|e\cos\omega|$;
- para una muestra con $\omega$ orientados al azar, $\langle|\cos\omega|\rangle=2/\pi$, así que la
  distribución de $|e\cos\omega|$ sigue a la de $e$ salvo un factor conocido.

**(a) Clasificación por $e\cos\omega$.** Si $|e\cos\omega|/\sigma>5$, la órbita es excéntrica con
certeza. Si no, se da el límite superior al 95 %, $|e\cos\omega|_{95}$, y la órbita es
**consistente con circular** hasta ese nivel en la componente que los datos sí miden.

**(b) $e$ completa y factor de Bayes (depende de $e\sin\omega$).** Con el prior uniforme en el
disco unitario de $(h,k)=(\sqrt e\cos\omega,\sqrt e\sin\omega)$, la masa previa dentro de
$e<e_0$ es exactamente $e_0$. Por eso el factor de Bayes de Savage–Dickey (Dickey 1971) para
"$e\approx0$" se estima **contando** muestras:

$$\mathrm{BF}_{e<e_0}=\frac{P(e<e_0\mid\text{datos})}{P(e<e_0)}=\frac{N(e<e_0)/N}{e_0},\qquad e_0=0.01 .$$

Si $N(e<e_0)=0$, se reporta la cota $\mathrm{BF}<3/(N e_0)$ (límite de Poisson al 95 %). Este
valor hereda la fragilidad de $e\sin\omega$: úsalo como información secundaria y compáralo
entre sectores.

**(c) Chequeo contra `1b`.** $e\cos\omega$ del posterior contra
$\tfrac{\pi}{2}(\phi_2-\tfrac12)$ medido en `1b`. La precisión de `1b` está limitada por el
tamaño de la ventana, así que se esperan diferencias de hasta ~0.01.

**(d) Sincronización.** En una órbita excéntrica, las mareas llevan la rotación al periodo
**pseudo-sincrónico** (Hut 1981, ec. 42), no al orbital:

$$\frac{P_{\rm orb}}{P_{\rm ps}}=\frac{1+\tfrac{15}{2}e^2+\tfrac{45}{8}e^4+\tfrac{5}{16}e^6}
{\left(1+3e^2+\tfrac38e^4\right)\left(1-e^2\right)^{3/2}} .$$

$P_{\rm ps}$ se calcula con el posterior completo de $e$ y también con la cota
$e_{\min}=|e\cos\omega|$, que es la versión robusta. Se comparan con $P_{\rm rot}$ del
Lomb-Scargle de la sección 11 (y su armónico $2P$). Todo se guarda en
`<Label>_circularizacion.csv` para el análisis de la muestra.
""")

code(r"""
hk = post["B_hk"].values.reshape(-1, 2)
N = len(hk)
e_s = hk[:, 0] ** 2 + hk[:, 1] ** 2
w_s = np.arctan2(hk[:, 1], hk[:, 0])
ecosw_s, esinw_s = e_s * np.cos(w_s), e_s * np.sin(w_s)
q = lambda v: np.percentile(v, [16, 50, 84])

# (a) clasificación por e cos w
ecosw_med, ecosw_sig = np.median(ecosw_s), np.std(ecosw_s)
signif_ecosw = abs(ecosw_med) / ecosw_sig
ecosw_95 = np.percentile(np.abs(ecosw_s), 95)
if signif_ecosw > 5:
    clase = f"EXCÉNTRICA: |e cos w| = {abs(ecosw_med):.4f} ({signif_ecosw:.0f} sigma), e >= {abs(ecosw_med):.4f}"
else:
    clase = f"consistente con CIRCULAR en e cos w: |e cos w| < {ecosw_95:.4f} (95 %)"

# (b) factor de Bayes por conteo (Savage-Dickey con prior uniforme en e)
E0 = 0.01
n0 = int((e_s < E0).sum())
if n0 > 0:
    BF = (n0 / N) / E0
    texto_bf = f"BF(e<{E0}) = {BF:.3g} (ln BF = {np.log(BF):+.2f})"
else:
    BF = np.nan
    texto_bf = f"BF(e<{E0}) < {3 / (N * E0):.3g} (ninguna muestra con e < {E0})"

# (c) chequeo contra 1b
ECOSW_1B = {{ECOSW_1B}}

# (d) sincronización (Hut 1981)
def cociente_ps(e):
    return (1 + 7.5 * e**2 + 5.625 * e**4 + 0.3125 * e**6) / ((1 + 3 * e**2 + 0.375 * e**4) * (1 - e**2) ** 1.5)

P_orb_s = post["B_porb"].values.reshape(-1)
P_ps_s = P_orb_s / cociente_ps(e_s)
P_ps_min_s = P_orb_s / cociente_ps(np.abs(ecosw_s))

filas_res = {"e_cos_w": q(ecosw_s), "e_sin_w": q(esinw_s), "e": q(e_s), "w_deg": q(np.rad2deg(w_s) % 360),
             "P_orb": q(P_orb_s), "P_pseudo_sinc": q(P_ps_s), "P_pseudo_sinc_emin": q(P_ps_min_s)}
res = pd.DataFrame(filas_res, index=["p16", "p50", "p84"]).T
display(res)

print(f"(a) {clase}")
print(f"(b) e = {np.median(e_s):.4f} [{q(e_s)[0]:.4f}, {q(e_s)[2]:.4f}];  {texto_bf}   <- depende de e sin w")
print(f"(c) e cos w: posterior {ecosw_med:+.5f}  vs  fase del secundario (1b) {ECOSW_1B:+.5f}")
for nombre, prot in [("P_rot (LS)", best_period), ("2 x P_rot (LS)", 2 * best_period)]:
    print(f"(d) {nombre} = {prot:.4f} d:  P_rot/P_orb = {prot / np.median(P_orb_s):.3f},  "
          f"P_rot/P_ps = {prot / np.median(P_ps_s):.3f},  P_rot/P_ps(e_min) = {prot / np.median(P_ps_min_s):.3f}")

fila_csv = {"TIC": TIC, "Sector": SECTOR, "clasificacion_ecosw": "excentrica" if signif_ecosw > 5 else "circular_en_ecosw",
            "signif_ecosw": signif_ecosw, "abs_ecosw_95": ecosw_95, "BF_e_lt_0.01": BF, "n_e_lt_0.01": n0,
            "ecosw_fase_1b": ECOSW_1B, "TEXP": TEXP, "PRIOR_RADIOS": PRIOR_RADIOS,
            "P_rot_LS": best_period, "P_rot_LS_x2": 2 * best_period, "P_rot_prior_1b": {{PROT}}}
for k, (a, b, c) in filas_res.items():
    fila_csv.update({f"{k}_p16": a, f"{k}_p50": b, f"{k}_p84": c})
pd.DataFrame([fila_csv]).to_csv(OUT_DIR / f"{Label}_circularizacion.csv", index=False)
print(f"Guardado: {OUT_DIR / f'{Label}_circularizacion.csv'}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ax[0].plot(ecosw_s, esinw_s, ",", alpha=0.3)
ax[0].axvline(0, color="0.6", lw=0.8); ax[0].axhline(0, color="0.6", lw=0.8)
ax[0].plot(0, 0, "r+", ms=14)
ax[0].set(xlabel=r"$e\cos\omega$", ylabel=r"$e\sin\omega$", title=r"Posterior: $e\cos\omega$ fijo, $e\sin\omega$ libre")
ax[1].hist(ecosw_s, bins=60, color="#4C6A4A", alpha=0.85)
ax[1].axvline(0, color="r", ls="--")
ax[1].set(xlabel=r"$e\cos\omega$", title=clase.split(":")[0])
ax[2].hist(e_s, bins=60, color="#B07D4E", alpha=0.85)
ax[2].axvline(abs(ecosw_med), color="r", ls="--", label=r"$|e\cos\omega|$ (cota inferior)")
ax[2].set(xlabel="e", title="e completa (depende de e sin w)")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / f"{Label}_excentricidad.png", dpi=200, bbox_inches="tight")
plt.show()
""")


# ---------------------------------------------------------------------------
# generación
# ---------------------------------------------------------------------------
def crear(tic, sector, v, avisos, fuente, file_lc, npts, cadencia, out_dir, test=False):
    bin_min = "None"
    avisos = list(avisos)
    if npts > 5000:
        avisos.append(f"Sector pesado ({npts} puntos). Si el muestreo es demasiado lento, usar BIN_MIN = 10 "
                      f"(~{int(npts * cadencia / 10)} puntos).")
    reemplazos = {
        "TIC": str(tic), "SECTOR": str(sector), "ARCHIVO": file_lc.name, "NPUNTOS": str(npts),
        "CADENCIA": f"{cadencia:.1f}", "RAMA": v["rama"], "EDAD": f"{v['edad']:.3g}",
        "FILE_LC": str(file_lc), "OUT_DIR": str(out_dir),
        "OUT_REL": str(out_dir.relative_to(RUTA_ORION)) if not test else str(out_dir),
        "AVISOS": ("**Avisos:**\n\n" + "\n".join(f"- ⚠️ {a}" for a in avisos)) if avisos else "Sin avisos.",
        "FUENTES": "\n".join([
            f"- **Primaria**: {fuente['A']}.",
            f"- **R_B** = R_A·sqrt(Depth_primary), Depth_primary (v2) = {v['D']:.5f} ± {v['sD']:.5f} "
            f"(v1 era {v['D_v1']:.5f}; sesgo sistemático máximo del estimador v2: {v['sesgo']:.0%}). "
            f"sd_r_B propaga el error de R_A y el de la profundidad.",
            f"- **M_B**: {fuente['m_B']}.",
            f"- **amp**: {fuente['amp']}. amp_A = 1 − amp_B.",
            f"- **P_orb, t0**: priors de la curva de luz de este sector (sd_t0 = duración del eclipse).",
            f"- Teff_A = {v['T1']:.0f} K, Teff_B = "
            + (f"{v['T2']:.0f} K" if np.isfinite(v["T2"]) else "sin estimación") + ".",
            f"- **ecc, w**: {fuente['ecc']}.",
            "- inc, u: mismos valores genéricos del notebook de ejemplo.",
        ]),
        "BIN_MIN": bin_min, "PROT": fmt(v.get("prot", np.nan), 5),
        "TUNE": "30" if test else "1500", "DRAWS": "30" if test else "1500",
        "CHAINS": "2" if test else "4", "CORES": "2" if test else "5",
    }
    for key in ["amp_A", "sd_amp_A", "r_A", "sd_r_A", "m_A", "sd_m_A", "amp_B", "sd_amp_B", "r_B", "sd_r_B",
                "k", "sd_k", "m_B", "sd_m_B", "D", "sD", "dur", "ecc", "w"]:
        reemplazos[key] = fmt(v[key], 5)
    reemplazos["porb"] = fmt(v["porb"], 9)
    reemplazos["t0"] = f"{v['t0']:.6f}"
    reemplazos["TEXP"] = f"{cadencia / 1440:.5f}"
    reemplazos["FASE2"] = f"{v['fase2']:.4f}"
    reemplazos["MODO"] = v["modo"]
    reemplazos["TEXP_DEFAULT"] = f"{cadencia / 1440:.5f}" if cadencia >= 9 else "None"
    reemplazos["ECOSW_1B"] = f"{v['ecosw_1b']:.5f}" if np.isfinite(v["ecosw_1b"]) else "np.nan"
    reemplazos["D2"] = f"{v['D2']:.5f}" if (np.isfinite(v["D2"]) and v["sec_det"]) else "np.nan"

    celdas = []
    for tipo, src in CELDAS:
        for k, val in reemplazos.items():
            src = src.replace("{{" + k + "}}", val)
        assert not re.search(r"\{\{[A-Za-z_]+\}\}", src), re.findall(r"\{\{[A-Za-z_]+\}\}", src)
        celdas.append(nbf.v4.new_markdown_cell(src) if tipo == "markdown" else nbf.v4.new_code_cell(src))
    nb = nbf.v4.new_notebook(cells=celdas)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
    return nb


def ya_ejecutado(path):
    if not path.exists():
        return False
    viejo = nbf.read(path, as_version=4)
    return any(c.get("outputs") for c in viejo.cells if c.cell_type == "code")


filas_indice = []
for _, r in resumen.sort_values(["TIC", "Sector"]).iterrows():
    tic, sector = int(r["TIC"]), int(r["Sector"])
    p = priors.loc[(priors["TIC"] == tic) & (priors["Sector"] == sector)].iloc[0]
    if not np.isfinite(p["Depth_primary"]):
        print(f"[omitido] TIC {tic} S{sector}: sin Depth_primary")
        continue
    file_lc = RUTA_ORION / "Light_Curve_Files" / f"tic {tic}_S_{sector}_LC.txt"
    t = pd.read_csv(file_lc, sep=r"\s+", usecols=["#Time"], nrows=50)["#Time"].to_numpy()
    cadencia = np.median(np.diff(t)) * 1440
    npts = n_puntos(file_lc)
    v, avisos, fuente = priors_de(tic, sector)
    v["prot"] = p["P_rot"]
    base = Path(__file__).resolve().parent / "test_starry" if TEST else DESTINO
    out_dir = base / f"TIC_{tic}" / f"S{sector}"
    nb = crear(tic, sector, v, avisos, fuente, file_lc, npts, cadencia, out_dir, test=TEST)
    nb_path = base / f"TIC_{tic}" / f"full_{tic}_s{sector}.ipynb"
    if ya_ejecutado(nb_path):
        # no se pisa un notebook con resultados: la versión nueva va aparte y escribe en S<sector>_ecc/
        nb_path = nb_path.with_name(f"full_{tic}_s{sector}_ecc.ipynb")
        out_dir = out_dir.with_name(f"S{sector}_ecc")
        nb = crear(tic, sector, v, avisos, fuente, file_lc, npts, cadencia, out_dir, test=TEST)
        print(f"[conservado] full_{tic}_s{sector}.ipynb tiene salidas; la versión nueva es {nb_path.name}")
    if ya_ejecutado(nb_path):
        print(f"[omitido] {nb_path.name} ya tiene salidas; no se sobreescribe")
        continue
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, nb_path)
    filas_indice.append(dict(TIC=tic, Sector=sector, puntos=npts, cadencia_min=round(cadencia, 2),
                             notebook=str(nb_path) if TEST else str(nb_path.relative_to(RUTA_ORION)), n_avisos=len(avisos),
                             avisos=" | ".join(avisos)))
    if TEST:
        break

if not TEST:
    ind = pd.DataFrame(filas_indice)
    ind.to_csv(DESTINO / "indice_notebooks.csv", index=False)
    print(f"{len(ind)} notebooks en {DESTINO}")
    print(ind[["TIC", "Sector", "puntos", "n_avisos"]].to_string(index=False))
