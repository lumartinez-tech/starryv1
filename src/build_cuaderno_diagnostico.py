"""Genera 5_diagnostico_resultados_starry.ipynb (kernel python3 del entorno starry). Uso: python build_cuaderno_diagnostico.py"""
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
AQUI = Path(__file__).resolve().parent

celdas = [
md(r"""# Diagnóstico de los resultados de starry en Orión

Cuaderno de revisión de **todas las corridas de starry** (19 sistemas, 54 combinaciones de sistema y sector): convergencia,
calidad del ajuste, parámetros orbitales y físicos, relaciones entre ellos y **casos inconsistentes**, con propuestas de mejora
que no alteran el flujo de trabajo actual. Está pensado para la reunión con el asesor.

**Cómo se leen las figuras.** Un punto es una corrida (un sistema en un sector); el marcador indica el sector.
Los números junto a los puntos son el índice del sistema de `indice_sistemas.csv` (el mismo de la gráfica de Prot vs Porb).
**Azul** = corrida sin problemas propios; **naranja** = corrida con problema (convergencia, ajuste del eclipse o física dudosa).

**Todos los valores son la media posterior** (la excentricidad queda ligeramente sesgada al alza cerca de 0 porque no puede ser negativa;
ω es una variable circular y se promedia como tal). Donde hay intervalos: **punto = media, trazo grueso = intervalo creíble de 68 %, trazo fino = 95 %**
(los *forest* o *caterpillar plots* de ArviZ y MCMCvis [5][6]). Donde solo hay barras, son ±1 desviación estándar.

**Guía de diseño de las figuras** (autores que muestran resultados de MCMC):
- Gabry et al. [1]: la traza sola no basta; mostrar pares con las divergencias marcadas, comparar prior y posterior, y comprobar el modelo contra los datos.
- Vehtari et al. [2]: sustituir las trazas por *rank plots* de las cadenas (planos si las cadenas mezclan bien).
- Foreman-Mackey, *corner.py* [3]: matriz de dispersión de las muestras, el estándar en astronomía.
- Kay et al. [4]: los *quantile dotplots* (20 puntos de 5 % cada uno) se entienden mejor que una densidad para estimar probabilidades.
- Hogg y Foreman-Mackey [7]: introducción pedagógica al MCMC en astronomía.

**Alcance.** Es un diagnóstico de lo que ya está corrido. No modifica ninguna corrida ni la metodología:
sigue suponiendo (1) una estrella dominante cuyos parámetros secundarios se estiman con massage, (2) edad compartida y
(3) profundidad del eclipse medida con la fotometría de TESS. La tercera luz queda para la discusión, no para el modelo."""),
code(r"""import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/luciam/Documents/starry/Resultados_Orion/diagnostico_resultados")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import diagnostico_orion as D

pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 140); pd.set_option("display.max_rows", 200)
FIG = D.AQUI / "figuras"; FIG.mkdir(exist_ok=True)
def mostrar(fig, nombre):
    fig.savefig(FIG / f"{nombre}.png", dpi=130, bbox_inches="tight"); plt.show()

T, todas, alt = D.tabla_maestra()               # corrida vigente por (TIC, sector) + todas las candidatas + rondas A y C
C = D.consistencia(T)                           # consistencia entre sectores
T = D.marcar_atipicas(T, C)
S = D.por_sistema(T)                            # una fila por sistema (mediana entre sectores)
A = D.tabla_atipicos(T, C)
A.to_csv(D.AQUI / "casos_atipicos.csv", index=False)
print(f"{len(todas)} corridas candidatas · {len(T)} vigentes · {T['TIC'].nunique()} sistemas · ejecutado {time.strftime('%Y-%m-%d %H:%M')}")"""),
md(r"""## Regla para elegir la corrida vigente
Para cada (sistema, sector) hay a veces varias corridas (original, corregida, ronda 2, edad por vecindad, ronda 3 de muestreo,
ronda C de fase). **Vigente = la última por fecha, entre las cuatro primeras versiones, que converge** (R̂ máximo ≤ 1.05 y ESS mínimo ≥ 100);
si ninguna converge, la de menor R̂. Una corrida de la **ronda 3** (más muestreo)
la reemplaza solo si su R̂ máximo no es peor y su ESS mínimo no es menor; una de la **ronda C** (prior de e cosω) solo si la vigente tenía la fase
del secundario a más de 0.02 de la observada y la nueva la deja a menos de 0.02 sin empeorar el R̂ en más de 0.05. Así una corrida que sale peor **no** pasa a los resultados (ver la tabla de rondas A y C más abajo)."""),
code(r"""n_conv = int((~T["f_conv"]).sum())
excentricas = T[T["clasificacion_ecosw"] == "excentrica"]
val = S[S["rot_valida"]]; sinc = val[(val["prot_porb"] - 1).abs() < 0.05]; sin_rot = S[~S["rot_valida"]]
pseudo = val[((val["P_rot_consenso"] / (val["P"] / D.hut(val["e"])) - 1).abs() < 0.05) & ~val.index.isin(sinc.index)]
otros_rot = val[~val.index.isin(sinc.index) & ~val.index.isin(pseudo.index)]
inc_fuerte = C[D.inconsistentes(C)]
r_menor = T[(T["z_prior_A_r"] < -3)]
display(Markdown(f'''## Resumen ejecutivo

- **Cobertura:** los **{T["TIC"].nunique()} sistemas** con eclipse tienen corrida de starry ({len(T)} combinaciones de sistema y sector vigentes).
- **Convergencia:** {n_conv} de {len(T)} corridas cumplen R̂ ≤ 1.02 y ESS ≥ 400; {int(T["f_conv"].sum())} no (detalle en la sección 1). {int(T["f_div"].sum())} con más de 5 % de pasos divergentes.
- **Ajuste modelo vs datos:** {int((T["ajuste_prim"].astype(str).str.startswith("bueno") & T["ajuste_sec"].astype(str).str.contains("bueno|no detectado")).sum())} corridas ajustan bien primario y secundario (o el secundario no se detecta); {int(T["f_fase"].sum())} con el secundario del modelo en una fase equivocada y {int((T["f_ajuste"] & ~T["f_fase"]).sum())} con un déficit de profundidad significativo (sección 2).
- **Excentricidad:** {len(excentricas)} corridas ({excentricas["TIC"].nunique()} sistemas) tienen e cosω significativa; el resto es consistente con órbitas circulares. {int(T["f_e_alta_P_corto"].sum())} corridas tienen e > 0.1 con P < 3 d, por debajo del periodo de circularización empírico (sección 3).
- **Rotación:** de los {len(val)} sistemas con rotación medible, {len(sinc)} están sincronizados (Prot/Porb = 1 ± 5 %) y {len(pseudo)} es pseudo-sincrónico (sistema {", ".join(pseudo["etiqueta"].astype(str))}); los demás ({", ".join(otros_rot["etiqueta"].astype(str))}) no cumplen ninguna de las dos. En {len(sin_rot)} sistemas ({", ".join(sin_rot["etiqueta"].astype(str))}) la modulación a Porb/2 es variación elipsoidal o no es interpretable, y no se cuenta como rotación (sección 4).
- **Masas:** los datos **casi no aprenden** sobre las masas (contracción mediana {T["contr_A_m"].median():.2f} en M_A y {T["contr_B_m"].median():.2f} en M_B): son prioritariamente las de massage. Los radios sí aprenden (R_A {T["contr_A_r"].median():.2f}, R_B {T["contr_B_r"].median():.2f}) y la amplitud de B más (sección 5).
- **Radios por debajo del prior:** en {len(r_menor)} corridas el R_A posterior queda más de 3σ por debajo del prior; el signo es el que produciría una profundidad diluida, pero no lo prueba.
- **Consistencia entre sectores:** {len(inc_fuerte)} de {len(C)} sistemas con más de un sector tienen algún parámetro (k, R_A, R_B, i, e o e cosω) que cambia más de 5σ de un sector a otro **y** por encima de un piso físico (k 0.05, radios 5 %, i 0.5°, e 0.03, e cosω 0.01; sección 6).
- **Casos atípicos:** {A["TIC"].nunique()} sistemas con al menos un motivo de revisión, listados con acción sugerida en la sección 8.'''))"""),
md(r"""## 1. Convergencia
R̂ ≤ 1.02 y ESS ≥ 400 como criterio. Las corridas que no lo cumplen se listan abajo con el parámetro peor muestreado."""),
code(r"""mostrar(D.fig_convergencia(T), "01_convergencia")
cols = ["etiqueta", "TIC", "Sector", "ver", "rhat_max", "rhat_peor", "ess_min", "ess_peor", "div_frac"]
display(T.loc[T["f_conv"], cols].sort_values(["etiqueta", "Sector"]).round(3).reset_index(drop=True))"""),
md(r"""## 2. Calidad del ajuste: modelo vs datos en cada eclipse
Se pliega con P y t₀ del posterior y se mide el residuo (datos − modelo) en el núcleo de cada eclipse como fracción de su profundidad.
**Criterio de significancia:** el secundario solo se evalúa si 1b lo detectó (`signif_emp > 5`); el residuo cuenta solo si supera 3 veces
el ruido empírico σ_emp (dispersión de la mediana del residuo en ventanas del mismo ancho fuera de eclipse, que incluye el ruido correlacionado
por manchas); si es significativo se clasifica por fracción de la profundidad (< 5 % bueno, 5–15 % regular, > 15 % malo); y la fase del secundario
del modelo debe caer a menos de 0.02 de la observada en 1b."""),
code(r"""mostrar(D.fig_ajuste(T), "02_ajuste")
cols = ["etiqueta", "TIC", "Sector", "ver", "ajuste_prim", "ajuste_sec", "res_prim_pct", "res_sec_pct", "z_res_sec", "fase_sec_ref", "fase_sec_modelo"]
display(T.loc[T["f_ajuste"] | T["f_fase"] | T["f_sec_no_det"], cols].sort_values(["etiqueta", "Sector"]).round(3).reset_index(drop=True))"""),
md(r"""## 3. Excentricidad, periodo orbital y ω
Excentricidad frente a periodo orbital (diagrama de circularización) y el plano (e cosω, e sinω). La banda azul marca los periodos de circularización
empíricos de Kepler y TESS (Zanazzi 2021: envolvente ≈ 3 d y núcleo frío ≈ 6 d). Un punto con e alta a P corto es un candidato a tercer cuerpo o a un
modelo mal restringido, no necesariamente un error."""),
code(r"""mostrar(D.fig_e_P(T), "03_e_vs_P")
mostrar(D.fig_omega(T), "04_omega")
f = D.fig_puntos_e(T)
if f is not None: mostrar(f, "04b_dotplot_e")
cols = ["etiqueta", "TIC", "Sector", "P", "e", "e_lo", "e_hi", "w", "ecosw", "ecosw_1b", "clasificacion_ecosw", "signif_ecosw"]
display(T.loc[T["f_e_alta_P_corto"] | (T["clasificacion_ecosw"] == "excentrica"), cols].sort_values(["etiqueta", "Sector"]).round(3).reset_index(drop=True))"""),
md(r"""## 4. Rotación frente a órbita
Mismo gráfico definitivo de Prot vs Porb, junto con el cociente Prot/Porb. El color indica la excentricidad de starry."""),
code(r"""mostrar(D.fig_rotacion(S), "05_rotacion")
mostrar(D.fig_rotacion_pseudo(S), "05b_rotacion_pseudosincronia")
from IPython.display import Image
print("Rotación del sistema 16 (427395536) en cuatro fotometrías, con controles (sistemas 8 y 10):")
display(Image(filename=str(FIG / "rotacion_fotometrias.png")))
display(pd.read_csv(D.AQUI / "rotacion_fotometrias.csv").round(3))
print("Sistemas con Prot ≈ Porb/2 (3, 4, 11, 19): modulación fuera de eclipse plegada en el periodo orbital")
display(Image(filename=str(FIG / "armonicos_prot_porb.png")))
display(pd.read_csv(D.AQUI / "armonicos_prot_porb.csv").round(3))
display(S[["etiqueta", "TIC", "P_orb_consenso", "P_rot_consenso", "prot_porb", "e", "n_sectores"]].sort_values("prot_porb").round(3).reset_index(drop=True))"""),
md(r"""## 5. Masas, radios y geometría
Las masas y los radios de A y B frente a los priors que usó cada corrida (extraídos de su propio notebook), la razón de radios k frente a la razón de masas q,
y la geometría (radios fraccionarios e inclinación)."""),
code(r"""mostrar(D.fig_masas(T, S), "06_masas")
mostrar(D.fig_radios(T, S), "07_radios")
mostrar(D.fig_contraccion(T), "08_contraccion")
mostrar(D.fig_geometria(T), "09_geometria")
cols = ["etiqueta", "TIC", "Sector", "A_r", "prior_A_r", "z_prior_A_r", "B_r", "prior_B_r", "z_prior_B_r", "B_m", "prior_B_m", "z_prior_B_m"]
display(T.loc[T["f_vs_prior"], cols].sort_values(["etiqueta", "Sector"]).round(3).reset_index(drop=True))"""),
md(r"""## 6. Consistencia entre sectores
Un mismo sistema observado en distintos sectores debería dar la misma geometría. En el gráfico de bosque (figura 1), cada fila es un sistema y cada
marcador un sector: **si los marcadores de una fila no se solapan, el parámetro cambió de un sector a otro**. El mapa de calor (figura 2) resume las diferencias
máximas entre sectores en unidades del error combinado; el contorno naranja marca las inconsistencias fuertes."""),
code(r"""mostrar(D.fig_forest(T), "10_forest")
mostrar(D.fig_consistencia(C), "11_consistencia")
display(C.round(1).reset_index(drop=True))"""),
md(r"""## 7. Relaciones entre parámetros
Correlaciones de Spearman entre las medianas por sistema. **Casi todas las correlaciones fuertes son triviales:** definiciones (q = M_B/M_A),
efectos de selección (solo vemos sistemas que eclipsan, así que la inclinación se acopla a R/a) o la relación masa-radio esperada. Se separan
en la lista; solo las marcadas *informativas* aportan algo, y con n = 19 y decenas de pares probados **ninguna sobrevive la corrección por comparaciones
múltiples**. Son pistas exploratorias, no resultados."""),
code(r"""fig, pares = D.fig_correlaciones(S); mostrar(fig, "12_correlaciones")
f2 = D.fig_relaciones(S, pares)
if f2 is not None: mostrar(f2, "13_relaciones")
display(pares.round(3).reset_index(drop=True))"""),
md(r"""### Relaciones orbitales
Periodo y excentricidad frente a masa de la secundaria, rotación, cociente de masas y edad. Cada figura trae la recta de Theil–Sen
(en log10 si el eje es logarítmico), la banda del IC 95 % de su pendiente y ρ de Spearman. **Ojo:** e es la media entre sectores y
arrastra e sin ω, que está mal restringida cuando el secundario es poco profundo."""),
code(r"""M = D.muestra_completa(S, T)                    # Orión (21) + Pléyades (4)
display(M[["cumulo", "etiqueta", "TIC", "corrida", "P", "e", "ecosw", "ecosw_sd", "ecosw_fuente", "q", "B_m", "P_rot_consenso", "edad_Myr"]].round(4))
for nombre, fig in D.figs_pares_orbitales(M).items():
    mostrar(fig, nombre)"""),
md(r"""## 8. Casos atípicos y qué hacer con cada uno
Cada fila es una corrida con al menos un motivo de revisión, con la acción que propongo **dentro de la metodología actual**
(sin tercera luz, sin liberar luminosidades, sin ajuste conjunto de sectores)."""),
code(r"""display(A.sort_values(["sistema", "sector"]).reset_index(drop=True))"""),
md(r"""## 8b. Galería de diagnóstico por corrida
Una ficha por cada corrida con convergencia, ajuste o divergencias problemáticas. **Arriba:** modelo (mediana y banda 16–84 %) sobre los datos (media por bin) en cada eclipse, y el *rank plot* por cadena del parámetro peor muestreado
(barras planas = cadenas que mezclan bien; una cadena con barras fuera de la banda gris = esa cadena explora otra región). **Abajo:** traza, par de parámetros con las divergencias en naranja
(si se agrupan en un borde, es un embudo en el posterior) y prior (línea discontinua) frente a posterior (azul): si se parecen, los datos no aprenden ese parámetro."""),
code(r"""casos = T[T["f_conv"] | T["f_div"] | T["f_fase"] | T["f_ajuste"]].sort_values(["sistema", "Sector"])
(FIG / "casos").mkdir(exist_ok=True)
print(f"{len(casos)} corridas en la galería")
for _, r in casos.iterrows():
    try:
        fig = D.fig_caso(r); fig.savefig(FIG / "casos" / f"caso_{r['etiqueta']}_S{int(r['Sector'])}.png", dpi=120, bbox_inches="tight"); plt.show()
    except Exception as e:
        print("sin ficha:", r["etiqueta"], int(r["Sector"]), e)"""),
md(r"""### Matrices de dispersión (corner) de los casos con peor convergencia
Los pares de masa, radio, inclinación y excentricidad de las corridas con mayor R̂; las divergencias en naranja."""),
code(r"""peores = T[T["f_conv"]].sort_values("rhat_max", ascending=False).head(2)
for _, r in peores.iterrows():
    fig = D.fig_corner(r); fig.savefig(FIG / "casos" / f"corner_{r['etiqueta']}_S{int(r['Sector'])}.png", dpi=120, bbox_inches="tight"); plt.show()"""),
md(r"""## 9. Rondas de mejora A (más muestreo) y C (prior de e cosω)
Resultado de las rondas ya terminadas y si su corrida reemplaza a la vigente. Las que aún no terminan no aparecen: este cuaderno se puede volver a ejecutar
y las incorpora solo si cumplen la regla de selección."""),
code(r"""display(alt.round(3).reset_index(drop=True) if len(alt) else Markdown("Aún no hay rondas A o C terminadas."))
import glob, os
pend = [os.path.basename(f) for f in glob.glob(str(D.RAIZ / "starry_correcciones/ronda3/TIC_*/full_*_r3.ipynb")) + glob.glob(str(D.RAIZ / "starry_correcciones/rondaC/TIC_*/full_*_rC.ipynb"))]
hechas = set(zip(alt["TIC"], alt["Sector"], alt["ronda"])) if len(alt) else set()
print(f"{len(pend)} notebooks de rondas A y C generados; {len(alt)} con resultados hasta ahora.")"""),
code(r"""def lista(df, col="etiqueta"):
    return ", ".join(sorted(df[col].astype(str).unique(), key=lambda x: (len(x), x))) or "ninguno"
art = set(T.loc[T["f_fase"] | T["f_sec_no_det"], "TIC"])            # ecosw poco fiable: fase mal puesta o secundario no detectado
ec_cambia = C[C["fuerte_ecosw"] & (~C["TIC"].isin(art))]       # > 5σ y cambio absoluto > 0.01 (si no, es ruido numérico en órbitas circulares)
ec_art = C[C["fuerte_ecosw"] & (C["TIC"].isin(art))]
ep_corto = T[T["f_e_alta_P_corto"] & ~T["f_fase"]]
nodet = T[T["f_sec_no_det"]]
radios_bajos = T[T["z_prior_A_r"] < -3]
deg = T[(T["f_conv"]) & (T["contr_B_m"] < 0.2)]
if len(ec_cambia):
    q4 = (f"En {len(ec_cambia)} sistemas ({lista(ec_cambia)}) e cosω cambia más de 5σ y más de 0.01 en valor absoluto entre sectores, sin problema de fase ni de detección: podría ser movimiento apsidal real "
          f"o sistemáticos de cada sector. ¿Vale la pena evaluar la escala de tiempo del movimiento apsidal?")
else:
    q4 = ("Ningún sistema cambia su e cosω entre sectores de forma física: los cambios formales de más de 5σ son diferencias menores de 0.01 en órbitas casi circulares, y los casos con cambios grandes "
          f"(sistemas {lista(ec_art)}) vienen de una fase mal puesta o de un secundario no detectado. No hay evidencia de movimiento apsidal; ¿se confirma que no se pretende buscarlo?")
display(Markdown(f'''## 10. Preguntas para el asesor
Surgen de los resultados de esta ejecución (las listas se recalculan al volver a ejecutar).

1. **Degeneración de masas.** Los datos de TESS casi no restringen las masas (contracción mediana {T["contr_A_m"].median():.2f} en M_A y {T["contr_B_m"].median():.2f} en M_B): el posterior reproduce el prior de massage.
   ¿Se acepta presentar las masas como *priors propagados*, y las constricciones solo en radios, inclinación y excentricidad? Hay {len(deg)} corridas que no cumplen la convergencia y tienen la masa de B dominada por el prior (sistemas {lista(deg)}): la degeneración de masas es un sospechoso, pero solo la demostré en 264483228 S98,
   donde con más muestreo una cadena se separó a otra región de masa. ¿Se reparametrizan las masas (no centradas) o se conserva la corrida vigente con la advertencia?
2. **Secundarios no detectados.** En {len(nodet)} corridas (sistemas {lista(nodet)}) el secundario no se detectó en 1b y la amplitud de B sale del cuerpo negro de massage; no hay fase observada.
   ¿Se reportan como solo primario, con la excentricidad sin medir?
3. **Radios por debajo del prior.** En {len(radios_bajos)} corridas (sistemas {lista(radios_bajos)}) el radio posterior de A queda más de 3σ por debajo del prior. Es el signo que produciría una profundidad diluida,
   pero también cabría un prior de radio sesgado. Como la tercera luz no entra en el modelo, ¿se presenta esto en la discusión como evidencia indirecta?
4. **e cosω entre sectores.** {q4}
5. **Excentricidad a periodo corto.** {len(ep_corto)} corridas (sistemas {lista(ep_corto)}) tienen e > 0.1 con P < 3 d (sin contar las de fase del secundario mal puesta), donde la circularización predice órbitas circulares.
   ¿Se discuten como candidatos a tercer cuerpo o se tratan como incertidumbre del modelo?
6. **Ajuste conjunto de sectores.** Es la única mejora que forzaría la consistencia geométrica entre sectores, pero cambia la metodología. ¿Se descarta o se explora solo para los sistemas más inconsistentes ({lista(inc_fuerte)})?
7. **Criterio de calidad.** Se mide el ajuste con la significancia empírica (σ_emp) y la fracción de profundidad, en lugar de σ de ruido blanco. ¿Es el criterio adecuado para la tesis?'''))"""),
md(r"""## Referencias de visualización
[1] [Visualization in Bayesian workflow](https://consensus.app/papers/details/0959ffafc71b5647be75e35d376953ba/?utm_source=claude_desktop) (Gabry et al., JRSS A)
[2] [Rank-normalization, folding, and localization: an improved R̂ for assessing convergence of MCMC](https://consensus.app/papers/details/9143d3730b545b2d94fd3e85beb249dc/?utm_source=claude_desktop) (Vehtari et al., Bayesian Analysis)
[3] [corner.py: Scatterplot matrices in Python](https://joss.theoj.org/papers/10.21105/joss.00024) (Foreman-Mackey 2016, JOSS)
[4] [When(ish) is my bus? User-centered visualizations of uncertainty in everyday, mobile predictive systems](https://idl.uw.edu/papers/when-ish-is-my-bus) (Kay, Kola, Hullman y Munson, CHI 2016)
[5] [ArviZ: a modular and flexible library for exploratory analysis of Bayesian models](https://consensus.app/papers/details/d25421691b0757eb96dcd1d2786be9c0/?utm_source=claude_desktop) (Martin et al., JOSS)
[6] [MCMCvis: tools to visualize, manipulate, and summarize MCMC output](https://consensus.app/papers/details/7da83d15fffd5d71b30b408fbf17f2f9/?utm_source=claude_desktop) (Youngflesh 2018, JOSS)
[7] [Data analysis recipes: Using Markov Chain Monte Carlo](https://arxiv.org/abs/1710.06068) (Hogg y Foreman-Mackey 2018, ApJS)"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (starry)", "language": "python", "name": "python3"}
nbf.write(nb, AQUI / "5_diagnostico_resultados_starry.ipynb")
print("escrito", AQUI / "5_diagnostico_resultados_starry.ipynb")
