"""Genera 4_seleccion_visual.ipynb: cuaderno interactivo de selección visual (estilo ciencia ciudadana).
Kernel: python3 del entorno starry (tiene ipywidgets y JupyterLab). Uso: python build_cuaderno_seleccion.py
"""
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
AQUI = Path(__file__).resolve().parent

celdas = [
md(r"""# Pléyades 4: selección visual interactiva

Cada curva se muestra **plegada y binada** con las curvas ya corregidas por cotrending. Clasifica con los botones
o con el teclado (**1–6**, **←** para volver). Cada respuesta se guarda al instante en `seleccion_visual.csv`, así que
puedes cerrar y seguir después.

**Qué mirar** (rúbrica):
1. El eclipse **se repite** en los distintos sectores/campañas y cae en la **misma fase**.
2. Profundidad claramente por encima del ruido (los puntos binados se separan de la línea base).
3. **Forma** plausible de eclipse (fondo plano o en V), no una oscilación continua.
4. Si hay secundario, cerca de la **fase 0.5**; mínimos impares y pares parecidos (panel de 2P).
5. **No** es una oscilación suave (rotación, pulsación) ni un salto que aparece en un solo sector (sistemático).
6. Contaminación (`Rcont`) baja, o razón para creer que el eclipse es de esta estrella.

**Modo ciego:** la cola mezcla las candidatas del triage con binarias conocidas (sin etiqueta) y con curvas al azar
que el triage descartó. Al final, la última celda compara tus respuestas con esas referencias para estimar
cuántas binarias recuperas y cuántos falsos positivos dejas pasar."""),
code(r"""%matplotlib inline
import re, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, clear_output, Javascript

BASE = Path("/home/luciam/Documents/starry/Resultados_Pleyades")
COT = BASE / "cotrending"; TRI = COT / "triage"; EST = TRI / "estrellas"
ORIG = BASE / "tessextractor_local/resultados/dat"; CORR = COT / "dat_cotrended"
SEL_CSV = COT / "seleccion_visual.csv"

# --- parámetros de la cola (ajustables) ---
SNR_MIN, SDE_MIN = 5.0, 6.0      # umbrales del triage: profundidad/error del primario y SDE de TLS
P_MAX, N_EVENTOS_MIN = 12.0, 3     # los periodos de 15-30 d son casi siempre periodos de 10-15 d duplicados por ruido rojo; con < 3 eventos no hay repetición
N_NEGATIVAS = 40                  # curvas al azar descartadas por el triage, como control de falsos positivos
CIEGO = True                      # oculta TIC y marcas de catálogo hasta el final
ORDEN = "aleatorio"               # "aleatorio" (recomendado en modo ciego) o "por_snr"
SEMILLA = 7"""),
code(r"""ruta, parcial = TRI / "triage_tabla.csv", TRI / "triage_tabla_parcial.csv"
if not ruta.exists() and not parcial.exists():
    raise FileNotFoundError("Aún no existe el triage. Ejecuta primero (en la terminal): "
                            "python cotrending/triage_cotrending.py   y espera a que cree cotrending/triage/triage_tabla.csv")
if not ruta.exists():
    print("AVISO: usando la tabla PARCIAL del triage (aún no termina); la cola cambiará al terminar.")
tab = pd.read_csv(ruta if ruta.exists() else parcial)
if "error" in tab: tab = tab[tab["error"].isna()]
tab = tab[(TRI / "estrellas").exists() & tab["TIC"].apply(lambda t: (EST / f"TIC_{int(t)}.npz").exists())].copy()
cat = pd.read_csv(BASE / "membresia/binarias_eclipsantes_conocidas.csv").dropna(subset=["TIC"])
conocidas = set(cat["TIC"].astype(int))
tab["conocida"] = tab["TIC"].isin(conocidas)

pasa = (tab["snr_prim"] >= SNR_MIN) & (tab["SDE_TLS"] >= SDE_MIN) & (~tab["P_orbita_TESS"].astype(bool)) & (tab["P"] <= P_MAX) & (tab["n_eventos"] >= N_EVENTOS_MIN)
cola = pd.concat([tab[pasa].assign(origen="candidata"),
                  tab[~pasa & tab["conocida"]].assign(origen="control_conocida"),
                  tab[~pasa & ~tab["conocida"]].sample(min(N_NEGATIVAS, int((~pasa & ~tab["conocida"]).sum())),
                                                       random_state=SEMILLA).assign(origen="control_negativa")])
cola = (cola.sample(frac=1, random_state=SEMILLA) if ORDEN == "aleatorio" else cola.sort_values("snr_prim", ascending=False)).reset_index(drop=True)
cola["n"] = np.arange(1, len(cola) + 1)
print(f"{len(tab)} estrellas en el triage; {int(pasa.sum())} pasan los umbrales (de ellas {int((pasa & tab['conocida']).sum())} conocidas)")
print(cola["origen"].value_counts().to_string())
print(f"cola total: {len(cola)} curvas")"""),
code(r"""CLASES = [("1", "Binaria eclipsante", "EB", "#2e7d32"), ("2", "Candidata", "CAND", "#f9a825"),
          ("3", "Variable (rotación/pulsación)", "VAR", "#1565c0"), ("4", "Sistemático/artefacto", "SIST", "#6d4c41"),
          ("5", "Sin señal", "NADA", "#757575"), ("6", "No sé", "NS", "#8e24aa")]

def cargar_serie(tic, carpeta):
    out = []
    for f in sorted(carpeta.glob(f"TIC_{int(tic)}_S_*_LC.dat")):
        d = np.loadtxt(f, usecols=(0, 1), ndmin=2)
        d = d[np.isfinite(d).all(axis=1)]
        out.append((int(re.search(r"_S_(\d+)_", f.name)[1]), d[:, 0], 10 ** (-0.4 * (d[:, 1] - np.median(d[:, 1])))))
    return out

def binear_t(t, y, horas=1.0):
    k = np.floor((t - t.min()) / (horas / 24)).astype(int)
    g = pd.DataFrame(dict(k=k, t=t, y=y)).groupby("k")
    return g["t"].mean().to_numpy(), g["y"].median().to_numpy()

def binear_fase(ph, r, nb=60, lo=-0.25, hi=0.75):
    b = np.linspace(lo, hi, nb + 1); k = np.digitize(ph, b) - 1
    med = np.full(nb, np.nan); err = np.full(nb, np.nan)
    for i in range(nb):
        v = r[k == i]
        if len(v) >= 5:
            med[i] = np.median(v); err[i] = 1.2533 * 1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(len(v))
    return 0.5 * (b[1:] + b[:-1]), med, err

COL = plt.cm.tab10.colors
def dibujar(fila):
    e = np.load(EST / f"TIC_{int(fila.TIC)}.npz")
    ph, pho, r, s, P = e["ph"], e["ph_otro"], e["r"], e["s"], float(e["P"])
    sec = sorted(np.unique(s))
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.42, wspace=0.22)
    # (a) series temporales por campaña: original (gris) y con cotrending (color), binadas a 1 h
    corr, orig = cargar_serie(fila.TIC, CORR), cargar_serie(fila.TIC, ORIG)
    orden = sorted(range(len(corr)), key=lambda i: corr[i][1].min()); grupos = []
    for i in orden:
        if grupos and corr[i][1].min() - max(corr[j][1].max() for j in grupos[-1]) < 60: grupos[-1].append(i)
        else: grupos.append([i])
    anchos = [max(corr[j][1].max() for j in g) - min(corr[j][1].min() for j in g) + 2 for g in grupos]
    sub = gs[0, :].subgridspec(1, len(grupos), width_ratios=anchos, wspace=0.04)
    axs = []
    for k, g in enumerate(grupos):
        ax = fig.add_subplot(sub[0, k], sharey=axs[0] if axs else None); axs.append(ax)
        for i in g:
            sc, tc, fc = corr[i]
            to = [o for o in orig if o[0] == sc]
            if to:
                tb, yb = binear_t(to[0][1], to[0][2]); ax.plot(tb, yb, ".", ms=2, color="0.75", label="original" if (k == 0 and i == g[0]) else None)
            tb, yb = binear_t(tc, fc); ax.plot(tb, yb, ".", ms=3, color=COL[sec.index(sc) % 10] if sc in sec else "C0", label=f"S{sc}")
        ax.legend(ncol=3, fontsize=7, loc="upper right"); ax.set_xlabel("tiempo [BTJD]", fontsize=8)
        if k: plt.setp(ax.get_yticklabels(), visible=False)
    axs[0].set_ylabel("flujo (1 h)"); axs[0].set_title("gris: original   color: con cotrending", fontsize=9, loc="left")
    # (b)(c)(d) plegados (60 bins, banda = error de la mediana)
    def plegado(a, p, rr, ss, titulo):
        for i, sc in enumerate(sec):
            m = ss == sc; a.plot(p[m], 1 + rr[m], ".", ms=1.5, alpha=0.25, color=COL[i % 10])
        x, y, ey = binear_fase(p, rr); a.fill_between(x, 1 + y - ey, 1 + y + ey, color="k", alpha=0.25, lw=0); a.plot(x, 1 + y, "k-", lw=1.3)
        a.axvline(0, c="r", lw=0.6, ls=":"); a.axvline(0.5, c="b", lw=0.6, ls=":")
        yy = y[np.isfinite(y)]; d = (yy.max() - yy.min()) or 1e-3
        a.set_ylim(1 + yy.min() - 0.7 * d, 1 + yy.max() + 0.7 * d); a.set_xlim(-0.25, 0.75); a.set_title(titulo, fontsize=9); a.set_xlabel("fase")
    plegado(fig.add_subplot(gs[1, 0]), ph, r, s, f"plegado en P = {P:.4f} d")
    dup = bool(fila.periodo_duplicado)
    plegado(fig.add_subplot(gs[1, 1]), pho, r, s, f"plegado en {'P/2' if dup else '2P'} (impares/pares)")
    a = fig.add_subplot(gs[1, 2]); plegado(a, ph, r, s, "zoom (solo la curva binada)")
    x, y, ey = binear_fase(ph, r); ok = np.isfinite(y); lo, hi = np.min((y - ey)[ok]), np.max((y + ey)[ok]); d = (hi - lo) or 1e-3
    a.set_ylim(1 + lo - 0.15 * d, 1 + hi + 0.15 * d)
    titulo = f"Curva #{int(fila.n)}" if CIEGO else f"Curva #{int(fila.n)} · TIC {int(fila.TIC)}"
    info = (f"Tmag {fila.Tmag:.2f}   Rcont {fila.Rcont:.3f}   P = {P:.4f} d   dur = {fila.dur_h:.1f} h   SDE = {fila.SDE_TLS:.1f}   "
            f"prof/err = {fila.snr_prim:.1f}   sec = {fila.sig_sec:.1f}σ   impar-par = {fila.sig_impar_par:.1f}σ   "
            f"P_rot = {fila.P_rot:.2f} d ({fila.amp_rot_ppt:.1f} ppt)   sectores {fila.sectores}")
    if not CIEGO and fila.conocida: info += "   [binaria conocida]"
    fig.suptitle(titulo + "\n" + info, fontsize=10, y=0.995)
    return fig"""),
code(r"""# --- estado y guardado ---
cols = ["n", "TIC", "clase", "confianza", "nota", "origen", "conocida", "segundos", "hora"]
if SEL_CSV.exists():
    previo = pd.read_csv(SEL_CSV)
else:
    previo = pd.DataFrame(columns=cols); previo.to_csv(SEL_CSV, index=False)
hechas = set(previo["TIC"].astype(int))      # se identifica por TIC: la cola puede cambiar cuando termine el triage
pendientes = [i for i in range(len(cola)) if int(cola.loc[i, "TIC"]) not in hechas]
estado = dict(pos=0, t0=time.time())

salida = W.Output()
progreso = W.IntProgress(min=0, max=len(cola), value=min(len(hechas), len(cola)), description="avance", layout=W.Layout(width="420px"))
etiqueta = W.HTML()
conf = W.ToggleButtons(options=[("segura", 3), ("probable", 2), ("dudosa", 1)], value=2, description="confianza")
nota = W.Text(placeholder="nota opcional (p. ej. 'secundario en 0.5', 'salto en S44')", layout=W.Layout(width="520px"))
botones = []

def mostrar():
    with salida:
        clear_output(wait=True)
        if estado["pos"] >= len(pendientes):
            print("Terminaste la cola. Ejecuta la última celda para ver el resumen."); return
        fila = cola.loc[pendientes[estado["pos"]]]
        fig = dibujar(fila); plt.show(); plt.close(fig)
    progreso.value = len(cola) - (len(pendientes) - estado["pos"])
    etiqueta.value = f"<b>{progreso.value}/{len(cola)}</b> clasificadas · quedan {len(pendientes) - estado['pos']}"
    estado["t0"] = time.time()

def guardar(clase):
    if estado["pos"] >= len(pendientes): return
    fila = cola.loc[pendientes[estado["pos"]]]
    reg = dict(n=int(fila.n), TIC=int(fila.TIC), clase=clase, confianza=conf.value, nota=nota.value, origen=fila.origen,
               conocida=bool(fila.conocida), segundos=round(time.time() - estado["t0"], 1), hora=time.strftime("%Y-%m-%d %H:%M:%S"))
    pd.DataFrame([reg]).to_csv(SEL_CSV, mode="a", header=False, index=False)
    nota.value = ""; conf.value = 2
    estado["pos"] += 1; mostrar()

def atras(_=None):
    if estado["pos"] == 0: return
    d = pd.read_csv(SEL_CSV); d = d.iloc[:-1]; d.to_csv(SEL_CSV, index=False)
    estado["pos"] -= 1; mostrar()

for tecla, nombre, clave, color in CLASES:
    b = W.Button(description=f"{tecla} · {nombre}", layout=W.Layout(width="auto"), style=dict(button_color=color, text_color="white"))
    b.add_class(f"sel-{tecla}"); b.on_click(lambda _, c=clave: guardar(c)); botones.append(b)
bat = W.Button(description="← atrás", layout=W.Layout(width="90px")); bat.add_class("sel-atras"); bat.on_click(atras)

display(W.VBox([W.HBox([progreso, etiqueta]), salida, W.HBox(botones + [bat]), W.HBox([conf, nota])]))
display(Javascript('''
if (!window.__sel_keys) { window.__sel_keys = true;
  document.addEventListener("keydown", function(e) {
    const tag = document.activeElement.tagName; if (tag === "INPUT" || tag === "TEXTAREA") return;
    let b = null;
    if (e.key >= "1" && e.key <= "6") b = document.querySelector(".sel-" + e.key);
    if (e.key === "ArrowLeft") b = document.querySelector(".sel-atras");
    if (b) { b.click(); e.preventDefault(); }
  });
}'''))
mostrar()"""),
md(r"""## Resumen y calibración (ejecutar al terminar)
Compara tus respuestas con las referencias ocultas: ¿qué fracción de las **binarias conocidas** marcaste como
EB/candidata?, ¿cuántas de las curvas **al azar** marcaste como EB/candidata (falsos positivos)? y ¿cuántas
de las **candidatas del triage** conservas?"""),
code(r"""d = pd.read_csv(SEL_CSV)
if len(d) == 0:
    print("Aún no hay clasificaciones.")
else:
    pos = d["clase"].isin(["EB", "CAND"])
    print(d["clase"].value_counts().to_string()); print()
    for og, nombre in [("control_conocida", "binarias conocidas (control)"), ("control_negativa", "curvas al azar (control)"), ("candidata", "candidatas del triage")]:
        s = d[d["origen"] == og]
        if len(s): print(f"{nombre}: {int(pos[s.index].sum())}/{len(s)} marcadas EB/candidata ({pos[s.index].mean():.0%})")
    kn = d[d["conocida"]]
    if len(kn): print(f"\nconocidas recuperadas (cualquier origen): {int(pos[kn.index].sum())}/{len(kn)}")
    print(f"\ntiempo mediano por curva: {d['segundos'].median():.0f} s")
    display(d[pos].sort_values("confianza", ascending=False)[["n", "TIC", "clase", "confianza", "nota", "origen"]])"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (starry)", "language": "python", "name": "python3"}
nbf.write(nb, AQUI / "4_seleccion_visual.ipynb")
print("escrito", AQUI / "4_seleccion_visual.ipynb")
