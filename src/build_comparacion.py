"""Construye comparacion_original_vs_corregida.ipynb."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
celdas = [
md(r"""# Corridas de starry: original vs corregida

Compara cada corrida de `starry_correcciones/` con la misma corrida de `starry_por_tic/`, con las mismas
métricas (`metricas_starry.py`):

- **Convergencia:** $\hat R$ máximo (bien si $<1.01$), ESS mínimo (bien si $>400$) y fracción de transiciones
  divergentes (bien si $<1\,\%$).
- **Ajuste:** residuo mediano (datos − modelo) en el núcleo de cada eclipse, en unidades de su error
  ($\sigma$ del ruido fuera de eclipse $/\sqrt{n}$), y $\chi^2$ por punto dentro del eclipse. Un residuo de
  $|r|>3\sigma$ indica que el modelo no reproduce la profundidad.
- **Consistencia entre sectores:** dispersión de $k=R_B/R_A$, de la inclinación y de $e\cos\omega$,
  $e\sin\omega$ entre los sectores de un mismo sistema. La geometría no cambia de un sector a otro, así que
  una dispersión mayor que los errores indica un problema del ajuste.

En el conjunto "corregido" de cada sistema se usan las corridas nuevas donde existen y las originales
(que ya estaban bien) en los demás sectores."""),
code(r"""import sys
from pathlib import Path
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AQUI = Path("/home/luciam/Documents/starry/Resultados_Orion/starry_correcciones")
sys.path.insert(0, str(AQUI))
import metricas_starry as ms

orig = ms.tabla(AQUI.parent / "starry_por_tic")
corr = ms.tabla(AQUI)
indice = pd.read_csv(AQUI / "indice_correcciones.csv")
# en starry_por_tic puede haber dos corridas del mismo sector (S6 y S6_ecc): se usa la más reciente
orig["mtime"] = [Path(c).stat().st_mtime for c in orig["carpeta"]]
orig = orig.sort_values("mtime").drop_duplicates(["TIC", "Sector"], keep="last")
print(f"{len(orig)} corridas originales, {len(corr)} corregidas")"""),
md("## 1) Corrida por corrida"),
code(r"""cols = ["rhat_max", "ess_min", "div_frac", "res_prim_sigma", "res_sec_sigma", "chi2_prim", "chi2_sec",
        "k", "B_inc_red", "B_ecc", "B_w", "B_amp"]
comp = indice[["TIC", "Sector"]].merge(orig[["TIC", "Sector"] + cols], on=["TIC", "Sector"], how="left") \
    .merge(corr[["TIC", "Sector"] + cols], on=["TIC", "Sector"], how="left", suffixes=("_orig", "_corr"))


def estado(r, suf):
    if not np.isfinite(r[f"rhat_max_{suf}"]):
        return "sin corrida"
    conv = r[f"rhat_max_{suf}"] < 1.01 and r[f"ess_min_{suf}"] > 400 and r[f"div_frac_{suf}"] < 0.01
    marg = r[f"rhat_max_{suf}"] < 1.05 and r[f"ess_min_{suf}"] > 100
    res = max(abs(np.nan_to_num(r[f"res_prim_sigma_{suf}"])), abs(np.nan_to_num(r[f"res_sec_sigma_{suf}"])))
    ajuste = "ajuste ok" if res < 3 else ("ajuste regular" if res < 6 else "ajuste malo")
    return ("converge" if conv else "marginal" if marg else "NO converge") + " · " + ajuste


comp["estado_orig"] = comp.apply(lambda r: estado(r, "orig"), axis=1)
comp["estado_corr"] = comp.apply(lambda r: estado(r, "corr"), axis=1)
comp.to_csv(AQUI / "comparacion_corridas.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
comp[["TIC", "Sector", "estado_orig", "estado_corr", "rhat_max_orig", "rhat_max_corr", "ess_min_orig",
      "ess_min_corr", "div_frac_orig", "div_frac_corr", "res_sec_sigma_orig", "res_sec_sigma_corr"]].round(3)"""),
code(r"""# figura: cada flecha va de la corrida original a la corregida
fig, axs = plt.subplots(1, 4, figsize=(20, 0.33 * len(comp) + 2), sharey=True)
etiquetas = [f"TIC {t} S{s}" for t, s in zip(comp["TIC"], comp["Sector"])]
y = np.arange(len(comp))
for ax, (col, titulo, escala, umbral) in zip(axs, [
        ("rhat_max", r"$\hat R$ máximo", "log", 1.01), ("ess_min", "ESS mínimo", "log", 400),
        ("div_frac", "fracción divergente", "symlog", 0.01),
        ("res_sec_sigma", r"residuo en el secundario [$\sigma$]", "symlog", 3)]):
    a, b = comp[f"{col}_orig"].to_numpy(float), comp[f"{col}_corr"].to_numpy(float)
    if col == "rhat_max":
        a, b = a - 1, b - 1
        umbral = umbral - 1
        titulo = r"$\hat R - 1$"
    if col == "res_sec_sigma":
        a, b = np.abs(a), np.abs(b)
        titulo = r"|residuo| en el secundario [$\sigma$]"
    for yi, ai, bi in zip(y, a, b):
        if np.isfinite(ai) and np.isfinite(bi):
            ax.plot([ai, bi], [yi, yi], "-", color="0.6", lw=1, zorder=1)
    ax.plot(a, y, "o", color="#c0392b", label="original")
    ax.plot(b, y, "o", color="#1b7f3a", label="corregida")
    ax.axvline(umbral, color="0.3", ls="--", lw=1)
    ax.set_xscale(escala, **({"linthresh": 1e-3} if escala == "symlog" and col == "div_frac" else
                             {"linthresh": 1} if escala == "symlog" else {}))
    ax.set_title(titulo); ax.grid(alpha=0.25)
axs[0].set_yticks(y); axs[0].set_yticklabels(etiquetas, fontsize=8); axs[0].invert_yaxis()
axs[0].legend(loc="lower right", fontsize=8)
fig.suptitle("Original (rojo) → corregida (verde); la línea discontinua es el umbral de aceptación", y=1.0)
fig.tight_layout()
fig.savefig(AQUI / "comparacion_metricas.png", dpi=120, bbox_inches="tight")
plt.show()"""),
md(r"""## 2) Consistencia entre sectores

Para cada sistema corregido: $k$, inclinación, $e\cos\omega$ y $e\sin\omega$ por sector, en el conjunto
original y en el corregido. Se cuantifica con la dispersión entre sectores dividida por el error típico
de un sector ($\chi^2_\nu$ respecto de la media pesada); $\approx1$ significa consistente."""),
code(r"""def conjunto(tabla_base, tabla_nueva, tic):
    base = tabla_base.loc[tabla_base["TIC"] == tic].set_index("Sector")
    if tabla_nueva is not None:
        nueva = tabla_nueva.loc[tabla_nueva["TIC"] == tic].set_index("Sector")
        base = pd.concat([base.drop(index=nueva.index, errors="ignore"), nueva]).sort_index()
    return base


def chi2_entre(valores, errores):
    v, e = np.asarray(valores, float), np.asarray(errores, float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    v, e = v[ok], e[ok]
    if len(v) < 2:
        return np.nan
    w = 1 / e ** 2
    m = np.sum(w * v) / w.sum()
    return float(np.sum(w * (v - m) ** 2) / (len(v) - 1))


def errores_derivados(t):
    t = t.copy()
    t["k_sd"] = t["k"] * np.hypot(t["A_r_sd"] / t["A_r"], t["B_r_sd"] / t["B_r"])
    # error aproximado de e cos w y e sin w a partir de e y w
    w = np.radians(t["B_w"]); sw = np.radians(t["B_w_sd"])
    t["ecosw_sd"] = np.hypot(t["B_ecc_sd"] * np.cos(w), t["B_ecc"] * np.sin(w) * sw)
    t["esinw_sd"] = np.hypot(t["B_ecc_sd"] * np.sin(w), t["B_ecc"] * np.cos(w) * sw)
    return t


orig_e, corr_e = errores_derivados(orig), errores_derivados(corr)
filas = []
for tic in sorted(indice["TIC"].unique()):
    for nombre, t in (("original", conjunto(orig_e, None, tic)), ("corregido", conjunto(orig_e, corr_e, tic))):
        fila = dict(TIC=tic, conjunto=nombre, sectores=",".join(map(str, t.index)))
        for p, e in (("k", "k_sd"), ("B_inc_red", "B_inc_sd"), ("ecosw", "ecosw_sd"), ("esinw", "esinw_sd")):
            fila[f"{p}"] = " / ".join(f"{x:.3f}" for x in t[p])
            fila[f"chi2_{p}"] = chi2_entre(t[p], t[e])
        filas.append(fila)
cons = pd.DataFrame(filas)
cons.to_csv(AQUI / "consistencia_entre_sectores.csv", index=False)
cons.round(2)"""),
code(r"""tics = sorted(indice["TIC"].unique())
fig, axs = plt.subplots(len(tics), 4, figsize=(18, 2.6 * len(tics)), squeeze=False)
for i, tic in enumerate(tics):
    for nombre, t, col, dx in (("original", conjunto(orig_e, None, tic), "#c0392b", -0.08),
                               ("corregido", conjunto(orig_e, corr_e, tic), "#1b7f3a", 0.08)):
        x = np.arange(len(t)) + dx
        for j, (p, e, et) in enumerate((("k", "k_sd", "k = R_B/R_A"), ("B_inc_red", "B_inc_sd", "inclinación [°]"),
                                        ("ecosw", "ecosw_sd", "e cos ω"), ("esinw", "esinw_sd", "e sin ω"))):
            axs[i][j].errorbar(x, t[p], yerr=t[e], fmt="o", color=col, ms=5, capsize=2, label=nombre)
            axs[i][j].set_xticks(np.arange(len(t))); axs[i][j].set_xticklabels([f"S{s}" for s in t.index])
            if i == 0:
                axs[i][j].set_title(et)
    axs[i][0].set_ylabel(f"TIC {tic}", fontsize=9)
axs[0][0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(AQUI / "comparacion_consistencia.png", dpi=110, bbox_inches="tight")
plt.show()"""),
md(r"""## 3) Modelo frente a datos

Curva plegada de cada corrida corregida (datos en negro, modelo original en rojo, corregido en verde),
centrada en el primario y en el secundario."""),
code(r"""def plegar_modelo(carpeta, P, t0):
    out = Path(carpeta)
    Q = pd.read_csv(next(out.glob("*_q50.csv")))
    D = pd.read_csv(next(out.glob("*_trending.csv")))
    M = pd.merge_asof(D.sort_values("time"), Q.sort_values("time").rename(columns={"flux": "modelo"}),
                      on="time", direction="nearest", tolerance=1e-4).dropna()
    return (((M.time - t0) / P) % 1).to_numpy(), M.flat_flux.to_numpy(), M.modelo.to_numpy()


n = len(comp)
fig, axs = plt.subplots(n, 2, figsize=(13, 2.4 * n), squeeze=False)
for i, r in enumerate(comp.itertuples()):
    co = corr.loc[(corr.TIC == r.TIC) & (corr.Sector == r.Sector)]
    og = orig.loc[(orig.TIC == r.TIC) & (orig.Sector == r.Sector)]
    if co.empty:
        continue
    P, t0 = co["B_porb"].iloc[0], co["B_t0"].iloc[0]
    ph, f, mc = plegar_modelo(co["carpeta"].iloc[0], P, t0)
    for ax, c in ((axs[i][0], 0.0), (axs[i][1], 0.5)):
        x = (ph - c + 0.5) % 1 - 0.5
        o = np.argsort(x)
        ax.plot(x, f, ".", color="0.3", ms=1.5, alpha=0.4)
        ax.plot(x[o], mc[o], "-", color="#1b7f3a", lw=1.5, label="corregido")
        if len(og):
            pho, fo, mo = plegar_modelo(og["carpeta"].iloc[0], P, t0)
            xo = (pho - c + 0.5) % 1 - 0.5
            oo = np.argsort(xo)
            ax.plot(xo[oo], mo[oo], "-", color="#c0392b", lw=1, alpha=0.8, label="original")
        ax.set_xlim(-0.12, 0.12)
    axs[i][0].set_ylabel(f"TIC {r.TIC} S{r.Sector}", fontsize=8)
axs[0][0].set_title("primario"); axs[0][1].set_title("secundario (fase 0.5)")
axs[0][0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(AQUI / "comparacion_modelos.png", dpi=100, bbox_inches="tight")
plt.show()"""),
md("## 4) Resumen"),
code(r"""resumen = comp.groupby(["estado_orig", "estado_corr"]).size().rename("corridas").reset_index()
print(resumen.to_string(index=False))
mejora = cons.pivot(index="TIC", columns="conjunto", values=["chi2_k", "chi2_B_inc_red", "chi2_ecosw", "chi2_esinw"])
mejora.round(1)"""),
]
nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, "/home/luciam/Documents/starry/Resultados_Orion/starry_correcciones/comparacion_original_vs_corregida.ipynb")
