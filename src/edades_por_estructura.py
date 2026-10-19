"""Edades de las grandes estructuras de Orión (Sánchez-Sanjuán et al. 2024, ec. 2) frente a las edades de MassAge
de nuestras primarias con cuatro modelos (MIST, Somers17, BHAC15 y PARSEC) y sus intervalos del 68 %.

Entradas:
- ~/Documents/catalogos_tesis/catalogo_sergio_orion.csv: miembros de las 13 grandes estructuras de S24.
- documentacion/antecedentes_sistemas/estructuras_s24_por_sistema.csv: estructura asignada a cada sistema (vecinos 5D).
- datos_estrellas_massage_orion/yceb_SpT_PM13_{mist,somers00,somers17,parsec}_results.csv: MassAge con tipo espectral
  (Somers00 y Somers17: modelos SPOTS sin manchas y con 17 % de cobertura).
- ~/Documents/massage_workshop/massage.py/orion_baraffe_results.csv: MassAge con BHAC15 (solo 10 primarias frías).
- edades_vecindad/edades_asignadas.csv: edad de grupo (HDBSCAN) de los sistemas 7 y 20.

Ec. 2 de S24: log(τ/Myr) = 1.499 − 0.665 ΔRP, con ΔRP = (2.6 + 2.1 (BP−RP)) − M_RP (supuesto: positivo por encima de la
línea de 30 Myr), solo para 1.0 < BP−RP < 3.5 y sin corrección por extinción.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diagnostico_orion as D

RAIZ = D.RAIZ
CAT = Path.home() / "Documents/catalogos_tesis/catalogo_sergio_orion.csv"
ASIG = RAIZ / "documentacion/antecedentes_sistemas/estructuras_s24_por_sistema.csv"
MA = RAIZ / "datos_estrellas_massage_orion"
BHAC = Path.home() / "Documents/massage_workshop/massage.py/orion_baraffe_results.csv"
GRUPO = RAIZ / "edades_vecindad/edades_asignadas.csv"
NOMBRES = {1: "λ Ori", 2: "Ori-North", 3: "Briceño-1A", 4: "Briceño-1B", 5: "Ori-East", 6: "OBP-Far", 7: "σ Ori",
           8: "OBP-b", 9: "OBP-d", 10: "OBP-Near", 11: "ONC", 12: "Ori-South", 13: "Orion Y"}
MODELOS = [  # (nombre, color, marcador, relleno)
    ("MIST", D.AZUL, "o", True), ("Somers00", D.NARANJA, "s", False), ("Somers17", D.NARANJA, "s", True),
    ("BHAC15", D.AQUA, "^", True), ("PARSEC", "#6a6a65", "D", True)]


def edades_estructuras():
    c = pd.read_csv(CAT)
    col = c.BP - c.RP
    mrp = c.RP + 5 * np.log10(c.PLX / 100)
    c["logt"] = 1.499 - 0.665 * ((2.6 + 2.1 * col) - mrp)
    s = c[(col > 1.0) & (col < 3.5)]
    E = s.groupby("Cluster").logt.agg(n="size", p16=lambda v: 10 ** np.percentile(v, 16),
                                      med=lambda v: 10 ** np.median(v), p84=lambda v: 10 ** np.percentile(v, 84))
    E["estructura"] = E.index.map(NOMBRES)
    return E.reset_index()


def _massage(ruta):
    d = pd.read_csv(ruta, dtype=str)                      # los source_id de Gaia pierden precisión como float
    d.columns = d.columns.str.strip()
    d["OBJECT"] = d["OBJECT"].str.strip().str.replace(r"\.0$", "", regex=True)
    d = d.set_index("OBJECT")[["age_median", "age_low", "age_upp"]]
    return d.apply(pd.to_numeric, errors="coerce")       # algunas filas traen "nan" como texto


def edades_sistemas():
    A = pd.read_csv(ASIG)
    A = A[A.clasificacion.isin(["miembro consistente", "borde del grupo"])].copy()
    macro = pd.read_csv(MA / "Macro_21aug2026.csv", dtype={"Source": str})[["TIC", "Source"]].set_index("TIC")
    A["Source"] = A.TIC.map(macro.Source)
    A["TIC_txt"] = A.TIC.astype(str)
    # sin tipo espectral (sistema 8): corridas con T_eff del TIC. Se usan los archivos por modelo y NO
    # catalogo_completo_mist_sommers.csv, que para 8, 13 y 14 tiene MIST y Somers17 intercambiados.
    def con_teff(spt, teff):
        a, b = _massage(MA / spt), _massage(MA / teff)
        return pd.concat([a, b[~b.index.isin(a.dropna().index)]]).groupby(level=0).first()
    tablas = {"MIST": (con_teff("yceb_SpT_PM13_mist_results.csv", "yceb_Teff_mist_results.csv"), "Source"),
              "Somers17": (con_teff("yceb_SpT_PM13_somers17_results.csv", "yceb_Teff_somers17_results.csv"), "Source"),
              "Somers00": (_massage(MA / "yceb_SpT_PM13_somers00_results.csv"), "Source"),
              "PARSEC": (_massage(MA / "yceb_SpT_PM13_parsec_results.csv"), "Source"),
              "BHAC15": (_massage(BHAC), "TIC_txt")}
    for m, (t, clave) in tablas.items():
        for q in ("age_median", "age_low", "age_upp"):
            A[f"{m}_{q.split('_')[1]}"] = A[clave].map(t[q])
    g = pd.read_csv(GRUPO).set_index("TIC")
    A["grupo_med"], A["grupo_p16"], A["grupo_p84"] = (A.TIC.map(g.edad_grupo_Myr), A.TIC.map(g.p16_Myr),
                                                     A.TIC.map(g.p84_Myr))
    return A


def figura(E, A):
    D.estilo()
    E = E.set_index("estructura")
    A = A.assign(med_est=A.grande.map(E.med)).sort_values(["med_est", "grande", "sistema"])
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(A) + 2.2))
    y = np.arange(len(A))[::-1]
    offs = {"MIST": 0.28, "Somers00": 0.14, "Somers17": 0.0, "BHAC15": -0.14, "PARSEC": -0.28}
    previo = None
    for yi, (_, r) in zip(y, A.iterrows()):
        e = E.loc[r.grande]
        ax.fill_betweenx([yi - 0.38, yi + 0.38], e.p16, e.p84, color="#e6e5e1", lw=0, zorder=1)
        ax.plot([e.med, e.med], [yi - 0.38, yi + 0.38], color=D.TINTA2, lw=2, zorder=2)
        if previo is not None and previo != r.grande:
            ax.axhline(yi + 0.5, color="#cfcdc7", lw=0.8, zorder=0)
        previo = r.grande
        for m, color, mk, lleno in MODELOS:
            med, lo, up = r[f"{m}_median"], r[f"{m}_low"], r[f"{m}_upp"]
            if pd.notna(med) and med > 0:
                xe = [[max(med - lo, 0)], [max(up - med, 0)]] if pd.notna(lo) and pd.notna(up) else None
                ax.errorbar(med, yi + offs[m], xerr=xe, fmt=mk, ms=7, color=color, mfc=color if lleno else D.SUPERFICIE,
                            mec=D.SUPERFICIE if lleno else color, mew=0.8 if lleno else 1.5, elinewidth=1.2,
                            ls=(0, (2, 1.5)) if not lleno else "-", capsize=0, zorder=4)
        if pd.notna(r.grupo_med):
            ax.errorbar(r.grupo_med, yi, xerr=[[r.grupo_med - r.grupo_p16], [r.grupo_p84 - r.grupo_med]], fmt="*",
                        ms=14, color=D.TINTA, mec=D.SUPERFICIE, mew=0.8, elinewidth=1.4, capsize=0, zorder=5)
    borde = A.clasificacion.eq("borde del grupo").to_numpy()
    sin_edad = A[[f"{m}_median" for m, *_ in MODELOS] + ["grupo_med"]].isna().all(axis=1).to_numpy()
    etiquetas = [f"sist. {s}{' (borde)' if b else ''}{' (sin edad)' if n else ''}" for s, b, n in zip(A.sistema, borde, sin_edad)]
    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas, fontsize=9.5)
    # nombre de la estructura a la derecha, una vez por grupo
    for est, sub in A.assign(y=y).groupby("grande", sort=False):
        e = E.loc[est]
        ax.annotate(f"{est}\n{e.med:.1f} Myr [{e.p16:.1f}–{e.p84:.1f}]", (1.01, (sub.y.mean() - y.min() + 0.5) / len(A)),
                    xycoords="axes fraction", fontsize=9, color=D.TINTA, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 600)
    ax.set_ylim(-0.6, len(A) - 0.4)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30", "100", "300"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("edad [Myr]")
    ax.set_title("Edad de la estructura de S24 (banda gris: 16–84 %, línea: mediana) frente a MassAge por modelo", fontsize=11)
    h = [plt.Line2D([], [], marker=mk, ls="-", color=c, mfc=c if ll else D.SUPERFICIE, mec=D.SUPERFICIE if ll else c,
                    mew=0.8 if ll else 1.5, ms=7, label=f"MassAge {m} (mediana, 68 %)") for m, c, mk, ll in MODELOS]
    h += [plt.Line2D([], [], marker="*", ls="-", color=D.TINTA, ms=12, label="edad de nuestro grupo HDBSCAN (7, 20)"),
          plt.matplotlib.patches.Patch(color="#e6e5e1", label="estructura S24, ec. 2: percentiles 16–84 %"),
          plt.Line2D([], [], color=D.TINTA2, lw=2, label="estructura S24, ec. 2: mediana")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.09), ncol=3, fontsize=8.5, frameon=False)
    fig.text(0.01, 0.005, "Ec. 2 de S24 aplicada por nosotros a los miembros de las grandes estructuras (signo de ΔRP supuesto; "
             "sin corrección por extinción).\nSomers00, Somers17 y BHAC15 no cubren primarias calientes; BHAC15 (no está en datos_estrellas_massage_orion/) viene de la corrida de mayo en massage_workshop/, solo 10 primarias.\nSistema 8: MassAge con T_eff (sin tipo espectral).",
             fontsize=7.5, color=D.TINTA2)
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    return fig


if __name__ == "__main__":
    E, A = edades_estructuras(), edades_sistemas()
    E.round(3).to_csv(D.AQUI / "edades_estructuras_s24.csv", index=False)
    cols = ["sistema", "TIC", "grande", "clasificacion"] + [f"{m}_{q}" for m, *_ in MODELOS for q in ("median", "low", "upp")] + [
        "grupo_med", "grupo_p16", "grupo_p84"]
    A[cols].round(3).to_csv(D.AQUI / "edades_sistemas_por_modelo.csv", index=False)
    fig = figura(E, A)
    for ext in ("png", "pdf"):
        fig.savefig(D.AQUI / "figuras" / f"15_edades_por_estructura.{ext}", dpi=170, bbox_inches="tight")
    pd.set_option("display.width", 250)
    print(A[cols].round(2).to_string(index=False))
