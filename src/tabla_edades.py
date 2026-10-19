"""Tabla de edades por sistema y por modelo/corrida de MassAge (Orión y Pléyades).
Salidas: edades_largo.csv (una fila por sistema y fuente) y edades_ancho.csv (resumen por sistema)."""
import pandas as pd, numpy as np
from pathlib import Path
O = Path("/home/luciam/Documents/starry/Resultados_Orion"); M = O / "datos_estrellas_massage_orion"
P = Path("/home/luciam/Documents/starry/Resultados_Pleyades/pipeline/massage")
OUT = Path(__file__).parent

def leer(f):
    d = pd.read_csv(f, sep=None, engine="python"); d.columns = [c.strip() for c in d.columns]; return d

muestra = pd.read_csv(O / "diagnostico_resultados/muestra_completa_orion_pleyades.csv")
cs = [c for c in muestra.columns if "sistema" in c.lower()][0]
muestra = muestra.rename(columns={cs: "sistema"})[["sistema", "TIC"]]
ple = {26017840: "P1", 125754991: "P2", 35159657: "P3", 346541355: "P4?"}
muestra["sistema"] = [ple.get(int(t), (str(int(s)) if pd.notna(s) else "?")) for s, t in zip(muestra.sistema, muestra.TIC)]
cat = pd.read_csv(O / "catalogo_completo_mist_sommers.csv")
gaia2tic = dict(zip(cat.OBJECT.astype("int64"), cat.TIC.astype("int64")))

filas = []
def add(tic, fuente, entrada, fase, modelo, med, lo, up, std=np.nan, masa=np.nan, T=np.nan):
    filas.append(dict(TIC=int(tic), fuente=fuente, entrada=entrada, fase=fase, modelo=modelo,
                      edad_med=med, edad_lo=lo, edad_up=up, edad_std=std, masa_med=masa, T_med=T))

# 1) Corridas de Jesús con tipo espectral (Gaia DR3 id)
for mod in ["mist", "parsec", "somers00", "somers17"]:
    d = leer(M / f"yceb_SpT_PM13_{mod}_results.csv")
    for _, r in d.iterrows():
        tic = gaia2tic.get(int(r.OBJECT))
        if tic: add(tic, "Jesús yceb_SpT_PM13", "SpT", "ms*", mod.upper() if mod != "parsec" else "PARSEC",
                    r.age_median, r.age_low, r.age_upp, r.age_std, r.mass_median, r.T_median)
for mod in ["mist", "somers17"]:
    d = leer(M / f"yceb_Teff_{mod}_results.csv")
    for _, r in d.iterrows():
        tic = gaia2tic.get(int(r.OBJECT))
        if tic: add(tic, "Jesús yceb_Teff", "Teff", "?", mod.upper(), r.age_median, r.age_low, r.age_upp, r.age_std, r.mass_median, r.T_median)
# 2) Catálogo que usa el pipeline (MIST = _1, Somers17 = _2; entrada Teff)
for suf, mod in (("1", "MIST"), ("2", "SOMERS17")):
    for _, r in cat.iterrows():
        add(r.TIC, "pipeline catalogo_completo_mist_sommers", "Teff", "?", mod,
            r[f"age_median_{suf}"], r.get(f"age_low_{suf}", np.nan), r.get(f"age_upp_{suf}", np.nan),
            r.get(f"age_std_{suf}", np.nan), r.get(f"mass_median_{suf}", np.nan), r.get(f"T_median_{suf}", np.nan))
# 3) Corrida pre-MS de 8, 13 y 14
d = pd.read_csv(M / "corrida_pms_8_13_14_20261013/resumen_modelos.csv")
for _, r in d.iterrows():
    m = r.modelo.split("_"); fase = m[1] if len(m) > 1 else "—"
    add(r.TIC, "corrida_pms_8_13_14 (13-oct)", "SpT equiv. (Teff TIC)", fase, m[0].upper(), r.edad, r.e_lo, r.e_up, np.nan, r.M, r["T"])
# 4) Edades por vecindad (3, 7, 11, 20) y PARSEC de la primaria
v = pd.read_csv(O / "edades_vecindad/edades_asignadas.csv")
for _, r in v.iterrows():
    if pd.notna(r.edad_grupo_Myr):
        add(r.TIC, "vecindad cinemática (grupo)", "miembros", "—", "PARSEC pms (miembros)", r.edad_grupo_Myr, r.p16_Myr, r.p84_Myr, r.err_edad_grupo_Myr)
    for col, fase in (("edad_primaria_PARSEC_pms", "pms"), ("edad_primaria_PARSEC_all", "all")):
        if pd.notna(r[col]): add(r.TIC, "edades_vecindad (primaria, SpT literatura)", "SpT lit.", fase, "PARSEC", r[col], np.nan, np.nan)
# 5) Pléyades
for f, ent in (("pleyades4eb_mist_results.csv", "Teff"), ("pleyades4eb_somers17_results.csv", "Teff"),
               ("pleyades3eb_SpT_mist_results.csv", "SpT"), ("pleyades3eb_SpT_somers17_results.csv", "SpT")):
    d = leer(P / f); mod = "SOMERS17" if "somers17" in f else "MIST"
    for _, r in d.iterrows():
        add(r.OBJECT, f"Pléyades {f.split('_results')[0]}", ent, "ms*", mod, r.age_median, r.age_low, r.age_upp, r.age_std, r.mass_median, r.T_median)
for tic in ple: add(tic, "literatura (cúmulo)", "—", "—", "Li depletion (Stauffer+98)", 125, 117, 133)

L = pd.DataFrame(filas).merge(muestra, on="TIC", how="left")
for c in ["edad_med", "edad_lo", "edad_up", "edad_std", "masa_med", "T_med"]:
    L[c] = pd.to_numeric(L[c], errors="coerce")
# edad usada en las corridas vigentes
t = pd.read_csv("/home/luciam/Documents/starry/preliminares_revision1/datos/tabla_maestra.csv")
usada = t.groupby("sistema").agg(edad_usada=("edad_Myr", lambda x: ", ".join(sorted({f"{v:.3g}" for v in x.dropna()}))),
                                 fuente_usada=("edad_fuente", lambda x: ", ".join(sorted(set(x.dropna().astype(str))))))
usada.index = usada.index.astype(str)
L.to_csv(OUT / "edades_largo.csv", index=False)

def fmt(r):
    if pd.isna(r.edad_med): return ""
    s = f"{r.edad_med:.3g}"
    if pd.notna(r.edad_lo) and pd.notna(r.edad_up): s += f" [{r.edad_lo:.3g}–{r.edad_up:.3g}]"
    return s
L["celda"] = L.apply(fmt, axis=1)
L["col"] = L.modelo + " (" + L.entrada + (", " + L.fase).where(L.fase.ne("—"), "") + ")"
A = L.pivot_table(index=["sistema", "TIC"], columns="col", values="celda", aggfunc=lambda x: " / ".join(sorted(set(x)))).reset_index()
A = A.merge(usada, left_on="sistema", right_index=True, how="left")
orden = {str(i): i for i in range(1, 22)}; orden.update({"P1": 22, "P2": 23, "P3": 24, "P4?": 25})
A = A.sort_values("sistema", key=lambda s: s.map(orden).fillna(99))
A.to_csv(OUT / "edades_ancho.csv", index=False)
print(L.groupby("col").size()); print(A.shape)
