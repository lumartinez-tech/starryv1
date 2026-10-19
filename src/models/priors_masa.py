"""Prior de masa para TIC 50897998 (sistema B0.7 + B) por dos vías independientes:
  (a) diagrama HR: (Teff, logL) de MassAge (SpT de la literatura) sobre isócronas MIST;
  (b) interpolador: R_A y R_B medidos por starry en S32 y S98 (sectores convergidos) invertidos con MIST a la
      edad del sistema (MIST/Somers del catálogo: mediana 12.4 Myr; se recorre 5–20 Myr).
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
R = "/home/luciam/Documents/starry/Resultados_Orion"
sys.path.insert(0, f"{R}/datos_estrellas_massage_orion")
from isocronas_interp import load_isochrone_model, IsochroneInterpolator, invert_radius_to_mass
MIST = IsochroneInterpolator(load_isochrone_model("MIST",
    f"{R}/datos_estrellas_massage_orion/modelos/iso_mist/MIST_iso_6a9a687177c08_todas_edades.csv"), "MIST")
rng = np.random.default_rng(1)
EDADES = (5, 8, 12.4, 15, 20)

# (a) HR: masa del punto de la isócrona más cercano en (logTe, logL), Monte Carlo en T y L
ma = pd.read_csv(f"{R}/starry_correcciones/priors_50897998/massage_primaria_all_results.csv", skipinitialspace=True)
ma.columns = [c.strip() for c in ma.columns]
lgL, sl = float(ma.lgL_median[0]), float(ma.lgL_mad[0]) * 1.4826
masas = np.linspace(3, 40, 800)
print(f"(a) MassAge: logL = {lgL:.2f} ± {sl:.2f}; Teff de SpT B0.7 ± 1.7 subtipos")
filas = []
for edad in EDADES:
    lg = np.log10(edad * 1e6)
    T = np.array([MIST.teff(lg, m) for m in masas]); L = np.array([MIST.logL(lg, m) for m in masas])
    ok = np.isfinite(T) & np.isfinite(L)
    # Teff: B0.7 ± 1.7 subtipos ≈ 27 000 (+5000/−6000) K en Pecaut & Mamajek; se usa log-normal 0.08 dex
    lt = rng.normal(np.log10(27000), 0.08, 3000); ll = rng.normal(lgL, sl, 3000)
    d2 = ((np.log10(T[ok])[None] - lt[:, None]) / 0.08) ** 2 + ((L[ok][None] - ll[:, None]) / sl) ** 2
    mm = masas[ok][np.argmin(d2, axis=1)]
    filas.append(dict(via="HR (MassAge)", edad_Myr=edad, componente="A", m50=np.median(mm),
                      m16=np.percentile(mm, 16), m84=np.percentile(mm, 84)))

# (b) interpolador con los radios de starry (media de S32 y S98)
rad = {}
for s in ("S32", "S98"):
    t = pd.read_csv(f"{R}/starry_por_tic/TIC_50897998/{s}/TIC_50897998_{s}_Summary.csv", index_col=0)
    for c in ("A_r", "B_r"):
        rad.setdefault(c, []).append((t.loc[c, "mean"], t.loc[c, "sd"]))
for c, comp in (("A_r", "A"), ("B_r", "B")):
    mu = np.mean([v[0] for v in rad[c]]); sd = np.sqrt(np.mean([v[1] ** 2 for v in rad[c]]))
    print(f"(b) starry {comp}: R = {mu:.2f} ± {sd:.2f} R☉")
    for edad in EDADES:
        lg = np.log10(edad * 1e6)
        mm = np.array([invert_radius_to_mass(lg, r, MIST) for r in rng.normal(mu, sd, 400)])
        mm = mm[np.isfinite(mm)]
        if len(mm) > 50:
            filas.append(dict(via="interpolador R→M", edad_Myr=edad, componente=comp, m50=np.median(mm),
                              m16=np.percentile(mm, 16), m84=np.percentile(mm, 84),
                              Teff=float(MIST.teff(lg, np.median(mm)))))
t = pd.DataFrame(filas).round(2)
t.to_csv(f"{R}/starry_correcciones/priors_50897998/priors_masa_50897998.csv", index=False)
print(t.to_string(index=False))

# prior final: edad uniforme en 8–20 Myr (a 5 Myr la secundaria sería pre-SP con ~7100 K, incompatible con su
# brillo superficial en el eclipse: B_amp/k² ≈ 0.53 → T_B ≈ 14 000 K) y radios de S32/S98
fin = {}
for c, comp in (("A_r", "A"), ("B_r", "B")):
    mu = np.mean([v[0] for v in rad[c]]); sd = np.sqrt(np.mean([v[1] ** 2 for v in rad[c]]))
    mm = np.array([invert_radius_to_mass(np.log10(rng.uniform(8, 20) * 1e6), rng.normal(mu, sd), MIST) for _ in range(1500)])
    mm = mm[np.isfinite(mm)]
    fin[comp] = (np.median(mm), (np.percentile(mm, 84) - np.percentile(mm, 16)) / 2)
    print(f"prior final m_{comp} = {fin[comp][0]:.2f} ± {fin[comp][1]:.2f} M☉ (edad 8–20 Myr, n={len(mm)})")
pd.DataFrame([dict(componente=k, m=v[0], sd=v[1]) for k, v in fin.items()]).round(3) \
  .to_csv(f"{R}/starry_correcciones/priors_50897998/prior_final_masas.csv", index=False)
