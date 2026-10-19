"""¿Alguno de los sistemas (21 de Orión + 4 EB de las Pléyades) tiene observaciones de JWST en MAST?
Busca registros JWST a menos de 3' de cada objeto y comprueba si el objeto cae dentro de la huella (s_region) de la observación."""
import re, warnings
import numpy as np, pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations, Catalogs
from matplotlib.path import Path as MPath
warnings.filterwarnings("ignore")
RAIZ = "/home/luciam/Documents/starry/Resultados_Orion"
m = pd.read_csv(f"{RAIZ}/datos_estrellas_massage_orion/Macro_21aug2026.csv", usecols=["TIC", "RA", "DEC", "Tmag"]).drop_duplicates("TIC")
ind = pd.read_csv(f"{RAIZ}/resultados_pipeline/periodos/indice_sistemas.csv")[["indice", "TIC"]]
t = ind.merge(m, on="TIC", how="left").assign(muestra="Orión")
p = pd.read_csv("/home/luciam/Documents/massage_workshop/massage.py/massageXtesss_pleyades.csv").drop_duplicates("TIC")
pl = p[p.TIC.isin([26017840, 35159657, 125754991, 346541355])][["TIC", "ra", "dec", "Tmag"]].rename(columns={"ra": "RA", "dec": "DEC"})
pl = pl.assign(indice=pl.TIC.map({26017840: 22, 125754991: 23, 35159657: 24, 346541355: 25}), muestra="Pléyades")
T = pd.concat([t, pl], ignore_index=True)
for i, r in T[T.RA.isna()].iterrows():          # coordenadas faltantes desde el TIC
    c = Catalogs.query_criteria(catalog="Tic", ID=int(r.TIC))
    T.loc[i, ["RA", "DEC", "Tmag"]] = [float(c["ra"][0]), float(c["dec"][0]), float(c["Tmag"][0])]
    print("coordenadas de TIC", int(r.TIC), "desde el catálogo TIC")


def dentro(ra, dec, s_region):
    for poly in re.findall(r"POLYGON\s+ICRS\s+([-\d.eE+\s]+)", str(s_region)):
        v = np.array(poly.split(), float).reshape(-1, 2)
        if len(v) >= 3 and MPath(v).contains_point((ra, dec)):
            return True
    return False


filas = []
for _, r in T.iterrows():
    c = SkyCoord(r.RA * u.deg, r.DEC * u.deg)
    try:
        o = Observations.query_region(c, radius=3 * u.arcmin).to_pandas()
        o = o[o.obs_collection == "JWST"]
    except Exception as e:
        print("error", int(r.TIC), str(e)[:80]); o = pd.DataFrame()
    n_cerca = len(o); n_dentro = 0
    for _, x in o.iterrows():
        d = dentro(r.RA, r.DEC, x.s_region); n_dentro += d
        sep = SkyCoord(x.s_ra * u.deg, x.s_dec * u.deg).separation(c).arcsec
        filas.append(dict(indice=int(r.indice), TIC=int(r.TIC), muestra=r.muestra, Tmag=r.Tmag, proposal_id=x.proposal_id, instrumento=x.instrument_name, filtro=x.filters,
                          objetivo=x.target_name, sep_arcsec=round(sep, 1), en_huella=bool(d), t_exp_s=x.t_exptime, calib=x.calib_level, derechos=x.dataRights, fecha_mjd=x.t_min, obs_id=x.obs_id))
    print(f"[{r.muestra} {int(r.indice):>2}] TIC {int(r.TIC)}: {n_cerca} registros JWST a < 3', {n_dentro} con el objeto dentro de la huella", flush=True)
R = pd.DataFrame(filas); R.to_csv("jwst_registros.csv", index=False)
resumen = T.merge(R.groupby("TIC").agg(n_registros=("obs_id", "size"), n_en_huella=("en_huella", "sum")).reset_index(), on="TIC", how="left").fillna({"n_registros": 0, "n_en_huella": 0})
resumen.to_csv("jwst_resumen.csv", index=False)
print("\nCon el objeto dentro de la huella de una observación JWST:")
print(R[R.en_huella][["indice", "TIC", "proposal_id", "instrumento", "filtro", "objetivo", "sep_arcsec", "t_exp_s", "derechos"]].to_string(index=False) if R.en_huella.any() else "  ninguno")
