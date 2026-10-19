"""TGLC con promedio temporal de las imágenes antes del ajuste del ePSF.

Para los sectores de 200 s (S56 en adelante) el ajuste del ePSF imagen por imagen domina el tiempo de
cálculo. Promediar de a `n` imágenes buenas antes del ajuste reduce el costo en un factor `n` y deja la
cadencia efectiva en n × 200 s (n = 3 → 600 s, igual que S27–S55).
"""
import numpy as np
from threadpoolctl import threadpool_limits


def binear_source(source, n):
    # promedia de a n imágenes consecutivas usando solo las de calidad 0 y tiempo finito
    t = np.asarray(source.time, float)
    q = np.asarray(source.quality).astype(np.int64)
    cad = np.asarray(source.cadence)
    flux = np.asarray(source.flux)
    err = np.asarray(source.flux_err)
    m = len(t) // n * n
    idx = np.arange(m).reshape(-1, n)
    bueno = (q[:m] == 0) & np.isfinite(t[:m])
    w = bueno[idx].astype(float)                    # (nb, n)
    nw = w.sum(axis=1)
    sin_buenas = nw == 0
    w[sin_buenas] = 1.0                             # grupo sin imágenes buenas: promedio simple, queda marcado
    nw = w.sum(axis=1)
    W = w[:, :, None, None]
    f_b = np.nansum(flux[idx] * W, axis=1) / nw[:, None, None]
    e_b = np.sqrt(np.nansum((err[idx] ** 2) * W, axis=1)) / nw[:, None, None]
    tt = np.where(np.isfinite(t[idx]), t[idx], 0.0)
    t_b = (tt * w).sum(axis=1) / np.maximum((w * np.isfinite(t[idx])).sum(axis=1), 1)
    t_b[(w * np.isfinite(t[idx])).sum(axis=1) == 0] = np.nan
    q_b = np.where(sin_buenas, np.bitwise_or.reduce(q[idx], axis=1), 0)
    source.time = t_b
    source.flux = f_b.astype(flux.dtype)
    source.flux_err = e_b.astype(err.dtype)
    source.quality = q_b.astype(np.asarray(source.quality).dtype)
    source.cadence = cad[idx][:, 0]
    return source


def descargar_source(tic, local_directory, sector, size=90, limit_mag=16):
    # corte TESScut + catálogo Gaia; si ya existe el .pkl en local_directory/source, se reusa
    from tglc.ffi_cut import ffi_cut
    source = ffi_cut(target=f"TIC {tic}", size=size, local_directory=local_directory, sector=sector,
                     limit_mag=limit_mag, ffi="SPOC", mast_timeout=3600)
    source.select_sector(sector=sector)
    return source


def nombre_gaia(source, ra, dec, radio_arcsec=3.0):
    # fuente Gaia DR3 más brillante a menos de radio_arcsec de (ra, dec)
    g = source.gaia
    dra = (np.asarray(g["ra"], float) - ra) * np.cos(np.radians(dec)) * 3600
    ddec = (np.asarray(g["dec"], float) - dec) * 3600
    cerca = np.flatnonzero(np.hypot(dra, ddec) < radio_arcsec)
    if not len(cerca):
        raise ValueError(f"ninguna fuente Gaia DR3 a menos de {radio_arcsec} arcsec")
    return str(g["DESIGNATION"][cerca[np.argmin(np.asarray(g["tess_mag"])[cerca])]])


def ajustar(source, tic, local_directory, n=1, limit_mag=16, ra=None, dec=None):
    # ePSF y curvas. El objetivo se identifica por TIC con el cruce TIC -> Gaia DR3 del propio corte; si
    # ese cruce es ambiguo (una fuente DR2 que se divide en dos en DR3), se usa la fuente DR3 más brillante
    # a menos de 3" de la posición del TIC
    from tglc.target_lightcurve import epsf
    with threadpool_limits(limits=1, user_api="blas"):
        if n > 1:
            binear_source(source, n)
        nombre = str(tic)
        try:
            matches = np.flatnonzero(np.asarray(source.tic["TIC"]).astype(str) == nombre)
            gid = source.tic["dr3_source_id"][matches[0]] if len(matches) else None
            if gid is None or np.ma.is_masked(gid):
                raise ValueError("cruce TIC -> DR3 ambiguo")
        except Exception as e:
            if ra is None:
                raise
            nombre = nombre_gaia(source, ra, dec)
            print(f"TIC {tic}: {e}; se usa {nombre}", flush=True)
        return epsf(source, factor=2, sector=source.sector, target=f"TIC {tic}_bin{n}",
                    local_directory=local_directory, name=nombre, limit_mag=limit_mag, save_aper=True,
                    ffi="SPOC") or []


def tglc_binado(tic, local_directory, sector, n=3, size=90, limit_mag=16):
    # equivalente a tglc_lc(tic, sector=sector) con el paso de binning antes del ePSF
    source = descargar_source(tic, local_directory, sector, size=size, limit_mag=limit_mag)
    return ajustar(source, tic, local_directory, n=n, limit_mag=limit_mag)
