"""
isocronas_interp.py

Interpolador Teff(R, t): dada la edad t de una estrella y su radio R,
regresa la temperatura efectiva que predicen las isocronas en `modelos/`.

Extiende el patrón de binaria_masa/isochrone_binary_fit.ipynb (loader ->
DataFrame estandarizado -> IsochroneInterpolator sobre (logAge, mass)),
agregando loaders para MIST y Somers et al. (2017) además de BHAC15/Baraffe,
y agrega la inversión R -> Teff a edad fija que faltaba para este uso.

Método:
    1. A edad fija (logAge dado), R es función de la masa: R(M).
    2. Se invierte R(M) = R_obs por búsqueda de raíz (brentq) para obtener
       la masa M* que reproduce el radio observado a esa edad.
    3. Teff(R, t) := Teff(logAge, M*).

Esto asume que R(M) es monótona a edad fija dentro del rango de masas del
modelo. Es razonable en la pre-secuencia principal para el rango de masas
de baja masa (K/M) típico de Orion, pero conviene revisarlo si se usan
edades post-MS o rangos de masa muy amplios.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import brentq

T_SUN = 5772.0  # K


# ---------------------------------------------------------------------------
# 1. Carga de modelos -> DataFrame estandarizado
#    columnas: logAge [log10(yr)], mass [Msun], Teff [K], radius [Rsun], logL [log10(Lsun)]
# ---------------------------------------------------------------------------

def _load_mist(path) -> pd.DataFrame:
    """
    Lee un CSV de isocronas MIST ya combinado (p. ej. el
    "*_todas_edades.csv" que exporta el paquete de isocronas de MIST, con
    columnas log10_isochrone_age_yr, star_mass, log_Teff, log_R, log_L).

    Se pide un archivo específico (no un directorio) porque en
    modelos/iso_mist/ conviven grids con distinta cobertura de edad (p. ej.
    dos descargas distintas); concatenarlos automáticamente mezclaría mallas
    no equivalentes.
    """
    path = Path(path)
    iso = pd.read_csv(path)

    required = {"log10_isochrone_age_yr", "star_mass", "log_Teff", "log_R", "log_L"}
    missing = required - set(iso.columns)
    if missing:
        raise ValueError(f"Columnas MIST ausentes en {path}: {missing}")

    out = pd.DataFrame({
        "logAge": iso["log10_isochrone_age_yr"].astype(float),
        "mass": iso["star_mass"].astype(float),
        "Teff": 10.0 ** iso["log_Teff"].astype(float),
        "radius": 10.0 ** iso["log_R"].astype(float),
        "logL": iso["log_L"].astype(float),
    })
    out["model"] = "MIST"
    return out


def _load_somers17(path) -> pd.DataFrame:
    """
    Lee un track de Somers et al. (2017) (p. ej. f017track_ms.csv), con
    columnas Mass, Age(Gyr), log(R/Rsun), log(Teff).
    """
    iso = pd.read_csv(path)

    required = {"Mass", "Age(Gyr)", "log(R/Rsun)", "log(Teff)"}
    missing = required - set(iso.columns)
    if missing:
        raise ValueError(f"Columnas Somers17 ausentes en {path}: {missing}")

    out = pd.DataFrame({
        "logAge": np.log10(iso["Age(Gyr)"].astype(float) * 1e9),
        "mass": iso["Mass"].astype(float),
        "Teff": 10.0 ** iso["log(Teff)"].astype(float),
        "radius": 10.0 ** iso["log(R/Rsun)"].astype(float),
    })
    out["logL"] = (
        iso["log(L/Lsun)"].astype(float) if "log(L/Lsun)" in iso.columns else np.nan
    )
    out["model"] = "Somers17"
    return out


def _load_bhac15(path) -> pd.DataFrame:
    """
    Lee el archivo BHAC15_iso.txt (Baraffe et al. 2015): bloques separados
    por líneas "!  t (Gyr) = ...", cada uno con columnas
    M/Ms Teff L/Ls g R/Rs Li/Li0 <fotometría...>.

    Aquí "radius" es una salida directa del modelo (columna propia),
    independiente de Teff y logL.
    """
    cols = [
        "mass", "Teff", "logL", "logg", "radius", "Li",
        "F33", "F33B", "F41", "F45B", "F47", "F51", "FHa", "F57", "F63B",
        "F67", "F75", "F78", "F82", "F82B", "F89", "G_RSV", "G", "G_BP", "G_RP",
    ]
    rows, tables = [], []
    current_age_gyr = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("!  t"):
                if rows and current_age_gyr is not None:
                    df = pd.DataFrame(rows, columns=cols)
                    df["logAge"] = np.log10(current_age_gyr * 1e9)  # Gyr -> yr
                    tables.append(df)
                    rows = []
                current_age_gyr = float(line.split("=")[1])
                continue
            if line.startswith("!") or line == "" or line.startswith("BHAC"):
                continue
            if line[0].isdigit():
                values = line.split()
                if len(values) >= len(cols):
                    rows.append([float(v) for v in values[: len(cols)]])

    if rows and current_age_gyr is not None:
        df = pd.DataFrame(rows, columns=cols)
        df["logAge"] = np.log10(current_age_gyr * 1e9)
        tables.append(df)

    if not tables:
        raise ValueError(f"No se pudieron extraer isocronas de {path}. Revisa el formato del archivo.")

    iso = pd.concat(tables, ignore_index=True)
    out = iso[["logAge", "mass", "Teff", "logL", "radius"]].copy()
    out["model"] = "BHAC15"
    return out


_LOADERS = {"MIST": _load_mist, "Somers17": _load_somers17, "BHAC15": _load_bhac15}


def load_isochrone_model(model: str, path) -> pd.DataFrame:
    """
    Punto de entrada único para cargar isocronas. `model` es "MIST",
    "Somers17" o "BHAC15"; para agregar una familia nueva basta con escribir
    un loader que regrese las mismas columnas estandarizadas y añadirlo a
    `_LOADERS`, el resto del pipeline no cambia.
    """
    if model not in _LOADERS:
        raise ValueError(f"Modelo '{model}' no reconocido. Opciones: {list(_LOADERS)}")
    grid = _LOADERS[model](path)
    _report_grid_coverage(grid, model)
    return grid


def _report_grid_coverage(grid: pd.DataFrame, model: str) -> None:
    age_min_myr = 10 ** grid["logAge"].min() / 1e6
    age_max_myr = 10 ** grid["logAge"].max() / 1e6
    print(f"[{model}] cobertura de edad:  {age_min_myr:.3g} - {age_max_myr:.3g} Myr")
    print(f"[{model}] cobertura de masa:  {grid['mass'].min():.3f} - {grid['mass'].max():.3f} Msun")


# ---------------------------------------------------------------------------
# 2. Interpolación (edad, masa) -> propiedades estelares
# ---------------------------------------------------------------------------

@dataclass
class IsochroneInterpolator:
    grid: pd.DataFrame
    model: str

    def __post_init__(self):
        pts = np.column_stack([self.grid["logAge"].values, self.grid["mass"].values])
        self._teff = LinearNDInterpolator(pts, self.grid["Teff"].values)
        self._radius = LinearNDInterpolator(pts, self.grid["radius"].values)
        self._logL = LinearNDInterpolator(pts, self.grid["logL"].values)
        self.logAge_min, self.logAge_max = float(self.grid["logAge"].min()), float(self.grid["logAge"].max())
        self.mass_min, self.mass_max = float(self.grid["mass"].min()), float(self.grid["mass"].max())

    def teff(self, logAge, mass):
        return self._eval(self._teff, logAge, mass, "Teff")

    def radius(self, logAge, mass):
        return self._eval(self._radius, logAge, mass, "radius")

    def logL(self, logAge, mass):
        return self._eval(self._logL, logAge, mass, "logL")

    def _eval(self, interp, logAge, mass, label):
        logAge_arr = np.atleast_1d(logAge).astype(float)
        mass_arr = np.atleast_1d(mass).astype(float)
        out = np.asarray(interp(logAge_arr, mass_arr), dtype=float)

        out_of_range = (
            (logAge_arr < self.logAge_min) | (logAge_arr > self.logAge_max)
            | (mass_arr < self.mass_min) | (mass_arr > self.mass_max)
        )
        outside_hull = np.isnan(out) & ~out_of_range
        if np.any(out_of_range):
            warnings.warn(
                f"{np.sum(out_of_range)} punto(s) fuera del rango cubierto por la malla "
                f"de isocronas ({self.model}) al interpolar '{label}'. Se devuelve NaN "
                "en vez de extrapolar.",
                RuntimeWarning,
            )
        if np.any(outside_hull):
            warnings.warn(
                f"{np.sum(outside_hull)} punto(s) dentro del rango nominal pero fuera "
                f"del casco convexo de la malla ({self.model}, '{label}'); probablemente "
                "una combinación (edad, masa) sin puntos de la malla cerca. Resultado NaN.",
                RuntimeWarning,
            )
        return out if out.size > 1 else out.item()


def invert_radius_to_mass(logAge, R_target, interp: IsochroneInterpolator,
                           mass_bounds: Optional[tuple] = None, n_scan: int = 400) -> float:
    """
    A edad fija, encuentra la masa M tal que radius(logAge, M) = R_target
    mediante búsqueda de raíz (brentq) sobre una malla fina de masas.

    Regresa np.nan si R_target no está acotado por el rango de radios
    evaluado a esa edad -- no extrapola.
    """
    mlo, mhi = mass_bounds or (interp.mass_min, interp.mass_max)
    m_scan = np.linspace(mlo, mhi, n_scan)
    r_scan = np.atleast_1d(interp.radius(logAge, m_scan))

    valid = np.isfinite(r_scan)
    if valid.sum() < 2:
        return np.nan
    m_scan, r_scan = m_scan[valid], r_scan[valid]

    diff = r_scan - R_target
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        return np.nan  # R_target fuera del rango de radios cubierto a esta edad
    i = sign_changes[0]

    def f(m):
        val = interp.radius(logAge, m)
        return (val if np.isfinite(val) else np.nan) - R_target

    try:
        return brentq(f, m_scan[i], m_scan[i + 1])
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# 3. Teff(R, t) -- lo que pediste
# ---------------------------------------------------------------------------

def teff_de_radio_edad(radio_Rsun: float, edad_Myr: float, interp: IsochroneInterpolator,
                        mass_bounds: Optional[tuple] = None) -> float:
    """
    Interpola Teff(R, t): dado el radio (Rsun) y la edad (Myr) de una
    estrella, regresa la Teff (K) que predice la isocrona `interp`.

    Internamente invierte R(M) a la edad dada para obtener la masa
    correspondiente y evalúa Teff en esa masa -- ver el docstring del módulo
    para la justificación del método y su supuesto de monotonía.

    Regresa np.nan si (radio, edad) cae fuera de lo que cubre la malla.
    """
    logAge = np.log10(edad_Myr * 1e6)
    mass = invert_radius_to_mass(logAge, radio_Rsun, interp, mass_bounds=mass_bounds)
    if np.isnan(mass):
        return np.nan
    return float(interp.teff(logAge, mass))


def teff_de_radio_edad_batch(radios_Rsun, edades_Myr, interp: IsochroneInterpolator,
                              mass_bounds: Optional[tuple] = None) -> np.ndarray:
    """Versión vectorizada de `teff_de_radio_edad` para arreglos/columnas de un catálogo."""
    radios_Rsun = np.atleast_1d(radios_Rsun)
    edades_Myr = np.atleast_1d(edades_Myr)
    if radios_Rsun.shape != edades_Myr.shape:
        radios_Rsun, edades_Myr = np.broadcast_arrays(radios_Rsun, edades_Myr)

    out = np.full(radios_Rsun.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # se reporta NaN por punto, no hace falta el warning por cada fila
        for i in range(radios_Rsun.size):
            out.flat[i] = teff_de_radio_edad(
                radios_Rsun.flat[i], edades_Myr.flat[i], interp, mass_bounds=mass_bounds
            )
    return out
