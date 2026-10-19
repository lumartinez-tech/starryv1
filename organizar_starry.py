#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ORGANIZADOR DEL PROYECTO STARRY (MEJORADO)
=========================================

Reorganiza el proyecto de análisis de estrellas binarias eclipsantes
en una estructura modular y reproducible.
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path.home() / "Documents" / "starry"

# Directorios que NUNCA deben explorarse para mover sus contenidos.
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
}

# Directorios de la nueva estructura.
DIRECTORIES = [
    "config",
    "data/raw",
    "data/raw/models",
    "data/processed",
    "docs",
    "notebooks/00_exploracion",
    "notebooks/01_pleyades",
    "notebooks/02_orion",
    "results/figures",
    "results/summaries",
    "src",
]

# Archivos y carpetas que deben permanecer en la raíz.
ROOT_FILES = {
    ".gitignore",
    "README.md",
    "requirements.txt",
    "environment.yml",
    "organizar_starry.py",
    "LICENSE",
}

# Archivos de modelos evolutivos conocidos.
MODEL_FILES = {
    "mist_v2.csv",
    "f017track_ms.csv",
}

# Catálogos y resúmenes conocidos.
SUMMARY_FILES = {
    "catalogo_completo_mist_sommers.csv",
    "yceb_spt_pm13_somers17_results.csv",
}

# Archivos de priors conocidos.
PRIOR_FILES = {
    "priors.csv",
    "priors_completosv1.csv",
}

# Extensiones típicas de datos científicos y astronómicos.
RAW_EXTENSIONS = {
    ".dat",
    ".fits",
    ".fit",
    ".fts",
    ".txt",
    ".fz",
    ".gz",
    ".h5",
    ".hdf5",
}

# Documentación y referencias.
DOC_EXTENSIONS = {
    ".pdf",
    ".bib",
    ".tex",
    ".docx",
    ".doc",
    ".rst",
}

# Archivos que suelen ser resultados gráficos.
FIGURE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".eps",
    ".pdf",  # Algunas figuras exportadas son PDF
}


# ============================================================
# 2. ESTRUCTURA BASE
# ============================================================

def create_directory_structure(root, apply=False):
    """Crea la estructura propuesta y asegura carpetas con .gitkeep."""
    print("\n" + "=" * 65)
    print("ESTRUCTURA DEL PROYECTO")
    print("=" * 65)

    for directory in DIRECTORIES:
        path = root / directory

        if path.exists():
            print(f"[EXISTE]   {directory}/")
        elif apply:
            path.mkdir(parents=True, exist_ok=True)
            # Agregar .gitkeep para que Git rastree directorios vacíos de datos
            gitkeep = path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
            print(f"[CREADO]   {directory}/")
        else:
            print(f"[SIMULAR] Crear {directory}/")


# ============================================================
# 3. CLASIFICACIÓN DE ARCHIVOS
# ============================================================

def classify_file(path, root):
    """Decide el destino de un archivo según su nombre y extensión."""
    name = path.name.lower()
    suffix = path.suffix.lower()

    # Descartar temporales de Jupyter/sistema
    if name.startswith(".~") or name.startswith("._"):
        return None

    try:
        relative = path.relative_to(root)
    except ValueError:
        return None

    parts = relative.parts
    parent_names = [part.lower() for part in parts[:-1]]

    # Ignorar la raíz o archivos que ya están en su lugar
    if len(parts) == 1 and path.name in ROOT_FILES:
        return None

    if parts[0] in {"config", "data", "docs", "notebooks", "results", "src"}:
        return None

    # Modelos evolutivos
    if name in MODEL_FILES or "modelos" in parent_names or "models" in parent_names:
        if suffix in {".csv", ".dat", ".txt", ".fits"}:
            return Path("data/raw/models") / path.name

    # Catálogos y resúmenes
    if name in SUMMARY_FILES or (name.startswith("tic_") and "summary" in name) or name.startswith("catalogo_"):
        return Path("results/summaries") / path.name

    # Priors
    if name in PRIOR_FILES or ("prior" in name and suffix == ".csv"):
        return Path("data/processed") / path.name

    # Notebooks
    if suffix == ".ipynb":
        if any(token in str(relative).lower() for token in ["pleyades", "pleiades", "tic125"]):
            return Path("notebooks/01_pleyades") / path.name
        if any(token in str(relative).lower() for token in ["orion", "yceb", "massage"]):
            return Path("notebooks/02_orion") / path.name
        return Path("notebooks/00_exploracion") / path.name

    # Código Python
    if suffix == ".py":
        return Path("src") / path.name

    # Figuras
    if suffix in FIGURE_EXTENSIONS and any(token in parent_names for token in ["plots", "figures", "figuras"]):
        return Path("results/figures") / path.name

    # Documentación
    if suffix in DOC_EXTENSIONS and not name.endswith(".ipynb"):
        return Path("docs") / path.name

    # Curvas de luz / Datos crudos
    if "light_curve_files" in parent_names or "light_curves" in parent_names or suffix in RAW_EXTENSIONS:
        return Path("data/raw") / path.name

    # Datos procesados
    if suffix == ".npz" or (suffix == ".csv" and "corrected" in name):
        return Path("data/processed") / path.name

    if suffix == ".csv" and any(token in parent_names for token in ["resultados_orion", "tables", "tablas"]):
        return Path("results/summaries") / path.name

    return None


# ============================================================
# 4. DETECCIÓN DE COLISIONES
# ============================================================

def unique_destination(destination):
    """Evita sobrescribir archivos agregando un contador."""
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    counter = 1

    while True:
        candidate = parent / f"{stem}_organizado_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ============================================================
# 5. INVENTARIO Y MOVIMIENTO
# ============================================================

def discover_files(root):
    """Explora el directorio omitiendo exclusiones."""
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue

        if relative.parts[0] in {"config", "data", "docs", "notebooks", "results", "src"}:
            continue

        if path.name == "organizar_starry.py":
            continue

        files.append(path)
    return sorted(files)


def organize_files(root, apply=False):
    """Planifica y ejecuta el movimiento de archivos."""
    files = discover_files(root)
    plan = []
    unclassified = []

    for source in files:
        relative_destination = classify_file(source, root)
        if relative_destination is None:
            unclassified.append(source)
            continue

        destination = root / relative_destination
        if source.resolve() == destination.resolve():
            continue

        plan.append((source, destination))

    print("\n" + "=" * 65)
    print("PLAN DE ORGANIZACIÓN")
    print("=" * 65)

    if not plan:
        print("No se encontraron archivos que necesiten moverse.")

    for source, destination in plan:
        print(f"\nORIGEN:  {source.relative_to(root)}")
        print(f"DESTINO: {destination.relative_to(root)}")

    print("\n" + "-" * 65)
    print(f"Archivos a mover:         {len(plan)}")
    print(f"Archivos no clasificados: {len(unclassified)}")

    if unclassified:
        print("\nArchivos no clasificados (permanecerán en su lugar):")
        for path in unclassified[:15]:
            print(f"  - {path.relative_to(root)}")
        if len(unclassified) > 15:
            print(f"  ... y otros {len(unclassified) - 15} archivos.")

    if not apply:
        print("\n[MODO SIMULACIÓN] Para ejecutar los cambios utiliza: python organizar_starry.py --apply")
        return

    # Mover archivos
    log_path = root / "organizacion_log.txt"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n=========================================================\nOrganización: {datetime.now().isoformat(timespec='seconds')}\n")
        moved, errors = 0, 0
        for source, destination in plan:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                final_destination = unique_destination(destination)
                shutil.move(str(source), str(final_destination))
                msg = f"MOVIDO: {source.relative_to(root)} -> {final_destination.relative_to(root)}"
                print(msg)
                log.write(msg + "\n")
                moved += 1
            except Exception as error:
                msg = f"ERROR al mover {source}: {error}"
                print(msg)
                log.write(msg + "\n")
                errors += 1

        print("\n" + "=" * 65)
        print("RESUMEN")
        print("=" * 65)
        print(f"Movidos: {moved} | Errores: {errors} | Log: {log_path}")


# ============================================================
# 6. ARCHIVOS BÁSICOS DEL PROYECTO
# ============================================================

def create_project_files(root, apply=False):
    """Genera .gitignore, README.md, environment.yml y config/targets.json si no existen."""
    templates = {
        ".gitignore": """# Datos pesados o binarios
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Archivos comprimidos
*.zip
*.tar.gz

# Python & Jupyter
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.~*

# Entornos virtuales
.venv/
venv/
env/

# Salidas MCMC / Cadenas pesadas
*.nc
*.pkl
*.pickle
""",
        "environment.yml": """name: starry_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pandas
  - matplotlib
  - astropy
  - lightkurve
  - wotan
  - pip:
    - starry
""",
        "README.md": """# Starry - Modelado de Binarias Eclipsantes

Análisis fotométrico de estrellas binarias eclipsantes en las regiones de Orión y Pléyades utilizando datos de TESS y Gaia DR3.

## Estructura
- `config/`: Configuración general de objetivos (TICs).
- `data/raw/`: Curvas de luz originales y modelos teóricos.
- `data/processed/`: Curvas corregidas (detrended) y matrices intermedias.
- `docs/`: Artículos y notas.
- `notebooks/`: Flujos de trabajo organizados por región (`01_pleyades`, `02_orion`).
- `results/`: Figuras científicas y tablas/resúmenes de parámetros posteriormes.
- `src/`: Funciones y módulos en Python.
""",
        "config/targets.json": json.dumps(
            {
                "project": "starry",
                "regions": {"orion": [], "pleyades": []}
            }, indent=4
        ) + "\n",
        "src/__init__.py": '"""Módulos del proyecto Starry."""\n',
    }

    print("\n" + "=" * 65)
    print("ARCHIVOS BÁSICOS")
    print("=" * 65)

    for relative_path, content in templates.items():
        path = root / relative_path
        if path.exists():
            print(f"[CONSERVADO] {relative_path}")
        elif apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"[CREADO]     {relative_path}")
        else:
            print(f"[SIMULAR]   Crear {relative_path}")


# ============================================================
# 7. EJECUCIÓN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Organizador del proyecto Starry.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Ruta del proyecto.")
    parser.add_argument("--apply", action="store_true", help="Aplicar los cambios.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directorio no válido: {root}")

    print("=" * 65)
    print("ORGANIZADOR DEL PROYECTO STARRY")
    print(f"Ruta: {root} | Modo: {'APLICAR' if args.apply else 'SIMULACIÓN'}")
    print("=" * 65)

    create_directory_structure(root, apply=args.apply)
    create_project_files(root, apply=args.apply)
    organize_files(root, apply=args.apply)


if __name__ == "__main__":
    main()