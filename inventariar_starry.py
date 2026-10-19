#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Genera un inventario de Starry sin modificar los archivos del proyecto."""

import argparse
import csv
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
}

def main():
    parser = argparse.ArgumentParser(
        description="Inventariar archivos de un proyecto sin modificarlos."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "Documents" / "starry",
        help="Directorio raíz del proyecto.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "starry_inventario.csv",
        help="Ruta del CSV de salida.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(
            f"No se encontró el directorio del proyecto: {root}"
        )

    # Evitar crear el inventario dentro del propio proyecto.
    if output == root or root in output.parents:
        raise ValueError(
            "El CSV de salida debe guardarse fuera del proyecto."
        )

    rows = []

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)

        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue

        if not path.is_file():
            continue

        try:
            stat = path.stat()
        except OSError as exc:
            print(f"No se pudo consultar {relative}: {exc}")
            continue

        rows.append({
            "ruta": str(relative),
            "extension": path.suffix.lower() or "[sin extensión]",
            "tamano_bytes": stat.st_size,
            "directorio_principal": (
                relative.parts[0] if len(relative.parts) > 1
                else "[raíz]"
            ),
        })

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ruta",
                "extension",
                "tamano_bytes",
                "directorio_principal",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Archivos inventariados: {len(rows)}")
    print(f"Inventario guardado en: {output}")
    print("No se movieron ni modificaron archivos del proyecto.")


if __name__ == "__main__":
    main()
