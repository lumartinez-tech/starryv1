# Starry: Photometric Analysis & TESS Pipeline

Este repositorio contiene la arquitectura de código y los cuadernos de análisis para el proyecto científico **Starry**, centrado en el estudio de fotometría TESS, inyección/completitud y análisis de sistemas binarios y rotación en cúmulos jóvenes (Pléyades y Orión).

## 🛠️ Estructura del Repositorio

El código fuente principal está modularizado en `src/`:

```text
starry/
├── src/
│   ├── data/          # Extracción y corrección de fotometría TESS / TGLC
│   ├── models/        # Priors, interpoladores de isocronas y modelos teóricos
│   ├── simulation/    # Scripts de inyección, recuperación y completitud
│   ├── core/          # Ejecución principal del modelo Starry y métricas
│   ├── analysis/      # Selección de candidatas, diagnósticos y reportes
│   └── legacy/        # Versiones de archivo y scripts históricos
├── notebooks/
│   ├── 00_exploracion/ # Cuadernos de pruebas iniciales y prototipos
│   ├── 01_pleyades/    # Cuadernos de análisis del cúmulo Pléyades
│   └── 02_orion/       # Cuadernos de análisis del cúmulo Orión
├── docs/               # Documentación complementaria y figuras de soporte
├── config/             # Archivos de configuración (.json, .yml)
└── .gitignore          # Reglas para excluir datos pesados y resultados localmente
```

## 🚀 Instalación y Entorno

1. **Clonar el repositorio:**
   ```bash
   git clone git@github.com:lumartinez-tech/starryv1.git
   cd starryv1
   ```

2. **Recrear el entorno de Conda:**
   ```bash
   conda env create -f environment.yml
   conda activate starry
   ```

3. **Uso de módulos en Python:**
   Al incluir los archivos `__init__.py`, puedes importar cualquier submódulo desde un script o cuaderno:
   ```python
   from src.data import tessextractor
   from src.core import genera_starry
   ```