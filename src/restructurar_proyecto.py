import os
import re
import shutil

# Directorio base (actual)
BASE_DIR = os.getcwd()

# Patron para reconocer el TIC y Sector de un archivo/nombre
# Ejemplo de coincidencias: TIC_264483228_S98, full_264741359_s32
TIC_PATTERN = re.compile(r'(?:TIC[_\s]?|full_)(\d{8,10})', re.IGNORECASE)
SECTOR_PATTERN = re.compile(r'[_\s](S\d{1,2}(?:_ecc)?)', re.IGNORECASE)

def obtener_destino(filename):
    """
    Determina la subcarpeta adecuada dentro de un sector
    según el tipo o extensión del archivo.
    """
    ext = os.path.splitext(filename)[1].lower()
    fname = filename.lower()
    
    if ext in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
        if 'parameter' in fname or 'table' in fname or 'summary' in fname:
            return 'Tables'
        return 'Figures'
    elif ext in ['.ipynb']:
        return 'notebooks'
    elif ext in ['.csv', '.nc', '.mp4', '.npz']:
        return ''  # Se quedan en la raíz del Sector
    return None

def organizar_archivos_sueltos():
    print("🚀 Iniciando organización de archivos...")
    
    # 1. Organizar archivos PNG sueltos en la raíz principal
    archivos_raiz = [f for f in os.listdir(BASE_DIR) if os.path.isfile(os.path.join(BASE_DIR, f))]
    
    for archivo in archivos_raiz:
        tic_match = TIC_PATTERN.search(archivo)
        sec_match = SECTOR_PATTERN.search(archivo)
        
        if tic_match and sec_match:
            tic_id = tic_match.group(1)
            sector = sec_match.group(1).upper()
            
            subcategoria = obtener_destino(archivo)
            if subcategoria is not None:
                # Construir ruta de destino
                destino_dir = os.path.join(BASE_DIR, "starry_por_tic", f"TIC_{tic_id}", sector)
                if subcategoria:
                    destino_dir = os.path.join(destino_dir, subcategoria)
                
                os.makedirs(destino_dir, exist_ok=True)
                origen = os.path.join(BASE_DIR, archivo)
                destino = os.path.join(destino_dir, archivo)
                
                print(f"📦 Moviendo {archivo}  ➔  {os.path.relpath(destino, BASE_DIR)}")
                shutil.move(origen, destino)

    # 2. Reorganizar Notebooks dentro de cada carpeta TIC si están sueltos ahí
    starry_dir = os.path.join(BASE_DIR, "starry_por_tic")
    if os.path.exists(starry_dir):
        for tic_folder in os.listdir(starry_dir):
            tic_path = os.path.join(starry_dir, tic_folder)
            if os.path.isdir(tic_path):
                notebooks_dir = os.path.join(tic_path, "notebooks")
                for item in os.listdir(tic_path):
                    if item.endswith(".ipynb"):
                        os.makedirs(notebooks_dir, exist_ok=True)
                        origen = os.path.join(tic_path, item)
                        destino = os.path.join(notebooks_dir, item)
                        print(f"📓 Moviendo notebook {item}  ➔  {os.path.relpath(destino, BASE_DIR)}")
                        shutil.move(origen, destino)

    # 3. Mover archivos en ./Tables/ sueltos si existen
    tables_dir = os.path.join(BASE_DIR, "Tables")
    if os.path.exists(tables_dir):
        for archivo in os.listdir(tables_dir):
            tic_match = TIC_PATTERN.search(archivo)
            sec_match = SECTOR_PATTERN.search(archivo)
            
            tic_id = tic_match.group(1) if tic_match else None
            sector = sec_match.group(1).upper() if sec_match else "S6" # default si no se especifica
            
            if tic_id:
                destino_dir = os.path.join(BASE_DIR, "starry_por_tic", f"TIC_{tic_id}", sector, "Tables")
                os.makedirs(destino_dir, exist_ok=True)
                origen = os.path.join(tables_dir, archivo)
                destino = os.path.join(destino_dir, archivo)
                print(f"📊 Moviendo tabla {archivo}  ➔  {os.path.relpath(destino, BASE_DIR)}")
                shutil.move(origen, destino)

    print("\n✅ ¡Organización completada con éxito!")

if __name__ == "__main__":
    organizar_archivos_sueltos()