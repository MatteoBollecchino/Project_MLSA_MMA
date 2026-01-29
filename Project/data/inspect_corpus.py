import os
import json
import gzip
import glob
import re

# --- CONFIGURAZIONE ---
# Sostituisci con il percorso dove hai salvato le cartelle (es. "python", "java", etc.)
JSONL_BASE_DIR = "Project/Datasets/python/final/jsonl" 
NUM_SAMPLES = 5  # Quanti esempi vuoi stampare

# --- TUE FUNZIONI DI PULIZIA (Copia-Incolla dal tuo codice) ---
def clean_docstring(doc):
    """ Cleans a docstring by removing noise such as Sphinx/doctest tags and taking only the first line. """
    if not doc: return ""
    first_line = doc.split('\n\n')[0].split('\n')[0]
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)
    clean = re.sub(r'>>>.*', '', clean)
    return clean.strip()

# --- FUNZIONE DI ISPEZIONE ---
def inspect_temp_content(base_dir, samples=5):
    """
    Replica la logica di prepare_vocab ma stampa a video invece di salvare su file.
    """
    search_pattern = os.path.join(base_dir, "train", "*.jsonl.gz")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print(f"ERRORE: Nessun file trovato in {search_pattern}")
        return

    print(f"Trovati {len(train_files)} file. Analisi del primo file: {os.path.basename(train_files[0])}...\n")
    print("="*60)

    count = 0
    
    # Apriamo solo il primo file per fare un controllo rapido
    with gzip.open(train_files[0], 'rt', encoding='utf-8') as f:
        for line in f:
            if count >= samples:
                break
            
            try:
                item = json.loads(line)
                
                # Estrazione (tua logica originale)
                code = item.get('code', item.get('func_code_string', ''))
                doc = item.get('docstring', item.get('func_documentation_string', ''))
                
                # Pulizia (tua logica originale)
                clean_doc = clean_docstring(doc)
                
                # Filtri (tua logica originale)
                if len(clean_doc) > 10 and len(code.split()) < 200:
                    count += 1
                    
                    # --- SIMULAZIONE OUTPUT NEL FILE TEMP ---
                    # Nel file temp verrebbe scritto: code + "\n" + clean_doc + "\n"
                    
                    print(f"--- ESEMPIO {count} ---")
                    print("[CONTENUTO CHE VERREBBE SCRITTO NEL FILE TEMP]:")
                    print("-" * 20)
                    print(f"{code}")      # Riga 1: Codice
                    print(f"{clean_doc}") # Riga 2: Docstring pulita
                    print("-" * 20)
                    print("\n")
                    
            except json.JSONDecodeError:
                continue

    print("="*60)
    print("Ispezione completata.")

# --- ESECUZIONE ---
if __name__ == "__main__":
    if os.path.exists(JSONL_BASE_DIR):
        inspect_temp_content(JSONL_BASE_DIR, NUM_SAMPLES)
    else:
        print(f"Attenzione: La cartella '{JSONL_BASE_DIR}' non esiste. Modifica la variabile JSONL_BASE_DIR.")