"""
================================================================================
CORPUS INSPECTION TOOL - Debug Utility
================================================================================
ROLE: Visual Validation of Data Cleaning and Extraction Logic.

DESIGN RATIONALE:
- Transparency: Provides a real-time console preview of the 'purified' output 
  that would otherwise be hidden inside temporary text files or BPE tokens.
- Heuristic Testing: Allows the developer to verify if the length filters 
  (e.g., code < 200 words) are correctly excluding structural noise.
- Logic Decoupling: Replicates the production cleaning functions in a safe, 
  read-only environment.
================================================================================
"""

import os
import json
import gzip
import glob
import re

# NON-FUNCTIONAL FILE: Used exclusively for debugging and structural inspection.

# --- CONFIGURATION ---
# Define the root directory for raw JSONL data across various languages.
JSONL_BASE_DIR = "Project/Datasets/python/final/jsonl" 
NUM_SAMPLES = 5  # Control parameter for inspection depth.

# --- DATA PURIFICATION LOGIC (Replicated from Production) ---
def clean_docstring(doc):
    """ 
    CLEANING LAYER: Strips structural noise and metadata from docstrings. 
    
    Technical Details:
    - Slicing: Targets only the first functional paragraph to avoid bloating.
    - Regex Filtering: Removes common documentation tags (:param, @return) 
      which are statistically redundant for summarization tasks.
    - Doctest Purge: Removes lines starting with '>>>' to prevent training 
      on execution examples.
    """
    if not doc: return ""
    # Capture only the head of the documentation.
    first_line = doc.split('\n\n')[0].split('\n')[0]
    
    # Strip formal parameter definitions.
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)
    # Strip Python console examples.
    clean = re.sub(r'>>>.*', '', clean)
    return clean.strip()

# --- INSPECTION ENGINE ---
def inspect_temp_content(base_dir, samples=5):
    """
    Simulates the vocabulary preparation phase but redirects output to the 
    standard output (Stdout) instead of the filesystem.
    """
    # Glob-based discovery to find compressed training shards.
    search_pattern = os.path.join(base_dir, "train", "*.jsonl.gz")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print(f"ERRORE: Nessun file trovato in {search_pattern}")
        return

    print(f"Trovati {len(train_files)} file. Analisi del primo file: {os.path.basename(train_files[0])}...\n")
    print("="*60)

    count = 0
    
    # OPEN STREAM: Read the compressed file in 'rt' (Read-Text) mode.
    # Logic: Efficiently decompress on-the-fly to minimize RAM footprint.
    with gzip.open(train_files[0], 'rt', encoding='utf-8') as f:
        for line in f:
            # Throttle output to the requested sample count.
            if count >= samples:
                break
            
            try:
                item = json.loads(line)
                
                # SCHEMA ALIGNMENT: Handle potential variations in JSON keys.
                code = item.get('code', item.get('func_code_string', ''))
                doc = item.get('docstring', item.get('func_documentation_string', ''))
                
                # Apply the same sanitization used in the main pipeline.
                clean_doc = clean_docstring(doc)
                
                # FILTRATION AUDIT:
                # Simulates the threshold logic: ignore short docstrings or overly complex code.
                if len(clean_doc) > 10 and len(code.split()) < 200:
                    count += 1
                    
                    # --- OUTPUT SIMULATION ---
                    # Visually separates the Code (Input) from the Docstring (Ground Truth).
                    print(f"--- SAMPLE #{count} ---")
                    print("[CONTENUTO CHE VERREBBE SCRITTO NEL FILE TEMP]:")
                    print("-" * 20)
                    print(f"{code}")      # Phase 1: Raw Code block.
                    print(f"{clean_doc}") # Phase 2: Sanitized documentation string.
                    print("-" * 20)
                    print("\n")
                    
            except json.JSONDecodeError:
                continue # Gracefully skip malformed JSON shards.

    print("="*60)
    print("Ispezione completata.")

# --- EXECUTION GATE ---
if __name__ == "__main__":
    # Ensure directory existence before initiating the audit.
    if os.path.exists(JSONL_BASE_DIR):
        inspect_temp_content(JSONL_BASE_DIR, NUM_SAMPLES)
    else:
        print(f"Attenzione: La cartella '{JSONL_BASE_DIR}' non esiste. Modifica la variabile JSONL_BASE_DIR.")