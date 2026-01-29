import os
import gzip
import json
import glob
import re
import hashlib
import logging
import io
import tokenize 
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DatasetCleaner:
    def __init__(self, min_code=20, max_code=250, min_doc=3, max_doc=50):
        self.min_code = min_code
        self.max_code = max_code
        self.min_doc = min_doc
        self.max_doc = max_doc
        self.global_seen_hashes = set()

    @staticmethod
    def clean_docstring(doc):
        """Pulisce la docstring (Target)."""
        if not doc: return ""
        doc = doc.split('\n\n')[0]
        doc = re.sub(r'(:param|:type|:return|:rtype|@param|@return|@throws|Args:|Returns:|Raises:).*', '', doc, flags=re.MULTILINE)
        doc = re.sub(r'>>>.*', '', doc)
        doc = re.sub(r'http\S+', '', doc)
        return ' '.join(doc.split()).strip()

    @staticmethod
    def remove_docstring_from_code(code):
        """
        Rimuove la docstring dal codice sorgente in modo robusto.
        Tenta prima con AST (più preciso), poi con Tokenizer, infine Regex.
        """
        if not code: return ""

        # --- TENTATIVO 1: AST (Abstract Syntax Tree) - Il metodo più sicuro ---
        # Richiede codice sintatticamente valido. Ottimo per Python 3.9+
        try:
            import ast
            parsed = ast.parse(code)
            
            # Naviga nell'albero per trovare funzioni/classi
            for node in ast.walk(parsed):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    if not node.body: continue
                    
                    # Controlla se il primo elemento del body è una stringa (Docstring)
                    if isinstance(node.body[0], ast.Expr) and \
                       isinstance(node.body[0].value, (ast.Str, ast.Constant)):
                        
                        # Check compatibilità versioni Python
                        val = node.body[0].value
                        is_docstring = False
                        if isinstance(val, ast.Str): is_docstring = True # Python < 3.8
                        elif isinstance(val, ast.Constant) and isinstance(val.value, str): is_docstring = True # Python >= 3.8
                        
                        if is_docstring:
                            node.body.pop(0) # Rimuovi il nodo docstring

            # Ricostruisci il codice (Normalizza anche l'indentazione!)
            if hasattr(ast, 'unparse'):
                return ast.unparse(parsed)
                
        except (SyntaxError, IndentationError, ImportError, AttributeError):
            # Se il codice è rotto o AST fallisce, passiamo al metodo manuale
            pass

        # --- TENTATIVO 2: Tokenizer (Fallback robusto) ---
        io_obj = io.BytesIO(code.encode('utf-8'))
        out = ""
        last_lineno = -1
        last_col = 0
        
        try:
            tokgen = tokenize.tokenize(io_obj.readline)
            for tok in tokgen:
                token_type = tok.type
                token_string = tok.string
                start_line, start_col = tok.start
                end_line, end_col = tok.end

                # Salta le stringhe di documentazione
                if token_type == tokenize.STRING:
                    if token_string.startswith('"""') or token_string.startswith("'''"):
                        continue
                
                # Ricostruzione
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += " " * (start_col - last_col)
                
                out += token_string
                last_col = end_col
                last_lineno = end_line
                
        except (tokenize.TokenError, IndentationError, SyntaxError):
            # --- TENTATIVO 3: Regex (Ultima spiaggia) ---
            # Se anche il tokenizer fallisce (codice molto rotto), usa la regex
            # Rimuove triple quotes grossolanamente
            out = re.sub(r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\')', '', code, count=1)
            return out.strip()

        return out.strip()

    def process_split(self, split_name, input_dir, output_dir, mode="filter"):
        search_path = os.path.join(input_dir, split_name, "*.jsonl.gz")
        files = glob.glob(search_path)
        
        if not files:
            logger.warning(f"Nessun file trovato per: {split_name}")
            return

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        
        stats = {"total": 0, "kept": 0, "removed_len": 0, "removed_dupe": 0, "removed_doc": 0}
        
        logger.info(f"Processing {split_name.upper()} -> {output_file}")

        with open(output_file, 'w', encoding='utf-8') as out_f:
            for file_path in tqdm(files, desc=f"Cleaning {split_name}"):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        stats["total"] += 1
                        try:
                            data = json.loads(line)
                            
                            raw_code = data.get('code', data.get('func_code_string', ''))
                            raw_doc = data.get('docstring', data.get('func_documentation_string', ''))
                            
                            if not raw_code: continue

                            # 1. Pulisci il TARGET (Docstring)
                            clean_doc = self.clean_docstring(raw_doc)
                            if not clean_doc or not any(c.isalnum() for c in clean_doc):
                                stats["removed_doc"] += 1
                                continue
                            
                            # 2. Pulisci l'INPUT (Codice) - RIMUOVI LA DOCSTRING DAL CODICE
                            clean_code = self.remove_docstring_from_code(raw_code)
                            
                            # Se rimuovendo la docstring non rimane codice (funzione vuota), scartala
                            if not clean_code.strip() or len(clean_code.strip()) < 10:
                                stats["removed_doc"] += 1 # Contiamo come removed bad code/doc
                                continue

                            # 3. Filtri Lunghezza (sui dati puliti)
                            code_len = len(clean_code.split())
                            doc_len = len(clean_doc.split())
                            
                            if not (self.min_code <= code_len <= self.max_code):
                                stats["removed_len"] += 1
                                continue
                            if not (self.min_doc <= doc_len <= self.max_doc):
                                stats["removed_len"] += 1
                                continue

                            # 4. Deduplicazione Cross-Split
                            code_norm = re.sub(r'\s+', '', clean_code) # Hash sul codice PULITO (senza docstring)
                            code_hash = hashlib.md5(code_norm.encode('utf-8')).hexdigest()
                            
                            if code_hash in self.global_seen_hashes:
                                stats["removed_dupe"] += 1
                                continue
                            
                            if mode == "build":
                                self.global_seen_hashes.add(code_hash)

                            # 5. Scrittura
                            # NOTA: Salviamo 'clean_code' come 'code', non 'raw_code'!
                            entry = {
                                "code": clean_code, 
                                "docstring": clean_doc
                            }
                            out_f.write(json.dumps(entry) + "\n")
                            stats["kept"] += 1

                        except json.JSONDecodeError:
                            continue
        
        logger.info(f"[{split_name.upper()}] Kept: {stats['kept']} | Dupes: {stats['removed_dupe']} | Len: {stats['removed_len']}")

    def run(self, input_dir, output_dir):
        self.process_split("train", input_dir, output_dir, mode="build")
        self.process_split("valid", input_dir, output_dir, mode="filter")
        self.process_split("test", input_dir, output_dir, mode="filter")

# --- MAIN ---
def main():

    DATASET_ROOT = "Project/Datasets/python/final/jsonl"
    OUTPUT_DIR = "Project/Datasets/processed"

    if not os.path.exists(DATASET_ROOT):
        logger.error(f"Cartella dataset non trovata: {DATASET_ROOT}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set globale per tracciare gli hash attraverso TUTTI gli split
    # Questo è il segreto per evitare data leakage.
    global_seen_hashes = set()

    cleaner = DatasetCleaner()
    cleaner.run(DATASET_ROOT, OUTPUT_DIR)
    
    """
    # L'ordine è FONDAMENTALE:
    # 1. Train (Definisce la conoscenza base)
    DatasetCleaner.process_split("train", DATASET_ROOT, OUTPUT_DIR, global_seen_hashes)
    
    # 2. Valid (Non deve contenere nulla che sia nel Train)
    DatasetCleaner.process_split("valid", DATASET_ROOT, OUTPUT_DIR, global_seen_hashes)
    
    # 3. Test (Non deve contenere nulla che sia in Train o Valid)
    DatasetCleaner.process_split("test", DATASET_ROOT, OUTPUT_DIR, global_seen_hashes)
    
    logger.info("Preprocessing Completo. File salvati in: " + OUTPUT_DIR)
    """

if __name__ == "__main__":
    main()