import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import gzip
import os
import glob
import re
import hashlib
import multiprocessing
from tokenizers import Tokenizer

# --- [FASE 1] DATA SANITIZATION (Principio GIGO: Garbage In, Garbage Out) TEST MARCO---
def clean_docstring(doc):
    """Estrae l'essenza della docstring rimuovendo il rumore tecnico."""
    if not doc: return ""
    first_line = doc.split('\n\n')[0].split('\n')[0]
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)
    clean = re.sub(r'>>>.*', '', clean)
    return clean.strip()

def compute_content_hash(code, doc):
    """Fingerprint deterministico per la deduplicazione."""
    normalized = (re.sub(r'\s+', '', code) + doc).encode('utf-8')
    return hashlib.sha256(normalized).hexdigest()

# --- [FASE 2] OPTIMIZED COLLATOR ---
class SmartCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        code_seqs, doc_seqs = zip(*batch)
        # Il padding dinamico al batch_max ottimizza la memoria rispetto al padding statico
        code_padded = pad_sequence(code_seqs, batch_first=True, padding_value=self.pad_id)
        doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=self.pad_id)
        return code_padded, doc_padded

# --- [FASE 3] ASYMMETRIC DATASET ENGINE ---
class CodeSummaryDataset(Dataset):
    def __init__(self, data_dir, split_type, tokenizer_path, max_len_code=256, max_len_doc=64, subset=None):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len_code = max_len_code
        self.max_len_doc = max_len_doc
        self.data = []
        self.seen_hashes = set()

        search_pattern = os.path.join(data_dir, split_type, "*.jsonl.gz")
        files = glob.glob(search_pattern)
        
        if not files:
            raise FileNotFoundError(f"❌ Sorgente dati non trovata: {search_pattern}")

        print(f"🔍 [IO_AUDIT] Caricamento '{split_type}' in corso...")

        for file_path in files:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        code = item.get('code', item.get('func_code_string', ''))
                        doc = item.get('docstring', item.get('func_documentation_string', ''))
                        
                        clean_doc = clean_docstring(doc)
                        
                        # Filtro di Qualità R&D
                        if len(clean_doc) > 10 and 10 < len(code.split()) < 250:
                            f_hash = compute_content_hash(code, clean_doc)
                            if f_hash not in self.seen_hashes:
                                self.seen_hashes.add(f_hash)
                                self.data.append({'code': code, 'doc': clean_doc})
                        
                        if subset and len(self.data) >= subset: break
                    except: continue
                if subset and len(self.data) >= subset: break
        
        print(f"✅ Set '{split_type}' pronto: {len(self.data)} campioni univoci.")
        
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.sos_id = self.tokenizer.token_to_id("<SOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenizzazione asimmetrica
        c_tokens = self.tokenizer.encode(item['code']).ids
        code_ids = [self.sos_id] + c_tokens[:self.max_len_code-2] + [self.eos_id]
        
        d_tokens = self.tokenizer.encode(item['doc']).ids
        doc_ids = [self.sos_id] + d_tokens[:self.max_len_doc-2] + [self.eos_id]
        
        return torch.tensor(code_ids), torch.tensor(doc_ids)

# --- [FASE 4] HIGH-THROUGHPUT FACTORY ---
def get_dataloader(data_dir, split_type, tokenizer_path, batch_size=32, shuffle=True, subset=None):
    dataset = CodeSummaryDataset(
        data_dir, split_type, tokenizer_path, 
        max_len_code=256, 
        max_len_doc=64, 
        subset=subset
    )
    
    collator = SmartCollator(dataset.pad_id)
    
    # --- LOGICA DI SBLOCCO GPU (Mental Model: Parallel Prefetching) ---
    # Non usiamo più workers=0 su Windows. Forza l'uso della CPU per preparare i dati.
    # Regola empirica: num_workers = num_cores_cpu / 2
    cpu_count = multiprocessing.cpu_count()
    workers = min(cpu_count, 4) if torch.cuda.is_available() else 0
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collator,
        num_workers=workers,           # Sblocca il caricamento parallelo
        pin_memory=True,                # Accelera il trasferimento RAM -> VRAM
        prefetch_factor=2 if workers > 0 else None, # Pre-carica i batch successivi
        persistent_workers=True if workers > 0 else False # Mantiene i worker vivi tra le epoche
    )