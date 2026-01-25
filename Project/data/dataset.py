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
from tqdm import tqdm
from tokenizers import Tokenizer

# --- [FASE 1] DATA SANITIZATION (GIGO Principle) o vediamo se ora funziona eh ---
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
        # Padding dinamico al batch_max per efficienza computazionale
        code_padded = pad_sequence(code_seqs, batch_first=True, padding_value=self.pad_id)
        doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=self.pad_id)
        return code_padded, doc_padded

# --- [FASE 3] ASYMMETRIC CACHING ENGINE ---
class CodeSummaryDataset(Dataset):
    def __init__(self, data_dir, split_type, tokenizer_path, max_len_code=256, max_len_doc=64, subset=None):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len_code = max_len_code
        self.max_len_doc = max_len_doc
        self.data = []
        self.seen_hashes = set()

        # Pre-fetch ID token speciali
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.sos_id = self.tokenizer.token_to_id("<SOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

        search_pattern = os.path.join(data_dir, split_type, "*.jsonl.gz")
        files = glob.glob(search_pattern)
        
        if not files:
            raise FileNotFoundError(f"❌ Sorgente dati non trovata: {search_pattern}")

        print(f"🔍 [MEM_CACHE] Caricamento e Tokenizzazione '{split_type}'...")

        for file_path in files:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                # Usiamo tqdm per monitorare il caricamento iniziale (che sarà più lento, ma salva l'epoca)
                for line in f:
                    try:
                        item = json.loads(line)
                        code = item.get('code', item.get('func_code_string', ''))
                        doc = item.get('docstring', item.get('func_documentation_string', ''))
                        clean_doc = clean_docstring(doc)
                        
                        if len(clean_doc) > 10 and 10 < len(code.split()) < 250:
                            f_hash = compute_content_hash(code, clean_doc)
                            if f_hash not in self.seen_hashes:
                                self.seen_hashes.add(f_hash)
                                
                                # --- [CRITICO] TOKENIZZAZIONE A MONTE ---
                                # Trasformiamo subito in tensori per liberare la CPU durante il training
                                c_tokens = self.tokenizer.encode(code).ids
                                d_tokens = self.tokenizer.encode(clean_doc).ids
                                
                                code_tensor = torch.tensor([self.sos_id] + c_tokens[:self.max_len_code-2] + [self.eos_id])
                                doc_tensor = torch.tensor([self.sos_id] + d_tokens[:self.max_len_doc-2] + [self.eos_id])
                                
                                self.data.append((code_tensor, doc_tensor))
                        
                        if subset and len(self.data) >= subset: break
                    except: continue
                if subset and len(self.data) >= subset: break
        
        print(f"✅ Cache Pronta: {len(self.data)} campioni caricati in RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Restituiamo i tensori pre-calcolati: Zero overhead CPU durante l'epoca
        return self.data[idx]

# --- [FASE 4] HIGH-THROUGHPUT FACTORY ---
def get_dataloader(data_dir, split_type, tokenizer_path, batch_size=32, shuffle=True, subset=None):
    dataset = CodeSummaryDataset(
        data_dir, split_type, tokenizer_path, 
        max_len_code=256, 
        max_len_doc=64, 
        subset=subset
    )
    
    collator = SmartCollator(dataset.pad_id)
    
    # Su Windows con RTX 40, usiamo 4 workers e prefetch aggressivo
    cpu_count = multiprocessing.cpu_count()
    workers = min(cpu_count, 4) if torch.cuda.is_available() else 0
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True, # Velocizza il trasferimento PCIe verso la GPU
        prefetch_factor=2 if workers > 0 else None,
        persistent_workers=True if workers > 0 else False
    )