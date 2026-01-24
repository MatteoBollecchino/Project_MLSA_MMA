import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import gzip
import os
import glob
import re
import hashlib
from tokenizers import Tokenizer

# --- [FASE 1] DATA SANITIZATION (GIGO Principle) ---
def clean_docstring(doc):
    """Estrae l'essenza (Headline) della docstring rimuovendo il boilerplate tecnico."""
    if not doc: return ""
    # 1. Prendiamo solo la prima riga (Headline)
    first_line = doc.split('\n\n')[0].split('\n')[0]
    # 2. Pulizia tag tecnici (Sphinx/Google style)
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)
    # 3. Rimozione residui di doctest
    clean = re.sub(r'>>>.*', '', clean)
    return clean.strip()

def compute_content_hash(code, doc):
    """Genera un fingerprint per la deduplicazione deterministica."""
    # Normalizzazione per evitare che spazi diversi creino hash diversi
    normalized = (re.sub(r'\s+', '', code) + doc).encode('utf-8')
    return hashlib.sha256(normalized).hexdigest()

# --- [FASE 2] SERIALIZABLE COLLATOR (Windows/Linux Agnostic) ---
class SmartCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        code_seqs, doc_seqs = zip(*batch)
        # Il padding viene fatto al valore massimo del BATCH corrente (efficienza)
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
            raise FileNotFoundError(f"❌ Nessun file trovato in {search_pattern}")

        print(f"🔍 [AUDIT {split_type.upper()}] Caricamento asimmetrico in corso...")

        for file_path in files:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        code = item.get('code', item.get('func_code_string', ''))
                        doc = item.get('docstring', item.get('func_documentation_string', ''))
                        
                        clean_doc = clean_docstring(doc)
                        
                        # FILTRI DI QUALITÀ (Review: 10 parole < codice < 200 parole)
                        # Il codice viene tagliato a 256 token, quindi carichiamo campioni ragionevoli
                        if len(clean_doc) > 10 and 10 < len(code.split()) < 250:
                            f_hash = compute_content_hash(code, clean_doc)
                            if f_hash not in self.seen_hashes:
                                self.seen_hashes.add(f_hash)
                                self.data.append({'code': code, 'doc': clean_doc})
                        
                        if subset and len(self.data) >= subset: break
                    except: continue
                if subset and len(self.data) >= subset: break
        
        print(f"✅ {split_type.upper()} pronto: {len(self.data)} campioni univoci.")
        
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.sos_id = self.tokenizer.token_to_id("<SOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encoding asimmetrico: più spazio al codice, meno al riassunto
        c_tokens = self.tokenizer.encode(item['code']).ids
        code_ids = [self.sos_id] + c_tokens[:self.max_len_code-2] + [self.eos_id]
        
        d_tokens = self.tokenizer.encode(item['doc']).ids
        doc_ids = [self.sos_id] + d_tokens[:self.max_len_doc-2] + [self.eos_id]
        
        return torch.tensor(code_ids), torch.tensor(doc_ids)

# --- [FASE 4] DATALOADER FACTORY ---
def get_dataloader(data_dir, split_type, tokenizer_path, batch_size=32, shuffle=True, subset=None):
    # Passiamo le lunghezze asimmetriche al dataset
    dataset = CodeSummaryDataset(
        data_dir, split_type, tokenizer_path, 
        max_len_code=256, 
        max_len_doc=64, 
        subset=subset
    )
    
    collator = SmartCollator(dataset.pad_id)
    
    # Adaptive Multiprocessing
    is_windows = os.name == 'nt'
    workers = 0 if (is_windows or (subset and subset < 100)) else 2

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if workers > 0 else None
    )