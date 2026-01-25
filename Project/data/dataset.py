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

# --- [FASE 1] DATA SANITIZATION & INTEGRITÀ (Principio GIGO) ---

def clean_docstring(doc):
    """
    Applica un filtro passa-basso semantico.
    Rimuove il 'rumore' tecnico (tag di documentazione, test) per lasciare
    solo la descrizione logica della funzione.
    """
    if not doc: return ""

    # Prendiamo solo la prima riga: è qui che risiede l'essenza (Headline)
    first_line = doc.split('\n\n')[0].split('\n')[0]
    
    # Regex per amputare tag Sphinx/Google-style (:param, @return, ecc.)
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)
    
    # Rimozione dei residui di test interattivi (doctest)
    clean = re.sub(r'>>>.*', '', clean)

    return clean.strip()

def compute_content_hash(code, doc):
    """
    Genera una 'impronta digitale' SHA-256 per la deduplicazione atomica.
    Impedisce al modello di barare imparando a memoria campioni duplicati.
    """
    # Normalizziamo gli spazi per evitare che la formattazione influenzi l'hash
    normalized = (re.sub(r'\s+', '', code) + doc).encode('utf-8')
    return hashlib.sha256(normalized).hexdigest()


# --- [FASE 2] OPTIMIZED COLLATOR (Padding Dinamico) ---

class SmartCollator:
    """
    Trasforma una lista di campioni in un Batch matematicamente coerente.
    """
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        # Separiamo codice e docstring dal batch
        code_seqs, doc_seqs = zip(*batch)

        # [OTTIMIZZAZIONE] Padding Dinamico:
        # Invece di riempire tutto a 256, riempiamo solo fino alla lunghezza 
        # massima del batch corrente. Questo accelera l'Attention di ordini di grandezza.
        code_padded = pad_sequence(code_seqs, batch_first=True, padding_value=self.pad_id)
        doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=self.pad_id)

        return code_padded, doc_padded


# --- [FASE 3] ASYMMETRIC CACHING ENGINE (Dataset Core) ---

class CodeSummaryDataset(Dataset):
    """
    Il magazzino dei dati. Implementa il caching in RAM per eliminare
    la latenza del filesystem durante il loop di training.
    """
    def __init__(self, data_dir, split_type, tokenizer_path, max_len_code=256, max_len_doc=64, subset=None):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len_code = max_len_code
        self.max_len_doc = max_len_doc
        self.data = []
        self.seen_hashes = set()

        # Estrazione ID speciali per evitare chiamate ripetute al tokenizer
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
                for line in f:
                    try:
                        item = json.loads(line)
                        code = item.get('code', item.get('func_code_string', ''))
                        doc = item.get('docstring', item.get('func_documentation_string', ''))
                        
                        clean_doc = clean_docstring(doc)
                        
                        # Filtro di Qualità: escludiamo frammenti troppo corti o troppo lunghi
                        if len(clean_doc) > 10 and 10 < len(code.split()) < 250:
                            f_hash = compute_content_hash(code, clean_doc)
                            
                            # Deduplicazione
                            if f_hash not in self.seen_hashes:
                                self.seen_hashes.add(f_hash)
                                
                                # --- [CRITICO] TOKENIZZAZIONE A MONTE ---
                                # Eseguiamo la trasformazione testo -> numeri QUI.
                                # Questo libera la CPU durante il training vero e proprio.
                                c_tokens = self.tokenizer.encode(code).ids
                                d_tokens = self.tokenizer.encode(clean_doc).ids
                                
                                # Assemblaggio tensore con Start-of-Sequence e End-of-Sequence
                                code_tensor = torch.tensor([self.sos_id] + c_tokens[:self.max_len_code-2] + [self.eos_id])
                                doc_tensor = torch.tensor([self.sos_id] + d_tokens[:self.max_len_doc-2] + [self.eos_id])
                                
                                self.data.append((code_tensor, doc_tensor))
                        
                        # Gestione Subset (per debugging rapido)
                        if subset and len(self.data) >= subset: break
                    except: continue
                if subset and len(self.data) >= subset: break
        
        print(f"✅ Cache Pronta: {len(self.data)} campioni caricati in RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Accesso diretto alla RAM: complessità O(1). Zero overhead CPU.
        return self.data[idx]


# --- [FASE 4] HIGH-THROUGHPUT FACTORY (DataLoader) ---

def get_dataloader(data_dir, split_type, tokenizer_path, batch_size=32, shuffle=True, subset=None):
    """
    Configura la pompa dei dati verso la GPU.
    """
    dataset = CodeSummaryDataset(
        data_dir, split_type, tokenizer_path, 
        max_len_code=256, 
        max_len_doc=64, 
        subset=subset
    )
    
    collator = SmartCollator(dataset.pad_id)
    
    # [OTTIMIZZAZIONE HARDWARE]
    # Su Windows usiamo workers paralleli e pin_memory per velocizzare
    # il trasferimento tramite il bus PCIe verso i Tensor Core della GPU.
    cpu_count = multiprocessing.cpu_count()
    workers = min(cpu_count, 4) if torch.cuda.is_available() else 0
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,            # Memoria RAM bloccata per trasferimento DMA rapido
        prefetch_factor=2 if workers > 0 else None, # Prepara i batch in anticipo
        persistent_workers=True if workers > 0 else False # Non distrugge i processi tra epoche
    )