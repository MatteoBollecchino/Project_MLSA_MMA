import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import gzip
import os
import glob
import re
from tokenizers import Tokenizer

def clean_docstring(doc):
    """
    Stessa logica di pulizia usata nel preprocessing.
    Garantisce che il modello non veda 'rumore' Sphinx o doctest.
    """
    if not doc: return ""
    # Prendi solo la prima riga/frase
    first_line = doc.split('\n\n')[0].split('\n')[0]
    # Rimuovi tag tecnici (:param, @return, ecc.)
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)
    # Rimuovi esempi di codice doctest
    clean = re.sub(r'>>>.*', '', clean)
    return clean.strip()

class CodeSummaryDataset(Dataset):
    def __init__(self, data_dir, split_type, tokenizer_path, max_len=128, subset=None):
        """
        data_dir: Deve essere la sandbox (Project/Datasets/python/final/jsonl)
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.data = []

        # 1. Ricerca selettiva (No Junk Files)
        # Cerchiamo solo nella sottocartella dello split specifico
        search_pattern = os.path.join(data_dir, split_type, "*.jsonl.gz")
        files = glob.glob(search_pattern)
        
        if not files:
            raise FileNotFoundError(f"ERRORE: Nessun file .jsonl.gz trovato in {search_pattern}")

        print(f"--- Caricamento {split_type} set: {len(files)} file rilevati ---")

        # 2. Caricamento e Sanitizzazione
        for file_path in files:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    raw_code = item.get('code', item.get('func_code_string', ''))
                    raw_doc = item.get('docstring', item.get('func_documentation_string', ''))
                    
                    # Applichiamo il filtro di qualità
                    clean_doc = clean_docstring(raw_doc)
                    
                    # Criteri di inclusione: solo coppie "sensate"
                    if len(clean_doc) > 10 and 10 < len(raw_code.split()) < 200:
                        self.data.append({'code': raw_code, 'doc': clean_doc})
                    
                    if subset and len(self.data) >= subset:
                        break
            if subset and len(self.data) >= subset:
                break
        
        print(f"Esempi caricati per {split_type}: {len(self.data)}")

        # Special Tokens
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.sos_id = self.tokenizer.token_to_id("<SOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # TRANSFORM: Da testo pulito a ID numerici
        code_tokens = self.tokenizer.encode(item['code']).ids
        code_ids = [self.sos_id] + code_tokens[:self.max_len-2] + [self.eos_id]
        
        doc_tokens = self.tokenizer.encode(item['doc']).ids
        doc_ids = [self.sos_id] + doc_tokens[:self.max_len-2] + [self.eos_id]
        
        return torch.tensor(code_ids), torch.tensor(doc_ids)

def collate_fn(batch, pad_id=0):
    code_seqs, doc_seqs = zip(*batch)
    code_padded = pad_sequence(code_seqs, batch_first=True, padding_value=pad_id)
    doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=pad_id)
    return code_padded, doc_padded

def get_dataloader(data_dir, split_type, tokenizer_path, batch_size=32, shuffle=True, subset=None):
    dataset = CodeSummaryDataset(data_dir, split_type, tokenizer_path, subset=subset)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda b: collate_fn(b, dataset.pad_id)
    )