import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import gzip
import os
import glob
import re
from tokenizers import Tokenizer

# DATA PREPROCESSING 
def clean_docstring(doc):
    """
    Removes noise from docstrings (GIGO Principle).
    Takes only the first line and removes Sphinx/doctest tags.
    """
    if not doc: return ""

    # 1. Take only the first part before an empty line or a period
    first_line = doc.split('\n\n')[0].split('\n')[0]

    # 2. Remove tags like :param:, :return:, @param etc.
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)

    # 3. Remove any leftover doctest residues (>>>)
    clean = re.sub(r'>>>.*', '', clean)

    return clean.strip()

class CodeSummaryDataset(Dataset):
    def __init__(self, data_dir, split_type, tokenizer_path, max_len=128, subset=None):
        """
        data_dir: Should be the sandbox (Project/Datasets/python/final/jsonl)
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.data = []

        # 1. Selective search (No Junk Files)
        # We look only in the specific split subfolder
        search_pattern = os.path.join(data_dir, split_type, "*.jsonl.gz")
        files = glob.glob(search_pattern)
        
        if not files:
            raise FileNotFoundError(f"ERROR: No .jsonl.gz files found in {search_pattern}")

        print(f"--- Loading {split_type} set: {len(files)} files detected ---")

        # 2. Loading and Sanitization
        for file_path in files:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    # Load JSON line
                    item = json.loads(line)

                    # Extract raw code and docstring
                    raw_code = item.get('code', item.get('func_code_string', ''))
                    raw_doc = item.get('docstring', item.get('func_documentation_string', ''))
                    
                    # Apply quality filter
                    clean_doc = clean_docstring(raw_doc)
                    
                    # Inclusion criteria: only "sensible" pairs
                    if len(clean_doc) > 10 and 10 < len(raw_code.split()) < 200:
                        self.data.append({'code': raw_code, 'doc': clean_doc})
                    
                    if subset and len(self.data) >= subset:
                        break
            if subset and len(self.data) >= subset:
                break
        
        print(f"Examples loaded for {split_type}: {len(self.data)}")
        
        # Special Tokens
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.sos_id = self.tokenizer.token_to_id("<SOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

    # LENGTH OF DATASET
    def __len__(self):
        return len(self.data)

    # GET ITEM BY INDEX
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # TRANSFORM: From clean text to numeric IDs
        code_tokens = self.tokenizer.encode(item['code']).ids
        code_ids = [self.sos_id] + code_tokens[:self.max_len-2] + [self.eos_id]
        
        doc_tokens = self.tokenizer.encode(item['doc']).ids
        doc_ids = [self.sos_id] + doc_tokens[:self.max_len-2] + [self.eos_id]
        
        return torch.tensor(code_ids), torch.tensor(doc_ids)

# COLLATE FUNCTION FOR DATALOADER
def collate_fn(batch, pad_id=0):

    # Unzip the batch
    code_seqs, doc_seqs = zip(*batch)

    # Pad sequences to the maximum length in the batch
    code_padded = pad_sequence(code_seqs, batch_first=True, padding_value=pad_id)
    doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=pad_id)

    return code_padded, doc_padded

# DATA LOADER CREATION FUNCTION
def get_dataloader(data_dir, split_type, tokenizer_path, batch_size=32, shuffle=True, subset=None):

    # Initialize the dataset
    dataset = CodeSummaryDataset(data_dir, split_type, tokenizer_path, subset=subset)

    # Return the DataLoader
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda b: collate_fn(b, dataset.pad_id)
    )