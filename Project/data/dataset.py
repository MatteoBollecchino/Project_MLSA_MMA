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

# --- [PHASE 1] DATA SANITIZATION ---

# Extracts the essence of the docstring by removing technical noise.
def clean_docstring(doc):
    if not doc: return ""

    # Focus on the first line and remove common annotations
    first_line = doc.split('\n\n')[0].split('\n')[0]
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)
    clean = re.sub(r'>>>.*', '', clean)

    return clean.strip()

# Deterministic fingerprint for deduplication.
def compute_content_hash(code, doc):
    normalized = (re.sub(r'\s+', '', code) + doc).encode('utf-8')
    return hashlib.sha256(normalized).hexdigest()

# --- [PHASE 2] OPTIMIZED COLLATOR ---
class SmartCollator:
    # Stores the padding token ID (usually 0). This number will be used to "fill" empty gaps.
    def __init__(self, pad_id):
        self.pad_id = pad_id

    # Dynamic padding based on the longest sequence in the batch
    def __call__(self, batch):
        code_seqs, doc_seqs = zip(*batch)

        # Dynamic padding to batch_max for computational efficiency
        code_padded = pad_sequence(code_seqs, batch_first=True, padding_value=self.pad_id)
        doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=self.pad_id)

        return code_padded, doc_padded

# --- [PHASE 3] ASYMMETRIC CACHING ENGINE ---
class CodeSummaryDataset(Dataset):

    # Initializes the dataset by loading and tokenizing data into memory.
    def __init__(self, data_dir, split_type, tokenizer_path, max_len_code=256, max_len_doc=64, subset=None):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len_code = max_len_code
        self.max_len_doc = max_len_doc
        self.data = []
        self.seen_hashes = set()

        # Pre-fetch special token IDs
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.sos_id = self.tokenizer.token_to_id("<SOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

        # Load and tokenize data into memory
        search_pattern = os.path.join(data_dir, split_type, "*.jsonl.gz")
        files = glob.glob(search_pattern)
        
        if not files:
            raise FileNotFoundError(f"❌ Data source not found: {search_pattern}")

        print(f"🔍 [MEM_CACHE] Loading and Tokenizing '{split_type}'...")

        for file_path in files:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                # Use tqdm to monitor the initial loading (which will be slower but saves the epoch)
                for line in f:
                    try:

                        # Parse JSON line and extract relevant fields
                        item = json.loads(line)
                        code = item.get('code', item.get('func_code_string', ''))
                        doc = item.get('docstring', item.get('func_documentation_string', ''))

                        # Data sanitization
                        clean_doc = clean_docstring(doc)
                        
                        if len(clean_doc) > 10 and 10 < len(code.split()) < 250:
                            f_hash = compute_content_hash(code, clean_doc)
                            if f_hash not in self.seen_hashes:
                                self.seen_hashes.add(f_hash)
                                
                                # --- [CRITICAL] PRE-TOKENIZATION ---
                                # Immediately convert to tensors to free up CPU during training
                                c_tokens = self.tokenizer.encode(code).ids
                                d_tokens = self.tokenizer.encode(clean_doc).ids
                                
                                # Create tensors with SOS and EOS tokens
                                code_tensor = torch.tensor([self.sos_id] + c_tokens[:self.max_len_code-2] + [self.eos_id])
                                doc_tensor = torch.tensor([self.sos_id] + d_tokens[:self.max_len_doc-2] + [self.eos_id])
                                
                                self.data.append((code_tensor, doc_tensor))
                        
                        if subset and len(self.data) >= subset: break
                    except: continue
                if subset and len(self.data) >= subset: break
        
        print(f"✅ Cache Ready: {len(self.data)} samples loaded into RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return pre-calculated tensors: Zero CPU overhead during epoch
        return self.data[idx]

# --- [PHASE 4] HIGH-THROUGHPUT FACTORY ---

# Returns the dataloader
def get_dataloader(data_dir, split_type, tokenizer_path, batch_size=32, shuffle=True, subset=None):
    dataset = CodeSummaryDataset(
        data_dir, split_type, tokenizer_path, 
        max_len_code=256, 
        max_len_doc=64, 
        subset=subset
    )
    
    collator = SmartCollator(dataset.pad_id)
    
    # On Windows with RTX 40, we use 4 workers and aggressive prefetching
    cpu_count = multiprocessing.cpu_count()
    workers = min(cpu_count, 4) if torch.cuda.is_available() else 0
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True, # Speeds up PCIe transfer to GPU
        prefetch_factor=2 if workers > 0 else None,
        persistent_workers=True if workers > 0 else False
    )