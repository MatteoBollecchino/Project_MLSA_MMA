import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import gzip
import os
import multiprocessing
from tokenizers import Tokenizer

# --- [PHASE 1] OPTIMIZED COLLATOR (Dynamic Padding) ---
# (Questa parte rimane identica ed è eccellente)
class SmartCollator:
    """
    Transforms a list of samples into a mathematically coherent Batch.
    Uses Dynamic Padding to minimize computation.
    """
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        # Separate code and docstring from the batch
        code_seqs, doc_seqs = zip(*batch)

        # [OPTIMIZATION] Dynamic Padding:
        # Pad only to the longest sequence in THIS batch, not the global max.
        code_padded = pad_sequence(code_seqs, batch_first=True, padding_value=self.pad_id)
        doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=self.pad_id)

        return code_padded, doc_padded


# --- [PHASE 2] STREAMLINED DATASET (Consumer Mode) ---

class CodeSummaryDataset(Dataset):
    """
    Lean & Mean Dataset Class.
    Since data is already preprocessed (cleaned, deduped, filtered),
    this class focuses purely on Tokenization and RAM Caching.
    """
    def __init__(self, data_dir, split_type, tokenizer_path, max_len_code=256, max_len_doc=64, subset=None):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len_code = max_len_code
        self.max_len_doc = max_len_doc
        self.data = []

        # Extract special IDs to avoid repeated lookups
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.sos_id = self.tokenizer.token_to_id("<SOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

        # Direct path to the preprocessed file (e.g., processed/train.jsonl.gz)
        file_path = os.path.join(data_dir, f"{split_type}.jsonl.gz")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Processed data not found: {file_path}")
        
        print(f"🔍 [MEM_CACHE] Loading processed '{split_type}' data...")

        # --- FAST LOADING LOOP ---
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    
                    # Direct access: Data is guaranteed to have these keys by DatasetCleaner
                    code = item.get('code', '')
                    doc = item.get('docstring', '')
                    
                    if not code or not doc: continue

                    # --- TOKENIZATION ---
                    # Transform text to integers using the pre-trained BPE tokenizer
                    c_tokens = self.tokenizer.encode(code).ids
                    d_tokens = self.tokenizer.encode(doc).ids
                    
                    # --- TENSORIZATION & TRUNCATION ---
                    # We still truncate here to ensure GPU memory safety, 
                    # even if the cleaner filtered by word count.
                    
                    # Create Code Tensor: [SOS] + tokens + [EOS]
                    code_ids = [self.sos_id] + c_tokens[:self.max_len_code-2] + [self.eos_id]
                    code_tensor = torch.tensor(code_ids, dtype=torch.long)
                    
                    # Create Doc Tensor: [SOS] + tokens + [EOS]
                    doc_ids = [self.sos_id] + d_tokens[:self.max_len_doc-2] + [self.eos_id]
                    doc_tensor = torch.tensor(doc_ids, dtype=torch.long)
                    
                    # Add to RAM Cache
                    self.data.append((code_tensor, doc_tensor))
                    
                    # Subset for debugging
                    if subset and len(self.data) >= subset: 
                        break
                        
                except json.JSONDecodeError:
                    continue

        print(f"✅ Cache Ready: {len(self.data)} samples loaded into RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Zero overhead access
        return self.data[idx]


# --- [PHASE 3] DATALOADER FACTORY ---

def get_dataloader(data_dir, split_type, tokenizer_path, batch_size=32, shuffle=True, subset=None):
    """
    Configures the data pipeline.
    """
    # Initialize the lean dataset
    dataset = CodeSummaryDataset(
        data_dir, 
        split_type, 
        tokenizer_path, 
        max_len_code=256, 
        max_len_doc=64, 
        subset=subset
    )
    
    collator = SmartCollator(dataset.pad_id)
    
    # Hardware Optimization
    cpu_count = multiprocessing.cpu_count()
    # On Windows/Mac sometimes too many workers cause overhead. 4 is a safe sweet spot.
    workers = min(cpu_count, 4) if torch.cuda.is_available() else 0
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,            # Speed up CPU -> GPU transfer
        prefetch_factor=2 if workers > 0 else None,
        persistent_workers=True if workers > 0 else False
    )