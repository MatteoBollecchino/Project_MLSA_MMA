"""
================================================================================
DATA PIPELINE & BATCHING UNIT
================================================================================
ROLE: High-Efficiency Data Loading and Tensor Synthesis.

DESIGN RATIONALE:
- Computational Economy: Uses Dynamic Padding via a custom Collator to minimize 
  the number of operations on <PAD> tokens.
- Latency Reduction: Implements RAM Caching for tokenized sequences, removing 
  JSON parsing and BPE encoding from the main training loop.
- Multi-Processing: Configures asynchronous data pre-fetching to ensure the 
  GPU 'never waits' for the next batch (Data Starvation prevention).
================================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import gzip
import os
import multiprocessing
from tokenizers import Tokenizer

# --- [PHASE 1] OPTIMIZED COLLATOR (Dynamic Padding) ---

class SmartCollator:
    """
    BATCH SYNTHESIZER: Implements the Dynamic Padding strategy.
    
    Standard padding to a fixed global length is computationally wasteful. 
    This class intercepts each batch and pads sequences to the length 
    of the LONGEST element in that specific batch.
    """
    def __init__(self, pad_id):
        """
        Args:
            pad_id (int): The integer ID used for <PAD> tokens (usually 0).
        """
        self.pad_id = pad_id

    def __call__(self, batch):
        """
        Transforms a raw list of samples into synchronized tensors.
        
        Logic:
        1. De-structure the batch into separate code and docstring streams.
        2. Apply 'pad_sequence' to create rectangular tensors [Batch, Max_Batch_Len].
        """
        # Unzip the batch into separate lists of code and docstring tensors.
        code_seqs, doc_seqs = zip(*batch)

        # [OPTIMIZATION] Dynamic Padding Logic:
        # If the longest code in this batch is 50 tokens, we pad to 50, even if 
        # the global maximum is 256. This reduces Transformer/LSTM overhead significantly.
        code_padded = pad_sequence(code_seqs, batch_first=True, padding_value=self.pad_id)
        doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=self.pad_id)

        return code_padded, doc_padded


# --- [PHASE 2] STREAMLINED DATASET (Consumer Mode) ---

class CodeSummaryDataset(Dataset):
    """
    MEM-RESIDENT DATASET: Optimized for high-speed gradient descent.
    
    Since the data has been purified by 'DatasetCleaner', this class focuses 
    on one-time tokenization and persistent RAM storage.
    """
    def __init__(self, data_dir, split_type, tokenizer_path, max_len_code=256, max_len_doc=64, subset=None):
        """
        Args:
            data_dir: Path to the 'processed/' directory.
            split_type: 'train', 'valid', or 'test'.
            tokenizer_path: Path to the pre-trained BPE model.
            max_len_code/doc: Hard safety caps for sequence length.
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len_code = max_len_code
        self.max_len_doc = max_len_doc
        self.data = []

        # Cache special IDs locally to eliminate repeated dictionary lookups.
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.sos_id = self.tokenizer.token_to_id("<SOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

        file_path = os.path.join(data_dir, f"{split_type}.jsonl.gz")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Refinery Error: Processed data not found at {file_path}")
        
        print(f"[MEM_CACHE] Loading refined '{split_type}' data into RAM...")

        # --- FAST LOADING & TOKENIZATION LOOP ---
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    # JSON Parsing: Each line is a JSON object with 'code' and 'docstring' fields.
                    item = json.loads(line)
                    
                    code = item.get('code', '')
                    doc = item.get('docstring', '')
                    
                    if not code or not doc: continue

                    # TOKENIZATION: Convert text to BPE sub-word units.
                    c_tokens = self.tokenizer.encode(code).ids
                    d_tokens = self.tokenizer.encode(doc).ids
                    
                    # TENSOR SYNTHESIS: Add structural tags (<SOS>/<EOS>) and truncate.
                    # Logic: [SOS] + Tokens + [EOS]. Hard truncation at max_len - 2.
                    code_ids = [self.sos_id] + c_tokens[:self.max_len_code-2] + [self.eos_id]
                    code_tensor = torch.tensor(code_ids, dtype=torch.long)
                    
                    doc_ids = [self.sos_id] + d_tokens[:self.max_len_doc-2] + [self.eos_id]
                    doc_tensor = torch.tensor(doc_ids, dtype=torch.long)
                    
                    # Persist the tensor pair in RAM for the duration of the training session.
                    self.data.append((code_tensor, doc_tensor))
                    
                    # If a subset size is specified, stop loading after reaching that number of samples.
                    if subset and len(self.data) >= subset: 
                        break
                        
                except json.JSONDecodeError:
                    continue

        print(f"Cache Ready: {len(self.data)} samples ready for stochastic gradient descent.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Zero-overhead access: Simply retrieves pre-computed tensors from RAM. """
        return self.data[idx]


# --- [PHASE 3] DATALOADER FACTORY ---

def get_dataloader(data_dir, split_type, tokenizer_path, batch_size=32, shuffle=True, subset=None):
    """
    PIPELINE CONFIGURATOR: Fuses the Dataset, Collator, and Multiprocessing logic.
    """
    # Initialize the high-speed dataset handler.
    dataset = CodeSummaryDataset(
        data_dir, 
        split_type, 
        tokenizer_path, 
        max_len_code=256, 
        max_len_doc=64, 
        subset=subset
    )
    
    # Inject the Dynamic Padding logic.
    collator = SmartCollator(dataset.pad_id)
    
    # HARDWARE OPTIMIZATION:
    # Logic: Use 4 parallel workers for pre-fetching if a GPU is available.
    # num_workers > 0 prevents the GPU from 'stalling' while waiting for CPU batch prep.
    cpu_count = multiprocessing.cpu_count()
    workers = min(cpu_count, 4) if torch.cuda.is_available() else 0
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,            # Allocates tensors in 'page-locked' memory for faster CPU->GPU transfer.
        prefetch_factor=2 if workers > 0 else None, # Pre-loads 2 batches per worker.
        persistent_workers=True if workers > 0 else False
    )