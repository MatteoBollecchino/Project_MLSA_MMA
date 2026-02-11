"""
================================================================================
TOKENIZATION ENGINE
================================================================================
ROLE: Linguistic Pre-processing and Vocabulary Synthesis.

DESIGN RATIONALE:
- Byte-Pair Encoding (BPE): Balanced approach between character-level and 
  word-level tokenization. Essential for Python where variable names are 
  highly dynamic.
- Atomic Data Handling: Reads directly from compressed streams to build a 
  monolithic training corpus for the BPE trainer.
- Structural Integrity: Reserves specific slots for control tokens (<PAD>, 
  <SOS>, etc.) to ensure the Seq2Seq logic remains deterministic.
================================================================================
"""

import os
import json
import gzip
import logging
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Module-level logger for auditing the vocabulary construction process.
logger = logging.getLogger(__name__)

# Target vocabulary size: A trade-off between semantic resolution and VRAM usage.
VOCAB_SIZE = 20000

def train_bpe_tokenizer(files, save_path, vocab_size=VOCAB_SIZE):
    """ 
    Executes the training of the sub-word tokenizer using the HuggingFace library.
    
    Args:
        files (list): List of paths to plain text corpus files.
        save_path (str): Final destination for the serialized .json tokenizer.
        vocab_size (int): Total unique tokens permitted in the vocabulary.
    """
    # 1. ARCHITECTURE INITIALIZATION: 
    # Logic: Start with a BPE model that handles unknown tokens via a dedicated <UNK> tag.
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    
    # 2. PRE-TOKENIZATION LAYER:
    # ByteLevel splits by whitespace but protects unique characters. 
    # This is standard for source code to maintain operational logic symbols.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 3. TRAINER PROTOCOL:
    # Special tokens are registered at indices 0-3 to ensure consistent tensor alignment 
    # during the padding and sequence initiation phases of the Transformer/LSTM.
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"],
        show_progress=True
    )
    
    logger.info(f"Training BPE Tokenizer (Target Vocab: {vocab_size})...")

    # 4. TRAINING EXECUTION:
    # Iteratively merges the most frequent adjacent character pairs into sub-words.
    tokenizer.train(files, trainer)

    # 5. PERSISTENCE:
    # Serialize the trained state to disk to avoid re-training in future sessions.
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    tokenizer.save(save_path)
    logger.info(f"Tokenizer saved to: {save_path}")

def build_tokenizer(processed_data_dir, save_path, vocab_size=VOCAB_SIZE):
    """ 
    CORPUS REFINERY: Prepares a monolithic text stream from pre-processed JSONL 
    data to feed the BPE trainer.
    
    Logic: Prioritizes Training Data to ensure the model's 'world-view' is based 
    solely on the allowed optimization set.
    """
    
    # TARGETING TRAIN SPLIT: Using valid/test for vocab building is a form of 
    # implicit Data Leakage. We strictly use the refined train.jsonl.gz.
    train_file_path = os.path.join(processed_data_dir, "train.jsonl.gz")
    
    if not os.path.exists(train_file_path):
        logger.error(f"Critical Error: Processed training file missing at {train_file_path}")
        return

    # INTERMEDIATE CORPUS STORAGE: A temporary text file to hold the raw text 
    # before the BPE trainer scans it.
    temp_corpus_file = "temp_corpus.txt"
    logger.info(f"Extracting corpus from {train_file_path}...")
    
    try:
        with open(temp_corpus_file, 'w', encoding='utf-8') as out_f:
            # STREAM DECOMPRESSION: Processing the GZIP stream line-by-line 
            # to minimize the memory footprint of the refinery process.
            with gzip.open(train_file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        # JSON PARSING: Each line is expected to be a JSON object with 'code' and 'docstring' fields (or their variants).
                        data = json.loads(line)
                        
                        # DATA EXTRACTION:
                        # Logic: Merge both Code logic and Docstring summaries into the 
                        # corpus so the tokenizer learns common tokens for both domains.
                        code = data.get('code', '')
                        doc = data.get('docstring', '')
                        
                        if code and doc:
                            # Concatenate with newlines to ensure distinct samples.
                            out_f.write(code + "\n" + doc + "\n")
                            
                    except json.JSONDecodeError:
                        continue # Defensive skip for corrupted JSON shards.
        
        # TRANSITION TO TRAINING: Execute the mathematical merging process.
        train_bpe_tokenizer([temp_corpus_file], save_path, vocab_size)
        
    finally:
        # VOLATILE CLEANUP: Remove the large intermediate text file to free disk space.
        if os.path.exists(temp_corpus_file):
            os.remove(temp_corpus_file)

# --- MASTER EXECUTION BLOCK ---
if __name__ == "__main__":
    """
    Standalone utility configuration. 
    Can be run independently of the C2Orchestrator for rapid vocab prototyping.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Path to refined data (Output of DatasetCleaner).
    PROCESSED_DIR = "Project/Datasets/processed"
    
    # Target path for the final Symbolic Mapping file.
    TOKENIZER_PATH = "Project/tokenizer.json"
    
    build_tokenizer(PROCESSED_DIR, TOKENIZER_PATH, vocab_size=VOCAB_SIZE)