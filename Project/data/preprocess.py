import os
import json
import gzip
import glob
import logging
import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# LOGGING CONFIGURATION
logger = logging.getLogger(__name__)

# DATA PREPROCESSING 
# Removes noise from docstrings (GIGO Principle). Takes only the first line and removes Sphinx/doctest tags.
def clean_docstring(doc):
    if not doc: return ""

    # 1. Take only the first part before an empty line or a period
    first_line = doc.split('\n\n')[0].split('\n')[0]

    # 2. Remove tags like :param:, :return:, @param etc.
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)

    # 3. Remove any leftover doctest residues (>>>)
    clean = re.sub(r'>>>.*', '', clean)

    return clean.strip()

# TOKENIZER TRAINING
def train_bpe_tokenizer(files, save_path, vocab_size=20000):

    # Initialize a Byte-Level BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    # Configure pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Setup trainer with special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    )
    
    logger.info(f"Training BPE Tokenizer (Vocab: {vocab_size})...")

    # Train the tokenizer on the provided files
    tokenizer.train(files, trainer)

    # Save the trained tokenizer to disk
    tokenizer.save(save_path)

# PREPARE VOCABULARY FROM DATASET
# Scans ONLY the Python data folder and extracts the clean corpus.
def prepare_vocab(jsonl_base_dir, tokenizer_path):
    
    # Sandboxing: we target only the train folder of the extracted data
    search_pattern = os.path.join(jsonl_base_dir, "train", "*.jsonl.gz")
    train_files = glob.glob(search_pattern)
    
    if not train_files:
        logger.error(f"Empty sandbox! No files found in {search_pattern}")
        return

    # Temporary file to hold the clean corpus
    temp_text_file = "temp_corpus.txt"
    logger.info(f"Processing {len(train_files)} files in sandbox...")
    
    try:
        # Extract clean code-doc pairs and write to temp file
        with open(temp_text_file, 'w', encoding='utf-8') as out:

            # Iterate through all training files
            for file_path in train_files:
                # rt = read text mode
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        # Load JSON line
                        item = json.loads(line)

                        # Extract code and docstring fields
                        code = item.get('code', item.get('func_code_string', ''))
                        doc = item.get('docstring', item.get('func_documentation_string', ''))
                        
                        # Clean the docstring
                        clean_doc = clean_docstring(doc)
                        
                        # Quality Filter: avoid too short noise or huge code
                        if len(clean_doc) > 10 and len(code.split()) < 200:
                            out.write(code + "\n" + clean_doc + "\n")
        
        # Train and save the BPE tokenizer on training data
        train_bpe_tokenizer([temp_text_file], tokenizer_path)
        
    finally:
        if os.path.exists(temp_text_file):
            os.remove(temp_text_file)
