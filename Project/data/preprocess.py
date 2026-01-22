import os
import json
import gzip
import glob
import logging
import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

logger = logging.getLogger(__name__)

def clean_docstring(doc):
    """
    Rimuove il rumore dalle docstring (GIGO Principle).
    Prende solo la prima riga e rimuove tag Sphinx/doctest.
    """
    if not doc: return ""
    # 1. Prendi solo la prima parte prima di una riga vuota o un punto fermo
    first_line = doc.split('\n\n')[0].split('\n')[0]
    # 2. Rimuove tag come :param:, :return:, @param ecc.
    clean = re.sub(r'(:param|:type|:return|:rtype|@param|@return).*', '', first_line)
    # 3. Rimuove eventuali residui di doctest (>>>)
    clean = re.sub(r'>>>.*', '', clean)
    return clean.strip()

def train_bpe_tokenizer(files, save_path, vocab_size=10000):
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    )
    
    logger.info(f"Training BPE Tokenizer (Vocab: {vocab_size})...")
    tokenizer.train(files, trainer) #SIIUUUm
    tokenizer.save(save_path)

def prepare_vocab(jsonl_base_dir, tokenizer_path):
    """
    Scansiona SOLO la cartella dei dati Python ed estrae il corpus pulito.
    """
    # Sandboxing: puntiamo solo alla cartella train dei dati estratti
    search_pattern = os.path.join(jsonl_base_dir, "train", "*.jsonl.gz")
    train_files = glob.glob(search_pattern)
    
    if not train_files:
        logger.error(f"Sandbox vuota! Nessun file in {search_pattern}")
        return

    temp_text_file = "temp_corpus.txt"
    logger.info(f"Processing {len(train_files)} files in sandbox...")
    
    try:
        with open(temp_text_file, 'w', encoding='utf-8') as out:
            for file_path in train_files:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        code = item.get('code', item.get('func_code_string', ''))
                        doc = item.get('docstring', item.get('func_documentation_string', ''))
                        
                        clean_doc = clean_docstring(doc)
                        
                        # Filtro di Qualità: evitiamo rumore troppo corto o codice immenso
                        if len(clean_doc) > 10 and len(code.split()) < 200:
                            out.write(code + "\n" + clean_doc + "\n")
        
        train_bpe_tokenizer([temp_text_file], tokenizer_path)
        
    finally:
        if os.path.exists(temp_text_file):
            os.remove(temp_text_file)
