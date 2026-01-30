import os
import json
import gzip
import logging
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

logger = logging.getLogger(__name__)

VOCAB_SIZE = 20000

def train_bpe_tokenizer(files, save_path, vocab_size=VOCAB_SIZE):
    """ 
    Trains a Byte-Pair Encoding (BPE) tokenizer using HuggingFace library.
    """
    # 1. Initialize BPE Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    
    # 2. Pre-tokenizer: Splits by whitespace/punctuation (ByteLevel is standard for code)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 3. Trainer Configuration
    # Special tokens are crucial for Seq2Seq models (Encoder-Decoder)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"],
        show_progress=True
    )
    
    logger.info(f"Training BPE Tokenizer (Target Vocab: {vocab_size})...")

    # 4. Train
    tokenizer.train(files, trainer)

    # 5. Save
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    tokenizer.save(save_path)
    logger.info(f"Tokenizer saved to: {save_path}")

def build_tokenizer(processed_data_dir, save_path, vocab_size=VOCAB_SIZE):
    """ 
    Prepares the corpus from ALREADY PROCESSED data and trains the tokenizer.
    """
    
    # Puntiamo SOLO al file di train processato.
    # Non usiamo valid/test per l'addestramento del vocabolario (Best Practice).
    train_file_path = os.path.join(processed_data_dir, "train.jsonl.gz")
    
    if not os.path.exists(train_file_path):
        logger.error(f"File di training non trovato: {train_file_path}")
        return

    # File temporaneo per accumulare tutto il testo (Corpus)
    temp_corpus_file = "temp_corpus.txt"
    logger.info(f"Extracting corpus from {train_file_path}...")
    
    try:
        with open(temp_corpus_file, 'w', encoding='utf-8') as out_f:
            # Leggiamo il file GZ processato
            with gzip.open(train_file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        
                        # PRENDIAMO I DATI DIRETTAMENTE
                        # Non serve pulire o filtrare, lo ha già fatto DatasetCleaner!
                        code = data.get('code', '')
                        doc = data.get('docstring', '')
                        
                        if code and doc:
                            # Scriviamo entrambi nel corpus per imparare i token di codice e linguaggio naturale
                            out_f.write(code + "\n" + doc + "\n")
                            
                    except json.JSONDecodeError:
                        continue
        
        # Avvia il training sul file temporaneo creato
        train_bpe_tokenizer([temp_corpus_file], save_path, vocab_size)
        
    finally:
        # Pulizia del file temporaneo
        if os.path.exists(temp_corpus_file):
            os.remove(temp_corpus_file)

# --- MAIN ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Path della cartella processata (dove DatasetCleaner ha salvato i file)
    PROCESSED_DIR = "Project/Datasets/processed"
    
    # Dove salvare il tokenizer finito
    TOKENIZER_PATH = "Project/tokenizer.json"
    
    build_tokenizer(PROCESSED_DIR, TOKENIZER_PATH, vocab_size=VOCAB_SIZE)