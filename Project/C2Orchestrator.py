import os
import argparse
import logging
import torch
from data.download_dataset import download_codesearchnet_robust
from data.inspect_dataset import save_human_readable_samples
from data.preprocess import prepare_vocab
from data.dataset import get_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeSummarizationPipeline:
    def __init__(self, config):
        self.config = config
        self.data_root = "Project/Datasets"
        self.jsonl_base = os.path.join(self.data_root, "python", "final", "jsonl")
        self.tokenizer_path = "Project/tokenizer.json"
        self.preview_subdir = "Human_readable_sample"

    def run(self):
        logger.info("--- MASTER PIPELINE START ---")

        # FASE 1: ACQUISIZIONE
        if not os.path.exists(self.jsonl_base) or self.config.force_download:
            logger.info("Fase 1: Download e scompattamento...")
            download_codesearchnet_robust(self.data_root)
        else:
            logger.info("Fase 1: Dataset già presente localmente.")

        # FASE 1.1: ISPEZIONE UMANA (Il tuo "Sanity Check" visivo)
        # Genera il file .txt per permetterti di leggere docstring e codice pulito
        logger.info(f"Fase 1.1: Generazione campioni leggibili in {self.preview_subdir}...")
        preview_file = save_human_readable_samples(self.data_root, self.preview_subdir)
        if preview_file:
            logger.info(f"Audit visivo pronto: {preview_file}")

        # FASE 2: PREPROCESSING (FIT del Tokenizer)
        if not os.path.exists(self.tokenizer_path) or self.config.force_preprocess:
            logger.info("Fase 2: Addestramento Tokenizer BPE (FIT)...")
            prepare_vocab(self.jsonl_base, self.tokenizer_path)
        else:
            logger.info("Fase 2: Tokenizer già esistente.")

        # FASE 2.1: VALIDAZIONE TRASFORMAZIONE (TRANSFORM Check)
        # Verifichiamo che la catena Stringa -> ID -> Tensor funzioni
        logger.info("Fase 2.1: Verifica della trasformazione numerica (TRANSFORM)...")
        try:
            test_loader = get_dataloader(
                data_dir=self.jsonl_base, 
                split_type="train", 
                tokenizer_path=self.tokenizer_path,
                batch_size=2,
                subset=5 
            )
            code_batch, doc_batch = next(iter(test_loader))
            
            logger.info("SUCCESS: Il DataLoader ha prodotto tensori validi.")
            logger.info(f"Esempio di sequenza ID (Codice): {code_batch[0][:10].tolist()}...")
            logger.info(f"Dimensione Batch: {code_batch.shape}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Errore nel caricamento dati: {e}")
            return

        logger.info("--- PIPELINE DATI VALIDATA E PRONTA PER IL TRAINING ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_preprocess", action="store_true")
    
    args = parser.parse_args()
    pipeline = CodeSummarizationPipeline(args)
    pipeline.run()