import os
import argparse
import logging
import torch
from datetime import datetime

# Import interni
from data.download_dataset import download_codesearchnet_robust
from data.inspect_dataset import save_human_readable_samples
from data.preprocess import prepare_vocab
from data.dataset import get_dataloader
from models.factory import get_model_architecture
from scripts.train import train_model 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeSummarizationPipeline:
    def __init__(self, config):
        self.config = config
        
        # --- LOGICA DI AUTO-IDENTIFICAZIONE (Mental Model: Context Encapsulation) ---
        # Otteniamo la cartella dove si trova FISICAMENTE questo script (C2Orchestrator.py)
        self.root = os.path.dirname(os.path.abspath(__file__))
        
        # Definiamo i percorsi partendo dalla root dello script
        self.data_root = os.path.join(self.root, "Datasets")
        self.jsonl_base = os.path.join(self.data_root, "python", "final", "jsonl")
        self.tokenizer_path = os.path.join(self.root, "tokenizer.json")
        self.checkpoint_dir = os.path.join(self.root, "models", "checkpoints")
        self.preview_subdir = "Human_readable_sample"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"[*] Pipeline Root identificata in: {self.root}")
        logger.info(f"[*] Hardware Target: {self.device}")

    def run(self):
        logger.info("--- MASTER PIPELINE START ---")

        # FASE 1 & 2: INFRASTRUTTURA DATI
        if not os.path.exists(self.jsonl_base) or self.config.force_download:
            logger.info("Fase 1: Acquisizione dataset...")
            download_codesearchnet_robust(self.data_root)

        save_human_readable_samples(self.data_root, self.preview_subdir)

        if not os.path.exists(self.tokenizer_path) or self.config.force_preprocess:
            logger.info("Fase 2: Fitting del Tokenizer...")
            prepare_vocab(self.jsonl_base, self.tokenizer_path)

        if self.config.skip_train:
            logger.info("Fase 4: Modalità ispezione attiva. Fine corsa.")
            return

        # FASE 3: TRAINING
        experiment_queue = ["lstm_attention"]
        
        # Inizializzazione loader con i percorsi dinamici
        train_loader = get_dataloader(self.jsonl_base, "train", self.tokenizer_path, self.config.batch_size, subset=self.config.subset)
        valid_loader = get_dataloader(self.jsonl_base, "valid", self.tokenizer_path, self.config.batch_size, subset=self.config.subset // 5 if self.config.subset else None)

        for model_tag in experiment_queue:
            # Versioning immutabile con Timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            subset_tag = f"sub{self.config.subset}" if self.config.subset else "all"
            filename = f"{timestamp}_{model_tag}_{subset_tag}.pt"
            model_save_path = os.path.join(self.checkpoint_dir, filename)
            
            logger.info(f"\n" + "-"*60)
            logger.info(f"TARGET: {model_tag.upper()} | OUTPUT: {filename}")
            
            try:
                model = get_model_architecture(model_tag, self.device)
                
                logger.info(f"STATUS: Avvio addestramento...")
                train_model(
                    model=model, 
                    train_loader=train_loader, 
                    valid_loader=valid_loader, 
                    config=self.config, 
                    device=self.device
                )
                
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"SUCCESS: Modello salvato in {model_save_path}")

            except Exception as e:
                logger.error(f"FAILURE: Errore nell'esperimento {model_tag}: {e}")
                continue 

        logger.info("\n--- PIPELINE ESEGUITA ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2 Orchestrator V3")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    
    args = parser.parse_args()
    pipeline = CodeSummarizationPipeline(args)
    pipeline.run()