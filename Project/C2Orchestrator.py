import os
import argparse
import logging
import torch
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
        self.data_root = "Project/Datasets"
        self.jsonl_base = os.path.join(self.data_root, "python", "final", "jsonl")
        self.tokenizer_path = "Project/tokenizer.json"
        self.preview_subdir = "Human_readable_sample"
        self.checkpoint_dir = "Project/models/checkpoints"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Hardware Target: {self.device}")

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

        # FASE 3: BENCHMARKING MULTI-MODELLO
        experiment_queue = ["lstm_attention", "transformer"]
        
        # Inizializzazione flussi (ottimizzata: una sola volta)
        train_loader = get_dataloader(self.jsonl_base, "train", self.tokenizer_path, self.config.batch_size, subset=self.config.subset)
        valid_loader = get_dataloader(self.jsonl_base, "valid", self.tokenizer_path, self.config.batch_size, subset=self.config.subset // 5 if self.config.subset else None)

        for model_tag in experiment_queue:
            model_save_path = os.path.join(self.checkpoint_dir, f"best_{model_tag}.pt")
            
            logger.info(f"\n" + "-"*60)
            logger.info(f"TARGET: {model_tag.upper()}")
            
            # --- GUARDIA DI IDEMPOTENZA ---
            if os.path.exists(model_save_path) and not self.config.force_train:
                logger.info(f"STATUS: Checkpoint rilevato in {model_save_path}. SKIP TRAINING.")
                continue
            
            try:
                # 1. Istanziazione Dinamica
                model = get_model_architecture(model_tag, self.device)
                
                # 2. Addestramento
                logger.info(f"STATUS: Avvio addestramento per {model_tag}...")
                train_model(
                    model=model, 
                    train_loader=train_loader, 
                    valid_loader=valid_loader, 
                    config=self.config, 
                    device=self.device
                )
                
                # 3. Cristallizzazione dello stato
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"SUCCESS: Modello {model_tag} salvato.")

            except NotImplementedError:
                logger.warning(f"SKIP: L'architettura '{model_tag}' è ancora un'idea teorica (Placeholder).")
            except Exception as e:
                logger.error(f"FAILURE: Errore nell'esperimento {model_tag}: {e}")
                continue 

        logger.info("\n--- PIPELINE ESEGUITA ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2 Orchestrator")
    
    # Scaling e Velocità
    parser.add_argument("--subset", type=int, default=None, help="Sottoinsieme per test rapidi (es. 1000)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Controllo Idempotenza
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--force_train", action="store_true", help="Ignora i checkpoint e ri-addestra tutto")
    parser.add_argument("--skip_train", action="store_true")
    
    args = parser.parse_args()
    
    pipeline = CodeSummarizationPipeline(args)
    pipeline.run()