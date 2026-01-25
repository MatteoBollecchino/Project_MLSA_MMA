import os
import argparse
import logging
import torch
import time
from datetime import datetime
from tokenizers import Tokenizer

# Import interni
from data.download_dataset import download_codesearchnet_robust
from data.inspect_dataset import save_human_readable_samples
from data.preprocess import prepare_vocab
from data.dataset import get_dataloader
from models.factory import get_model_architecture
from scripts.train import train_model 
from scripts.log_manager import ExecutionLogger 
from evaluate import BatchEvaluator 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeSummarizationPipeline:
    def __init__(self, config):
        self.config = config
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.data_root = os.path.join(self.root, "Datasets")
        self.jsonl_base = os.path.join(self.data_root, "python", "final", "jsonl")
        self.tokenizer_path = os.path.join(self.root, "tokenizer.json")
        self.checkpoint_dir = os.path.join(self.root, "models", "checkpoints")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mapping Fedeltà Valutazione
        self.eval_map = {"instant": 100, "fast": 200, "deep": 1000}
        self.eval_samples = self.eval_map[self.config.evaluation]

    def run(self):
        logger.info(f"--- MASTER PIPELINE START | Mode: {self.config.mode.upper()} ---")
        
        if self.config.mode == "train":
            if not self.config.model:
                logger.error("❌ Errore: In modalità 'train' il parametro --model è OBBLIGATORIO.")
                return
            self._execute_training()
        else:
            self._execute_audit()

    def _execute_training(self):
        """Gestisce l'intera pipeline di addestramento e telemetria."""
        # Inizializzazione Telemetria specifica per il training
        telemetry = ExecutionLogger(self.root, self.config.model, self.config.subset)
        telemetry.log_sys_info()

        # FASE 1: DATA INFRASTRUCTURE
        t0 = time.time()
        if not os.path.exists(self.jsonl_base) or self.config.force_download:
            download_codesearchnet_robust(self.data_root)
        save_human_readable_samples(self.data_root, "Human_readable_sample")
        telemetry.log_phase("data_infrastructure", time.time() - t0)

        # FASE 2: TOKENIZER
        t0 = time.time()
        reused = os.path.exists(self.tokenizer_path) and not self.config.force_preprocess
        if not reused:
            prepare_vocab(self.jsonl_base, self.tokenizer_path)
        tokenizer_duration = time.time() - t0
        
        tmp_tok = Tokenizer.from_file(self.tokenizer_path)
        vocab_size = tmp_tok.get_vocab_size()
        telemetry.log_tokenizer_info(vocab_size, reused=reused, duration=tokenizer_duration)

        # FASE 3: TRAINING
        model_tag = self.config.model
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{model_tag}_sub{self.config.subset if self.config.subset else 'all'}.pt"
        model_save_path = os.path.join(self.checkpoint_dir, filename)

        train_loader = get_dataloader(self.jsonl_base, "train", self.tokenizer_path, self.config.batch_size, subset=self.config.subset)
        valid_loader = get_dataloader(self.jsonl_base, "valid", self.tokenizer_path, self.config.batch_size, subset=self.config.subset // 5 if self.config.subset else None)

        try:
            t_train_start = time.time()
            model = get_model_architecture(model_tag, self.device, vocab_size=vocab_size)
            train_model(model, train_loader, valid_loader, self.config, self.device, telemetry=telemetry)
            
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            telemetry.log_phase("model_training", time.time() - t_train_start)

            # FASE 4: AUTO-EVALUATION (Fast)
            t_eval_start = time.time()
            evaluator = BatchEvaluator(self.device, self.tokenizer_path, self.checkpoint_dir, self.jsonl_base, subset_size=self.eval_samples)
            df_results = evaluator.run_all(specific_file=filename)
            telemetry.log_phase(f"evaluation_{self.config.evaluation}", time.time() - t_eval_start)

            if df_results is not None:
                telemetry.log_final_metrics(df_results, self.config.evaluation)

        except Exception as e:
            logger.error(f"FAILURE durante il training: {e}")
            telemetry._write_to_file(f"CRITICAL ERROR: {str(e)}\n")
        finally:
            telemetry.finalize()

    def _execute_audit(self):
        """Esegue l'audit (valutazione) su una selezione di checkpoint esistenti."""
        logger.info(f"🧐 Avvio Audit Mode: {self.config.evaluation.upper()} fidelity")
        
        if not os.path.exists(self.checkpoint_dir):
            logger.error(f"❌ Directory checkpoints non trovata in {self.checkpoint_dir}")
            return

        # Recupero file LIFO (Last-In, First-Out)
        all_ckpts = sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")], reverse=True)
        
        if not all_ckpts:
            logger.error("❌ Nessun checkpoint disponibile per la valutazione.")
            return

        # Selezione dei target tramite --neval
        targets = []
        if self.config.neval.lower() == "all":
            targets = all_ckpts
        else:
            try:
                # Converte "1,2,4" in indici di lista [0, 1, 3]
                indices = [int(i.strip()) - 1 for i in self.config.neval.split(",")]
                targets = [all_ckpts[i] for i in indices if 0 <= i < len(all_ckpts)]
            except (ValueError, IndexError) as e:
                logger.error(f"❌ Errore nel formato --neval '{self.config.neval}': {e}")
                return

        if not targets:
            logger.warning("⚠️ Nessun checkpoint corrispondente agli indici forniti.")
            return

        logger.info(f"🔎 Modelli in esame: {targets}")

        # Inizializzazione Evaluator
        evaluator = BatchEvaluator(
            self.device, self.tokenizer_path, self.checkpoint_dir, 
            self.jsonl_base, subset_size=self.eval_samples
        )

        t_audit_start = time.time()
        for ckpt in targets:
            logger.info(f"--- Analisi: {ckpt} ---")
            df_res = evaluator.run_all(specific_file=ckpt)
            if df_res is not None:
                print(f"\n{df_res.to_string(index=False)}\n")
        
        logger.info(f"✅ Audit completato in {time.time() - t_audit_start:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2 Orchestrator V7 - Multitask AI Pipeline")
    
    # Switch principale di modalità
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], 
                        help="Scegli tra addestramento o valutazione pura")
    
    # Parametri Modello & Training
    parser.add_argument("--model", type=str, choices=["transformer", "lstm_attention"], 
                        help="Architettura (Obbligatoria solo in --mode train)")
    parser.add_argument("--subset", type=int, default=None, help="Dimensione dataset training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Parametri Audit / Evaluation
    parser.add_argument("--neval", type=str, default="1", 
                        help="'all' per tutti i checkpoint, oppure indici separati da virgola (es. 1,2,4) basati su ordine cronologico inverso")
    parser.add_argument("--evaluation", type=str, default="fast", choices=["instant", "fast", "deep"], 
                        help="Profondità della valutazione (100, 200, 1000 campioni)")

    # Flags Infrastruttura
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--skip_train", action="store_true", help="Vecchio flag per compatibilità, preferire --mode eval")
    
    args = parser.parse_args()
    
    # Override per compatibilità se l'utente usa ancora --skip_train
    if args.skip_train:
        args.mode = "eval"

    pipeline = CodeSummarizationPipeline(args)
    pipeline.run()