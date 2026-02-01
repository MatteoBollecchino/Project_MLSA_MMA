import os
import argparse
import logging
import torch
import time
import sys
from datetime import datetime
from tokenizers import Tokenizer

# --- [GLOBAL CONFIGURATION] ---
PIPELINE_VERSION = "4.4.0-Origin"  # Focus: Regularization Tuning & Fail-Safe Shifting

# Importing project modules
from data.download_dataset import download_codesearchnet_robust
from data.inspect_dataset import save_human_readable_samples
from data.tokenizer import build_tokenizer
from data.dataset import get_dataloader
from models.factory import get_model_architecture
from scripts.train import train_model 
from scripts.log_manager import ExecutionLogger 
from evaluate import BatchEvaluator 
from data.dataset_cleaner import DatasetCleaner

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [MASTER PIPELINE CLASS] ---
class CodeSummarizationPipeline:
    """ Pipeline orchestrator for code summarization tasks - C2 Project. """

    def __init__(self, config):
        self.config = config
        self.root = os.path.dirname(os.path.abspath(__file__))

        # Paths configuration
        self.dataset_root = os.path.join(self.root, "Datasets")
        self.jsonl_base = os.path.join(self.dataset_root, "python", "final", "jsonl")
        self.tokenizer_path = os.path.join(self.root, "tokenizer.json")
        self.checkpoint_dir = os.path.join(self.root, "models", "checkpoints")
        self.processed_dir = os.path.join(self.dataset_root, "processed")

        print(f"\n" + "="*60)
        print(f"🤖 C2 ORCHESTRATOR | Version: {PIPELINE_VERSION}")
        print(f"📅 Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # Audit Hardware
        print(f"[*] Inspecting Hardware capabilities...")
        device_count = torch.cuda.device_count()
        if device_count > 0:
            for i in range(device_count):
                print(f"    - GPU Index {i}: {torch.cuda.get_device_name(i)}")
            # Forza l'utilizzo della GPU ad alte prestazioni (Indice 1)
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            print(f"[!] Strict Device Isolation: CUDA_VISIBLE_DEVICES forced to '1'")
        else:
            print("    - No GPU detected. Falling back to CPU (Throughput limited).")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Target Compute Device: {self.device}")
        
        # Evaluation Fidelity Mapping
        self.eval_map = {"instant": 100, "fast": 200, "deep": 1000}
        self.eval_samples = self.eval_map[self.config.evaluation]

    def run(self):
        """ Main execution method of the pipeline. """
        logger.info(f"--- MASTER PIPELINE START | v{PIPELINE_VERSION} | Mode: {self.config.mode.upper()} ---")
        
        if self.config.mode == "train":
            if not self.config.model:
                logger.error("❌ Error: In 'train' mode the --model parameter is MANDATORY.")
                return
            self._execute_training()
        else:
            self._execute_audit()

    def _execute_training(self):
        """ Manages the entire training and telemetry pipeline. """
        telemetry = ExecutionLogger(self.root, self.config.model, self.config.subset)
        
        # Iniezione versione nella telemetria
        telemetry._write_to_file(f"PIPELINE_VERSION: {PIPELINE_VERSION}\n")
        telemetry.log_sys_info()

        # --- PHASE 1: DATA INFRASTRUCTURE ---
        logger.info("📡 Phase 1: Validating Data Infrastructure...")
        t0 = time.time()

        if not os.path.exists(self.jsonl_base) or self.config.force_download:
            download_codesearchnet_robust(self.dataset_root)

        output_dir = os.path.join(self.dataset_root, "Human_readable_sample")
        save_human_readable_samples(self.jsonl_base, output_dir, "data_preview_raw.txt")
        telemetry.log_phase("data_infrastructure", time.time() - t0)

        # --- PHASE 2: DATA CLEANING & MD5 DEDUPLICATION ---
        logger.info("🧹 Phase 2: Sanitizing Source Code & Target Summaries...")
        t1 = time.time()
        expected_files = ["train.jsonl.gz", "valid.jsonl.gz", "test.jsonl.gz"]
        is_processed = all(os.path.exists(os.path.join(self.processed_dir, f)) for f in expected_files)
        
        if is_processed and not self.config.force_download:
            logger.info("✅ Verified Clean Data found. Skipping cleaning phase.")
        else:
            cleaner = DatasetCleaner()
            cleaner.run(self.jsonl_base, self.processed_dir)
            save_human_readable_samples(self.processed_dir, output_dir, "data_preview_processed.txt")
            
        telemetry.log_phase("data_cleaning", time.time() - t1)

        # --- PHASE 3: TOKENIZATION ---
        logger.info(f"🔢 Phase 3: Building Symbolic Mapping (BPE Vocab: 20000)...")
        t0 = time.time()
        reused = os.path.exists(self.tokenizer_path) and not self.config.force_preprocess

        if not reused:
            build_tokenizer(self.processed_dir, self.tokenizer_path, vocab_size=20000)
        
        tokenizer_duration = time.time() - t0
        tmp_tok = Tokenizer.from_file(self.tokenizer_path)
        vocab_size = tmp_tok.get_vocab_size()
        telemetry.log_tokenizer_info(vocab_size, reused=reused, duration=tokenizer_duration)

        # --- PHASE 4: ARCHITECTURAL SYNTHESIS & TRAINING ---
        logger.info(f"🏗️ Phase 4: Initializing {self.config.model.upper()}...")
        model_tag = self.config.model
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{model_tag}_sub{self.config.subset if self.config.subset else 'all'}.pt"
        model_save_path = os.path.join(self.checkpoint_dir, filename)

        train_loader = get_dataloader(self.processed_dir, "train", self.tokenizer_path, self.config.batch_size, subset=self.config.subset)
        valid_loader = get_dataloader(self.processed_dir, "valid", self.tokenizer_path, self.config.batch_size, subset=self.config.subset // 5 if self.config.subset else None)

        try:
            t_train_start = time.time()
            # La Factory ora inietta i parametri differenziati (Dropout 0.3 per Transformer)
            model = get_model_architecture(model_tag, self.device, vocab_size=vocab_size)
            
            # Il Train ora applica il Weight Decay differenziato (0.05 per Transformer)
            train_model(model, train_loader, valid_loader, self.config, self.device, telemetry=telemetry)
            
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"💾 Checkpoint saved: {filename}")
            telemetry.log_phase("model_training", time.time() - t_train_start)

            # --- PHASE 5: AUTO-EVALUATION ---
            logger.info(f"📊 Phase 5: Executing {self.config.evaluation.upper()} Audit...")
            t_eval_start = time.time()
            evaluator = BatchEvaluator(self.device, self.tokenizer_path, self.checkpoint_dir, self.processed_dir, subset_size=self.eval_samples)
            df_results = evaluator.run_all(specific_file=filename)
            telemetry.log_phase(f"evaluation_{self.config.evaluation}", time.time() - t_eval_start)
            
            if df_results is not None:
                telemetry.log_final_metrics(df_results, self.config.evaluation)

        except Exception as e:
            logger.error(f"❌ CRITICAL PIPELINE FAILURE: {e}")
            telemetry._write_to_file(f"CRITICAL ERROR | v{PIPELINE_VERSION}: {str(e)}\n")
        finally:
            telemetry.finalize()
            print(f"\n[!] Session Finalized. Pipeline Version {PIPELINE_VERSION} signing off.\n")

    def _execute_audit(self):
        """ Executes the audit (evaluation) on existing checkpoints."""
        logger.info(f"🧐 Audit Mode Active | Pipeline v{PIPELINE_VERSION}")
        
        if not os.path.exists(self.checkpoint_dir):
            logger.error(f"❌ Checkpoint directory not found at {self.checkpoint_dir}")
            return

        all_ckpts = sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")], reverse=True)
        if not all_ckpts:
            logger.error("❌ No checkpoints available for evaluation.")
            return

        # Selection of targets via --neval
        targets = []
        if self.config.neval.lower() == "all":
            targets = all_ckpts
        else:
            try:
                indices = [int(i.strip()) - 1 for i in self.config.neval.split(",")]
                targets = [all_ckpts[i] for i in indices if 0 <= i < len(all_ckpts)]
            except Exception as e:
                logger.error(f"❌ Invalid index format: {e}")
                return

        evaluator = BatchEvaluator(self.device, self.tokenizer_path, self.checkpoint_dir, self.processed_dir, subset_size=self.eval_samples)
        
        for ckpt in targets:
            print(f"\n🔍 Analyzing: {ckpt}")
            df_res = evaluator.run_all(specific_file=ckpt)
            if df_res is not None:
                print(df_res.to_string(index=False))
        
        logger.info(f"✅ Audit Mode Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"C2 Master Pipeline v{PIPELINE_VERSION}")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--model", type=str, choices=["transformer", "lstm_bahdanau", "lstm_dotproduct"])
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--neval", type=str, default="1")
    parser.add_argument("--evaluation", type=str, default="fast", choices=["instant", "fast", "deep"])
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    
    args = parser.parse_args()
    pipeline = CodeSummarizationPipeline(args)
    pipeline.run()