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

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [MASTER PIPELINE CLASS] ---
class CodeSummarizationPipeline:

    # Initialization of configuration of the pipeline
    def __init__(self, config):

        self.config = config

        # self-root = path of the orchestrator
        self.root = os.path.dirname(os.path.abspath(__file__))

        # Paths configuration
        self.dataset_root = os.path.join(self.root, "Datasets")
        self.jsonl_base = os.path.join(self.dataset_root, "python", "final", "jsonl")
        self.tokenizer_path = os.path.join(self.root, "tokenizer.json")
        self.checkpoint_dir = os.path.join(self.root, "models", "checkpoints")

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evaluation Fidelity Mapping
        self.eval_map = {"instant": 100, "fast": 200, "deep": 1000}
        self.eval_samples = self.eval_map[self.config.evaluation]

    # Execution of the pipeline
    def run(self):
        logger.info(f"--- MASTER PIPELINE START | Mode: {self.config.mode.upper()} ---")
        
        # Mode-based execution
        if self.config.mode == "train":
            if not self.config.model:
                logger.error("❌ Error: In 'train' mode the --model parameter is MANDATORY.")
                return
            self._execute_training()
        else:
            self._execute_audit()

    # Execution of the training pipeline
    def _execute_training(self):
        """Manages the entire training and telemetry pipeline."""
        # Initialization of telemetry specific to training
        telemetry = ExecutionLogger(self.root, self.config.model, self.config.subset)
        telemetry.log_sys_info()

        # PHASE 1: DATA INFRASTRUCTURE
        t0 = time.time()

        # Dataset download if not present
        if not os.path.exists(self.jsonl_base) or self.config.force_download:
            download_codesearchnet_robust(self.dataset_root)

        # Save human-readable samples for inspection
        save_human_readable_samples(self.dataset_root, "Human_readable_sample")
        telemetry.log_phase("data_infrastructure", time.time() - t0)

        # PHASE 2: TOKENIZER
        t0 = time.time()

        # Tokenizer preparation (reuse if already exists and no force flag)
        reused = os.path.exists(self.tokenizer_path) and not self.config.force_preprocess

        if not reused:
            # Tokenizer's Vocabulary creation
            prepare_vocab(self.jsonl_base, self.tokenizer_path)
        tokenizer_duration = time.time() - t0
        
        # Load tokenizer and get vocabulary size
        tmp_tok = Tokenizer.from_file(self.tokenizer_path)
        vocab_size = tmp_tok.get_vocab_size()

        # Logging tokenizer info
        telemetry.log_tokenizer_info(vocab_size, reused=reused, duration=tokenizer_duration)

        # PHASE 3: TRAINING

        # Model filename configuration
        model_tag = self.config.model
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{model_tag}_sub{self.config.subset if self.config.subset else 'all'}.pt"
        model_save_path = os.path.join(self.checkpoint_dir, filename)

        # DataLoader preparation
        train_loader = get_dataloader(self.jsonl_base, "train", self.tokenizer_path, self.config.batch_size, subset=self.config.subset)

        # Validation DataLoader (smaller subset for speed)
        valid_loader = get_dataloader(self.jsonl_base, "valid", self.tokenizer_path, self.config.batch_size, subset=self.config.subset // 5 if self.config.subset else None)

        try:
            t_train_start = time.time()
            model = get_model_architecture(model_tag, self.device, vocab_size=vocab_size)
            train_model(model, train_loader, valid_loader, self.config, self.device, telemetry=telemetry)
            
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            telemetry.log_phase("model_training", time.time() - t_train_start)

            # PHASE 4: AUTO-EVALUATION (Fast)
            t_eval_start = time.time()
            evaluator = BatchEvaluator(self.device, self.tokenizer_path, self.checkpoint_dir, self.jsonl_base, subset_size=self.eval_samples)
            df_results = evaluator.run_all(specific_file=filename)
            telemetry.log_phase(f"evaluation_{self.config.evaluation}", time.time() - t_eval_start)

            if df_results is not None:
                telemetry.log_final_metrics(df_results, self.config.evaluation)

        except Exception as e:
            logger.error(f"FAILURE during training: {e}")
            telemetry._write_to_file(f"CRITICAL ERROR: {str(e)}\n")
        finally:
            telemetry.finalize()

    # Execution of the audit pipeline
    def _execute_audit(self):
        """Executes the audit (evaluation) on a selection of existing checkpoints."""
        logger.info(f"🧐 Starting Audit Mode: {self.config.evaluation.upper()} fidelity")
        
        if not os.path.exists(self.checkpoint_dir):
            logger.error(f"❌ Checkpoint directory not found at {self.checkpoint_dir}")
            return

        # Retrieve files LIFO (Last-In, First-Out)
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
                # Converts "1,2,4" into list indices [0, 1, 3]
                indices = [int(i.strip()) - 1 for i in self.config.neval.split(",")]
                targets = [all_ckpts[i] for i in indices if 0 <= i < len(all_ckpts)]
            except (ValueError, IndexError) as e:
                logger.error(f"❌ Error in --neval format '{self.config.neval}': {e}")
                return

        if not targets:
            logger.warning("⚠️ No checkpoints matching the provided indices.")
            return

        logger.info(f"🔎 Models under review: {targets}")

        # Initializing Evaluator
        evaluator = BatchEvaluator(
            self.device, self.tokenizer_path, self.checkpoint_dir, 
            self.jsonl_base, subset_size=self.eval_samples
        )

        t_audit_start = time.time()
        for ckpt in targets:
            logger.info(f"--- Analysis: {ckpt} ---")
            df_res = evaluator.run_all(specific_file=ckpt)
            if df_res is not None:
                print(f"\n{df_res.to_string(index=False)}\n")
        
        logger.info(f"✅ Audit completed in {time.time() - t_audit_start:.2f}s")

if __name__ == "__main__":

    # Declaration of the parser for command line arguments
    parser = argparse.ArgumentParser(description="C2 Orchestrator V7 - Multitask AI Pipeline")
    
    # Modality switch of the pipeline (train / eval)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], 
                        help="Choose 'train' to train a new model or 'eval' to audit existing checkpoints")
    
    # Model parameters & Training
    parser.add_argument("--model", type=str, choices=["transformer", "lstm_attention"], 
                        help="Architecture (Required only in --mode train)")
    parser.add_argument("--subset", type=int, default=None, help="Training dataset size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Audit / Evaluation parameters
    parser.add_argument("--neval", type=str, default="1", 
                        help="'all' for all checkpoints, or comma-separated indices (e.g., 1,2,4) based on reverse chronological order")
    
    # When running, set to 'instant'
    parser.add_argument("--evaluation", type=str, default="fast", choices=["instant", "fast", "deep"], 
                        help="Depth of evaluation (100, 200, 1000 samples)")

    # INFRASTRUCTURE FLAGS
    # Force re-download of the dataset
    parser.add_argument("--force_download", action="store_true")

    # Force re-preprocessing of the dataset (vocab + tokenization)
    parser.add_argument("--force_preprocess", action="store_true")

    # Deprecated flag for compatibility
    parser.add_argument("--skip_train", action="store_true", help="Old flag for compatibility, prefer --mode eval")
    
    # Parsing of arguments
    args = parser.parse_args()
    
    # Override for compatibility if the user still uses --skip_train
    if args.skip_train:
        args.mode = "eval"

    # Declaration and execution of the pipeline
    pipeline = CodeSummarizationPipeline(args)
    pipeline.run()