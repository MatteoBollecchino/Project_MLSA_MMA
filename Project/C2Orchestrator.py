"""
================================================================================
C2 MASTER ORCHESTRATOR - (Code Analysis & Summarization System)
================================================================================
ROLE: Central Management Unit (CMU) for the Neural Pipeline.

DESIGN PATTERN: 
- Modular Pipeline: Strictly separates Data Ingestion, Refinery, and Training.
- Factory Pattern: Decouples architecture selection from the training loop.
- Phase-Gated Execution: Ensures prerequisites (e.g., tokenization) are met 
  before high-compute stages (training).

SYSTEM GOAL: Transform raw Python source code into concise natural language 
summaries by optimizing the signal-to-noise ratio in latent space.
================================================================================
"""

import os
import argparse
import logging
import torch
import time
import sys
from datetime import datetime
from tokenizers import Tokenizer

# --- [GLOBAL CONFIGURATION] ---
# Tracking logic evolution. 4.4.0 introduces Internal Target Shifting 
# and differentiated regularization (Dropout/Weight Decay) for Transformers.
PIPELINE_VERSION = "4.4.2-Origin" 

# --- [DOMAIN ISOLATION: MODULE IMPORTS] ---
from data.download_dataset import download_codesearchnet_robust
from data.inspect_dataset import save_human_readable_samples
from data.tokenizer import build_tokenizer
from data.dataset import get_dataloader
from models.factory import get_model_architecture
from scripts.train import train_model 
from scripts.log_manager import ExecutionLogger 
from evaluate import BatchEvaluator 
from data.dataset_cleaner import DatasetCleaner

# Standard Logger setup to track pipeline progress and critical errors.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [MASTER PIPELINE CLASS] ---
class CodeSummarizationPipeline:
    """ 
    Orchestrator Class managing the end-to-end Machine Learning lifecycle.
    Encapsulates filesystem hooks, hardware affinity, and phase transitions.
    """

    def __init__(self, config):
        """
        Constructor: Configures paths and audits local hardware capabilities.
        
        Args:
            config (argparse.Namespace): Object containing runtime flags 
                                         (model type, epochs, batch size, etc.)
        """
        self.config = config
        self.root = os.path.dirname(os.path.abspath(__file__))

        # Path Definitions: Strictly defined to ensure "Source of Truth" for data flow.
        self.dataset_root = os.path.join(self.root, "Datasets")
        self.jsonl_base = os.path.join(self.dataset_root, "python", "final", "jsonl")
        self.tokenizer_path = os.path.join(self.root, "tokenizer.json")
        self.checkpoint_dir = os.path.join(self.root, "models", "checkpoints")
        self.processed_dir = os.path.join(self.dataset_root, "processed")

        print(f"\n" + "="*60)
        print(f"🤖 C2 ORCHESTRATOR | Version: {PIPELINE_VERSION}")
        print(f"📅 Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # --- HARDWARE AUDIT & ISOLATION ---
        # Logic to detect and isolate compute resources.
        # Index '1' is often forced for dedicated high-performance GPUs.
        print(f"[*] Inspecting Hardware capabilities...")
        device_count = torch.cuda.device_count()
        if device_count > 0:
            for i in range(device_count):
                print(f"    - GPU Index {i}: {torch.cuda.get_device_name(i)}")
            
            # Forced Device Affinity: Ensures the pipeline targets the correct VRAM bank.
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            print(f"[!] Strict Device Isolation: CUDA_VISIBLE_DEVICES forced to '1'")
        else:
            print("    - No GPU detected. Falling back to CPU (Throughput limited).")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Target Compute Device: {self.device}")
        
        # Audit Fidelity Mapping: Controls the statistical significance of the evaluation.
        # 'deep' mode provides higher confidence but increases inference latency.
        self.eval_map = {"instant": 100, "fast": 200, "deep": 1000}
        self.eval_samples = self.eval_map[self.config.evaluation]

    def run(self):
        """ 
        Main Entry Point: Directs the flow between Training and Audit modes.
        """
        logger.info(f"--- MASTER PIPELINE START | v{PIPELINE_VERSION} | Mode: {self.config.mode.upper()} ---")
        
        if self.config.mode == "train":
            # Safety Interlock: Training requires a defined target architecture.
            if not self.config.model:
                logger.error("❌ Error: In 'train' mode the --model parameter is MANDATORY.")
                return
            self._execute_training()
        else:
            self._execute_audit()

    def _execute_training(self):
        """ 
        The "Training State Machine": Executes five sequential phases of ML development.
        Implements detailed telemetry logging for post-mortem performance analysis.
        """
        # Initialize telemetry manager to capture system logs and loss curves.
        telemetry = ExecutionLogger(self.root, self.config.model, self.config.subset)
        
        # Record Pipeline Metadata for reproducibility.
        telemetry._write_to_file(f"PIPELINE_VERSION: {PIPELINE_VERSION}\n")
        telemetry.log_sys_info()

        # --- PHASE 1: DATA INFRASTRUCTURE ---
        # Goal: Ensure raw data availability. Implements reliable retrieval from mirrors.
        logger.info("📡 Phase 1: Validating Data Infrastructure...")
        t0 = time.time()

        if not os.path.exists(self.jsonl_base) or self.config.force_download:
            download_codesearchnet_robust(self.dataset_root)

        # Generate readable previews of raw data for qualitative human inspection.
        output_dir = os.path.join(self.dataset_root, "Human_readable_sample")
        save_human_readable_samples(self.jsonl_base, output_dir, "data_preview_raw.txt")
        telemetry.log_phase("data_infrastructure", time.time() - t0)

        # --- PHASE 2: DATA REFINERY & MD5 DEDUPLICATION ---
        # Goal: Integrity. MD5 hashing prevents 'Data Contamination' (training on test data).
        # Sanitization removes comments/docstrings to prevent 'Model Cheating'.
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
        # Goal: Symbolic Representation. Converts text into high-dimensional integer IDs.
        # Uses Byte-Pair Encoding (BPE) to handle sub-word units and Python syntax.
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
        # Goal: Optimization. Translates data into neural weights.
        # Differentiated regularizers (Dropout 0.3 for Transformers) are injected here.
        logger.info(f"🏗️ Phase 4: Initializing {self.config.model.upper()}...")
        model_tag = self.config.model
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{model_tag}_sub{self.config.subset if self.config.subset else 'all'}.pt"
        model_save_path = os.path.join(self.checkpoint_dir, filename)

        # Dataloader Factory: Optimized with dynamic padding to minimize null computations.
        train_loader = get_dataloader(self.processed_dir, "train", self.tokenizer_path, self.config.batch_size, subset=self.config.subset)
        valid_loader = get_dataloader(self.processed_dir, "valid", self.tokenizer_path, self.config.batch_size, subset=self.config.subset // 5 if self.config.subset else None)

        try:
            t_train_start = time.time()
            # Construct model via Factory pattern to isolate architecture details.
            model = get_model_architecture(model_tag, self.device, vocab_size=vocab_size)
            
            # CORE OPTIMIZATION LOOP: Performs backpropagation and weight updates.
            train_model(model, train_loader, valid_loader, self.config, self.device, telemetry=telemetry)
            
            # Persist optimized weights to disk.
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"💾 Checkpoint saved: {filename}")
            telemetry.log_phase("model_training", time.time() - t_train_start)

            # --- PHASE 5: AUTO-EVALUATION ---
            # Goal: Metric Validation. Calculates BLEU and ROUGE-L on unseen data.
            # Crucial: Assesses the actual utility of the current session's model.
            logger.info(f"📊 Phase 5: Executing {self.config.evaluation.upper()} Audit...")
            t_eval_start = time.time()
            evaluator = BatchEvaluator(self.device, self.tokenizer_path, self.checkpoint_dir, self.processed_dir, subset_size=self.eval_samples)
            df_results = evaluator.run_all(specific_file=filename)
            telemetry.log_phase(f"evaluation_{self.config.evaluation}", time.time() - t_eval_start)
            
            if df_results is not None:
                telemetry.log_final_metrics(df_results, self.config.evaluation)

        except Exception as e:
            # Exception Handler: Catches and logs runtime errors to prevent telemetry loss.
            logger.error(f"❌ CRITICAL PIPELINE FAILURE: {e}")
            telemetry._write_to_file(f"CRITICAL ERROR | v{PIPELINE_VERSION}: {str(e)}\n")
        finally:
            # Atomic Log Finalization: Renames temp logs with metrics and metadata tags.
            telemetry.finalize()
            print(f"\n[!] Session Finalized. Pipeline Version {PIPELINE_VERSION} signing off.\n")

    def _execute_audit(self):
        """ 
        Audit Module: Executes comparative performance analysis on existing weights.
        Designed for LIFO (Last-In, First-Out) checkpoint review.
        """
        logger.info(f"🧐 Audit Mode Active | Pipeline v{PIPELINE_VERSION}")
        
        if not os.path.exists(self.checkpoint_dir):
            logger.error(f"❌ Checkpoint directory not found at {self.checkpoint_dir}")
            return

        # Retrieve and sort available .pt weights chronologically.
        all_ckpts = sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")], reverse=True)
        if not all_ckpts:
            logger.error("❌ No checkpoints available for evaluation.")
            return

        # Target Selection Logic: Allows batch auditing (e.g., '1,2,5' or 'all').
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

        # Run Audit: Re-initializes architecture and runs inference on test datasets.
        evaluator = BatchEvaluator(self.device, self.tokenizer_path, self.checkpoint_dir, self.processed_dir, subset_size=self.eval_samples)
        
        for ckpt in targets:
            print(f"\n🔍 Analyzing: {ckpt}")
            df_res = evaluator.run_all(specific_file=ckpt)
            if df_res is not None:
                print(df_res.to_string(index=False))
        
        logger.info(f"✅ Audit Mode Completed.")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Command Line Interface (CLI) Definition.
    parser = argparse.ArgumentParser(description=f"C2 Master Orchestrator - v{PIPELINE_VERSION}")
    
    # Fundamental Mode Switching.
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], 
                        help="Select training session or checkpoint auditing.")
    
    # Architectural Selection.
    parser.add_argument("--model", type=str, choices=["transformer", "lstm_bahdanau", "lstm_dotproduct"],
                        help="Neural backbone to be synthesized.")
    
    # Hyperparameter & Data Pressure Control.
    parser.add_argument("--subset", type=int, default=None, help="Number of samples to pull from refinery.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of complete passes through training data.")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per stochastic gradient step.")
    
    # Audit Control.
    parser.add_argument("--neval", type=str, default="1", help="Indices of checkpoints to audit (e.g. 1,2,5).")
    parser.add_argument("--evaluation", type=str, default="fast", choices=["instant", "fast", "deep"],
                        help="Depth of evaluation samples (higher is more accurate).")
    
    # Pipeline Flags.
    parser.add_argument("--force_download", action="store_true", help="Ignore local cache and pull raw data.")
    parser.add_argument("--force_preprocess", action="store_true", help="Regenerate tokenizer vocabulary.")
    parser.add_argument("--force_train", action="store_true", help="Overwrite existing sessions.")
    
    # Initialize execution logic.
    args = parser.parse_args()
    pipeline = CodeSummarizationPipeline(args)
    pipeline.run()