import os
import argparse
import logging
# import torch
from data.download_dataset import download_codesearchnet_robust
from data.inspect_dataset import save_human_readable_samples
from data.preprocess import prepare_vocab
from data.dataset import get_dataloader

# LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PIPELINE CLASS
class CodeSummarizationPipeline:

    # INITIALIZATION
    def __init__(self, config):
        self.config = config
        self.data_root = "Project/Datasets"
        self.jsonl_base = os.path.join(self.data_root, "python", "final", "jsonl")
        self.tokenizer_path = "Project/tokenizer.json"
        self.preview_subdir = "Human_readable_sample"

    # RUN THE ENTIRE PIPELINE
    def run(self):
        logger.info("--- MASTER PIPELINE START ---")

        # PHASE 1: DATA DOWNLOAD AND EXTRACTION
        if not os.path.exists(self.jsonl_base) or self.config.force_download:
            logger.info("Phase 1: Download and extraction...")
            download_codesearchnet_robust(self.data_root)
        else:
            logger.info("Phase 1: Dataset already present locally.")

        # PHASE 1.1: HUMAN INSPECTION (Your visual "Sanity Check")
        # Generate the .txt file to allow reading clean docstrings and code
        logger.info(f"Phase 1.1: Generating human-readable samples in {self.preview_subdir}...")
        preview_file = save_human_readable_samples(self.data_root, self.preview_subdir)
        if preview_file:
            logger.info(f"Visual audit ready: {preview_file}")

        # PHASE 2: PREPROCESSING (FIT the Tokenizer)
        if not os.path.exists(self.tokenizer_path) or self.config.force_preprocess:
            logger.info("Phase 2: Training BPE Tokenizer (FIT)...")
            prepare_vocab(self.jsonl_base, self.tokenizer_path)
        else:
            logger.info("Phase 2: Tokenizer already exists.")

        # PHASE 2.1: TRANSFORMATION VALIDATION (TRANSFORM Check)
        # Verify that the String -> ID -> Tensor chain works
        logger.info("Phase 2.1: Verifying numerical transformation (TRANSFORM)...")
        try:
            # Get a small DataLoader for testing
            test_loader = get_dataloader(
                data_dir=self.jsonl_base, 
                split_type="train", 
                tokenizer_path=self.tokenizer_path,
                batch_size=2,
                subset=5 
            )
            code_batch, doc_batch = next(iter(test_loader))
            
            logger.info("SUCCESS: The DataLoader produced valid tensors.")
            logger.info(f"Example ID sequence (Code): {code_batch[0][:10].tolist()}...")
            logger.info(f"Batch Size: {code_batch.shape}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Error loading data: {e}")
            return

        logger.info("--- DATA PIPELINE VALIDATED AND READY FOR TRAINING ---")

# MAIN EXECUTION
if __name__ == "__main__":

    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_preprocess", action="store_true")
    
    # Parse arguments and run pipeline
    args = parser.parse_args()
    pipeline = CodeSummarizationPipeline(args)
    pipeline.run()