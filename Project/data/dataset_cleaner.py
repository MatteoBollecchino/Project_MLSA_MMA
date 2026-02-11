"""
================================================================================
DATA REFINERY & SANITIZATION UNIT
================================================================================
ROLE: Ensuring High-Fidelity Training Data & Preventing Data Contamination.

DESIGN RATIONALE:
- Anti-Cheating Logic: Models often learn to "shortcut" by copying comments 
  rather than understanding code. This unit strips all existing metadata.
- Cross-Split Deduplication: Guarantees that the Test set is truly "unseen" 
  by hashing the Training corpus.
- Dimensional Filtering: Discards outliers (extremely short or long sequences) 
  that would destabilize the Transformer's attention variance.
================================================================================
"""

import os
import gzip
import json
import glob
import re
import hashlib
import logging
from tqdm import tqdm

# Configure module-level logger for refinery audit trails.
logger = logging.getLogger(__name__)

class DatasetCleaner:
    """
    Handles the preprocessing, cleaning, and deduplication of code-docstring pairs.
    Acts as a 'Firewall' between raw, noisy internet data and the neural engine.
    """

    def __init__(self, min_code=20, max_code=250, min_doc=3, max_doc=50):
        """
        Args:
            min_code/max_code: Bounds for source code word count.
            min_doc/max_doc: Bounds for summary length (Docstrings).
        """
        self.min_code = min_code
        self.max_code = max_code
        self.min_doc = min_doc
        self.max_doc = max_doc
        
        # GLOBAL HASH REGISTRY: Stores MD5 fingerprints of every processed code block. 
        # This is the primary defense against Cross-Split Leakage.
        self.global_seen_hashes = set()

    @staticmethod
    def clean_docstring(doc):
        """
        TARGET SANITIZATION: Strips noise from the ground-truth summaries.
        
        Logic:
        - Paragraph Slicing: Only keeps the high-level summary (first paragraph).
        - Tag Removal: Strips Sphinx/Doxygen artifacts (:param, @return, etc.) via Regex.
        - URL Neutralization: Removes web links to force the model into linguistic logic.
        """
        if not doc: return ""
        
        # Isolate the primary intent (summary line/paragraph).
        # Takes the first block of text before any double newline
        # which often contains detailed descriptions or parameter lists.
        doc = doc.split('\n\n')[0]
        
        # REGEX SHIELD: Remove metadata tags commonly found in automated documentation.
        # Removes lines starting with :param, :type, :return, @param, @return, Args:, Returns:, Raises: etc. that can be found on multiple lines.
        # r -> raw string to avoid escaping backslashes, ^ and $ for line anchors, .* for any characters, and re.MULTILINE to apply to each line.
        doc = re.sub(r'(:param|:type|:return|:rtype|@param|@return|@throws|Args:|Returns:|Raises:).*', '', doc, flags=re.MULTILINE)
        
        # Remove doctest snippets (>>> example) to prevent training on test code.
        doc = re.sub(r'>>>.*', '', doc)
        
        # Remove URLs to clean up the target vocabulary.
        doc = re.sub(r'http\S+', '', doc)
        
        # Final cleanup: Collapse multiple spaces and trim.
        return ' '.join(doc.split()).strip()
    
    @staticmethod
    def clean_code_body(code):
        """
        INPUT SANITIZATION: The "Cheat-Proof" filter for source code.
        
        Logic:
        1. Docstring Extraction: Deletes triple-quoted strings at the start of functions.
        2. Inline Comment Purge: Uses a callback-based regex to remove '#' comments 
           while safely preserving '#' characters inside actual string literals.
        3. Compactification: Removes empty lines to provide a dense signal to the Encoder.
        """
        if not code: return ""

        # STEP 1: Eliminate triple-quoted docstrings. Non-greedy match ([\s\S]*?).
        docstring_pattern = r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\')'

        # count=1 ensures only the first docstring is removed, which is typically the function-level docstring.
        code = re.sub(docstring_pattern, '', code, count=1)

        # STEP 2: Intelligent Comment removal.
        # Group 1: Captures strings ("...") - we want to KEEP these.
        # Group 2: Captures comments (#...) - we want to DELETE these.
        comment_pattern = r"(\".*?(?<!\\)\"|'.*?(?<!\\)')|(#.*)"

        def _replacer(match):
            # If the regex matched Group 2 (a comment), return an empty string.
            if match.group(2) is not None:
                return ""
            # If it matched Group 1 (a literal string), return it as is.
            else:
                return match.group(1)

        code = re.sub(comment_pattern, _replacer, code)

        # STEP 3: Structural cleanup to remove trailing whitespace and null lines.
        lines = [line for line in code.split('\n') if line.strip()]
        
        # Final compactification: Join lines with a single newline to create a dense code block.
        return "\n".join(lines).strip()

    def process_split(self, split_name, input_dir, output_dir, mode="filter"):
        """
        Processes a specific dataset split (train, valid, or test).
        
        Args:
            mode (str): 
                - 'build': Populates the global hash registry (Use for TRAIN).
                - 'filter': Tests against the registry to prevent leakage (Use for VALID/TEST).
        """

        # Glob pattern to find all JSONL.GZ files for the given split.
        search_path = os.path.join(input_dir, split_name, "*.jsonl.gz")

        # Captures the files and records processed for each split.
        files = glob.glob(search_path)
        
        if not files:
            logger.warning(f"No files found for split: {split_name}")
            return

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{split_name}.jsonl.gz")
        
        # Metrics for refinery audit.
        stats = {"total": 0, "kept": 0, "removed_len": 0, "removed_dupe": 0, "removed_doc": 0}
        
        logger.info(f"Refining {split_name.upper()} -> {output_file}")

        # Stream directly into GZIP to manage large datasets without memory saturation.
        with gzip.open(output_file, 'wt', encoding='utf-8') as out_f:

            # Process each file in the split sequentially to maintain a clear audit trail and allow for progress tracking.
            for file_path in tqdm(files, desc=f"Sanitizing {split_name}"):

                # Stream the input file line by line to handle large files efficiently.
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        stats["total"] += 1
                        try:
                            # JSON Parsing: Each line is expected to be a JSON object with 'code' and 'docstring' fields (or their variants).
                            data = json.loads(line)
                            
                            # Normalize key naming across different dataset versions (func_code vs code).
                            raw_code = data.get('code', data.get('func_code_string', ''))
                            raw_doc = data.get('docstring', data.get('func_documentation_string', ''))
                            
                            if not raw_code: continue

                            # 1. CLEAN TARGET: High-level summary extraction.
                            clean_doc = self.clean_docstring(raw_doc)
                            if not clean_doc or not any(c.isalnum() for c in clean_doc):
                                stats["removed_doc"] += 1
                                continue
                            
                            # 2. CLEAN INPUT: Strip comments and leakage-prone text.
                            clean_code = self.clean_code_body(raw_code)
                            if not clean_code.strip() or len(clean_code.strip()) < 10:
                                stats["removed_doc"] += 1 
                                continue

                            # 3. STATISTICAL FILTERING: Sequence length pruning.
                            code_len = len(clean_code.split())
                            doc_len = len(clean_doc.split())
                            
                            if not (self.min_code <= code_len <= self.max_code) or \
                               not (self.min_doc <= doc_len <= self.max_doc):
                                stats["removed_len"] += 1
                                continue

                            # 4. CROSS-SPLIT DEDUPLICATION: MD5 Fingerprinting.
                            # We strip all whitespace to catch formatting-only duplicates.
                            code_norm = re.sub(r'\s+', '', clean_code)

                            # Hashing the normalized code to create a unique fingerprint for deduplication.
                            code_hash = hashlib.md5(code_norm.encode('utf-8')).hexdigest()
                            
                            if code_hash in self.global_seen_hashes:
                                stats["removed_dupe"] += 1
                                continue
                            
                            if mode == "build":
                                self.global_seen_hashes.add(code_hash)

                            # 5. SERIALIZATION: Save purified JSON entry.
                            entry = {"code": clean_code, "docstring": clean_doc}
                            out_f.write(json.dumps(entry) + "\n")
                            stats["kept"] += 1

                        except json.JSONDecodeError:
                            continue 
        
        logger.info(f"[{split_name.upper()}] Kept: {stats['kept']} | Removed Dupes: {stats['removed_dupe']}")

    def run(self, input_dir, output_dir):
        """
        Executes the Refinery Pipeline in a strict causal order.
        Mental Model: Training data defines the 'Universe', Valid/Test are 'Unseen'.
        """
        # Step 1: Ingest Training data to build the Hash Registry.
        self.process_split("train", input_dir, output_dir, mode="build")
        
        # Step 2 & 3: Filter Valid and Test sets against the Registry.
        self.process_split("valid", input_dir, output_dir, mode="filter")
        self.process_split("test", input_dir, output_dir, mode="filter")

# --- STANDALONE EXECUTION ENTRY ---
def main():
    """ Utility for manual dataset cleaning outside the orchestrator. """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    DATASET_ROOT = "Project/Datasets/python/final/jsonl" 
    OUTPUT_DIR = "Project/Datasets/processed" 

    if not os.path.exists(DATASET_ROOT):
        logger.error(f"Refinery Error: Source directory missing: {DATASET_ROOT}")
        return

    cleaner = DatasetCleaner()
    cleaner.run(DATASET_ROOT, OUTPUT_DIR)
    logger.info(f"Refinery Shutdown. Purified data ready in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()