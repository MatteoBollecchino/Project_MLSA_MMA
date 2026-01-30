import os
import gzip
import json
import glob
import re
import hashlib
import logging
from tqdm import tqdm

# Configure module-level logger
logger = logging.getLogger(__name__)

class DatasetCleaner:
    """
    Handles the preprocessing, cleaning, and deduplication of code-docstring pairs.
    
    Key Features:
    1. Removes docstrings and comments from the source code to prevent Data Leakage.
    2. Cleans the target summaries (docstrings) by removing metadata/tags.
    3. Performs Cross-Split Deduplication using MD5 hashing to ensure no code 
       in Validation/Test sets has been seen in the Training set.
    4. Saves output directly to compressed .gz files to save space.
    """

    def __init__(self, min_code=20, max_code=250, min_doc=3, max_doc=50):
        # Thresholds for filtering data based on token length (whitespace split)
        self.min_code = min_code
        self.max_code = max_code
        self.min_doc = min_doc
        self.max_doc = max_doc
        
        # Global set to store MD5 hashes of code. 
        # Crucial for cross-split deduplication (Train -> Valid -> Test).
        self.global_seen_hashes = set()

    @staticmethod
    def clean_docstring(doc):
        """
        Cleans the target docstring (the summary).
        Removes metadata, noise, and structural tags often found in auto-generated docs.
        """
        if not doc: return ""
        
        # Take only the first paragraph (often the summary) and ignore detailed descriptions
        doc = doc.split('\n\n')[0]
        
        # Remove parameter definitions and return type descriptions using Regex
        # (e.g., :param x: ..., @return: ...)
        doc = re.sub(r'(:param|:type|:return|:rtype|@param|@return|@throws|Args:|Returns:|Raises:).*', '', doc, flags=re.MULTILINE)
        
        # Remove doctests (lines starting with >>>)
        doc = re.sub(r'>>>.*', '', doc)
        
        # Remove URLs
        doc = re.sub(r'http\S+', '', doc)
        
        return ' '.join(doc.split()).strip()
    
    @staticmethod
    def clean_code_body(code):
        """
        Sanitizes the source code (Input).
        It removes the original Docstring (to prevent the model from cheating/copying)
        and inline comments, while preserving strings that might contain '#' characters.
        """
        if not code: return ""

        # STEP 1: Remove the Docstring (Triple-quoted strings)
        # We use a non-greedy regex ([\s\S]*?) to capture everything between triple quotes.
        docstring_pattern = r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\')'
        
        # count=1 ensures we only remove the FIRST occurrence (the function docstring),
        # preserving other triple-quoted strings inside the function logic.
        code = re.sub(docstring_pattern, '', code, count=1)

        # STEP 2: Remove inline comments (#) intelligently
        # This regex has two capturing groups:
        # Group 1: Strings (double or single quotes). We want to KEEP these.
        # Group 2: Comments (starting with #). We want to REMOVE these.
        comment_pattern = r"(\".*?(?<!\\)\"|'.*?(?<!\\)')|(#.*)"

        def _replacer(match):
            """Callback function for re.sub"""
            # If Group 2 matched, it's a comment -> Replace with empty string
            if match.group(2) is not None:
                return ""
            # If Group 1 matched, it's a string -> Return the original string
            else:
                return match.group(1)

        code = re.sub(comment_pattern, _replacer, code)

        # STEP 3: Formatting cleanup
        # Remove empty lines resulting from comment deletion to compact the code
        lines = [line for line in code.split('\n') if line.strip()]
        
        return "\n".join(lines).strip()

    def process_split(self, split_name, input_dir, output_dir, mode="filter"):
        """
        Processes a specific dataset split (train, valid, or test).
        
        Args:
            mode (str): 
                - 'build': Adds hashes to the global set (Use for TRAIN).
                - 'filter': Checks hashes against the global set but doesn't add new ones (Use for VALID/TEST).
        """
        # Define search path for input files (usually .jsonl.gz)
        search_path = os.path.join(input_dir, split_name, "*.jsonl.gz")
        files = glob.glob(search_path)
        
        if not files:
            logger.warning(f"No files found for split: {split_name}")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        # Output file path (Saving directly as compressed GZIP)
        output_file = os.path.join(output_dir, f"{split_name}.jsonl.gz")
        
        # Statistics tracking
        stats = {"total": 0, "kept": 0, "removed_len": 0, "removed_dupe": 0, "removed_doc": 0}
        
        logger.info(f"Processing {split_name.upper()} -> {output_file}")

        # Open output file in 'wt' mode (Write Text), allowing us to write strings directly into the gzip archive
        with gzip.open(output_file, 'wt', encoding='utf-8') as out_f:
            
            # Iterate over all source files for this split
            for file_path in tqdm(files, desc=f"Cleaning {split_name}"):
                
                # Open input file (Read Text)
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        stats["total"] += 1
                        try:
                            data = json.loads(line)
                            
                            # Extract raw data handling potential schema variations
                            raw_code = data.get('code', data.get('func_code_string', ''))
                            raw_doc = data.get('docstring', data.get('func_documentation_string', ''))
                            
                            if not raw_code: continue

                            # 1. CLEAN TARGET (Docstring)
                            clean_doc = self.clean_docstring(raw_doc)
                            # Discard if docstring is empty or contains no alphanumeric characters
                            if not clean_doc or not any(c.isalnum() for c in clean_doc):
                                stats["removed_doc"] += 1
                                continue
                            
                            # 2. CLEAN INPUT (Code)
                            # Remove the docstring and comments from the code body
                            clean_code = self.clean_code_body(raw_code)
                            
                            # Discard if code became empty (e.g., it was only comments) or too short
                            if not clean_code.strip() or len(clean_code.strip()) < 10:
                                stats["removed_doc"] += 1 
                                continue

                            # 3. LENGTH FILTERING
                            # Simple whitespace tokenization for length check
                            code_len = len(clean_code.split())
                            doc_len = len(clean_doc.split())
                            
                            if not (self.min_code <= code_len <= self.max_code):
                                stats["removed_len"] += 1
                                continue
                            if not (self.min_doc <= doc_len <= self.max_doc):
                                stats["removed_len"] += 1
                                continue

                            # 4. DEDUPLICATION (Global Hashing)
                            # Normalize code (remove whitespace) to catch formatting variations
                            code_norm = re.sub(r'\s+', '', clean_code)
                            # Generate MD5 hash of the cleaned code
                            code_hash = hashlib.md5(code_norm.encode('utf-8')).hexdigest()
                            
                            # If hash exists in global set, it's a duplicate -> skip
                            if code_hash in self.global_seen_hashes:
                                stats["removed_dupe"] += 1
                                continue
                            
                            # If we are in 'build' mode (Training), add hash to the set
                            if mode == "build":
                                self.global_seen_hashes.add(code_hash)

                            # 5. WRITE TO FILE
                            entry = {
                                "code": clean_code, 
                                "docstring": clean_doc
                            }
                            # Write JSON line to compressed file
                            out_f.write(json.dumps(entry) + "\n")
                            stats["kept"] += 1

                        except json.JSONDecodeError:
                            continue # Skip malformed lines
        
        logger.info(f"[{split_name.upper()}] Kept: {stats['kept']} | Dupes: {stats['removed_dupe']} | Len: {stats['removed_len']}")

    def run(self, input_dir, output_dir):
        """
        Executes the pipeline in the strict order required for correct deduplication.
        Order: Train -> Valid -> Test.
        """
        # 1. Train: 'build' mode -> Populates the hash set with training data
        self.process_split("train", input_dir, output_dir, mode="build")
        
        # 2. Valid: 'filter' mode -> Removes samples that exist in Train
        self.process_split("valid", input_dir, output_dir, mode="filter")
        
        # 3. Test: 'filter' mode -> Removes samples that exist in Train or Valid
        self.process_split("test", input_dir, output_dir, mode="filter")


# --- MAIN EXECUTION BLOCK ---
def main():
    # Setup simple logging configuration for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Path Configuration
    DATASET_ROOT = "Project/Datasets/python/final/jsonl" # Source directory (Raw GZIPs)
    OUTPUT_DIR = "Project/Datasets/processed"            # Destination directory (Cleaned GZIPs)

    if not os.path.exists(DATASET_ROOT):
        logger.error(f"Dataset directory not found: {DATASET_ROOT}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize and run the cleaner
    logger.info("Starting preprocessing pipeline...")
    cleaner = DatasetCleaner()
    cleaner.run(DATASET_ROOT, OUTPUT_DIR)
    
    logger.info(f"Preprocessing Complete. Cleaned files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()