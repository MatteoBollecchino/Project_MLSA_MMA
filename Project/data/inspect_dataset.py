"""
================================================================================
DATASET INSPECTION UNIT
================================================================================
ROLE: Generating Human-Readable Snapshots from Compressed Binary Streams.

DESIGN RATIONALE:
- Observability: Translates high-density JSONL.GZ files into plain text formats, 
  allowing manual audit of the data refinery's effectiveness.
- Schema Agnosticism: Dynamically maps different field names to ensure 
  compatibility with both raw "CodeSearchNet" schemas and "Refined" versions.
- Defensive Slicing: Only reads a requested buffer size (num_examples) to avoid 
  unnecessary I/O overhead on large-scale datasets.
================================================================================
"""

import gzip
import json
import os
import glob

def save_human_readable_samples(input_dir, output_dir, output_filename, num_examples=100):
    """ 
    Orchestrates the extraction of compressed data for qualitative analysis. 
    
    Args:
        input_dir (str): Root path where .jsonl.gz shards are located.
        output_dir (str): Destination for the decoded text audit file.
        output_filename (str): Target filename for the preview output.
        num_examples (int): Number of top-tier samples to extract for the snapshot.
    """

    # --- [PHASE 1: DIRECTORY GOVERNANCE] ---
    # Ensure the target audit path exists to prevent OS-level I/O exceptions.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    full_output_path = os.path.join(output_dir, output_filename)

    # --- [PHASE 2: HEURISTIC FILE DISCOVERY] ---
    # Attempting to locate 'train' specific shards using recursive glob patterns.
    # Logic: Prioritize training data as it defines the model's future internal state.
    search_pattern = os.path.join(input_dir, "**", "*train*.jsonl.gz")
    files = glob.glob(search_pattern, recursive=True)

    # Fallback Strategy: If no explicit 'train' files exist, target any available JSONL.GZ.
    if not files:
        search_pattern = os.path.join(input_dir, "*.jsonl.gz")
        files = glob.glob(search_pattern, recursive=True)

    # Guard Clause: Prevent execution if the input pipe is empty.
    if not files:
        print(f"Warning: No valid .jsonl.gz streams found in {input_dir}")
        return None

    # Deterministic Selection: Targeting the first shard discovered by the OS.
    source_file = files[0]
    print(f"Generating system preview from: {source_file} -> {full_output_path}")

    # --- [PHASE 3: STREAM DECODING & FIELD MAPPING] ---
    try:
        # Opening the target audit file in UTF-8 to preserve code symbols.
        with open(full_output_path, "w", encoding="utf-8") as out:
            out.write(f"=== DATASET PREVIEW SNAPSHOT ===\n")
            out.write(f"Source Stream: {source_file}\n\n")

            # Binary Stream Context: Decompressing the GZIP payload in text-mode ('rt').
            with gzip.open(source_file, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    # Hard cap on sample extraction to maintain utility speed.
                    if i >= num_examples: break
                    
                    try:
                        data = json.loads(line)
                        # SCHEMA NORMALIZATION:
                        # Logic: Handle raw dataset keys (func_...) and refined keys (code/docstring).
                        # Implements a fallback to 'N/A' to avoid key-error crashes.

                        # First attempt to map to the refined schema, then fallback to raw schema keys.
                        code = data.get('code', data.get('func_code_string', 'N/A'))
                        doc = data.get('docstring', data.get('func_documentation_string', 'N/A'))

                        # Structure the snapshot for human readability.
                        out.write(f"--- SAMPLE #{i+1} ---\n")
                        out.write(f"TARGET SUMMARY (DOC): {doc.strip()}\n")
                        out.write(f"INPUT LOGIC (CODE):\n{code.strip()}\n")
                        out.write("-" * 50 + "\n\n")
                    except json.JSONDecodeError:
                        # Error Telemetry: Mark corrupt JSON objects without breaking the stream.
                        out.write(f"--- SAMPLE #{i+1} [CRITICAL: JSON DECODE ERROR] ---\n\n")
                        
        return full_output_path

    except Exception as e:
        # Low-level exception handling for filesystem/permission issues.
        print(f"I/O Exception during preview generation: {e}")
        return None


# --- STANDALONE EXECUTION BLOCK ---
if __name__ == "__main__":
    """
    Main Driver: Generates dual-previews for comparative logic verification.
    Focus: Inspecting the delta between RAW data and PROCESSED data.
    """
    OUTPUT_DIR = "Project/Datasets/Human_readable_sample"
    
    # Snapshot 1: Audit of the uncleaned, raw dataset.
    save_human_readable_samples(
        input_dir="Project/Datasets/python/final/jsonl",
        output_dir=OUTPUT_DIR,
        output_filename="data_preview_raw.txt"
    )

    # Snapshot 2: Audit of the refined dataset (post-DatasetCleaner execution).
    save_human_readable_samples(
        input_dir="Project/Datasets/processed",
        output_dir=OUTPUT_DIR,
        output_filename="data_preview_processed.txt"
    )