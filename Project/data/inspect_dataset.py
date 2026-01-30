import gzip
import json
import os
import glob

def save_human_readable_samples(input_dir, output_dir, output_filename, num_examples=100):
    """ 
    Saves human-readable samples from .jsonl.gz files for inspection. 
    Works for both Raw and Processed datasets.
    
    Args:
        input_dir (str): Root directory to search for .jsonl.gz files.
        output_dir (str): Directory where the .txt file will be saved.
        output_filename (str): Name of the output .txt file.
        num_examples (int): Number of lines to read.
    """

    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    full_output_path = os.path.join(output_dir, output_filename)

    # 2. Search for files
    # First, try to find specific "train" files recursively (common in Raw structure)
    search_pattern = os.path.join(input_dir, "**", "*train*.jsonl.gz")
    files = glob.glob(search_pattern, recursive=True)

    # Fallback: If no "train" specific files found, just grab any .jsonl.gz (common in Processed structure)
    if not files:
        search_pattern = os.path.join(input_dir, "*.jsonl.gz")
        files = glob.glob(search_pattern, recursive=True)

    if not files:
        print(f"Warning: No .jsonl.gz files found in {input_dir}")
        return None

    # We take the first file found as a sample source
    source_file = files[0]
    print(f"Generating preview from: {source_file} -> {full_output_path}")

    # 3. Write human-readable samples
    try:
        with open(full_output_path, "w", encoding="utf-8") as out:
            out.write(f"=== DATASET PREVIEW ===\n")
            out.write(f"Source File: {source_file}\n\n")

            with gzip.open(source_file, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_examples: break
                    
                    try:
                        data = json.loads(line)
                        # Handle keys for both raw (func_...) and processed (code/docstring) formats
                        code = data.get('code', data.get('func_code_string', 'N/A'))
                        doc = data.get('docstring', data.get('func_documentation_string', 'N/A'))

                        out.write(f"--- SAMPLE #{i+1} ---\n")
                        out.write(f"DOCSTRING: {doc.strip()}\n")
                        out.write(f"CODE:\n{code.strip()}\n")
                        out.write("-" * 50 + "\n\n")
                    except json.JSONDecodeError:
                        out.write(f"--- SAMPLE #{i+1} [ERROR DECODING JSON] ---\n\n")
                        
        return full_output_path

    except Exception as e:
        print(f"Error writing preview: {e}")
        return None


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    OUTPUT_DIR = "Project/Datasets/Human_readable_sample"
    
    # 1. Preview for RAW Data
    save_human_readable_samples(
        input_dir="Project/Datasets/python/final/jsonl",
        output_dir=OUTPUT_DIR,
        output_filename="data_preview_raw.txt"
    )

    # 2. Preview for PROCESSED Data
    save_human_readable_samples(
        input_dir="Project/Datasets/processed",
        output_dir=OUTPUT_DIR,
        output_filename="data_preview_processed.txt"
    )