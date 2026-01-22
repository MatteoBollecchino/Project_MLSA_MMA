import gzip
import json
import os
import glob

# GENERATE HUMAN-READABLE SAMPLES FOR DATA INSPECTION
def save_human_readable_samples(datasets_dir="Project/Datasets", output_subdir="Human_readable_sample", num_examples=100):

    # Define the full path for the folder and file
    full_output_dir = os.path.join(datasets_dir, output_subdir)
    output_file = os.path.join(full_output_dir, "data_preview.txt")
    
    # Create the subfolder (Directory Encapsulation)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    # Search for train files in all subdirectories
    search_pattern = os.path.join(datasets_dir, "**", "train", "*.jsonl.gz")

    # Find all matching files using glob
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        return None

    # Write human-readable samples to the output file
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"=== DATASET PREVIEW - CODE SEARCH NET (PYTHON) ===\n")
        out.write(f"Source File: {files[0]}\n\n")

        with gzip.open(files[0], 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_examples: break
                
                data = json.loads(line)
                code = data.get('code', data.get('func_code_string', 'N/A'))
                doc = data.get('docstring', data.get('func_documentation_string', 'N/A'))

                out.write(f"--- SAMPLE #{i+1} ---\n")
                out.write(f"DOCSTRING: {doc.strip()}\n")
                out.write(f"CODE:\n{code.strip()}\n")
                out.write("-" * 50 + "\n\n")
    
    return output_file

# MAIN EXECUTION
if __name__ == "__main__":
    save_human_readable_samples()