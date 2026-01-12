import os
import datetime

# --- CONFIGURATION ---
# The root directory to scan for code files.
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Project'))

# The directory where the output file will be saved.
EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'export')

# The name of the output file.
OUTPUT_FILENAME = "code_snapshot.txt"

# File extensions to include in the snapshot.
INCLUDED_EXTENSIONS = {'.py', '.md', '.txt', '.yml', '.yaml'}

# Directories to ignore completely.
IGNORE_DIRS = {'__pycache__', '.git', '.idea', 'venv', 'dataset'}

# Files to ignore completely.
IGNORE_FILES = {'.DS_Store'}
# --- END CONFIGURATION ---

def create_code_snapshot():
    """
    Scans the project directory, reads all relevant files, and compiles them
    into a single text file with headers.
    """
    # Ensure the export directory exists
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating export directory: {e}")
        return

    output_path = os.path.join(EXPORT_DIR, OUTPUT_FILENAME)
    
    print(f"Starting code snapshot for directory: {PROJECT_PATH}")
    print(f"Output will be saved to: {output_path}")

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # Write a main header for the snapshot file
            outfile.write("=" * 80 + "\n")
            outfile.write(f"Code Snapshot for Project: {os.path.basename(PROJECT_PATH)}\n")
            outfile.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write("=" * 80 + "\n\n")

            # Walk through the project directory
            for root, dirs, files in os.walk(PROJECT_PATH, topdown=True):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

                # Process and write files
                for filename in sorted(files):
                    if filename in IGNORE_FILES:
                        continue

                    file_ext = os.path.splitext(filename)[1]
                    if file_ext in INCLUDED_EXTENSIONS:
                        file_path = os.path.join(root, filename)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                                content = infile.read()
                            
                            # Write file-specific header
                            relative_path = os.path.relpath(file_path, PROJECT_PATH)
                            header = f"--- FILE: {filename} | PATH: Project\\{relative_path} ---"
                            
                            outfile.write(header + "\n")
                            outfile.write("-" * len(header) + "\n\n")
                            
                            # Write file content
                            outfile.write(content)
                            
                            # Add a separator for readability
                            outfile.write("\n\n" + "=" * 80 + "\n\n")
                            
                            print(f"  + Added: {relative_path}")

                        except Exception as e:
                            print(f"  - Error reading {file_path}: {e}")
    
    except IOError as e:
        print(f"Fatal Error: Could not write to output file {output_path}: {e}")
        return

    print("\nCode snapshot generation complete.")

def main():
    """Main function."""
    if not os.path.isdir(PROJECT_PATH):
        print(f"Error: Project directory not found at '{PROJECT_PATH}'")
        return
    create_code_snapshot()

if __name__ == '__main__':
    main()
