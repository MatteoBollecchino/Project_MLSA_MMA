import os

# --- CONFIGURATION ---
# The root directory to start the tree from.
# os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Project'))
# will point to the 'Project' directory, assuming 'Tools' and 'Project' are siblings.
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Project'))

# The directory where the output file will be saved.
EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'export')

# The name of the output file.
OUTPUT_FILENAME = "project_tree.txt"

# Directories and files to ignore during tree generation.
IGNORE_DIRS = {'__pycache__', '.git', '.idea', 'venv'}
IGNORE_FILES = {'.DS_Store'}
# --- END CONFIGURATION ---

def generate_tree(start_path):
    """Generates a directory tree structure as a list of strings."""
    tree_lines = []
    
    # Ensure the export directory exists
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
    except OSError as e:
        return [f"Error creating export directory: {e}"]

    for root, dirs, files in os.walk(start_path, topdown=True):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * level
        
        # Add the current directory to the tree
        tree_lines.append(f"{indent}├───{os.path.basename(root)}/")
        
        sub_indent = ' ' * 4 * (level + 1)
        
        # Add the files in the current directory
        for f in sorted(files):
            if f not in IGNORE_FILES:
                tree_lines.append(f"{sub_indent}├───{f}")
                
    return tree_lines

def main():
    """Main function to generate and save the project tree."""
    print(f"Generating directory tree for: {PROJECT_PATH}")
    
    if not os.path.isdir(PROJECT_PATH):
        print(f"Error: Project directory not found at '{PROJECT_PATH}'")
        return
        
    tree = generate_tree(PROJECT_PATH)
    
    output_path = os.path.join(EXPORT_DIR, OUTPUT_FILENAME)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Project Tree for: {PROJECT_PATH}\n")
            f.write("=" * (len(PROJECT_PATH) + 20))
            f.write("\n\n")
            # Write the project root name itself
            f.write(f"{os.path.basename(PROJECT_PATH)}/\n")
            # Write the rest of the tree, adjusting indentation
            for line in tree[1:]:
                 # Adjust indentation because os.walk starts inside the root
                base_level = line.count(' ' * 4)
                adjusted_line = ' ' * 4 * (base_level -1) + line.lstrip()
                f.write(adjusted_line + "\n")

        print(f"\nSuccessfully generated project tree at: {output_path}")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == '__main__':
    main()
