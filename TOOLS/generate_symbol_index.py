import os
import ast
import datetime

# --- CONFIGURATION ---
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Project'))
EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'export')
OUTPUT_FILENAME = "symbol_index.md"
IGNORE_DIRS = {'__pycache__', '.git', '.idea', 'venv', 'Datasets'}
# --- END CONFIGURATION ---

def format_args(args_node):
    """Formats ast.arguments into a readable string."""
    args = []
    # Positional and keyword-only arguments
    for arg in args_node.posonlyargs + args_node.args:
        args.append(arg.arg)
    
    # *args
    if args_node.vararg:
        args.append(f"*{args_node.vararg.arg}")
        
    # Keyword-only arguments
    for arg in args_node.kwonlyargs:
        args.append(arg.arg)
        
    # **kwargs
    if args_node.kwarg:
        args.append(f"**{args_node.kwarg.arg}")
        
    return ", ".join(args)

class SymbolVisitor(ast.NodeVisitor):
    """
    Visits an AST and extracts information about classes and functions.
    """
    def __init__(self):
        self.symbols = []

    def visit_ClassDef(self, node):
        docstring = ast.get_docstring(node) or "No docstring."
        self.symbols.append({
            "type": "Class",
            "name": node.name,
            "lineno": node.lineno,
            "docstring": docstring
        })
        # Do not visit children of classes to avoid listing methods as top-level functions
        # self.generic_visit(node) 

    def visit_FunctionDef(self, node):
        docstring = ast.get_docstring(node) or "No docstring."
        self.symbols.append({
            "type": "Function",
            "name": node.name,
            "lineno": node.lineno,
            "args": format_args(node.args),
            "docstring": docstring
        })
        # self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        docstring = ast.get_docstring(node) or "No docstring."
        self.symbols.append({
            "type": "Async Function",
            "name": node.name,
            "lineno": node.lineno,
            "args": format_args(node.args),
            "docstring": docstring
        })
        # self.generic_visit(node)
        
def analyze_code_symbols():
    """
    Walks the project path and analyzes symbols in each Python file.
    """
    project_symbols = {}
    for root, dirs, files in os.walk(PROJECT_PATH, topdown=True):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in sorted(files):
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, PROJECT_PATH)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ast.parse(content, filename=file)
                        visitor = SymbolVisitor()
                        visitor.visit(tree)
                        if visitor.symbols:
                            project_symbols[relative_path] = visitor.symbols
                except Exception as e:
                    print(f"Could not parse {file_path}: {e}")
    return project_symbols

def main():
    """
    Main function to generate and save the symbol index report.
    """
    print("Generating symbol index for the project...")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    project_symbols = analyze_code_symbols()
    
    output_path = os.path.join(EXPORT_DIR, OUTPUT_FILENAME)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Project Symbol Index\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if not project_symbols:
            f.write("No Python files with symbols found in the project.")
            return

        for file_path, symbols in sorted(project_symbols.items()):
            f.write(f"## `Project\\{file_path}`\n\n")
            for symbol in symbols:
                if symbol['type'] == 'Class':
                    f.write(f"### `class {symbol['name']}`\n")
                else: # Function or Async Function
                    f.write(f"### `def {symbol['name']}({symbol['args']})`\n")
                
                f.write(f"* **Type:** {symbol['type']}\n")
                f.write(f"* **Line:** {symbol['lineno']}\n")
                f.write(f"* **Docstring:**\n```\n{symbol['docstring']}\n```\n\n")
            f.write("---\n\n")

    print(f"Successfully generated symbol index at: {output_path}")

if __name__ == '__main__':
    main()
