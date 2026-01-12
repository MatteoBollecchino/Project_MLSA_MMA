import os
import ast
import sys
from collections import Counter
import datetime

# --- CONFIGURATION ---
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Project'))
EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'export')
OUTPUT_FILENAME = "dependency_report.txt"
IGNORE_DIRS = {'__pycache__', '.git', '.idea', 'venv', 'Datasets'}

# A list of standard library modules for Python 3.8+
# This is not exhaustive but covers the most common ones.
# A more robust solution would use a library like `stdlib-list`,
# but this avoids external dependencies for the tool itself.
STD_LIB_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asyncio', 'atexit', 'audioop',
    'base64', 'bdb', 'binascii', 'bisect', 'builtins', 'bz2', 'calendar', 'cgi',
    'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
    'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
    'contextvars', 'copy', 'copyreg', 'csv', 'ctypes', 'curses', 'dataclasses',
    'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest',
    'email', 'encodings', 'ensurepip', 'enum', 'errno', 'faulthandler', 'fcntl',
    'filecmp', 'fileinput', 'fnmatch', 'fractions', 'ftplib', 'functools',
    'gc', 'getopt', 'getpass', 'gettext', 'glob', 'grp', 'gzip', 'hashlib',
    'heapq', 'hmac', 'html', 'http', 'imaplib', 'imghdr', 'imp', 'importlib',
    'inspect', 'io', 'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3',
    'linecache', 'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal',
    'math', 'mimetypes', 'mmap', 'modulefinder', 'multiprocessing', 'netrc',
    'nis', 'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
    'parser', 'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil',
    'platform', 'plistlib', 'poplib', 'posix', 'pprint', 'pty', 'pwd', 'py_compile',
    'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're', 'readline', 'reprlib',
    'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select', 'selectors',
    'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib', 'sndhdr',
    'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl', 'stat', 'statistics',
    'string', 'stringprep', 'struct', 'subprocess', 'sunau', 'symbol', 'symtable',
    'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile',
    'termios', 'textwrap', 'threading', 'time', 'timeit', 'tkinter', 'token',
    'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo',
    'types', 'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv',
    'warnings', 'wave', 'weakref', 'webbrowser', 'wsgiref', 'xdrlib', 'xml',
    'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib'
}
# --- END CONFIGURATION ---

class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.relative_imports = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level > 0:  # Relative import
            if node.module:
                 self.relative_imports.append(node.module.split('.')[0])
        else:  # Absolute import
            if node.module:
                self.imports.append(node.module.split('.')[0])
        self.generic_visit(node)

def analyze_dependencies():
    all_imports = []
    all_relative_imports = []

    for root, dirs, files in os.walk(PROJECT_PATH, topdown=True):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ast.parse(content, filename=file)
                        visitor = ImportVisitor()
                        visitor.visit(tree)
                        all_imports.extend(visitor.imports)
                        all_relative_imports.extend(visitor.relative_imports)
                except Exception as e:
                    print(f"Could not parse {file_path}: {e}")
    
    return Counter(all_imports), Counter(all_relative_imports)

def main():
    print("Analyzing project dependencies...")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    imports_count, relative_imports_count = analyze_dependencies()
    
    std_lib_imports = {name: count for name, count in imports_count.items() if name in STD_LIB_MODULES}
    third_party_imports = {name: count for name, count in imports_count.items() if name not in STD_LIB_MODULES}
    
    output_path = os.path.join(EXPORT_DIR, OUTPUT_FILENAME)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Dependency Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        f.write("--- Third-Party Libraries ---\n")
        if third_party_imports:
            for name, count in sorted(third_party_imports.items()):
                f.write(f"{name} (imported {count} times)\n")
        else:
            f.write("No third-party libraries found.\n")
        
        f.write("\n\n--- Standard Library Imports ---\n")
        if std_lib_imports:
            for name, count in sorted(std_lib_imports.items()):
                f.write(f"{name} (imported {count} times)\n")
        else:
            f.write("No standard library imports found.\n")
            
        f.write("\n\n--- Project-Internal Imports ---\n")
        if relative_imports_count:
            for name, count in sorted(relative_imports_count.items()):
                f.write(f"{name} (imported {count} times)\n")
        else:
            f.write("No project-internal imports found.\n")
            
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("--- Suggested requirements.txt ---\n")
        if third_party_imports:
            # Note: This is a best-effort guess. Package names might differ from import names.
            # e.g., 'from bs4 import BeautifulSoup' -> import is 'bs4', package is 'beautifulsoup4'
            f.write("# Note: Package names may differ from import names (e.g., sklearn -> scikit-learn).\n")
            for name in sorted(third_party_imports.keys()):
                f.write(f"{name.lower()}\n")
        else:
            f.write("# No third-party packages detected.\n")

    print(f"Successfully generated dependency report at: {output_path}")

if __name__ == '__main__':
    main()
