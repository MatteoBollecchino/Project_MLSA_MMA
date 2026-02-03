import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

class ParametricLogParser:
    def __init__(self):
        # Regex base per il parsing del contenuto (come prima)
        self.epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)", re.IGNORECASE)
        self.bleu_pattern = re.compile(r'["\']?bleu["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        self.rouge_pattern = re.compile(r'["\']?rougeL["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        
        # --- NUOVO: Regex per estrarre parametri dal NOME DEL FILE ---
        # Cerca pattern tipo "sub50000", "sub_10k", "50000_samples"
        self.subset_pattern = re.compile(r"sub(?:set)?[-_]?(\d+)", re.IGNORECASE)

    def parse_file(self, filepath):
        filename = os.path.basename(filepath)
        
        data = {
            "filename": filename,
            "architecture": self._derive_architecture_name(filename), # Nome pulito del modello
            "subset_size": self._extract_subset_size(filename),       # Dimensione dataset
            "total_epochs": 0,
            "metrics": {
                "bleu": None,
                "rougeL": None,
                "best_val_loss": float('inf'),
                "final_train_loss": None
            }
        }

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Parsing Epoche
        max_epoch = 0
        final_train_loss = 0
        
        for match in self.epoch_pattern.finditer(content):
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            
            if epoch > max_epoch:
                max_epoch = epoch
                final_train_loss = train_loss
            
            if val_loss < data["metrics"]["best_val_loss"]:
                data["metrics"]["best_val_loss"] = val_loss
        
        data["total_epochs"] = max_epoch
        data["metrics"]["final_train_loss"] = final_train_loss

        # 2. Parsing Metriche Finali
        bleu_match = self.bleu_pattern.search(content)
        if bleu_match: data["metrics"]["bleu"] = float(bleu_match.group(1))

        rouge_match = self.rouge_pattern.search(content)
        if rouge_match: data["metrics"]["rougeL"] = float(rouge_match.group(1))

        return data

    def _extract_subset_size(self, filename):
        """Cerca 'sub12345' nel nome del file."""
        match = self.subset_pattern.search(filename)
        if match:
            return int(match.group(1))
        return 0 # Default se non trovato

    def _derive_architecture_name(self, filename):
        """
        Pulisce il nome del file per trovare l'architettura base.
        Es: '2024_lstm_sub5000.txt' -> 'lstm'
        """
        name = os.path.splitext(filename)[0]
        # Rimuove date, timestamp, 'subXXXX'
        name = re.sub(r'\d{8}[-_]\d{4}', '', name) # Toglie data YYYYMMDD_HHMM
        name = re.sub(r'sub(?:set)?[-_]?\d+', '', name) # Toglie sub5000
        name = name.replace('_', ' ').strip()
        return name if name else "Unknown"

def analyze_parametric(file_list):
    parser = ParametricLogParser()
    data_points = []

    print(f"Analisi parametrica su {len(file_list)} file...")

    for f in file_list:
        if os.path.exists(f):
            try:
                parsed = parser.parse_file(f)
                # Appiattiamo il dizionario per creare un DataFrame facile da usare
                flat_entry = {
                    "filename": parsed["filename"],
                    "architecture": parsed["architecture"],
                    "subset_size": parsed["subset_size"],
                    "epochs": parsed["total_epochs"],
                    "bleu": parsed["metrics"]["bleu"],
                    "rougeL": parsed["metrics"]["rougeL"],
                    "best_val_loss": parsed["metrics"]["best_val_loss"]
                }
                # Aggiungiamo solo se abbiamo dati validi
                if flat_entry["epochs"] > 0:
                    data_points.append(flat_entry)
            except Exception as e:
                print(f"Errore su {f}: {e}")

    if not data_points:
        print("Nessun dato trovato.")
        return

    df = pd.DataFrame(data_points)
    
    # Ordiniamo per subset size per avere linee pulite nei grafici
    df = df.sort_values(by="subset_size")
    
    print("\n--- Dati Estratti ---")
    print(df[["architecture", "subset_size", "epochs", "bleu"]].to_string(index=False))

    plot_parametric_results(df)

def plot_parametric_results(df):
    """
    Crea grafici parametrici.
    Cerca di capire quale sia il parametro variabile (Subset o Epoche).
    """
    
    # Identifichiamo i parametri unici per capire cosa plottare
    unique_subsets = df["subset_size"].nunique()
    unique_epochs = df["epochs"].nunique()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- LOGICA DI PLOTTING ---
    # Se abbiamo diverse dimensioni di subset, usiamo Subset Size sull'asse X
    if unique_subsets > 1:
        x_axis = "subset_size"
        x_label = "Dimensione Subset (n. campioni)"
        title_prefix = "Impatto della Dimensione del Dataset"
    # Altrimenti, se variano le epoche, usiamo Epoche sull'asse X
    elif unique_epochs > 1:
        x_axis = "epochs"
        x_label = "Numero Totale Epoche"
        title_prefix = "Impatto della Durata del Training"
    else:
        print("\n[INFO] Non c'è abbastanza varianza nei parametri (Subset o Epoche) per fare curve parametriche.")
        # Fallback: grafico a barre semplice
        df.set_index("filename")[["bleu", "rougeL"]].plot(kind="bar", ax=axes[0])
        return

    architectures = df["architecture"].unique()

    # GRAFICO 1: BLEU Score vs Parametro
    for arch in architectures:
        subset = df[df["architecture"] == arch]
        axes[0].plot(subset[x_axis], subset["bleu"], marker='o', label=arch, linewidth=2)
    
    axes[0].set_title(f"{title_prefix} su BLEU Score")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("BLEU Score")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # GRAFICO 2: Best Validation Loss vs Parametro
    for arch in architectures:
        subset = df[df["architecture"] == arch]
        axes[1].plot(subset[x_axis], subset["best_val_loss"], marker='s', label=arch, linewidth=2, linestyle='--')

    axes[1].set_title(f"{title_prefix} su Best Val Loss")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Min Validation Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# --- ESEMPIO PRATICO ---
if __name__ == "__main__":

    # Sarebbe comodo avere delle sottocartelle separate per i diversi modelli
    TRANSFORMER_LOG_DIR = "Project/logs/transformer"  # Cartella dei file di log Transformer
    BAHDANAU_LOG_DIR = "Project/logs/bahdanau"  # Cartella dei file di log Bahdanau
    DOTPRODUCT_LOG_DIR = "Project/logs/dotproduct"  # Cartella dei file di log DotProduct
    
    # Prendi TUTTI i file .log nella cartella indicata
    transformer_files = glob.glob(os.path.join(TRANSFORMER_LOG_DIR, "*.log"))
    bahdanau_files = glob.glob(os.path.join(BAHDANAU_LOG_DIR, "*.log"))
    dotproduct_files = glob.glob(os.path.join(DOTPRODUCT_LOG_DIR, "*.log"))

    files = transformer_files + bahdanau_files + dotproduct_files

    # Scansione automatica
    # files = glob.glob("*S*.log")

    analyze_parametric(files)