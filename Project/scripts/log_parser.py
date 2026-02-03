import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

class ParametricLogParser:
    def __init__(self):
        # Regex Contenuto
        self.epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)", re.IGNORECASE)
        self.bleu_pattern = re.compile(r'["\']?bleu["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        self.rouge_pattern = re.compile(r'["\']?rougeL["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        
        # Regex Nome File (per Subset)
        self.subset_pattern = re.compile(r"_S(\d+)", re.IGNORECASE)

    def parse_file(self, filepath):
        filename = os.path.basename(filepath)
        
        data = {
            "filename": filename,
            # Passiamo l'intero path per poter leggere anche il nome della cartella se serve
            "architecture": self._derive_architecture_name(filepath),
            "subset_size": self._extract_subset_size(filename),
            "total_epochs": 0,
            "metrics": {
                "bleu": None,
                "rougeL": None,
                "best_val_loss": float('inf'),
                "final_train_loss": None
            }
        }

        try:
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
            
        except Exception as e:
            print(f"Errore lettura file {filename}: {e}")

        return data

    def _extract_subset_size(self, filename):
        match = self.subset_pattern.search(filename)
        if match: return int(match.group(1))
        return 0

    def _derive_architecture_name(self, filepath):
        """
        Strategia Ibrida:
        1. Prova a leggerlo dal nome file (es. ..._LSTM__...)
        2. Se fallisce o è 'Unknown', usa il nome della CARTELLA padre.
        """
        filename = os.path.basename(filepath)
        arch_name = "Unknown"
        
        # TENTATIVO 1: Dal nome file (come prima)
        try:
            parts = filename.split('_')
            if len(parts) > 2:
                potential_name = parts[2]
                # Se il nome estratto ha senso (non è vuoto), lo teniamo
                if potential_name and not potential_name.startswith("S"): 
                    arch_name = potential_name
        except:
            pass

        # TENTATIVO 2: Dal nome della cartella padre se il file non è chiaro
        if arch_name == "Unknown":
            parent_dir = os.path.basename(os.path.dirname(filepath))
            if parent_dir:
                arch_name = parent_dir
        
        return arch_name

def analyze_parametric(root_folder):
    parser = ParametricLogParser()
    data_points = []

    # --- RICERCA RICORSIVA ---
    # Cerca tutti i file .log dentro root_folder e TUTTE le sue sottocartelle
    search_pattern = os.path.join(root_folder, "**", "*.log")
    file_list = glob.glob(search_pattern, recursive=True)

    print(f"Trovati {len(file_list)} file di log in '{root_folder}' e sottocartelle.")

    for f in file_list:
        parsed = parser.parse_file(f)
        
        flat_entry = {
            "filename": parsed["filename"],
            "architecture": parsed["architecture"],
            "subset_size": parsed["subset_size"],
            "epochs": parsed["total_epochs"],
            "bleu": parsed["metrics"]["bleu"],
            "rougeL": parsed["metrics"]["rougeL"],
            "best_val_loss": parsed["metrics"]["best_val_loss"]
        }
        
        if flat_entry["epochs"] > 0:
            data_points.append(flat_entry)

    if not data_points:
        print("Nessun dato valido trovato.")
        return

    df = pd.DataFrame(data_points)
    df = df.sort_values(by="subset_size")
    
    print("\n--- Riepilogo Dati ---")
    print(df[["architecture", "subset_size", "bleu", "best_val_loss"]].to_string(index=False))

    plot_parametric_results(df)

def plot_parametric_results(df):
    architectures = df["architecture"].unique()
    print(f"\nGenerazione grafici per: {list(architectures)}")

    for arch in architectures:
        subset_df = df[df["architecture"] == arch]
        if subset_df.empty: continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Modello: {arch}", fontsize=16, fontweight='bold')

        x_vals = subset_df["subset_size"]
        
        # 1. BLEU
        if subset_df["bleu"].notna().any():
            axes[0].plot(x_vals, subset_df["bleu"], marker='o', color='royalblue', linewidth=2)
            axes[0].set_title("BLEU Score")
            # Etichette sui punti
            for x, y in zip(x_vals, subset_df["bleu"]):
                if pd.notna(y): axes[0].annotate(f"{y:.2f}", (x, y), xytext=(0,10), textcoords="offset points", ha='center')

        # 2. ROUGE
        if subset_df["rougeL"].notna().any():
            axes[1].plot(x_vals, subset_df["rougeL"], marker='s', color='seagreen', linewidth=2)
            axes[1].set_title("ROUGE-L Score")

        # 3. LOSS
        if subset_df["best_val_loss"].notna().any():
            axes[2].plot(x_vals, subset_df["best_val_loss"], marker='x', color='firebrick', linestyle='--', linewidth=2)
            axes[2].set_title("Best Val Loss")

        for ax in axes:
            ax.set_xlabel("Dataset Size")
            ax.grid(True, linestyle='--', alpha=0.6)
            if len(x_vals) > 0: ax.set_xticks(x_vals)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    # Inserisci qui il percorso della cartella "madre" che contiene le sottocartelle dei modelli
    ROOT_LOGS = "Project/logs" 

    if os.path.exists(ROOT_LOGS):
        analyze_parametric(ROOT_LOGS)
    else:
        print(f"Attenzione: La cartella '{ROOT_LOGS}' non esiste.")