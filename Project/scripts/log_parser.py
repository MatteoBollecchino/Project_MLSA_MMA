import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

class LogParser:
    """
    Classe per analizzare i file di log.
    Mantiene la struttura ordinata della prima versione, ma usa 
    il parsing robusto (Regex) della seconda versione.
    """
    def __init__(self):
        # 1. Regex per le linee delle Epoche (Training & Validation)
        # Cerca: Epoch X | Loss: Y | Val: Z
        self.epoch_pattern = re.compile(
            r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)",
            re.IGNORECASE
        )
        
        # 2. Regex per le Metriche Finali (BLEU, ROUGE)
        # Cerca pattern tipo: "bleu": 0.123 oppure bleu: 0.123
        self.bleu_pattern = re.compile(r'["\']?bleu["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        self.rouge_pattern = re.compile(r'["\']?rougeL["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        
        # 3. Regex per il nome del modello (se presente nel testo)
        self.model_name_pattern = re.compile(r'["\']?model["\']?\s*[:=]\s*["\']([^"\']+)["\']', re.IGNORECASE)

    def parse_file(self, filepath):
        """
        Legge un file e restituisce un dizionario strutturato.
        """
        # Struttura dati base
        data = {
            "filename": os.path.basename(filepath),
            "model_name": "Unknown",
            "epochs": [],      # Lista di dizionari per ogni epoca
            "metrics": {       # Metriche finali
                "bleu": 0.0,
                "rougeL": 0.0,
                "best_val_loss": float('inf') 
            }
        }

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # --- A. Estrazione Dati Epoche ---
        for match in self.epoch_pattern.finditer(content):
            epoch_num = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            
            data["epochs"].append({
                "epoch": epoch_num,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
            
            # Calcolo dinamico della best validation loss
            if val_loss < data["metrics"]["best_val_loss"]:
                data["metrics"]["best_val_loss"] = val_loss

        # --- B. Estrazione Metriche Finali (Test) ---
        bleu_match = self.bleu_pattern.search(content)
        if bleu_match:
            data["metrics"]["bleu"] = float(bleu_match.group(1))

        rouge_match = self.rouge_pattern.search(content)
        if rouge_match:
            data["metrics"]["rougeL"] = float(rouge_match.group(1))

        # --- C. Identificazione Nome Modello ---
        name_match = self.model_name_pattern.search(content)
        if name_match:
            data["model_name"] = name_match.group(1)
        else:
            # Fallback: usa il nome del file senza estensione se non trova il nome nel testo
            data["model_name"] = os.path.splitext(data["filename"])[0]

        return data

def compare_models(log_files):
    """
    Funzione principale (Main Logic):
    Prende una lista di file, usa il Parser per leggerli, crea i DataFrame
    e chiama le funzioni di plotting.
    """
    parser = LogParser()
    all_models_data = []

    print(f"Analisi di {len(log_files)} file di log...")

    # 1. Parsing di tutti i file
    for file in log_files:
        if os.path.exists(file):
            try:
                parsed_data = parser.parse_file(file)
                # Includiamo il modello solo se ha dati di training (epoche)
                if parsed_data["epochs"]:
                    all_models_data.append(parsed_data)
                else:
                    print(f"  [SKIP] {file}: Nessun dato di training (epoche) trovato.")
            except Exception as e:
                print(f"  [ERR] Errore leggendo {file}: {e}")
        else:
            print(f"  [ERR] File non trovato: {file}")

    if not all_models_data:
        print("Nessun dato valido trovato per generare i grafici.")
        return

    # 2. Preparazione DataFrame per le Curve di Training (Loss/Val)
    df_epochs_list = []
    for model_data in all_models_data:
        df = pd.DataFrame(model_data["epochs"])
        df["model"] = model_data["model_name"]
        df_epochs_list.append(df)
    
    if df_epochs_list:
        df_all_epochs = pd.concat(df_epochs_list)
        plot_training_curves(df_all_epochs)
    
    # 3. Preparazione DataFrame per le Metriche Finali
    metrics_list = []
    for model_data in all_models_data:
        m = model_data["metrics"]
        m["model"] = model_data["model_name"]
        metrics_list.append(m)
    
    if metrics_list:
        df_metrics = pd.DataFrame(metrics_list)
        # Stampa tabella riassuntiva in console
        print("\n--- Riassunto Metriche Finali ---")
        cols = ["model", "bleu", "rougeL", "best_val_loss"]
        # Filtra colonne esistenti nel caso mancasse qualcosa
        cols = [c for c in cols if c in df_metrics.columns]
        print(df_metrics[cols].sort_values(by="bleu", ascending=False).to_string(index=False))
        
        plot_metrics_comparison(df_metrics)

def plot_training_curves(df):
    """
    Genera i grafici a linea per Loss di Training e Validation.
    """
    models = df["model"].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Grafico 1: Training Loss
    for model in models:
        subset = df[df["model"] == model]
        axes[0].plot(subset["epoch"], subset["train_loss"], label=model, marker='.', linewidth=1.5)
    
    axes[0].set_title("Training Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Grafico 2: Validation Loss
    for model in models:
        subset = df[df["model"] == model]
        axes[1].plot(subset["epoch"], subset["val_loss"], label=model, marker='o', linewidth=2)

    axes[1].set_title("Validation Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(df):
    """
    Genera i grafici a barre per il confronto finale (BLEU/ROUGE).
    """
    # Imposta 'model' come indice per facilitare il plotting
    df_plot = df.set_index("model")
    
    # Seleziona solo colonne rilevanti
    cols_to_plot = [c for c in ["bleu", "rougeL"] if c in df_plot.columns]
    
    if not cols_to_plot:
        return

    # Crea il grafico a barre
    ax = df_plot[cols_to_plot].plot(kind="bar", figsize=(10, 6), width=0.7, rot=45)
    
    plt.title("Confronto Performance Finali (Test Set)")
    plt.ylabel("Score")
    plt.xlabel("Modello")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Aggiungi i valori sopra le barre per leggibilità
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=9)
            
    plt.tight_layout()
    plt.show()

# --- ESEMPIO DI UTILIZZO ---
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
    # Esegui l'analisi
    compare_models(files)