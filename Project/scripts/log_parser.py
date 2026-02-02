import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

class LogParser:
    """
    Classe per analizzare il contenuto dei file di log di training.
    """
    def __init__(self):
        # Regex per le linee delle Epoche
        # Es: Epoch 01 | Loss: 7.4297 | Val: 7.4139 | LR: 0.000198 | Time: 703.95s
        self.epoch_pattern = re.compile(
            r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)\s+\|\s+LR:\s+([\d.e-]+)\s+\|\s+Time:\s+([\d.]+)s"
        )
        
    def parse_file(self, filepath):
        """
        Legge un file di log e restituisce un dizionario con i dati estratti.
        """
        data = {
            "model_name": "Unknown",
            "epochs": [],
            "metrics": {},
            "infrastructure": {},
            "total_duration": None
        }

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Estrazione Dati Epoche
        for match in self.epoch_pattern.finditer(content):
            data["epochs"].append({
                "epoch": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "val_loss": float(match.group(3)),
                "lr": float(match.group(4)),
                "time": float(match.group(5))
            })

        # 2. Estrazione JSON (Infrastructure e Evaluation)
        # Cerchiamo blocchi JSON specifici dopo gli header
        
        # Infrastructure
        infra_match = re.search(r"--- INFRASTRUCTURE ---\s*(\{.*?\})", content, re.DOTALL)
        if infra_match:
            try:
                data["infrastructure"] = json.loads(infra_match.group(1))
            except json.JSONDecodeError:
                pass

        # Final Evaluation
        eval_match = re.search(r"--- FINAL EVALUATION \(FAST\) ---\s*(\{.*?\})", content, re.DOTALL)
        if eval_match:
            try:
                eval_data = json.loads(eval_match.group(1))
                data["metrics"] = eval_data
                # Usiamo il nome del modello nel JSON, altrimenti il nome del file
                data["model_name"] = eval_data.get("model", os.path.basename(filepath))
            except json.JSONDecodeError:
                pass
        
        # Se non abbiamo trovato il nome del modello nel JSON, usiamo il nome del file
        if data["model_name"] == "Unknown":
            data["model_name"] = os.path.basename(filepath)

        # 3. Durata Totale
        duration_match = re.search(r"Total Duration:\s+([\d:.]+)", content)
        if duration_match:
            data["total_duration"] = duration_match.group(1)

        return data

def compare_models(log_files):
    """
    Analizza una lista di file di log e genera grafici comparativi.
    """
    parser = LogParser()
    all_models_data = []

    print(f"Analisi di {len(log_files)} file di log...")

    for file in log_files:
        if os.path.exists(file):
            parsed_data = parser.parse_file(file)
            all_models_data.append(parsed_data)
        else:
            print(f"Attenzione: File non trovato -> {file}")

    if not all_models_data:
        print("Nessun dato valido trovato.")
        return

    # --- Creazione DataFrame per Epoche (per i grafici) ---
    df_epochs_list = []
    for model in all_models_data:
        df = pd.DataFrame(model["epochs"])
        df["model"] = model["model_name"]
        df_epochs_list.append(df)
    
    if df_epochs_list:
        df_all_epochs = pd.concat(df_epochs_list)
        plot_training_curves(df_all_epochs)
    
    # --- Creazione DataFrame per Metriche Finali ---
    metrics_list = []
    for model in all_models_data:
        m = model["metrics"]
        m["model"] = model["model_name"]
        # Aggiungiamo la miglior validation loss trovata durante il training
        epochs_df = pd.DataFrame(model["epochs"])
        if not epochs_df.empty:
            m["best_val_loss"] = epochs_df["val_loss"].min()
        metrics_list.append(m)
    
    if metrics_list:
        df_metrics = pd.DataFrame(metrics_list)
        print("\n--- Riassunto Metriche Finali ---")
        print(df_metrics[["model", "bleu", "rougeL", "best_val_loss"]].to_string(index=False))
        plot_metrics_comparison(df_metrics)

def plot_training_curves(df):
    """Genera i grafici di Loss e Validation Loss"""
    models = df["model"].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Training Loss
    for model in models:
        subset = df[df["model"] == model]
        axes[0].plot(subset["epoch"], subset["train_loss"], label=f"{model} Train", marker='.')
    
    axes[0].set_title("Training Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Plot Validation Loss
    for model in models:
        subset = df[df["model"] == model]
        axes[1].plot(subset["epoch"], subset["val_loss"], label=f"{model} Val", marker='o', linestyle='--')

    axes[1].set_title("Validation Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(df):
    """Genera un grafico a barre per BLEU e ROUGE"""
    # Filtriamo solo le colonne che ci interessano e che esistono
    cols_to_plot = [c for c in ["bleu", "rougeL"] if c in df.columns]
    
    if not cols_to_plot:
        return

    df.set_index("model")[cols_to_plot].plot(kind="bar", figsize=(10, 6), rot=45)
    plt.title("Confronto Metriche Finali (BLEU & ROUGE-L)")
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- ESEMPIO DI UTILIZZO ---
# Puoi salvare il tuo log in un file chiamato 'log1.txt' e aggiungerne altri per testare.

if __name__ == "__main__":
    # Creiamo un file dummy per testare lo script immediatamente se non hai file pronti
    dummy_log_content = """
    --- INFRASTRUCTURE ---
    {"os": "Linux", "device": "Tesla T4"}
    Epoch 01 | Loss: 7.4297 | Val: 7.4139 | LR: 0.000198 | Time: 703.95s
    Epoch 02 | Loss: 6.6733 | Val: 7.4159 | LR: 0.000468 | Time: 703.81s
    Epoch 03 | Loss: 6.5031 | Val: 7.3485 | LR: 0.000486 | Time: 704.36s
    --- FINAL EVALUATION (FAST) ---
    {"model": "lstm_bahdanau", "bleu": 0.0024, "rougeL": 0.1770}
    Total Duration: 1:43:16.471708
    """
    
    dummy_log_content_2 = """
    --- INFRASTRUCTURE ---
    {"os": "Linux", "device": "Tesla T4"}
    Epoch 01 | Loss: 6.8000 | Val: 6.9000 | LR: 0.000200 | Time: 600.00s
    Epoch 02 | Loss: 5.5000 | Val: 6.1000 | LR: 0.000400 | Time: 600.00s
    Epoch 03 | Loss: 4.2000 | Val: 5.8000 | LR: 0.000300 | Time: 600.00s
    --- FINAL EVALUATION (FAST) ---
    {"model": "transformer_base", "bleu": 0.0045, "rougeL": 0.2100}
    Total Duration: 1:00:00.000000
    """

    # Salvataggio file temporanei
    with open("log_model_A.txt", "w") as f: f.write(dummy_log_content)
    with open("log_model_B.txt", "w") as f: f.write(dummy_log_content_2)

    # Eseguiamo l'analisi
    lista_files = ["log_model_A.txt", "log_model_B.txt"]
    compare_models(lista_files)
    
    # Pulizia file temporanei (opzionale)
    # os.remove("log_model_A.txt")
    # os.remove("log_model_B.txt")