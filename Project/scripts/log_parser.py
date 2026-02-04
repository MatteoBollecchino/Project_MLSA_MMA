import re
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import pandas as pd

class LogComparator:
    def __init__(self):
        # Regex standard
        # Group 1: Epoch, Group 2: Train Loss, Group 3: Val Loss
        self.epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)", re.IGNORECASE)
        self.bleu_pattern = re.compile(r'["\']?bleu["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        self.rouge_pattern = re.compile(r'["\']?rougeL["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        self.internal_filename_pattern = re.compile(r'"file"\s*:\s*"([^"]+)"', re.IGNORECASE)

    def _get_short_name(self, filepath):
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        name = re.sub(r'^\d{8}_\d{6}_', '', name) # Rimuove timestamp
        return name

    def parse_file(self, filepath):
        default_name = self._get_short_name(filepath)
        data = {
            "name": default_name,
            "epochs": [],
            "train_loss": [], # Per il plot singolo
            "val_loss": [],
            "bleu": None,
            "rougeL": None
        }

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Cerca nome interno
            name_match = self.internal_filename_pattern.search(content)
            if name_match:
                clean_name = name_match.group(1).replace(".pt", "")
                data["name"] = clean_name

            # Estrazione Loss
            for match in self.epoch_pattern.finditer(content):
                data["epochs"].append(int(match.group(1)))
                data["train_loss"].append(float(match.group(2))) # Train
                data["val_loss"].append(float(match.group(3)))   # Val

            # Estrazione Metriche
            bleu_match = self.bleu_pattern.search(content)
            if bleu_match: data["bleu"] = float(bleu_match.group(1))

            rouge_match = self.rouge_pattern.search(content)
            if rouge_match: data["rougeL"] = float(rouge_match.group(1))
            
        except Exception as e:
            print(f"Errore parsing {filepath}: {e}")
            return None
            
        return data

    def compare_logs(self, file_list):
        """Metodo principale che decide quale grafico mostrare."""
        parsed_data = []
        for f in file_list:
            if os.path.exists(f):
                d = self.parse_file(f)
                if d and d["epochs"]:
                    parsed_data.append(d)
            else:
                print(f"[WARN] File non trovato: {f}")

        if not parsed_data:
            print("Nessun dato valido trovato.")
            return

        # --- LOGICA DI SCELTA ---
        if len(parsed_data) == 1:
            print(f"Rilevato file singolo: Mostro Training vs Validation per '{parsed_data[0]['name']}'")
            self._plot_single_mode(parsed_data[0])
        else:
            print(f"Rilevati {len(parsed_data)} file: Mostro confronto comparativo.")
            self._plot_compare_mode(parsed_data)

    # --- MODALITÀ 1: SINGOLO FILE (Train vs Val) ---
    def _plot_single_mode(self, entry):
        epochs = entry["epochs"]
        train_loss = entry["train_loss"]
        val_loss = entry["val_loss"]
        
        plt.figure(figsize=(12, 6))
        
        # Curve
        plt.plot(epochs, train_loss, label='Training Loss', color='skyblue', linewidth=2)
        plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)

        # Best Model Marker
        min_val = min(val_loss)
        best_epoch = epochs[val_loss.index(min_val)]
        plt.scatter(best_epoch, min_val, color='red', s=100, zorder=5, label='Best Model')
        plt.annotate(f"Min: {min_val:.4f}", (best_epoch, min_val), 
                     textcoords="offset points", xytext=(0, -20), ha='center', color='red')

        # Box Metriche
        bleu_txt = f"{entry['bleu']:.4f}" if entry['bleu'] is not None else "N/A"
        rouge_txt = f"{entry['rougeL']:.4f}" if entry['rougeL'] is not None else "N/A"
        info_text = f"BLEU: {bleu_txt}\nROUGE-L: {rouge_txt}"
        
        plt.gca().text(0.98, 0.95, info_text, transform=plt.gca().transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                       fontsize=11)
        # Setup Asse X
        plt.xticks(epochs)

        plt.title(f"Training Dynamics: {entry['name']}", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    # --- MODALITÀ 2: MULTIPLI FILE (Confronto) ---
    def _plot_compare_mode(self, data_list):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle("Model Comparison: Dynamics & Metrics", fontsize=16, fontweight='bold')

        # 1. Curve Validation Loss
        ax_loss = axes[0]
        max_epochs_found = 0
        
        for entry in data_list:
            ax_loss.plot(entry["epochs"], entry["val_loss"], 
                         label=entry["name"], linewidth=2, marker='o', markersize=4, alpha=0.8)
            
            # Segna minimo
            min_loss = min(entry["val_loss"])
            best_epoch = entry["epochs"][entry["val_loss"].index(min_loss)]
            ax_loss.scatter(best_epoch, min_loss, s=60, edgecolors='black', zorder=10)
            
            max_epochs_found = max(max_epochs_found, max(entry["epochs"]))

        # Ticks asse X
        ax_loss.set_xticks(range(1, max_epochs_found + 1))

        ax_loss.set_title("Validation Loss Evolution")
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Validation Loss")
        ax_loss.legend(fontsize='small')
        ax_loss.grid(True, linestyle='--', alpha=0.5)

        # 2. Bar Chart Metriche
        ax_metrics = axes[1]
        names = [d["name"] for d in data_list]
        # Usa 0.0 se None per evitare crash nel plot
        bleus = [d["bleu"] if d["bleu"] else 0.0 for d in data_list]
        rouges = [d["rougeL"] if d["rougeL"] else 0.0 for d in data_list]

        x = np.arange(len(names))
        width = 0.35

        rects1 = ax_metrics.bar(x - width/2, bleus, width, label='BLEU', color='royalblue', alpha=0.8)
        rects2 = ax_metrics.bar(x + width/2, rouges, width, label='ROUGE-L', color='seagreen', alpha=0.8)

        ax_metrics.set_title("Final Evaluation Metrics")
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(names, rotation=15, ha="right")
        ax_metrics.legend()
        ax_metrics.grid(axis='y', linestyle='--', alpha=0.5)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax_metrics.annotate(f'{height:.5f}', # 5 decimali qui
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def extract_subset_size(filename):
    """Estrae la dimensione del subset dal nome del file."""
    if not filename: return 0
    match = re.search(r"sub(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def plot_metrics(filepath):
    # --- 1. CARICAMENTO DATI ---
    if not os.path.exists(filepath):
        print(f"File non trovato: {filepath}")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Errore lettura JSON: {e}")
        return

    df = pd.DataFrame(data)
    # Assumiamo che la funzione extract_subset_size sia definita altrove
    df['subset_size'] = df['file'].apply(extract_subset_size)
    
    # Ordiniamo per subset size per coerenza visiva (barre crescenti)
    df = df.sort_values(by='subset_size')

    # Identifichiamo modelli unici e dimensioni uniche
    models = df['model'].unique()
    subset_sizes = df['subset_size'].unique()
    subset_sizes.sort()
    
    n_models = len(models)
    n_subsets = len(subset_sizes)

    # --- CONFIGURAZIONE COLORI (per Subset Size) ---
    # Usiamo una mappa di colori diversa per distinguere le dimensioni dei dati
    # Es: gradazioni di blu o colori distinti
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_subsets)) 
    
    # --- 2. CONFIGURAZIONE GRIGLIA 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparison Model Performance', fontsize=16, fontweight='bold')

    ax_flat = axes.flatten()

    metrics_config = [
        {"col": "bleu",       "title": "BLEU Score (Higher is Better)", "ylabel": "Score"},
        {"col": "rougeL",     "title": "ROUGE-L Score (Higher is Better)", "ylabel": "Score"},
        {"col": "loss",       "title": "Validation Loss (Lower is Better)", "ylabel": "Loss"},
        {"col": "perplexity", "title": "Perplexity (Lower is Better)", "ylabel": "PPL"} 
    ]

    # --- CONFIGURAZIONE BARRE ---
    # Larghezza totale disponibile per ogni gruppo (modello) = 0.8
    # Larghezza singola barra = 0.8 / numero di subset
    total_width = 0.8
    bar_width = total_width / n_subsets
    
    # Indici asse X (uno per ogni modello)
    x_indices = np.arange(n_models)

    # --- 3. PLOTTING ---
    for i, config in enumerate(metrics_config):
        ax = ax_flat[i]
        metric_col = config["col"]
        
        # Iteriamo su ogni dimensione del dataset (ogni subset diventa una barra colorata)
        for j, size in enumerate(subset_sizes):
            # Filtriamo i valori per questa specifica dimensione
            subset_data = df[df['subset_size'] == size]
            
            # Dobbiamo allineare i dati all'ordine dei modelli in 'models'
            # Creiamo una lista di valori ordinata in base a 'models'
            values = []
            for model in models:
                row = subset_data[subset_data['model'] == model]
                if not row.empty:
                    values.append(row[metric_col].values[0])
                else:
                    values.append(0) # O np.nan se preferisci non plottare nulla

            # Calcolo offset per centrare il gruppo di barre
            # j va da 0 a n_subsets-1. 
            offset = (j - n_subsets / 2) * bar_width + (bar_width / 2)
            
            # Plot della barra
            rects = ax.bar(
                x_indices + offset, 
                values, 
                bar_width, 
                label=f"Size: {size}" if i == 0 else "", # Label solo nel primo grafico per la legenda unica
                color=colors[j],
                alpha=0.9,
                edgecolor='white'
            )

            # Annotazioni sopra le barre
            for rect in rects:
                height = rect.get_height()
                if height > 0: # Evita di annotare barre vuote
                    fmt = "{:.5f}"
                    
                    ax.annotate(fmt.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        # Configurazione Assi
        ax.set_title(config["title"], fontsize=10, fontweight='bold', color='#333333')
        ax.set_ylabel(config["ylabel"])
        
        # Imposta i nomi dei modelli sull'asse X
        ax.set_xticks(x_indices)
        ax.set_xticklabels(models, rotation=0, ha='center', fontsize=10, fontweight='medium')
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # --- LEGENDA E LAYOUT ---
    # La legenda ora mostra le dimensioni del dataset (Subset Sizes)
    handles, labels = ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Training Set Size", loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=n_subsets, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.0)
    plt.show()

def main():

    # Plot delle metriche di tutti i modelli
    json_file = "Project/logs/result_metrics.json"
    plot_metrics(json_file)

    """
    # 1. Definisci i file che vuoi confrontare manualmente
    files_to_compare = [
        "Project/logs/Bahdanau/20260201_210033_LSTM__S50000_B0.00.log"
    ]

    # 2. Crea il comparatore ed esegui
    comparator = LogComparator()
    comparator.compare_logs(files_to_compare)

    # ESEMPIO 1: CONFRONTO TRA DIVERSI MODELLI MA STESSO SUBSET SIZE
    files_to_compare = [
        "Project/logs/Bahdanau/20260201_215008_LSTM__S20000_B0.00.log",
        "Project/logs/DotProduct/20260201_215511_LSTM__S20000_B0.00.log",
        "Project/logs/Transformer/20260202_175409_TRANS_S20000_B0.00.log"
    ]
    
    comparator.compare_logs(files_to_compare)


    # ESEMPIO 2: CONFRONTO TRA STESSI MODELLI MA DIVERSI SUBSET SIZE
    files_to_compare = [
        "Project/logs/Bahdanau/20260202_155459_LSTM__S50000_B0.00.log",
        "Project/logs/Bahdanau/20260203_002032_LSTM__S70000_B0.01.log",
    ]

    comparator.compare_logs(files_to_compare)

    # ESEMPIO 3: CONFRONTO TRA STESSI MODELLI MA DIVERSI SUBSET SIZE
    files_to_compare = [
        "Project/logs/DotProduct/20260202_151709_LSTM__S20000_B0.00.log",
        "Project/logs/DotProduct/20260202_152825_LSTM__S50000_B0.00.log",
        "Project/logs/DotProduct/20260202_171614_LSTM__S70000_B0.00.log"
    ]

    comparator.compare_logs(files_to_compare)

    # ESEMPIO 4: CONFRONTO TRA STESSI MODELLI MA DIVERSI SUBSET SIZE
    files_to_compare = [
        "Project/logs/Transformer/20260202_175409_TRANS_S20000_B0.00.log",
        "Project/logs/Transformer/20260202_092805_TRANS_S50000_B0.00.log",
        "Project/logs/Transformer/20260202_112600_TRANS_S70000_B0.00.log"
    ]

    comparator.compare_logs(files_to_compare)
    """

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    main()
    