import re
import matplotlib.pyplot as plt
import os
import numpy as np

class LogComparator:
    def __init__(self):
        # Regex standard
        self.epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)", re.IGNORECASE)
        self.bleu_pattern = re.compile(r'["\']?bleu["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        self.rouge_pattern = re.compile(r'["\']?rougeL["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)

        # Cerca: "file": "QUALCOSA"
        # Spiegazione:
        #   "file"\s*:\s* -> cerca la chiave "file" seguita da due punti e spazi
        #   "([^"]+)"      -> cattura tutto ciò che sta tra le virgolette successive
        self.internal_filename_pattern = re.compile(r'"file"\s*:\s*"([^"]+)"', re.IGNORECASE)

    def _get_short_name(self, filepath):
        """Crea un'etichetta leggibile per la legenda, rimuovendo timestamp e percorsi."""
        filename = os.path.basename(filepath)
        # Rimuove estensione
        name = os.path.splitext(filename)[0]
        # Rimuove data/ora (es. 20260201_215511_) per accorciare la label
        name = re.sub(r'^\d{8}_\d{6}_', '', name)
        return name

    def parse_file(self, filepath):
        data = {
            "name": self._get_short_name(filepath),
            "epochs": [],
            "val_loss": [],
            "bleu": 0.0,
            "rougeL": 0.0
        }

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # --- PARTE NUOVA: Estrazione nome interno ---
            name_match = self.internal_filename_pattern.search(content)
            if name_match:
                # Se trovato, estraiamo il nome (es: "20260201...sub50000.pt")
                internal_name = name_match.group(1)
                
                # Opzionale: Pulizia estensioni (.pt, .pth) per rendere il grafico più bello
                internal_name = internal_name.replace(".pt", "").replace(".pth", "")
                
                # Sovrascriviamo il nome nel dizionario
                data["name"] = internal_name
            # ---------------------------------------------

            # Parsing delle epoche (come prima)
            for match in self.epoch_pattern.finditer(content):
                data["epochs"].append(int(match.group(1)))
                data["val_loss"].append(float(match.group(3)))

            # Parsing Metriche (come prima)
            bleu_match = self.bleu_pattern.search(content)
            if bleu_match: data["bleu"] = float(bleu_match.group(1))

            rouge_match = self.rouge_pattern.search(content)
            if rouge_match: data["rougeL"] = float(rouge_match.group(1))
            
        except Exception as e:
            print(f"Errore parsing {filepath}: {e}")
            return None
            
        return data

    def compare_logs(self, file_list):
        """
        Input: Lista di percorsi file (strings).
        Output: Grafico comparativo.
        """
        parsed_data = []
        for f in file_list:
            if os.path.exists(f):
                d = self.parse_file(f)
                if d and d["epochs"]: # Accetta solo se ci sono dati validi
                    parsed_data.append(d)
            else:
                print(f"[WARN] File non trovato: {f}")

        if not parsed_data:
            print("Nessun dato valido da confrontare.")
            return

        # --- SETUP GRAFICO (2 Sottografici) ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle("Model Comparison: Validation Dynamics & Final Metrics", fontsize=16, fontweight='bold')

        # --- 1. GRAFICO VALIDATION LOSS (Curve) ---
        ax_loss = axes[0]
        
        for entry in parsed_data:
            # Plotta la curva
            ax_loss.plot(entry["epochs"], entry["val_loss"], 
                         label=entry["name"], linewidth=2, marker='o', markersize=4, alpha=0.8)
            
            # Segna il punto minimo (Best Model)
            min_loss = min(entry["val_loss"])
            min_idx = entry["val_loss"].index(min_loss)
            best_epoch = entry["epochs"][min_idx]
            ax_loss.scatter(best_epoch, min_loss, s=80, edgecolors='black', zorder=10)

        # 1. Trova il numero massimo di epoche tra tutti i dati per definire il range
        # (Oppure usa la lista delle epoche se stai plottando un solo grafico)
        max_epoch = max(entry["epochs"]) if isinstance(entry["epochs"], list) else 0
        
        # Crea una sequenza di numeri interi da 1 a Max
        all_ticks = range(1, max_epoch + 1)

        # 2. Imposta i ticks sull'asse
        ax_loss.set_xticks(all_ticks)
        
        # Opzionale: Se i numeri si sovrappongono, riduci la dimensione del font
        ax_loss.tick_params(axis='x', labelsize=8)
    
        ax_loss.set_title("Validation Loss Evolution")
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Validation Loss")
        ax_loss.legend(fontsize='small')
        ax_loss.grid(True, linestyle='--', alpha=0.5)

        # --- 2. GRAFICO BARRE METRICHE (Bar Chart) ---
        ax_metrics = axes[1]
        
        names = [d["name"] for d in parsed_data]
        bleu_scores = [d["bleu"] for d in parsed_data]
        rouge_scores = [d["rougeL"] for d in parsed_data]

        x = np.arange(len(names))  # Posizioni etichette
        width = 0.35  # Larghezza barre

        # Disegna le barre
        rects1 = ax_metrics.bar(x - width/2, bleu_scores, width, label='BLEU', color='royalblue', alpha=0.8)
        rects2 = ax_metrics.bar(x + width/2, rouge_scores, width, label='ROUGE-L', color='seagreen', alpha=0.8)

        # Configurazione assi
        ax_metrics.set_title("Final Evaluation Metrics")
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(names, rotation=15, ha="right")
        ax_metrics.set_ylabel("Score")
        ax_metrics.legend()
        ax_metrics.grid(axis='y', linestyle='--', alpha=0.5)

        # Aggiunge i valori sopra le barre
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax_metrics.annotate(f'{height:.5f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    
    # 1. Definisci i file che vuoi confrontare manualmente
    # ESEMPIO 1: CONFRONTO TRA DIVERSI MODELLI MA STESSO SUBSET SIZE
    files_to_compare = [
        "Project/logs/Bahdanau/20260201_215008_LSTM__S20000_B0.00.log",
        "Project/logs/DotProduct/20260201_215511_LSTM__S20000_B0.00.log",
        "Project/logs/Transformer/20260202_175409_TRANS_S20000_B0.00.log"
    ]
    
    # 2. Crea il comparatore ed esegui
    comparator = LogComparator()
    
    # Nota: Assicurati che i file esistano, altrimenti lo script li salta con un warning
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