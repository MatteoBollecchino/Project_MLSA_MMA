import re
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# --- [CONFIGURATION] ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    'font.size': 11, 
    'figure.titlesize': 16,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10
})

class LogComparator:
    """CMU for individual log forensic analysis."""

    def __init__(self):
        self.epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)", re.IGNORECASE)
        self.bleu_pattern = re.compile(r'["\']?bleu["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        self.rouge_pattern = re.compile(r'["\']?rougeL["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE)
        self.internal_filename_pattern = re.compile(r'"file"\s*:\s*"([^"]+)"', re.IGNORECASE)

    def _get_short_name(self, filepath):
        name = os.path.splitext(os.path.basename(filepath))[0]
        return re.sub(r'^\d{8}_\d{6}_', '', name)

    def parse_file(self, filepath):
        data = {
            "name": self._get_short_name(filepath), 
            "epochs": [], 
            "train_loss": [], 
            "val_loss": [], 
            "bleu": None, 
            "rougeL": None
        }

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            name_match = self.internal_filename_pattern.search(content)

            if name_match: data["name"] = name_match.group(1).replace(".pt", "")

            for m in self.epoch_pattern.finditer(content):
                data["epochs"].append(int(m.group(1)))
                data["train_loss"].append(float(m.group(2)))
                data["val_loss"].append(float(m.group(3)))

            b_m = self.bleu_pattern.search(content)
            r_m = self.rouge_pattern.search(content)

            if b_m: data["bleu"] = float(b_m.group(1))
            if r_m: data["rougeL"] = float(r_m.group(1))

        except Exception as e:
            print(f"Parsing error: {e}"); return None
        
        return data

    def plot_best_model_trace(self, log_path):
        """Generates high-fidelity training trace for the best model."""

        data = self.parse_file(log_path)

        if not data: return

        plt.figure(figsize=(12, 7))

        plt.plot(data["epochs"], data["train_loss"], label='Training Loss', color='#3498db', linewidth=2.5, alpha=0.8)

        plt.plot(data["epochs"], data["val_loss"], label='Validation Loss', color='#e67e22', linewidth=2.5)
        
        min_loss = min(data["val_loss"])
        best_epoch = data["val_loss"].index(min_loss) + 1
        plt.axvline(x=best_epoch, color='#c0392b', linestyle='--', alpha=0.6, label=f'Optimal Stopping (Ep {best_epoch})')
        plt.scatter(best_epoch, min_loss, color='#c0392b', s=100, zorder=5)

        plt.title(f"Best Model Convergence Profile: {data['name']}", pad=20)
        plt.xlabel("Epochs"); plt.ylabel("Cross-Entropy Loss")
        plt.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(SCRIPT_DIR, "best_model_trace.png")
        plt.savefig(save_path, dpi=300); 
        plt.show()

# --- [VISUALIZATION ENGINE] ---

def extract_subset_size(filename):
    match = re.search(r"sub(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def plot_linguistic_metrics(df):
    """Section 1: Spaced 2x2 grid for Core NLP Metrics."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    metrics = [('bleu', 'BLEU Score'), ('rougeL', 'ROUGE-L Score'), ('loss', 'Final Val Loss'), ('perplexity', 'Perplexity')]
    models = df['model'].unique(); subsets = sorted(df['subset_size'].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(subsets)))
    x = np.arange(len(models)); w = 0.8 / len(subsets)

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx//2, idx%2]

        for i, s in enumerate(subsets):
            sub_df = df[df['subset_size'] == s]
            vals = [sub_df[sub_df['model']==m][col].values[0] if not sub_df[sub_df['model']==m].empty else 0 for m in models]
            ax.bar(x + (i - len(subsets)/2)*w + w/2, vals, w, color=colors[i], label=f"Size: {s}" if idx==0 else "", edgecolor='white', alpha=0.9)

        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xticks(x); ax.set_xticklabels(models); ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.legend(*(axes[0,0].get_legend_handles_labels()), loc='upper center', ncol=len(subsets), bbox_to_anchor=(0.5, 0.97), frameon=False)

    plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.9)
    plt.savefig(os.path.join(SCRIPT_DIR, "linguistic_audit.png"), dpi=300); 
    plt.show()

def plot_temporal_efficiency(df):
    """Section 2: Detailed Spaced Temporal Costs."""

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    configs = [('Mean Epoch (s)', 'Epoch Duration (s)'), ('throughput', 'Throughput (Samples/s)'), ('time_per_1k', 'Cost per 1k Samples (s)')]

    models = df['model'].unique(); subsets = sorted(df['subset_size'].unique())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(subsets)))
    x = np.arange(len(models)); w = 0.8 / len(subsets)

    for idx, (col, title) in enumerate(configs):
        ax = axes[idx]

        for i, s in enumerate(subsets):
            sub_df = df[df['subset_size'] == s]
            vals = [sub_df[sub_df['model']==m][col].values[0] if not sub_df[sub_df['model']==m].empty else 0 for m in models]
            ax.bar(x + (i - len(subsets)/2)*w + w/2, vals, w, color=colors[i], edgecolor='white')

        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=90)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.subplots_adjust(wspace=0.35, bottom=0.2)
    plt.savefig(os.path.join(SCRIPT_DIR, "temporal_efficiency.png"), dpi=300); 
    plt.show()

# --- [MAIN EXECUTION] ---

if __name__ == "__main__":
    json_path = "Project/logs/result_metrics.json"
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f: df = pd.DataFrame(json.load(f))
        
        # Feature Engineering
        df['subset_size'] = df['file'].apply(extract_subset_size)
        df['Mean Epoch (s)'] = pd.to_numeric(df['Mean Epoch (s)'], errors='coerce').fillna(0)
        df['throughput'] = df['subset_size'] / df['Mean Epoch (s)']
        df['time_per_1k'] = (df['Mean Epoch (s)'] / df['subset_size']) * 1000
        df = df.sort_values(by='subset_size')

        # Run Sectioned Audit
        plot_linguistic_metrics(df)
        plot_temporal_efficiency(df)
    else:
        print(f"[ERROR] Metrics file missing: {json_path}")

    # Final trace: Bahdanau 70k Champion
    best_log = "Project/logs/Bahdanau/20260203_002032_LSTM__S70000_B0.01.log"
    if os.path.exists(best_log):
        LogComparator().plot_best_model_trace(best_log)