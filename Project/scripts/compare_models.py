import matplotlib.pyplot as plt
import re
import os

# --- CONFIGURATION ---
# Define the models and their specific log files to compare
# Paths are relative to the project root
COMPARISON_CONFIG = {
    "Bahdanau (LSTM)": os.path.join("logs", "Bahdanau", "20260201_215008_LSTM__S20000_B0.00.log"),
    "DotProduct (LSTM)": os.path.join("logs", "DotProduct", "20260201_215511_LSTM__S20000_B0.00.log"),
    "Transformer": os.path.join("logs", "Transformer", "20260202_175409_TRANS_S20000_B0.00.log")
}

OUTPUT_DIR_NAME = ""
OUTPUT_FILENAME = "comparison_20k_subset.png"
# ---------------------

def get_project_root():
    """Returns the absolute path to the Project root (assuming script is in scripts/)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

def parse_log(file_path):
    """Parses a single log file to extract epochs and validation losses."""
    epochs = []
    val_losses = []
    
    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {file_path}")
        return [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        pattern = re.compile(r"Epoch\s+(\d+).*?Val:\s+([\d\.]+)")

        for line in lines:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return epochs, val_losses

def compare_models():
    root_dir = get_project_root()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    plt.figure(figsize=(12, 7))
    
    # Colors for distinct visibility
    colors = {"Bahdanau (LSTM)": "blue", "DotProduct (LSTM)": "orange", "Transformer": "green"}
    
    summary_data = []

    print(f"{'Model':<20} | {'Min Val Loss':<15} | {'Final Epoch':<12}")
    print("-" * 55)

    for model_name, relative_path in COMPARISON_CONFIG.items():
        full_path = os.path.join(root_dir, relative_path)
        epochs, losses = parse_log(full_path)
        
        if epochs:
            # Plotting the line
            plt.plot(epochs, losses, marker='o', markersize=4, linestyle='-', 
                     linewidth=2, label=model_name, color=colors.get(model_name, "black"))
            
            # Highlighting the best epoch
            min_loss = min(losses)
            best_epoch = epochs[losses.index(min_loss)]
            plt.scatter(best_epoch, min_loss, s=60, edgecolors='black', zorder=10)

            # Collecting stats for print summary
            min_loss = min(losses)
            final_epoch = epochs[-1]
            summary_data.append((model_name, min_loss))
            
            print(f"{model_name:<20} | {min_loss:.4f}          | {final_epoch}")
        else:
            print(f"{model_name:<20} | NO DATA FOUND")

    # X-axis ticks
    plt.xticks(epochs)

    # Chart Styling
    plt.title('Model Comparison: Subset 20k (Validation Loss)', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss (Lower is Better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Save the plot
    output_dir = os.path.join(script_dir, OUTPUT_DIR_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    plt.savefig(output_path)
    print("-" * 55)
    print(f"Comparison chart saved at: {output_path}")

if __name__ == "__main__":
    compare_models()