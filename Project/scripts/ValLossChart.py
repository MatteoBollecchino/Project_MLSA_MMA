import matplotlib.pyplot as plt
import re
import os

# --- CONFIGURATION ---
LOG_FILENAME = "20260201_210033_LSTM__S50000_B0.00.log"

# The relative path from the project root to the log folder
LOG_SUBDIR = os.path.join("logs", "Bahdanau")

OUTPUT_DIR_NAME = ""
# ---------------------

def get_absolute_paths():
    """
    Constructs absolute paths dynamically based on the script's location.
    Assumes the script is located in 'Project/scripts/' and logs are in 'Project/logs/'.
    """
    # Get the directory where this script is located (e.g., .../Project/scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to reach the project root (e.g., .../Project)
    project_root = os.path.dirname(script_dir)
    
    # Construct the full path to the log file
    full_log_path = os.path.join(project_root, LOG_SUBDIR, LOG_FILENAME)
    
    # Construct the full path for the output directory (inside scripts folder)
    output_dir = os.path.join(script_dir, OUTPUT_DIR_NAME)
    
    return full_log_path, output_dir

def parse_log_and_plot():
    log_path, output_dir = get_absolute_paths()
    
    epochs = []
    val_losses = []

    print(f"Reading file: {log_path} ...")

    # 1. Read file and extract data
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Regex to find lines like: "Epoch 01 | ... | Val: 7.3937"
        # Captures the Epoch number (Group 1) and Validation Loss (Group 2)
        pattern = re.compile(r"Epoch\s+(\d+).*?Val:\s+([\d\.]+)")

        for line in lines:
            match = pattern.search(line)
            if match:
                epoch_num = int(match.group(1))
                val_loss = float(match.group(2))
                
                epochs.append(epoch_num)
                val_losses.append(val_loss)

    except FileNotFoundError:
        print(f"ERROR: The file was not found at path: {log_path}")
        print("Check if the file name is correct and if the folder structure matches.")
        return

    if not epochs:
        print("WARNING: No data found. Please check the log file content format.")
        return

    # 2. Generate the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, marker='o', linestyle='-', color='#d62728', linewidth=2, label='Validation Loss')

    # Plot styling
    plt.title(f'Validation Loss Trend\n({LOG_FILENAME})', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure X-axis shows integer ticks for epochs
    plt.xticks(epochs) 
    plt.legend()

    # 3. Save the plot
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory: {e}")
            return

    # Construct output filename based on log filename
    output_filename = LOG_FILENAME.replace(".log", "_chart.png")
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path)
    plt.close() # Close the figure to free memory

    print("-" * 30)
    print(f"Success! Processed {len(epochs)} epochs.")
    print(f"Chart saved at: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    parse_log_and_plot()