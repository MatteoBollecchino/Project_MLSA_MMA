"""
================================================================================
EXECUTION LOGGER & TELEMETRY UNIT
================================================================================
ROLE: Centralized Observability and Performance Tracking.

DESIGN RATIONALE:
- Data Integrity: Captures snapshots of the system state before and after runs.
- Analytics Ready: Exports structured JSON for downstream graphing and audit tools.
- Post-Mortem Diagnostics: Detailed timing of every pipeline stage (refinery, 
  tokenization, training) to identify computational bottlenecks.
================================================================================
"""

import os
import torch
import platform
import json
import time
from datetime import datetime

# --- [LOGGING AND TELEMETRY MANAGER] ---
class ExecutionLogger:
    """ 
    Black-box recorder for the ML Pipeline. 
    Manages the lifecycle of log files from initialization to final archival.
    """

    def __init__(self, root_dir, model_tag, subset_size=None):
        """ 
        Initializes the telemetry buffer and prepares filesystem hooks. 
        
        Args:
            root_dir (str): Base project directory.
            model_tag (str): Architectural identifier (e.g., 'transformer').
            subset_size (int): Data pressure coefficient used for this run.
        """
        self.start_time = datetime.now()

        # Metadata for session identification.
        self.model_tag = model_tag
        self.subset_size = subset_size if subset_size else "ALL"
        
        # Ensure log directory existence without interrupting the pipeline flow.
        self.log_dir = os.path.join(root_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Temporary file creation to protect against incomplete data if the session crashes.
        self.temp_name = f"exec_{self.start_time.strftime('%Y%m%d_%H%M%S')}.tmp"
        self.file_path = os.path.join(self.log_dir, self.temp_name)
        
        # Structured Data Schema: Designed for direct serialization to JSON.
        self.data = {
            "metadata": {
                "model_type": model_tag,
                "subset": self.subset_size,
                "start_time": self.start_time.isoformat()
            },
            "durations": {}, # Stores execution time per phase (e.g., "data_cleaning")
            "infrastructure": {}, # Stores hardware specs
            "training_history": [], # Stores per-epoch metrics
            "evaluation": {} # Stores final BLEU/ROUGE scores
        }

    def log_sys_info(self):
        """ 
        Fingerprints the execution environment. 
        Captures OS, Python version, and GPU architecture to normalize performance benchmarks.
        """
        info = {
            "os": platform.system(),
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        }

        # VRAM monitoring: Crucial for auditing OOM (Out of Memory) conditions.
        if torch.cuda.is_available():
            info["vram_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            
        self.data["infrastructure"] = info
        self._write_to_file(f"--- INFRASTRUCTURE ---\n{json.dumps(info, indent=4)}\n")

    def log_phase(self, phase_name, duration_seconds):
        """ 
        Logs the temporal cost of specific pipeline stages. 
        Used to identify bottlenecks (e.g., slow data refinery vs. fast training).
        """
        dur_str = f"{duration_seconds:.2f}s"
        self.data["durations"][phase_name] = dur_str
        self._write_to_file(f"[TIMER] Phase '{phase_name}' completed in {dur_str}\n")

    def log_tokenizer_info(self, vocab_size, reused=True, duration=0):
        """ 
        Logs vocabulary constraints and tokenizer performance. 
        Ensures the model/tokenizer vocabulary match is tracked.
        """
        tok_info = {
            "vocab_size": vocab_size,
            "status": "REUSED" if reused else "NEWLY_TRAINED",
            "duration": f"{duration:.2f}s"
        }
        self.data["metadata"]["tokenizer"] = tok_info
        self._write_to_file(f"--- TOKENIZER ---\n{json.dumps(tok_info, indent=4)}\n")

    def log_epoch(self, epoch, train_loss, val_loss, lr, epoch_duration):
        """ 
        Captures granular epoch data. 
        Creates a time-series record of the learning process for curve analysis.
        """
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "duration": f"{epoch_duration:.2f}s"
        }
        self.data["training_history"].append(entry)
        self._write_to_file(
            f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {lr:.6f} | Time: {epoch_duration:.2f}s\n"
        )

    def log_final_metrics(self, df_metrics, mode="fast"):
        """
        Aggregates terminal performance scores. 
        Accepts a Pandas DataFrame from the Evaluator and flattens it for the log.
        """
        metrics = df_metrics.to_dict(orient='records')[0] if df_metrics is not None and not df_metrics.empty else {}
        metrics["eval_mode"] = mode
        self.data["evaluation"] = metrics
        self._write_to_file(f"\n--- FINAL EVALUATION ({mode.upper()}) ---\n{json.dumps(metrics, indent=4)}\n")

    def finalize(self):
        """
        Renames temporary files into permanent chronological records. 
        Includes the BLEU score in the filename for immediate visual audit of logs.
        """
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        self.data["durations"]["total_execution"] = str(total_duration)
        
        # BLEU Tagging Logic: Makes it easy to find the highest-performing checkpoints.
        bleu = self.data["evaluation"].get("bleu", 0)
        bleu_tag = f"B{bleu:.2f}" if bleu > 0.0001 else "NA"
        
        # Slug format: TIMESTAMP_ARCH_SUBSET_METRIC
        full_ts = self.start_time.strftime('%Y%m%d_%H%M%S')
        model_short = self.model_tag.upper()[:5]
        subset_tag = f"S{self.subset_size}"
        
        final_name = f"{full_ts}_{model_short}_{subset_tag}_{bleu_tag}"
        
        final_path_log = os.path.join(self.log_dir, f"{final_name}.log")
        final_path_json = os.path.join(self.log_dir, f"{final_name}.json")

        self._write_to_file(f"\n--- ALL PHASE DURATIONS ---\n{json.dumps(self.data['durations'], indent=4)}\n")
        self._write_to_file(f"\n--- FINISHED ---\nTotal Duration: {total_duration}\n")
        
        try:
            # Atomic Move: Prevents file corruption and ensures log persistence.
            if os.path.exists(self.file_path):
                os.rename(self.file_path, final_path_log)
            
            # Export structured data for visualization scripts.
            with open(final_path_json, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4)
                
            print(f"✅ Log Saved: {os.path.basename(final_path_log)}")
        except Exception as e:
            print(f"⚠️ Critical error during log archiving: {e}")

    def _write_to_file(self, text):
        """ Low-level I/O method to commit text to the physical storage device. """
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(text)