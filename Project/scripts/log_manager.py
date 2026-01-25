import os
import torch
import platform
import json
import time
from datetime import datetime

# --- [LOGGING AND TELEMETRY MANAGER] ---
class ExecutionLogger:

    # Initialization of the logger
    def __init__(self, root_dir, model_tag, subset_size=None):
        self.start_time = datetime.now()

        # Tag that identifies the model architecture
        self.model_tag = model_tag

        # Subset size used during training
        self.subset_size = subset_size if subset_size else "ALL"
        
        # Directory for saving logs
        self.log_dir = os.path.join(root_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Temporary filename (will be renamed in finalize)
        self.temp_name = f"exec_{self.start_time.strftime('%Y%m%d_%H%M%S')}.tmp"
        self.file_path = os.path.join(self.log_dir, self.temp_name)
        
        # Internal Data Structure (JSON Ready)
        self.data = {
            "metadata": {
                "model_type": model_tag,
                "subset": self.subset_size,
                "start_time": self.start_time.isoformat()
            },
            "durations": {}, # Phase -> "Xs"
            "infrastructure": {},
            "training_history": [],
            "evaluation": {}
        }

    # Fingerprints hardware per telemetric audit.
    def log_sys_info(self):

        # System Information
        info = {
            "os": platform.system(),
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        }

        if torch.cuda.is_available():
            info["vram_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            
        self.data["infrastructure"] = info
        self._write_to_file(f"--- INFRASTRUCTURE ---\n{json.dumps(info, indent=4)}\n")

    # Logs the duration of a macro-phase (e.g., data loading).
    def log_phase(self, phase_name, duration_seconds):
        dur_str = f"{duration_seconds:.2f}s"
        self.data["durations"][phase_name] = dur_str
        self._write_to_file(f"[TIMER] Phase '{phase_name}' completed in {dur_str}\n")

    # Details about the tokenizer pipeline.
    def log_tokenizer_info(self, vocab_size, reused=True, duration=0):
        tok_info = {
            "vocab_size": vocab_size,
            "status": "REUSED" if reused else "NEWLY_TRAINED",
            "duration": f"{duration:.2f}s"
        }
        self.data["metadata"]["tokenizer"] = tok_info
        self._write_to_file(f"--- TOKENIZER ---\n{json.dumps(tok_info, indent=4)}\n")

    # Granular logging for each training epoch.
    def log_epoch(self, epoch, train_loss, val_loss, lr, epoch_duration):
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

    # Logs the final evaluation metrics after training.
    def log_final_metrics(self, df_metrics, mode="fast"):
        metrics = df_metrics.to_dict(orient='records')[0] if df_metrics is not None and not df_metrics.empty else {}
        metrics["eval_mode"] = mode
        self.data["evaluation"] = metrics
        self._write_to_file(f"\n--- FINAL EVALUATION ({mode.upper()}) ---\n{json.dumps(metrics, indent=4)}\n")

    # Finalizes the logging by renaming the file with a timestamp prefix.
    # Renames the file with an ISO 8601 chronological prefix for LIFO ordering.
    def finalize(self):
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        self.data["durations"]["total_execution"] = str(total_duration)
        
        # Extraction of Metrics for the Slug
        bleu = self.data["evaluation"].get("bleu", 0)
        # If BLEU is almost zero or absent, use NA
        bleu_tag = f"B{bleu:.2f}" if bleu > 0.0001 else "NA"
        
        # Naming Slug: 20260125_143022_TRANS_S50000_B0.25
        full_ts = self.start_time.strftime('%Y%m%d_%H%M%S')
        model_short = self.model_tag.upper()[:5]
        subset_tag = f"S{self.subset_size}"
        
        final_name = f"{full_ts}_{model_short}_{subset_tag}_{bleu_tag}"
        
        final_path_log = os.path.join(self.log_dir, f"{final_name}.log")
        final_path_json = os.path.join(self.log_dir, f"{final_name}.json")

        # Final summary write
        self._write_to_file(f"\n--- ALL PHASE DURATIONS ---\n{json.dumps(self.data['durations'], indent=4)}\n")
        self._write_to_file(f"\n--- FINISHED ---\nTotal Duration: {total_duration}\n")
        
        try:
            # Closing and renaming (Atomic Move)
            if os.path.exists(self.file_path):
                os.rename(self.file_path, final_path_log)
            
            # Export metadata in JSON for future analysis (e.g., graphs)
            with open(final_path_json, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4)
                
            print(f"✅ Log Saved: {os.path.basename(final_path_log)}")
        except Exception as e:
            print(f"⚠️ Critical error during log archiving: {e}")

    # Internal method to append text to the log file.
    def _write_to_file(self, text):
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(text)