"""
================================================================================
DEEP AUDIT & QUALITATIVE INFERENCE UNIT
================================================================================
ROLE: High-Fidelity Model Assessment via Beam Search and Suite Testing.

DESIGN RATIONALE:
- Formal Verification: Generates a persistent audit log for post-mortem analysis.
- Deterministic Evaluation: Removes visual noise (emojis) to focus on 
  statistical consistency and sequence quality.
- Throughput Monitoring: Implements progress tracking for large checkpoint
  batches to monitor hardware efficiency during inference.
================================================================================
"""

import torch
import torch.nn.functional as F
import json
import gzip
import os
import sys
import random
from datetime import datetime
from tqdm import tqdm
from tokenizers import Tokenizer

# Path resolution to enable the internal 'models' package discovery.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.factory import get_model_architecture

def clean_output(text):
    """ Post-processing to normalize BPE spacing artifacts. """
    return text.replace('Ġ', ' ').replace('  ', ' ').strip()

def beam_search_decode(model, src_tensor, tokenizer, model_tag, beam_width=5, max_len=40, device="cpu", penalty=1.2):
    """ 
    Heuristic tree search for non-greedy sequence generation. 
    Implements repetition penalty to increase output diversity.
    """
    model.eval()
    sos_id = tokenizer.token_to_id("<SOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    pad_id = tokenizer.token_to_id("<PAD>")
    
    if "lstm" in model_tag:
        with torch.no_grad():
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
        beams = [([sos_id], 0.0, hidden, cell)]
    else:
        beams = [([sos_id], 0.0)]
        encoder_outputs = None

    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            for b in beams:
                if "lstm" in model_tag:
                    seq, score, h, c = b
                else:
                    seq, score = b
                
                if seq[-1] == eos_id:
                    all_candidates.append(b)
                    continue
                
                if "lstm" in model_tag:
                    input_token = torch.LongTensor([seq[-1]]).to(device)
                    output, next_h, next_c = model.decoder(input_token, h, c, encoder_outputs)
                else:
                    input_tensor = torch.LongTensor([seq]).to(device)
                    output = model(src_tensor, input_tensor)[:, -1, :] 
                
                log_probs = F.log_softmax(output, dim=-1).squeeze(0)
                
                # Apply repetition penalty to existing tokens in sequence.
                for idx in set(seq):
                    if idx not in [sos_id, eos_id, pad_id]:
                        log_probs[idx] -= penalty
                
                top_log_probs, top_indices = log_probs.topk(beam_width)
                for i in range(beam_width):
                    next_token = top_indices[i].item()
                    new_score = score + top_log_probs[i].item()
                    if "lstm" in model_tag:
                        all_candidates.append((seq + [next_token], new_score, next_h, next_c))
                    else:
                        all_candidates.append((seq + [next_token], new_score))
            
            # Global pruning: keep top K candidates.
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(b[0][-1] == eos_id for b in beams):
                break
                
    return beams[0][0]

def load_samples(data_dir, split, num_samples=10):
    """ Extracts random samples from processed splits for qualitative testing. """
    file_path = os.path.join(data_dir, f"{split}.jsonl.gz")
    if not os.path.exists(file_path):
        return []
    samples = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
            chosen = random.sample(lines, min(num_samples, len(lines)))
            for c in chosen:
                item = json.loads(c)
                samples.append({'code': item.get('code', ''), 'doc': item.get('docstring', '')})
    except Exception:
        pass
    return samples

def run_deep_audit():
    """ 
    Main entry point for model stress testing and log generation.
    Iterates through all checkpoints and generates comparative reports.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = os.path.join(project_root, "tokenizer.json")
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
    data_root = os.path.join(project_root, "Datasets", "processed")
    log_file = os.path.join(project_root, f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    if not os.path.exists(tokenizer_path):
        print("Error: Tokenizer file not found. Inference aborted.")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # Define multi-domain test suites.
    suites = {
        "TRAIN (Memorization)": load_samples(data_root, "train", 10),
        "TEST (Generalization)": load_samples(data_root, "test", 10),
        "CUSTOM (Logic Check)": [
            {"code": "def find_max(l):\n    return max(l) if l else None", "doc": "Finds the maximum value in a list."},
            {"code": "def save_file(data, path):\n    with open(path, 'w') as f:\n        f.write(data)", "doc": "Writes data to a file."}
        ]
    }

    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
    
    with open(log_file, "w", encoding="utf-8") as log:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = f"{'='*90}\nC2 DEEP AUDIT REPORT | {timestamp}\nDEVICE: {device} | MODE: BEAM SEARCH (k=5) | PENALTY: 1.2\n{'='*90}\n"
        print(header)
        log.write(header)

        # Progress bar for checkpoint iteration.
        for ckpt in tqdm(checkpoints, desc="Analyzing Weight Shards", unit="ckpt"):
            if "transformer" in ckpt.lower(): model_tag = "transformer"
            elif "lstm_dotproduct" in ckpt.lower(): model_tag = "lstm_dotproduct"
            elif "lstm_bahdanau" in ckpt.lower(): model_tag = "lstm_bahdanau"
            else: continue

            log.write(f"\n[CHECKPOINT] {ckpt} (Architecture: {model_tag})\n")
            
            try:
                # Factory synthesis and state loading.
                model = get_model_architecture(model_tag, device, vocab_size=vocab_size)
                model.load_state_dict(torch.load(os.path.join(checkpoint_dir, ckpt), map_location=device))
                model.eval()

                for suite_name, samples in suites.items():
                    if not samples: continue
                    log.write(f"\n--- SUITE: {suite_name} ---\n")
                    
                    for i, s in enumerate(samples):
                        # Tokenization and hardware mapping.
                        ids_input = tokenizer.encode(s['code']).ids
                        src_tensor = torch.LongTensor([1] + ids_input + [2]).unsqueeze(0).to(device)
                        
                        # Sequence decoding.
                        ids_pred = beam_search_decode(model, src_tensor, tokenizer, model_tag, beam_width=5, device=device)
                        prediction = clean_output(tokenizer.decode(ids_pred, skip_special_tokens=True))
                        
                        # Writing comparative data to log.
                        log.write(f"Sample {i+1}:\n")
                        log.write(f"  CODE: {s['code'].strip().replace(chr(10), ' ')[:60]}...\n")
                        log.write(f"  REAL: {s['doc'].strip()}\n")
                        log.write(f"  PRED: {prediction}\n")
                        log.write(f"  {'-'*15}\n")
            except Exception as e:
                log.write(f"  CRITICAL ERROR: Failed to audit {ckpt}. Reason: {str(e)}\n")

    print(f"\nAudit complete. Quantitative and qualitative report finalized at: {log_file}")

if __name__ == "__main__":
    run_deep_audit()