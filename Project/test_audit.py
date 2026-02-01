"""
================================================================================
DEEP AUDIT & QUALITATIVE INFERENCE UNIT
================================================================================
ROLE: High-Fidelity Model Assessment via Beam Search and Suite Testing.

DESIGN RATIONALE:
- Non-Greedy Exploration: Uses Beam Search (k=5) to navigate the probability 
  landscape, preventing the model from choosing short-term high-prob tokens 
   that lead to long-term semantic collapse.
- Universal Decoding: A polymorphic decoding function that adapts to the 
  recurrent nature of LSTMs or the attention-centric nature of Transformers.
- Heuristic Debugging: Includes a 'Custom Suite' for manually curated edge cases 
  (e.g., max calculation, file I/O) to verify functional logic understanding.
================================================================================
"""

import torch
import torch.nn.functional as F
import json
import gzip
import os
import glob
import random
import sys
from tokenizers import Tokenizer

# Path resolution to enable the internal 'models' package discovery.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.factory import get_model_architecture

# --- [PHASE 1] DETERMINISTIC NORMALIZATION ---

def clean_output(text):
    """
    POST-PROCESSING LAYER: Cleans BPE artifacts for human readability.
    
    Technical Detail: Byte-Level BPE often leaves 'Ġ' symbols (representing spaces) 
    in the raw string. This function normalizes them into standard UTF-8 spacing.
    """
    return text.replace('Ġ', ' ').replace('  ', ' ').strip()

# --- [PHASE 2] BEAM SEARCH DECODING LOGIC ---

def beam_search_decode(model, src_tensor, tokenizer, model_tag, beam_width=5, max_len=40, device="cpu", penalty=2.0):
    """
    HEURISTIC TREE SEARCH: Explores multiple candidate sequences in parallel.
    
    Logic:
    1. Tracking: Maintains a list of 'Beams'. Each beam is (sequence, score, [states]).
    2. Expansion: Each beam predicts the next V tokens. Total candidates = k * V.
    3. Pruning: Sorts by cumulative log-probability and keeps only the top k.
    4. Penalty: Subtracts a 'penalty' value from the logits of tokens already 
       present in the sequence to increase output entropy (diversity).
    """
    model.eval()
    sos_id = tokenizer.token_to_id("<SOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    pad_id = tokenizer.token_to_id("<PAD>")
    
    # --- BRANCH-SPECIFIC INITIALIZATION ---
    if "lstm" in model_tag:
        with torch.no_grad():
            # For LSTMs, the Encoder output and states are computed once.
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
        # Beams for LSTM carry the recurrent state (h, c) for each path.
        beams = [([sos_id], 0.0, hidden, cell)]
    else:
        # Transformers are stateless; they rely on the growing 'seq' for context.
        beams = [([sos_id], 0.0)]
        encoder_outputs = None

    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            
            for b in beams:
                # Unpack based on architectural phenotype.
                if "lstm" in model_tag:
                    seq, score, h, c = b
                else:
                    seq, score = b
                
                # If a beam hits the terminal <EOS> token, it stops expanding.
                if seq[-1] == eos_id:
                    all_candidates.append(b)
                    continue
                
                # --- LOGIT PREDICTION ---
                if "lstm" in model_tag:
                    input_token = torch.LongTensor([seq[-1]]).to(device)
                    output, next_h, next_c = model.decoder(input_token, h, c, encoder_outputs)
                else:
                    input_tensor = torch.LongTensor([seq]).to(device)
                    # Forward pass returns full sequence; we only need the last logit projection.
                    output = model(src_tensor, input_tensor)[:, -1, :] 
                
                # Use Log-Softmax for numerical stability (summing logs avoids floating point underflow).
                log_probs = F.log_softmax(output, dim=-1).squeeze(0)
                
                # --- REPETITION PENALTY ---
                # Penalize logits of tokens already used to prevent 'Stuttering' or infinite loops.
                for idx in set(seq):
                    if idx not in [sos_id, eos_id, pad_id]:
                        log_probs[idx] -= penalty
                
                # Top-K Expansion: Isolate the most promising branches for this specific beam.
                top_log_probs, top_indices = log_probs.topk(beam_width)
                
                for i in range(beam_width):
                    next_token = top_indices[i].item()
                    new_score = score + top_log_probs[i].item() # Path score accumulation.
                    
                    if "lstm" in model_tag:
                        all_candidates.append((seq + [next_token], new_score, next_h, next_c))
                    else:
                        all_candidates.append((seq + [next_token], new_score))
            
            # GLOBAL SELECTION: Rank all candidates across all beams and select the top K survivors.
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # CONVERGENCE CHECK: Stop if all survivors have reached the <EOS> token.
            if all(b[0][-1] == eos_id for b in beams):
                break
                
    return beams[0][0] # Return the sequence with the highest overall log-probability.

# --- [PHASE 3] DATA LOADING & SUITE PREPARATION ---

def load_samples(data_dir, split, num_samples=10):
    """ Loads a random slice of the dataset for qualitative auditing. """
    file_path = os.path.join(data_dir, f"{split}.jsonl.gz")
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: split file missing at {file_path}")
        return []
    
    samples = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
            # Stochastic Selection: Prevents bias toward the first files in the shard.
            chosen = random.sample(lines, min(num_samples, len(lines)))
            for c in chosen:
                item = json.loads(c)
                samples.append({
                    'code': item.get('code', ''),
                    'doc': item.get('docstring', '')
                })
    except Exception as e:
        print(f"❌ Error during sample ingestion: {e}")
    return samples

# --- [PHASE 4] CORE AUDIT ENGINE ---

def run_deep_audit():
    """ 
    Executes the 'Deep Audit' protocol. 
    Iterates through all available checkpoints in the model directory and 
    subjects them to memorization, generalization, and logic tests.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = os.path.join(project_root, "tokenizer.json")
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
    data_root = os.path.join(project_root, "Datasets", "processed")

    if not os.path.exists(tokenizer_path):
        print(f"❌ Error: Tokenizer not found. Vocabulary alignment impossible.")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # DEFINE TEST SUITES:
    suites = {
        "TRAIN (Memorization)": load_samples(data_root, "train", 10),
        "TEST (Generalization)": load_samples(data_root, "test", 10),
        "CUSTOM (Logic Check)": [
            {"code": "def find_max(l):\n    return max(l) if l else None", "doc": "Finds the maximum value in a list."},
            {"code": "def save_file(data, path):\n    with open(path, 'w') as f:\n        f.write(data)", "doc": "Writes data to a file at the specified path."}
        ]
    }

    # Discover weight shards.
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    print(f"\n{'='*90}\n🚀 AUDIT START | DEVICE: {device} | MODE: BEAM SEARCH (k=5) | PENALTY: ON\n{'='*90}")

    for ckpt in sorted(checkpoints):
        # ARCHITECTURAL MAPPING: Determine class based on filename string.
        if "transformer" in ckpt.lower(): model_tag = "transformer"
        elif "lstm_dotproduct" in ckpt.lower(): model_tag = "lstm_dotproduct"
        elif "lstm_bahdanau" in ckpt.lower(): model_tag = "lstm_bahdanau"
        else: continue # Skip untagged weights.

        print(f"\n🔍 SCRUTINIZING CHECKPOINT: {ckpt} (Phenotype: {model_tag})")
        
        try:
            # Reconstruct architecture from the project factory.
            model = get_model_architecture(model_tag, device, vocab_size=vocab_size)
            # Load weights using weights_only=True for enhanced security.
            model.load_state_dict(torch.load(os.path.join(checkpoint_dir, ckpt), map_location=device))
            model.eval()

            for suite_name, samples in suites.items():
                if not samples: continue
                print(f"\n   >>> SUITE: {suite_name}")
                for i, s in enumerate(samples):
                    # Convert raw Python text into token IDs.
                    ids_input = tokenizer.encode(s['code']).ids
                    # Structural Wrapping: [1] = <SOS>, [2] = <EOS>.
                    src_tensor = torch.LongTensor([1] + ids_input + [2]).unsqueeze(0).to(device)
                    
                    # RUN BEAM DECODING
                    ids_pred = beam_search_decode(model, src_tensor, tokenizer, model_tag, beam_width=5, device=device, penalty=1.2)
                    prediction = clean_output(tokenizer.decode(ids_pred, skip_special_tokens=True))
                    
                    # Log Comparative Output for qualitative verification.
                    print(f"    S#{i+1} | CODE: {s['code'].strip().replace(chr(10), ' ')[:45]}...")
                    print(f"        REAL: {s['doc'][:60].strip()}")
                    print(f"        PRED: {prediction}")
                    print(f"        {'-'*20}")
        except Exception as e:
            print(f"    ❌ Critical Runtime Error during audit of {ckpt}: {e}")

if __name__ == "__main__":
    run_deep_audit()