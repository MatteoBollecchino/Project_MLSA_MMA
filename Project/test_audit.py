"""
================================================================================
DEEP AUDIT & QUALITATIVE INFERENCE UNIT
================================================================================
ROLE: High-Fidelity Model Assessment via Beam Search and Suite Testing.

DESIGN RATIONALE:
- Formal Verification: Generates a persistent audit log for post-mortem analysis.
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

    # Clean up BPE artifacts: Replace 'Ġ' with space and collapse multiple spaces.
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
        # For LSTM-based models, we need to manage hidden states across beam expansions.
        with torch.no_grad():
            # Encode the source sequence once to reuse across beam expansions.
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Initialize beams with the start token and initial hidden states.
        beams = [([sos_id], 0.0, hidden, cell)]
    else:
        # For Transformer, we can encode once and reuse encoder outputs.
        beams = [([sos_id], 0.0)]
        encoder_outputs = None

    # Iteratively expand beams until max length or all beams end with EOS.
    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            for b in beams:
                if "lstm" in model_tag:
                    # Unpack sequence, score, and hidden states for LSTM-based models.
                    seq, score, h, c = b
                else:
                    # Unpack sequence and score for Transformer-based models.
                    seq, score = b
                
                # If the last token is EOS, we don't expand this beam further.
                if seq[-1] == eos_id:
                    all_candidates.append(b)
                    continue
                
                if "lstm" in model_tag:
                    # For LSTM, we need to pass the last token and hidden states to get the next output.
                    input_token = torch.LongTensor([seq[-1]]).to(device)
                    # output: [1, Vocab] for the next token probabilities.
                    output, next_h, next_c = model.decoder(input_token, h, c, encoder_outputs)
                else:
                    # For Transformer, we pass the entire sequence to get the next token probabilities.
                    input_tensor = torch.LongTensor([seq]).to(device)
                    # output: [1, Seq, Vocab], we take the last token's output for prediction.
                    output = model(src_tensor, input_tensor)[:, -1, :] 
                
                # Apply log softmax to get log probabilities for the next token.
                # log_probs -> we squeeze to remove batch dimension.
                log_probs = F.log_softmax(output, dim=-1).squeeze(0)
                
                # Apply repetition penalty to existing tokens in sequence.
                for idx in set(seq):
                    # Only apply penalty to non-special tokens to encourage diversity.
                    if idx not in [sos_id, eos_id, pad_id]:
                        # Penalize tokens that have already been generated in this beam to reduce repetition.
                        log_probs[idx] -= penalty
                
                # Get top K candidates from the log probabilities for the next token.
                top_log_probs, top_indices = log_probs.topk(beam_width)

                # Expand each candidate and add to the list of all candidates.
                for i in range(beam_width):
                    # Calculate the new score by adding the log probability of the next token to the existing score.
                    next_token = top_indices[i].item()
                    new_score = score + top_log_probs[i].item()
                    if "lstm" in model_tag:
                        # For LSTM, we need to carry forward the hidden states for each candidate beam.
                        all_candidates.append((seq + [next_token], new_score, next_h, next_c))
                    else:
                        # For Transformer, we only need to carry forward the sequence and score.
                        all_candidates.append((seq + [next_token], new_score))
            
            # Global pruning: keep top K candidates.
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # If all beams have ended with EOS, we can stop early.
            if all(b[0][-1] == eos_id for b in beams):
                break

    # Return the sequence from the best beam (highest score) after removing special tokens.           
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

    # Gather all checkpoint files for analysis.
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
                # Load the checkpoint state dict.
                model.load_state_dict(torch.load(os.path.join(checkpoint_dir, ckpt), map_location=device))
                model.eval()

                for suite_name, samples in suites.items():
                    if not samples: continue
                    log.write(f"\n--- SUITE: {suite_name} ---\n")
                    
                    for i, s in enumerate(samples):
                        # Tokenization and hardware mapping.

                        # ids_input contains the token IDs for the input code snippet, which is then wrapped with <SOS> and <EOS> tokens and converted to a tensor for model input.
                        ids_input = tokenizer.encode(s['code']).ids
                        
                        # src_tensor shape: [1, Seq] where Seq is the length of the tokenized input sequence including special tokens.
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

# --- STANDALONE EXECUTION BLOCK ---
if __name__ == "__main__":
    run_deep_audit()