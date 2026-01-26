import torch
import torch.nn.functional as F
import json
import gzip
import os
import glob
import random
import sys
import math
from tokenizers import Tokenizer

# --- [FASE 1] NORMALIZZAZIONE DETERMINISTICA ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.factory import get_model_architecture

def clean_prediction(text):
    """Rimuove artefatti BPE e normalizza la spaziatura."""
    return text.replace('Ġ', ' ').replace('  ', ' ').strip()

# --- LOGICA DI DECODIFICA UNIVERSALE ---
def autoregressive_decode(model, src_tensor, tokenizer, model_tag, max_len=30, device="cpu"):
    model.eval()
    sos_id = tokenizer.token_to_id("<SOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    
    with torch.no_grad():
        if model_tag == "lstm_attention":
            # Decoding per LSTM
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
            input_token = torch.LongTensor([sos_id]).to(device)
            predicted_indices = []
            for _ in range(max_len):
                output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
                top1 = output.argmax(1)
                if top1.item() == eos_id: break
                predicted_indices.append(top1.item())
                input_token = top1
            return predicted_indices
        
        else:
            # Decoding per Transformer
            # Il Transformer richiede la generazione sequenziale del target
            ys = torch.ones(1, 1).fill_(sos_id).type(torch.long).to(device)
            predicted_indices = []
            for _ in range(max_len):
                out = model(src_tensor, ys)
                # Prendiamo l'ultimo logit della sequenza prodotta
                next_word = out[:, -1].argmax(1).item()
                if next_word == eos_id: break
                predicted_indices.append(next_word)
                ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)
            return predicted_indices

# --- DATA LOADING ---
def load_random_samples(data_dir, split, num_samples=10):
    search_pattern = os.path.join(data_dir, split, "*.jsonl.gz")
    files = glob.glob(search_pattern)
    if not files: return []
    
    samples = []
    # Mescoliamo i file per non prendere sempre lo stesso chunk
    random.shuffle(files)
    for file_path in files:
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
                chosen = random.sample(lines, min(num_samples - len(samples), len(lines)))
                for c in chosen:
                    item = json.loads(c)
                    samples.append({
                        'code': item.get('code', item.get('func_code_string', '')),
                        'doc': item.get('docstring', item.get('func_documentation_string', ''))
                    })
            if len(samples) >= num_samples: break
        except: continue
    return samples[:num_samples]

# --- CORE AUDIT ENGINE ---
def run_deep_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = os.path.join(project_root, "tokenizer.json")
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
    data_root = os.path.join(project_root, "Datasets", "python", "final", "jsonl")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # Suite 10-10-10
    suites = {
        "TRAIN (Memorization Check)": load_random_samples(data_root, "train", 10),
        "TEST (Generalization Check)": load_random_samples(data_root, "test", 10),
        "CUSTOM (Zero-Shot Logic)": [
            {"code": "def add(a, b): return a + b", "doc": "Adds two numbers."},
            {"code": "def sub(a, b): return a - b", "doc": "Subtracts b from a."},
            {"code": "def mul(a, b): return a * b", "doc": "Multiplies two factors."},
            {"code": "def power(a, b): return a ** b", "doc": "Calculates exponentiation."},
            {"code": "def is_even(n): return n % 2 == 0", "doc": "Checks if a number is even."},
            {"code": "def get_len(x): return len(x)", "doc": "Returns the length of an object."},
            {"code": "def reverse_list(l): return l[::-1]", "doc": "Reverses a list."},
            {"code": "def to_upper(s): return s.upper()", "doc": "Converts string to uppercase."},
            {"code": "def check_none(obj): return obj is None", "doc": "Verifies if object is None."},
            {"code": "def find_max(l): return max(l)", "doc": "Finds the maximum value."}
        ]
    }

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    
    print(f"\n{'='*90}\n🚀 BATCH AUDIT ENGINE | VOCAB: {vocab_size} | DEVICE: {device}\n{'='*90}")

    for ckpt_name in sorted(checkpoints):
        print(f"\n🔍 TARGET: {ckpt_name}")
        model_tag = "transformer" if "transformer" in ckpt_name.lower() else "lstm_attention"
        
        try:
            model = get_model_architecture(model_tag, device, vocab_size=vocab_size)
            model.load_state_dict(torch.load(os.path.join(checkpoint_dir, ckpt_name), map_location=device))
            model.eval()

            for suite_name, samples in suites.items():
                print(f"\n--- SUITE: {suite_name} ---")
                for i, s in enumerate(samples):
                    encoded = tokenizer.encode(s['code']).ids
                    src_tensor = torch.LongTensor([1] + encoded + [2]).unsqueeze(0).to(device)
                    
                    ids = autoregressive_decode(model, src_tensor, tokenizer, model_tag, device=device)
                    prediction = clean_prediction(tokenizer.decode(ids, skip_special_tokens=True))
                    
                    # Formattazione scannable
                    code_snippet = s['code'].replace('\n', ' ')[:40]
                    print(f"[{i+1}/10] CODE: {code_snippet}...")
                    print(f"      REAL: {s['doc'][:60]}")
                    print(f"      PRED: {prediction}")
                    print(f"      {'-'*15}")
        except Exception as e:
            print(f"⚠️ Errore su {ckpt_name}: {e}")

if __name__ == "__main__":
    run_deep_audit()