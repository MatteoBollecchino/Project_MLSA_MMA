import torch
import json
import gzip
import os
import glob
import random
import sys
from tokenizers import Tokenizer

# Risoluzione percorsi per import interni
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.factory import get_model_architecture

# --- [PHASE 1] DETERMINISTIC NORMALIZATION ---

def clean_output(text):
    """Rimuove artefatti BPE e normalizza la spaziatura."""
    return text.replace('Ġ', ' ').replace('  ', ' ').strip()

# --- LOGICA DI DECODIFICA UNIVERSALE ---

def autoregressive_decode(model, src_tensor, tokenizer, model_tag, max_len=40, device="cpu"):
    model.eval()
    sos_id = tokenizer.token_to_id("<SOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    
    with torch.no_grad():
        # LOGICA PER VARIANTI LSTM (Bahdanau & DotProduct condividono la struttura encoder/decoder)
        if "lstm" in model_tag:
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
        
        # LOGICA PER TRANSFORMER
        else:
            ys = torch.ones(1, 1).fill_(sos_id).type(torch.long).to(device)
            predicted_indices = []
            for _ in range(max_len):
                out = model(src_tensor, ys)
                next_word = out[:, -1].argmax(1).item()
                if next_word == eos_id: break
                predicted_indices.append(next_word)
                ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)
            return predicted_indices

# --- DATA LOADING (Updated to Processed Directory) ---

def load_samples(data_dir, split, num_samples=10):
    """Carica i campioni dai file già puliti (processed)."""
    file_path = os.path.join(data_dir, f"{split}.jsonl.gz")
    if not os.path.exists(file_path):
        print(f"⚠️ File non trovato: {file_path}")
        return []
    
    samples = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
            chosen = random.sample(lines, min(num_samples, len(lines)))
            for c in chosen:
                item = json.loads(c)
                samples.append({
                    'code': item.get('code', ''),
                    'doc': item.get('docstring', '')
                })
    except Exception as e:
        print(f"❌ Errore caricamento campioni: {e}")
    return samples

# --- CORE AUDIT ENGINE ---

def run_deep_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = os.path.join(project_root, "tokenizer.json")
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
    
    # PUNTA ALLA CARTELLA PROCESSED (Decisione Architetturale corretta)
    data_root = os.path.join(project_root, "Datasets", "processed")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    suites = {
        "TRAIN (Memorization)": load_samples(data_root, "train", 10),
        "TEST (Generalization)": load_samples(data_root, "test", 10),
        "CUSTOM (Complex logic)": [
            {"code": "def find_max(l):\n    return max(l) if l else None", "doc": "Finds the maximum value in a list."},
            {"code": "def save_file(data, path):\n    with open(path, 'w') as f:\n        f.write(data)", "doc": "Writes data to a file at the specified path."}
        ]
    }

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    print(f"\n{'='*90}\n🚀 AUDIT START | DEVICE: {device} | DATA SOURCE: PROCESSED\n{'='*90}")

    for ckpt in sorted(checkpoints):
        # MAPPING DINAMICO BASATO SUI NUOVI TAG
        if "transformer" in ckpt.lower(): model_tag = "transformer"
        elif "lstm_dotproduct" in ckpt.lower(): model_tag = "lstm_dotproduct"
        elif "lstm_bahdanau" in ckpt.lower(): model_tag = "lstm_bahdanau"
        else: continue # Salta file non riconosciuti

        print(f"\n🔍 CHECKPOINT: {ckpt} (Tag: {model_tag})")
        
        try:
            model = get_model_architecture(model_tag, device, vocab_size=vocab_size)
            model.load_state_dict(torch.load(os.path.join(checkpoint_dir, ckpt), map_location=device))
            model.eval()

            for suite_name, samples in suites.items():
                if not samples: continue
                print(f"\n   >>> SUITE: {suite_name}")
                for i, s in enumerate(samples):
                    ids_input = tokenizer.encode(s['code']).ids
                    src_tensor = torch.LongTensor([1] + ids_input + [2]).unsqueeze(0).to(device)
                    
                    ids_pred = autoregressive_decode(model, src_tensor, tokenizer, model_tag, device=device)
                    prediction = clean_output(tokenizer.decode(ids_pred, skip_special_tokens=True))
                    
                    print(f"    S#{i+1} | CODE: {s['code'].strip().replace('\\n', ' ')[:45]}...")
                    print(f"        REAL: {s['doc'][:60].strip()}")
                    print(f"        PRED: {prediction}")
                    print(f"        {'-'*20}")
        except Exception as e:
            print(f"    ❌ Error analyzing {ckpt}: {e}")

if __name__ == "__main__":
    run_deep_audit()