import torch
import torch.nn.functional as F
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

# --- BEAM SEARCH DECODING LOGIC ---

def beam_search_decode(model, src_tensor, tokenizer, model_tag, beam_width=5, max_len=40, device="cpu", penalty=2.0):
    """
    Ricerca a fascio (Beam Search) per esplorare percorsi multipli.
    Ottimizzato per gestire sia LSTM che Transformer.
    """
    model.eval()
    sos_id = tokenizer.token_to_id("<SOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    pad_id = tokenizer.token_to_id("<PAD>")
    
    # Inizializzazione fasci: (sequenza, score, hidden, cell)
    # Lo score è in log-space (inizialmente 0.0)
    if "lstm" in model_tag:
        with torch.no_grad():
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
        beams = [([sos_id], 0.0, hidden, cell)]
    else:
        # Per il Transformer passiamo solo seq e score
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
                
                # Se il fascio è già terminato, lo manteniamo come candidato
                if seq[-1] == eos_id:
                    all_candidates.append(b)
                    continue
                
                # Predizione prossimo token
                if "lstm" in model_tag:
                    input_token = torch.LongTensor([seq[-1]]).to(device)
                    output, next_h, next_c = model.decoder(input_token, h, c, encoder_outputs)
                else:
                    input_tensor = torch.LongTensor([seq]).to(device)
                    # La forward del Transformer restituisce l'intera sequenza
                    output = model(src_tensor, input_tensor)[:, -1, :] 
                
                # Log-Softmax per stabilità numerica (sommiamo i log invece di moltiplicare)
                log_probs = F.log_softmax(output, dim=-1).squeeze(0)
                
                # Applichiamo la Repetition Penalty sottraendo dai log-probs
                for idx in set(seq):
                    if idx not in [sos_id, eos_id, pad_id]:
                        log_probs[idx] -= penalty
                
                # Prendiamo i migliori candidati per questo fascio specifico
                top_log_probs, top_indices = log_probs.topk(beam_width)
                
                for i in range(beam_width):
                    next_token = top_indices[i].item()
                    new_score = score + top_log_probs[i].item()
                    
                    if "lstm" in model_tag:
                        all_candidates.append((seq + [next_token], new_score, next_h, next_c))
                    else:
                        all_candidates.append((seq + [next_token], new_score))
            
            # Selezione globale: ordiniamo tutti i candidati e teniamo i top K
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Se tutti i fasci hanno toccato EOS, usciamo
            if all(b[0][-1] == eos_id for b in beams):
                break
                
    return beams[0][0] # Restituiamo la sequenza del miglior fascio (score più alto)

# --- DATA LOADING ---

def load_samples(data_dir, split, num_samples=10):
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
    data_root = os.path.join(project_root, "Datasets", "processed")

    if not os.path.exists(tokenizer_path):
        print(f"❌ Errore: Tokenizer non trovato")
        return

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
    print(f"\n{'='*90}\n🚀 AUDIT START | DEVICE: {device} | SOURCE: PROCESSED | MODE: BEAM SEARCH (k=5)\n{'='*90}")

    for ckpt in sorted(checkpoints):
        if "transformer" in ckpt.lower(): model_tag = "transformer"
        elif "lstm_dotproduct" in ckpt.lower(): model_tag = "lstm_dotproduct"
        elif "lstm_bahdanau" in ckpt.lower(): model_tag = "lstm_bahdanau"
        else: continue

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
                    
                    # Chiamata con Beam Search
                    ids_pred = beam_search_decode(model, src_tensor, tokenizer, model_tag, beam_width=5, device=device, penalty=1.2)
                    prediction = clean_output(tokenizer.decode(ids_pred, skip_special_tokens=True))
                    
                    print(f"    S#{i+1} | CODE: {s['code'].strip().replace(chr(10), ' ')[:45]}...")
                    print(f"        REAL: {s['doc'][:60].strip()}")
                    print(f"        PRED: {prediction}")
                    print(f"        {'-'*20}")
        except Exception as e:
            print(f"    ❌ Error analyzing {ckpt}: {e}")

if __name__ == "__main__":
    run_deep_audit()