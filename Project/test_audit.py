import torch
import torch.nn.functional as F
import json
import gzip
import os
import glob
import random
import sys
from tokenizers import Tokenizer

# --- [FASE 1] NORMALIZZAZIONE DETERMINISTICA ---
# Essendo lo script nella Root, project_root è la cartella dello script stesso
project_root = os.path.dirname(os.path.abspath(__file__))

if project_root not in sys.path:
    sys.path.append(project_root)

# Ora l'import dei tuoi modelli funzionerà senza errori di modulo
from models.factory import get_model_architecture

# --- LOGICA DI DECODIFICA AVANZATA ---
def beam_search_decode(model, src_tensor, tokenizer, beam_width=5, max_len=30, temp=1.2):
    device = src_tensor.device
    sos_id = tokenizer.token_to_id("<SOS>")
    eos_id = tokenizer.token_to_id("<EOS>")

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    beams = [(0.0, [sos_id], hidden, cell)]

    for _ in range(max_len):
        candidates = []
        for log_prob, tokens, h, c in beams:
            if tokens[-1] == eos_id:
                candidates.append((log_prob, tokens, h, c))
                continue

            input_token = torch.LongTensor([tokens[-1]]).to(device)
            with torch.no_grad():
                output, h_next, c_next = model.decoder(input_token, h, c, encoder_outputs)
            
            # Distribuzione di probabilità con Temperatura: P = softmax(logits / T)
            probs = F.log_softmax(output / temp, dim=1)
            topk_probs, topk_ids = probs.topk(beam_width)

            for i in range(beam_width):
                next_log_prob = log_prob + topk_probs[0][i].item()
                next_tokens = tokens + [topk_ids[0][i].item()]
                candidates.append((next_log_prob, next_tokens, h_next, c_next))

        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        if all(b[1][-1] == eos_id for b in beams):
            break

    return beams[0][1]

# --- DATA LOADING ---
def load_random_samples(data_dir, split, num_samples=2):
    search_pattern = os.path.join(data_dir, split, "*.jsonl.gz")
    files = glob.glob(search_pattern)
    if not files: return []
    
    samples = []
    try:
        with gzip.open(random.choice(files), 'rt', encoding='utf-8') as f:
            lines = f.readlines()
            chosen = random.sample(lines, min(num_samples, len(lines)))
            for c in chosen:
                item = json.loads(c)
                samples.append({
                    'code': item.get('code', item.get('func_code_string', '')),
                    'doc': item.get('docstring', item.get('func_documentation_string', ''))
                })
    except: pass
    return samples

# --- CORE AUDIT ENGINE ---
def run_deep_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mapping risorse basato sulla nuova Root
    tokenizer_path = os.path.join(project_root, "tokenizer.json")
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
    data_root = os.path.join(project_root, "Datasets", "python", "final", "jsonl")

    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer non trovato in: {tokenizer_path}")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 1. Recupero Checkpoints
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Directory checkpoints non trovata: {checkpoint_dir}")
        return
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        print("❌ Nessun file .pt trovato.")
        return

    # 2. Preparazione Suite di Test
    suites = {
        "TRAIN (Memorization)": load_random_samples(data_root, "train", 2),
        "TEST (Generalization)": load_random_samples(data_root, "test", 2),
        "CUSTOM (Injection)": [
            {"code": "def power(a, b): return a ** b", "doc": "Calculates power of a number."},
            {"code": "def check_empty(l): return len(l) == 0", "doc": "Checks if list is empty."}
        ]
    }

    print(f"\n{'='*80}\n🚀 STARTING BATCH AUDIT ON {len(checkpoints)} MODELS\n{'='*80}")

    for ckpt_name in sorted(checkpoints):
        print(f"\n🔍 ANALYZING CHECKPOINT: {ckpt_name}")
        
        # Determina architettura (Transformer se specificato nel nome, altrimenti LSTM)
        model_tag = "transformer" if "transformer" in ckpt_name.lower() else "lstm_attention"
        
        try:
            model = get_model_architecture(model_tag, device)
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            for suite_name, samples in suites.items():
                print(f"\n  >>> SUITE: {suite_name}")
                for i, s in enumerate(samples):
                    encoded = tokenizer.encode(s['code']).ids
                    # SOS=1, EOS=2 (verifica se i tuoi ID corrispondono)
                    src_tensor = torch.LongTensor([1] + encoded + [2]).unsqueeze(0).to(device)
                    
                    ids = beam_search_decode(model, src_tensor, tokenizer)
                    prediction = tokenizer.decode(ids, skip_special_tokens=True).replace('Ġ', ' ').strip()
                    
                    print(f"    SAMPLE #{i+1} | CODE: {s['code'].strip()[:50]}...")
                    print(f"    REAL: {s['doc'][:60]}...")
                    print(f"    PRED: {prediction}")
                    print(f"    {'-'*20}")
        except Exception as e:
            print(f"    ❌ Errore durante l'audit di {ckpt_name}: {e}")

if __name__ == "__main__":
    run_deep_audit()