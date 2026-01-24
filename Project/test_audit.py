import torch
import torch.nn.functional as F
import json
import gzip
import os
import glob
import random
from tokenizers import Tokenizer
from models.factory import get_model_architecture

# --- LOGICA DI GENERAZIONE (BEAM SEARCH CON TEMPERATURA) ---
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
            
            # Applichiamo la temperatura per ammorbidire la distribuzione
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

def load_random_samples(data_dir, split, num_samples=2):
    """Carica campioni casuali dai file compressi del dataset."""
    search_pattern = os.path.join(data_dir, split, "*.jsonl.gz")
    files = glob.glob(search_pattern)
    if not files: return []
    
    samples = []
    with gzip.open(random.choice(files), 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        chosen = random.sample(lines, min(num_samples, len(lines)))
        for c in chosen:
            item = json.loads(c)
            samples.append({
                'code': item.get('code', item.get('func_code_string', '')),
                'doc': item.get('docstring', item.get('func_documentation_string', ''))
            })
    return samples

def run_deep_audit():
    # --- CONFIG ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("Project/tokenizer.json")
    model = get_model_architecture("lstm_attention", device)
    checkpoint = "Project/models/checkpoints/best_lstm_attention.pt"
    
    if not os.path.exists(checkpoint):
        print("[!] Errore: Nessun checkpoint trovato.")
        return
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # --- SORGENTI DATI ---
    data_root = "Project/Datasets/python/final/jsonl"
    
    # 1. Test di Memorizzazione (da Train)
    train_samples = load_random_samples(data_root, "train", 2)
    # 2. Test di Generalizzazione (da Test/Valid)
    test_samples = load_random_samples(data_root, "test", 2)
    # 3. Test di Iniezione (Codice mai visto, scritto da noi)
    custom_samples = [
        {"code": "def power(a, b): return a ** b", "doc": "Calculates power of a number."},
        {"code": "def check_empty(l): return len(l) == 0", "doc": "Checks if list is empty."}
    ]

    suites = {"TRAIN (Memorization)": train_samples, "TEST (Generalization)": test_samples, "CUSTOM (Injection)": custom_samples}

    print(f"\n{'='*80}\nSTARTING MODEL AUDIT: {device}\n{'='*80}")

    for suite_name, samples in suites.items():
        print(f"\n>>> SUITE: {suite_name}")
        for i, s in enumerate(samples):
            # Preprocessing
            encoded = tokenizer.encode(s['code']).ids
            src_tensor = torch.LongTensor([1] + encoded + [2]).unsqueeze(0).to(device)
            
            # Predizione
            ids = beam_search_decode(model, src_tensor, tokenizer)
            prediction = tokenizer.decode(ids, skip_special_tokens=True).replace('Ġ', ' ').strip()
            
            print(f"\n  SAMPLE #{i+1}")
            print(f"  CODE: {s['code'].replace(chr(10), ' ')[:70]}...")
            print(f"  REAL: {s['doc'][:70]}...")
            print(f"  PRED: {prediction}")
            print(f"  {'-'*40}")

if __name__ == "__main__":
    run_deep_audit()