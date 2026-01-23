import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.factory import get_model_architecture

# --- LOGICA DI DECODIFICA AVANZATA ---
def beam_search_decode(model, src_tensor, tokenizer, beam_width=5, max_len=30):
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
            
            log_probs = F.log_softmax(output, dim=1)
            topk_probs, topk_ids = log_probs.topk(beam_width)

            for i in range(beam_width):
                next_log_prob = log_prob + topk_probs[0][i].item()
                next_tokens = tokens + [topk_ids[0][i].item()]
                candidates.append((next_log_prob, next_tokens, h_next, c_next))

        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        if all(b[1][-1] == eos_id for b in beams):
            break

    return beams[0][1]

# --- SUITE DI TEST ---
def run_injection_suite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tag = "lstm_attention"
    tokenizer_path = "Project/tokenizer.json"
    checkpoint_path = f"Project/models/checkpoints/best_{model_tag}.pt"

    print(f"[*] Bootstrapping Audit su {device}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = get_model_architecture(model_tag, device, vocab_size=10000)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print("[+] Checkpoint caricato. Analisi della varianza semantica avviata.")
    except Exception as e:
        print(f"[!] Errore critico: {e}")
        return

    # Iniezione di campioni con diversi "trigger" logici
    test_cases = {
        "Simple Arithmetic": "def add(a, b): return a + b",
        "Boolean Logic": "def is_valid(user): return user.active and user.has_permission",
        "List Comprehension": "def get_squares(n): return [i**2 for i in range(n)]",
        "Edge Case (Empty)": "def placeholder(): pass",
        "External Library": "import requests\ndef fetch(url): return requests.get(url).json()"
    }

    print("\n" + "="*60)
    for name, code in test_cases.items():
        # Preprocessing
        encoded = tokenizer.encode(code).ids
        src_tensor = torch.LongTensor([1] + encoded + [2]).unsqueeze(0).to(device)

        # Inferenza
        predicted_indices = beam_search_decode(model, src_tensor, tokenizer, beam_width=5)
        
        # Cleanup
        prediction = tokenizer.decode(predicted_indices, skip_special_tokens=True)
        prediction = prediction.replace('Ġ', ' ').strip()

        print(f"TEST: {name}")
        print(f"CODE: {code}")
        print(f"PRED: >>> {prediction}")
        print("-" * 60)

if __name__ == "__main__":
    run_injection_suite()