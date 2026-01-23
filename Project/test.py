import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.factory import get_model_architecture

def beam_search_decode(model, src_tensor, tokenizer, beam_width=5, max_len=30):
    device = src_tensor.device
    sos_id = tokenizer.token_to_id("<SOS>")
    eos_id = tokenizer.token_to_id("<EOS>")

    # 1. Encoding: L'encoder analizza il codice una sola volta
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    # 2. Inizializzazione del "Beam"
    # Ogni elemento: (punteggio_logaritmico, sequenza_token, hidden, cell)
    beams = [(0.0, [sos_id], hidden, cell)]

    for _ in range(max_len):
        candidates = []
        for log_prob, tokens, h, c in beams:
            # Se l'ipotesi è già terminata, la manteniamo così com'è
            if tokens[-1] == eos_id:
                candidates.append((log_prob, tokens, h, c))
                continue

            # Previsione del prossimo token
            input_token = torch.LongTensor([tokens[-1]]).to(device)
            with torch.no_grad():
                output, h_next, c_next = model.decoder(input_token, h, c, encoder_outputs)
            
            # Calcoliamo le log-probabilità per evitare l'underflow numerico
            log_probs = F.log_softmax(output, dim=1)
            
            # Prendiamo i top-k candidati per questo specifico ramo
            topk_probs, topk_ids = log_probs.topk(beam_width)

            for i in range(beam_width):
                next_log_prob = log_prob + topk_probs[0][i].item()
                next_tokens = tokens + [topk_ids[0][i].item()]
                candidates.append((next_log_prob, next_tokens, h_next, c_next))

        # 3. Selezione: Teniamo solo i 'beam_width' migliori tra tutti i rami espansi
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        # Se tutti i beam hanno raggiunto <EOS>, ci fermiamo
        if all(b[1][-1] == eos_id for b in beams):
            break

    # Restituiamo la sequenza con il punteggio più alto
    best_tokens = beams[0][1]
    return best_tokens

def run_quick_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tag = "lstm_attention"
    tokenizer_path = "Project/tokenizer.json"
    checkpoint_path = f"Project/models/checkpoints/best_{model_tag}.pt"
    
    code_to_summarize = "def add(a, b): return a + b"

    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = get_model_architecture(model_tag, device, vocab_size=10000)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    except:
        print("[!] Checkpoint non trovato.")
        return

    # Preprocessing
    encoded = tokenizer.encode(code_to_summarize).ids
    src_tensor = torch.LongTensor([1] + encoded + [2]).unsqueeze(0).to(device)

    # Generazione con Beam Search
    predicted_indices = beam_search_decode(model, src_tensor, tokenizer, beam_width=5)

    # Sanitizzazione (Soluzione per la 'Ġ' e rimozione token speciali)
    prediction = tokenizer.decode(predicted_indices, skip_special_tokens=True)
    prediction = prediction.replace('Ġ', ' ').strip()
    
    print(f"\nRESULT: {prediction}")

if __name__ == "__main__":
    run_quick_test()