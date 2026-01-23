import torch
from tokenizers import Tokenizer
from models.factory import get_model_architecture

def run_quick_test():
    # --- CONFIGURAZIONE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tag = "lstm_attention"
    tokenizer_path = "Project/tokenizer.json"
    checkpoint_path = f"Project/models/checkpoints/best_{model_tag}.pt"
    
    # --- IL TUO CODICE DA TESTARE ---
    # Scrivo io una funzione che filtra numeri pari da una lista
    code_to_summarize = """
def is_adult(age):
    if age >= 18:
        return True
    else:
        return False
    """

    print(f"[*] Caricamento componenti per {model_tag}...")
    
    # 1. Caricamento Tokenizer e Modello
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # Usiamo la factory per avere la stessa struttura usata nel C2
    model = get_model_architecture(model_tag, device, vocab_size=10000)
    
    # 2. Caricamento Pesi (Stato Cristallizzato)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[+] Pesi caricati correttamente da {checkpoint_path}")
    except FileNotFoundError:
        print(f"[!] ERRORE: Non trovo il checkpoint. Hai fatto girare il C2Orchestrator?")
        return

    model.eval()

    # 3. Trasformazione dell'Input (Symbolic Mapping)
    # Aggiungiamo manualmente i tag di inizio (1) e fine (2)
    encoded = tokenizer.encode(code_to_summarize).ids
    src_tensor = torch.LongTensor([1] + encoded + [2]).unsqueeze(0).to(device)

    print(f"[*] Analisi codice in corso...")

    # 4. Loop di Generazione (Greedy Search)
    with torch.no_grad():
        # L'Encoder crea il pensiero astratto (context)
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Iniziamo la frase con <SOS>
        input_token = torch.LongTensor([tokenizer.token_to_id("<SOS>")]).to(device)
        predicted_indices = []

        for _ in range(30): # Massimo 30 parole per il riassunto
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            
            # Prendiamo il token più probabile
            top1 = output.argmax(1)
            token_id = top1.item()
            
            if token_id == tokenizer.token_to_id("<EOS>"):
                break
                
            predicted_indices.append(token_id)
            input_token = top1 # Il predetto diventa il prossimo input

    # 5. Output Finale
    prediction = tokenizer.decode(predicted_indices)
    
    print("\n" + "="*30)
    print("CODICE INPUT:")
    print(code_to_summarize.strip())
    print("-" * 30)
    print("RIASSUNTO PREDETTO:")
    print(f">>> {prediction}")
    print("="*30)

if __name__ == "__main__":
    run_quick_test()