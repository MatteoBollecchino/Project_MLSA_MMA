import torch
from tokenizers import Tokenizer
from models.factory import get_model_architecture

def run_quick_test():
    # --- CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tag = "lstm_attention"
    tokenizer_path = "Project/tokenizer.json"
    checkpoint_path = f"Project/models/checkpoints/best_{model_tag}.pt"
    
    # --- YOUR CODE TO TEST ---
    # I write a function that filters even numbers from a list
    code_to_summarize = """
def is_adult(age):
    if age >= 18:
        return True
    else:
        return False
    """

    print(f"[*] Loading components for {model_tag}...")
    
    # 1. Loading Tokenizer and Model
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # We use the factory to have the same structure used in C2
    model = get_model_architecture(model_tag, device, vocab_size=10000)
    
    # 2. Loading Weights (Frozen State)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[+] Weights loaded successfully from {checkpoint_path}")
    except FileNotFoundError:
        print(f"[!] ERROR: Checkpoint not found. Did you run the C2Orchestrator?")
        return

    model.eval()
    # 3. Input Transformation (Symbolic Mapping)
    # Manually adding start (1) and end (2) tags
    encoded = tokenizer.encode(code_to_summarize).ids
    src_tensor = torch.LongTensor([1] + encoded + [2]).unsqueeze(0).to(device)

    print(f"[*] Code analysis in progress...")

    # 4. Generation Loop (Greedy Search)
    with torch.no_grad():
        # The Encoder creates the abstract thought (context)
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # We start the sentence with <SOS>
        input_token = torch.LongTensor([tokenizer.token_to_id("<SOS>")]).to(device)
        predicted_indices = []

        for _ in range(30): # Maximum 30 words for the summary
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            
            # We take the most probable token
            top1 = output.argmax(1)
            token_id = top1.item()
            
            if token_id == tokenizer.token_to_id("<EOS>"):
                break
                
            predicted_indices.append(token_id)
            input_token = top1 # The predicted token becomes the next input

    # 5. Final Output
    prediction = tokenizer.decode(predicted_indices)
    
    print("\n" + "="*30)
    print("INPUT CODE:")
    print(code_to_summarize.strip())
    print("-" * 30)
    print("PREDICTED SUMMARY:")
    print(f">>> {prediction}")
    print("="*30)

if __name__ == "__main__":
    run_quick_test()