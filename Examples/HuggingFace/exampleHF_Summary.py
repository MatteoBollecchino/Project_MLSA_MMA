import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configurazioni
MODEL_NAME = "Salesforce/codet5-base-multi-sum"  # Modello specifico per code summarization
INPUT_FILE = "Examples/HuggingFace/input_code.txt"

def generate_summary(file_path):
    """
    Legge un file di codice e genera un riassunto in linguaggio naturale.
    """
    if not os.path.exists(file_path):
        print(f"Errore: Il file '{file_path}' non esiste.")
        return

    # 1. Carica il Tokenizer e il Modello pre-addestrato
    print(f"Caricamento modello {MODEL_NAME} in corso...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 2. Legge il contenuto del file
    with open(file_path, "r", encoding="utf-8") as f:
        code_snippet = f.read()

    print("\n--- Codice Input ---")
    print(code_snippet.strip())
    print("--------------------\n")

    # 3. Prepara l'input per il modello (Tokenization)
    input_ids = tokenizer.encode(code_snippet, return_tensors="pt", max_length=512, truncation=True)

    # 4. Genera il riassunto (Inference)
    # num_beams=10 usa beam search per una qualità migliore
    summary_ids = model.generate(input_ids, max_length=150, num_beams=10, early_stopping=True)
    
    # 5. Decodifica l'output
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("--- Code Summary Generato ---")
    print(summary)
    print("-----------------------------")

if __name__ == "__main__":
    generate_summary(INPUT_FILE)