import sys
import os

# --- FIX PERCORSI (PRINCIPLE OF RELATIVE CONTEXT) ---
# Aggiunge la cartella genitore (Project) al sys.path affinché Python trovi 'models' e 'data'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import logging
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tokenizers import Tokenizer

# Ora gli import funzioneranno correttamente
from models.factory import get_model_architecture
from data.dataset import get_dataloader

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_tag="lstm_attention", subset_size=200):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_path = os.path.join(project_root, "tokenizer.json")
        self.checkpoint_path = os.path.join(project_root, "models", "checkpoints", f"best_{model_tag}.pt")
        self.data_path = os.path.join(project_root, "Datasets", "python", "final", "jsonl")
        
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.subset_size = subset_size
        
        # Caricamento Modello
        self.model = get_model_architecture(model_tag, self.device)
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print(f"[+] Pesi caricati da: {self.checkpoint_path}")
        else:
            print(f"[!] Checkpoint non trovato in {self.checkpoint_path}")
            
        self.model.eval()
        
        self.test_loader = get_dataloader(
            self.data_path, "test", 
            self.tokenizer_path, batch_size=1, subset=subset_size
        )

    def run_evaluation(self):
        references = []
        hypotheses = []
        
        print(f"[*] Avvio Valutazione Quantitativa su {self.subset_size} campioni...")
        
        with torch.no_grad():
            for src, trg in tqdm(self.test_loader):
                src = src.to(self.device)
                
                # Generazione Greedy
                encoder_outputs, hidden, cell = self.model.encoder(src)
                input_token = torch.LongTensor([self.tokenizer.token_to_id("<SOS>")]).to(self.device)
                
                predicted_indices = []
                for _ in range(30):
                    output, hidden, cell = self.model.decoder(input_token, hidden, cell, encoder_outputs)
                    top1 = output.argmax(1)
                    token_id = top1.item()
                    if token_id == self.tokenizer.token_to_id("<EOS>"): break
                    predicted_indices.append(token_id)
                    input_token = top1

                # Decodifica e pulizia (Byte-Level BPE fix)
                pred_text = self.tokenizer.decode(predicted_indices, skip_special_tokens=True).replace('Ġ', ' ').strip()
                real_text = self.tokenizer.decode(trg.squeeze().tolist(), skip_special_tokens=True).replace('Ġ', ' ').strip()
                
                hypotheses.append(pred_text.split())
                references.append([real_text.split()])

        # --- CALCOLO METRICHE ---
        bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge1 = sum(scorer.score(" ".join(r[0]), " ".join(h))['rouge1'].fmeasure for h, r in zip(hypotheses, references)) / len(hypotheses)
        
        self.print_report(bleu, rouge1, hypotheses)

    def print_report(self, bleu, rouge, hypotheses):
        # ... (stessa logica del report precedente)
        print("\n" + "="*50)
        print(f"BLEU: {bleu:.4f} | ROUGE: {rouge:.4f}")
        print("="*50)

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()