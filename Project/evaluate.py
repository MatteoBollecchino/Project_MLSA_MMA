import sys
import os
import torch
import logging
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tokenizers import Tokenizer

# --- [FASE 1] NORMALIZZAZIONE PERCORSI ---
# Essendo lo script nella root, project_root è la cartella dello script stesso
project_root = os.path.dirname(os.path.abspath(__file__))

if project_root not in sys.path:
    sys.path.append(project_root)

# Import interni (ora che il path è iniettato correttamente)
from models.factory import get_model_architecture
from data.dataset import get_dataloader

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class BatchEvaluator:
    def __init__(self, subset_size=200):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mapping deterministico delle risorse
        self.tokenizer_path = os.path.join(project_root, "tokenizer.json")
        self.checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
        self.data_path = os.path.join(project_root, "Datasets", "python", "final", "jsonl")
        
        # Verifica esistenza risorse critiche
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"❌ Tokenizer non trovato in: {self.tokenizer_path}")
            
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.subset_size = subset_size
        self.results = []

    def get_all_checkpoints(self):
        """Recupera tutti i file .pt nella cartella checkpoints"""
        if not os.path.exists(self.checkpoint_dir):
            return []
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        return sorted(files)

    def evaluate_single_checkpoint(self, checkpoint_name):
        # Inferenza architettura: cerca 'transformer' nel nome, altrimenti LSTM
        model_tag = "transformer" if "transformer" in checkpoint_name.lower() else "lstm_attention"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # 1. Caricamento Modello
        try:
            model = get_model_architecture(model_tag, self.device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.eval()
        except Exception as e:
            return {"file": checkpoint_name, "error": str(e)}
        
        # 2. Setup Dataloader (Test set)
        test_loader = get_dataloader(
            self.data_path, "test", 
            self.tokenizer_path, batch_size=1, subset=self.subset_size
        )

        references = []
        hypotheses = []
        
        with torch.no_grad():
            for src, trg in test_loader:
                src = src.to(self.device)
                
                # Decoding (Greedy)
                encoder_outputs, hidden, cell = model.encoder(src)
                input_token = torch.LongTensor([self.tokenizer.token_to_id("<SOS>")]).to(self.device)
                
                predicted_indices = []
                for _ in range(30):
                    output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
                    top1 = output.argmax(1)
                    token_id = top1.item()
                    if token_id == self.tokenizer.token_to_id("<EOS>"): break
                    predicted_indices.append(token_id)
                    input_token = top1

                # Clean Decoding
                pred_text = self.tokenizer.decode(predicted_indices, skip_special_tokens=True).replace('Ġ', ' ').strip()
                real_text = self.tokenizer.decode(trg.squeeze().tolist(), skip_special_tokens=True).replace('Ġ', ' ').strip()
                
                hypotheses.append(pred_text.split())
                references.append([real_text.split()])

        # --- CALCOLO METRICHE ---
        bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = sum(scorer.score(" ".join(r[0]), " ".join(h))['rougeL'].fmeasure for h, r in zip(hypotheses, references)) / len(hypotheses)
        
        return {"file": checkpoint_name, "bleu": bleu, "rougeL": rougeL}

    def run_all(self):
        checkpoints = self.get_all_checkpoints()
        if not checkpoints:
            print(f"❌ Nessun checkpoint trovato in {self.checkpoint_dir}")
            return

        print(f"🚀 Avvio Batch Evaluation su {len(checkpoints)} modelli...")
        for ckpt in checkpoints:
            print(f"🧐 Analisi di: {ckpt}")
            metrics = self.evaluate_single_checkpoint(ckpt)
            if "error" in metrics:
                print(f"   ⚠️ Errore: {metrics['error']}")
            else:
                self.results.append(metrics)
        
        self.show_summary()

    def show_summary(self):
        if not self.results: return
        df = pd.DataFrame(self.results)
        print("\n" + "="*70)
        print("🏆 CLASSIFICA MODELLI (Sorted by BLEU)")
        print("="*70)
        print(df.sort_values(by="bleu", ascending=False).to_string(index=False))
        print("="*70)

if __name__ == "__main__":
    evaluator = BatchEvaluator(subset_size=200)
    evaluator.run_all()