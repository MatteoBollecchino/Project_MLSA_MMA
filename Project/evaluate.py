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
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.factory import get_model_architecture
from data.dataset import get_dataloader

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class BatchEvaluator:
    def __init__(self, subset_size=200):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer_path = os.path.join(project_root, "tokenizer.json")
        self.checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
        self.data_path = os.path.join(project_root, "Datasets", "python", "final", "jsonl")
        
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"❌ Tokenizer non trovato in: {self.tokenizer_path}")
            
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        
        # --- [CRITICO] ESTRAZIONE VOCAB_SIZE DINAMICA ---
        self.vocab_size = self.tokenizer.get_vocab_size()
        logger.info(f"Detezione Vocabolario: {self.vocab_size} token.")
        
        self.subset_size = subset_size
        self.results = []

    def get_all_checkpoints(self):
        if not os.path.exists(self.checkpoint_dir):
            return []
        return sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")])

    def evaluate_single_checkpoint(self, checkpoint_name):
        model_tag = "transformer" if "transformer" in checkpoint_name.lower() else "lstm_attention"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # 1. Caricamento Modello con Vocab_Size CORRETTO
        try:
            model = get_model_architecture(model_tag, self.device, vocab_size=self.vocab_size)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.eval()
        except Exception as e:
            return {"file": checkpoint_name, "error": f"Size Mismatch o Corruzione: {str(e)}"}
        
        # 2. Setup Dataloader
        test_loader = get_dataloader(
            self.data_path, "test", 
            self.tokenizer_path, batch_size=1, subset=self.subset_size
        )

        references = []
        hypotheses = []
        
        sos_id = self.tokenizer.token_to_id("<SOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")

        with torch.no_grad():
            for src, trg in tqdm(test_loader, desc=f"Valutazione {model_tag}", leave=False):
                src = src.to(self.device)
                predicted_indices = []

                if model_tag == "lstm_attention":
                    # --- LOGICA DECODING LSTM ---
                    encoder_outputs, hidden, cell = model.encoder(src)
                    input_token = torch.LongTensor([sos_id]).to(self.device)
                    
                    for _ in range(30):
                        output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
                        top1 = output.argmax(1)
                        if top1.item() == eos_id: break
                        predicted_indices.append(top1.item())
                        input_token = top1
                
                else:
                    # --- LOGICA DECODING TRANSFORMER (Autoregressiva) ---
                    # Il Transformer ha bisogno di generare un token alla volta ricostruendo il target
                    ys = torch.ones(1, 1).fill_(sos_id).type(torch.long).to(self.device)
                    for _ in range(30):
                        out = model(src, ys) # Forward pass
                        next_word = out[:, -1].argmax(1).item()
                        if next_word == eos_id: break
                        predicted_indices.append(next_word)
                        ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(self.device)], dim=1)

                # Clean Decoding
                pred_text = self.tokenizer.decode(predicted_indices, skip_special_tokens=True).strip()
                real_text = self.tokenizer.decode(trg.squeeze().tolist(), skip_special_tokens=True).strip()
                
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
        # Formattazione per rendere i numeri leggibili
        df['bleu'] = df['bleu'].apply(lambda x: f"{x:.4f}")
        df['rougeL'] = df['rougeL'].apply(lambda x: f"{x:.4f}")
        print(df.sort_values(by="bleu", ascending=False).to_string(index=False))
        print("="*70)

if __name__ == "__main__":
    evaluator = BatchEvaluator(subset_size=200)
    evaluator.run_all()