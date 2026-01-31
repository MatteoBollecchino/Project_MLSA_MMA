import sys
import os
import torch
import logging
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tokenizers import Tokenizer
from models.factory import get_model_architecture
from data.dataset import get_dataloader

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

logger = logging.getLogger(__name__)

class BatchEvaluator:
    def __init__(self, device, tokenizer_path, checkpoint_dir, data_path, subset_size=200):
        self.device = device
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.checkpoint_dir = checkpoint_dir
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.subset_size = subset_size
        self.results = []

    def evaluate_single_checkpoint(self, checkpoint_name, penalty=2.0):
        # --- [MAPPING LOGIC] ---
        if "transformer" in checkpoint_name.lower():
            model_tag = "transformer"
        elif "lstm_dotproduct" in checkpoint_name.lower():
            model_tag = "lstm_dotproduct"
        else:
            model_tag = "lstm_bahdanau"

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        try:
            model = get_model_architecture(model_tag, self.device, vocab_size=self.vocab_size)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.eval()
        except Exception as e:
            return {"file": checkpoint_name, "error": f"Load error: {str(e)}"}
        
        test_loader = get_dataloader(
            self.data_path, "test", 
            self.tokenizer_path, batch_size=1, subset=self.subset_size
        )

        references, hypotheses = [], []
        sos_id = self.tokenizer.token_to_id("<SOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        pad_id = self.tokenizer.token_to_id("<PAD>")

        with torch.no_grad():
            for src, trg in tqdm(test_loader, desc=f"Audit {model_tag}", leave=False):
                src = src.to(self.device)
                predicted_indices = []

                # --- [INFERENCE LOGIC: LSTM VARIANTS] ---
                if "lstm" in model_tag:
                    encoder_outputs, hidden, cell = model.encoder(src)
                    input_token = torch.LongTensor([sos_id]).to(self.device)

                    for _ in range(30):
                        output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
                        
                        # --- PENALTY INJECTION ---
                        for idx in set(predicted_indices):
                            if idx not in [sos_id, eos_id, pad_id]:
                                output[0, idx] -= penalty 
                        
                        top1 = output.argmax(1)
                        if top1.item() == eos_id: break
                        predicted_indices.append(top1.item())
                        input_token = top1
                
                # --- [INFERENCE LOGIC: TRANSFORMER] ---
                else:
                    ys = torch.ones(1, 1).fill_(sos_id).type(torch.long).to(self.device)
                    for _ in range(30):
                        out = model(src, ys)
                        logits = out[:, -1, :] 
                        
                        # --- PENALTY INJECTION ---
                        for idx in set(predicted_indices):
                            if idx not in [sos_id, eos_id, pad_id]:
                                logits[0, idx] -= penalty

                        next_word = logits.argmax(1).item()
                        if next_word == eos_id: break
                        predicted_indices.append(next_word)
                        ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(self.device)], dim=1)

                pred_text = self.tokenizer.decode(predicted_indices, skip_special_tokens=True).strip()
                real_text = self.tokenizer.decode(trg.squeeze().tolist(), skip_special_tokens=True).strip()
                
                hypotheses.append(pred_text.split())
                references.append([real_text.split()])

        # Calcolo Metriche
        bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Protezione divisione per zero se tutte le predizioni sono vuote
        total_rouge = 0
        for h, r in zip(hypotheses, references):
            score = scorer.score(" ".join(r[0]), " ".join(h))
            total_rouge += score['rougeL'].fmeasure
        rougeL = total_rouge / len(hypotheses) if hypotheses else 0
        
        return {"file": checkpoint_name, "model": model_tag, "bleu": bleu, "rougeL": rougeL}

    def run_all(self, specific_file=None):
        files = [specific_file] if specific_file else [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        if not files: return None

        print(f"\n🚀 Analisi Metriche su {len(files)} modello/i (Penalty: ACTIVE)...")
        for ckpt in sorted(files):
            res = self.evaluate_single_checkpoint(ckpt)
            if "error" not in res: self.results.append(res)
        
        return pd.DataFrame(self.results)