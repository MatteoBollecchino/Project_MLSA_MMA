"""
================================================================================
BATCH EVALUATION & PROBABILISTIC AUDIT UNIT
================================================================================
ROLE: Assessing Model Generalization and Statistical Confidence.

DESIGN RATIONALE:
- Probabilistic Integrity: Integrates Perplexity (PPL) and Cross-Entropy Loss
  to measure the model's internal uncertainty (branching factor).
- Search Space Exploration: Implements Beam Search (k=3) for high-quality 
  sequence synthesis during qualitative assessment.
- Numerical Stability: Operates in log-probability space to prevent underflow 
  and ensure accurate cumulative scoring.
================================================================================
"""

import sys
import os
import torch
import logging
import math
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tokenizers import Tokenizer
from models.factory import get_model_architecture
from data.dataset import get_dataloader

# Path resolution to ensure the factory and data modules are discoverable.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

logger = logging.getLogger(__name__)

class BatchEvaluator:
    """
    Automated Audit Engine. 
    Performs a deep statistical analysis on model checkpoints using the Test split.
    """
    def __init__(self, device, tokenizer_path, checkpoint_dir, data_path, subset_size=200):
        self.device = device
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.checkpoint_dir = checkpoint_dir
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.subset_size = subset_size
        self.results = []

    def beam_decode(self, model, src, model_tag, beam_width=3, max_len=30):
        """ Heuristic search for the most likely sequence. """
        sos_id = self.tokenizer.token_to_id("<SOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        
        if "lstm" in model_tag:
            encoder_outputs, hidden, cell = model.encoder(src)
            beams = [([sos_id], 0.0, hidden, cell)]
        else:
            beams = [([sos_id], 0.0)]
            encoder_outputs = None

        for step in range(max_len):
            new_candidates = []
            for b in beams:
                if "lstm" in model_tag:
                    seq, score, h, c = b
                else:
                    seq, score = b
                
                if seq[-1] == eos_id:
                    new_candidates.append(b)
                    continue
                
                try:
                    if "lstm" in model_tag:
                        input_token = torch.LongTensor([seq[-1]]).to(self.device)
                        output, next_h, next_c = model.decoder(input_token, h, c, encoder_outputs)
                    else:
                        input_tensor = torch.LongTensor([seq]).to(self.device)
                        output = model(src, input_tensor)[:, -1, :]
                    
                    log_probs = F.log_softmax(output, dim=-1)
                    top_v, top_i = log_probs.topk(beam_width)
                    
                    for i in range(beam_width):
                        next_token = top_i[0, i].item()
                        new_score = score + top_v[0, i].item()
                        
                        if "lstm" in model_tag:
                            new_candidates.append((seq + [next_token], new_score, next_h, next_c))
                        else:
                            new_candidates.append((seq + [next_token], new_score))
                except Exception as e:
                    logger.error(f"Search Failure at step {step}: {e}")
                    raise e
            
            beams = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(b[0][-1] == eos_id for b in beams):
                break
                
        return beams[0][0]

    def evaluate_single_checkpoint(self, checkpoint_name):
        """ Performs full metric audit: BLEU, ROUGE, Loss, and Perplexity. """
        if "transformer" in checkpoint_name.lower(): model_tag = "transformer"
        elif "lstm_dotproduct" in checkpoint_name.lower(): model_tag = "lstm_dotproduct"
        else: model_tag = "lstm_bahdanau"

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        print(f"\nAuditing Performance: {model_tag} | {checkpoint_name}")
        
        try:
            model = get_model_architecture(model_tag, self.device, vocab_size=self.vocab_size)
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            return {"file": checkpoint_name, "error": str(e)}
        
        test_loader = get_dataloader(self.data_path, "test", self.tokenizer_path, batch_size=1, subset=self.subset_size)
        references, hypotheses = [], []
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for src, trg in tqdm(test_loader, desc=f"Audit {model_tag}", leave=False):
                src, trg = src.to(self.device), trg.to(self.device)
                
                # --- STATISTICAL ANALYSIS (Loss & PPL) ---
                # We perform a forward pass with the ground truth to measure 'Surprise'
                if "lstm" in model_tag:
                    output = model(src, trg[:, :-1]) # Standard teacher forcing pass
                else:
                    output = model(src, trg[:, :-1])
                
                # Reshape for CrossEntropy: [Batch * Seq, Vocab]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg_loss = trg[:, 1:].contiguous().view(-1)
                
                loss = F.cross_entropy(output, trg_loss, reduction='sum')
                total_loss += loss.item()
                total_tokens += trg_loss.numel()

                # --- QUALITATIVE ANALYSIS (Decoding) ---
                predicted_indices = self.beam_decode(model, src, model_tag, beam_width=3)
                pred_text = self.tokenizer.decode(predicted_indices, skip_special_tokens=True).strip()
                real_text = self.tokenizer.decode(trg.squeeze().tolist(), skip_special_tokens=True).strip()
                
                hypotheses.append(pred_text.split())
                references.append([real_text.split()])

        # --- METRIC SYNTHESIS ---
        # 1. Linguistic Metrics
        bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        total_rouge = sum(scorer.score(" ".join(r[0]), " ".join(h))['rougeL'].fmeasure for h, r in zip(hypotheses, references))
        rougeL = total_rouge / len(hypotheses) if hypotheses else 0
        
        # 2. Probabilistic Metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return {
            "file": checkpoint_name, 
            "model": model_tag, 
            "bleu": bleu, 
            "rougeL": rougeL, 
            "loss": avg_loss, 
            "perplexity": perplexity
        }

    def run_all(self, specific_file=None):
        files = [specific_file] if specific_file else [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        if not files: return None

        print(f"\nSTART BATCH EVALUATION (Probabilistic Audit)")
        for ckpt in sorted(files):
            res = self.evaluate_single_checkpoint(ckpt)
            if "error" not in res: 
                self.results.append(res)
        
        return pd.DataFrame(self.results)