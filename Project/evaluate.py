"""
================================================================================
BATCH EVALUATION & BEAM SEARCH UNIT
================================================================================
ROLE: Assessing Model Generalization through Advanced Decoding.

DESIGN RATIONALE:
- Search Space Exploration: Implements Beam Search ($k=3$) to avoid the 'myopic' 
  nature of Greedy decoding, allowing the model to correct early lexical errors.
- Numerical Stability: Operates in log-probability space to prevent underflow 
  during the cumulative scoring of long sequences.
- Statistical Rigor: Uses SmoothingFunction (method1) for BLEU to ensure 
  non-zero scores for short summaries, providing a more granular audit.
================================================================================
"""

import sys
import os
import torch
import logging
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
    Loads neural weights and runs a controlled inference battery on the Test set.
    """
    def __init__(self, device, tokenizer_path, checkpoint_dir, data_path, subset_size=200):
        """
        Args:
            device: Hardware target (CUDA/CPU).
            tokenizer_path: Path to the BPE vocabulary.
            checkpoint_dir: Directory containing .pt weight files.
            data_path: Source of the refined test dataset.
            subset_size: Number of samples to audit (Balance between speed and confidence).
        """
        self.device = device
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.checkpoint_dir = checkpoint_dir
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.subset_size = subset_size
        self.results = []

    def beam_decode(self, model, src, model_tag, beam_width=3, max_len=30):
        """
        BEAM SEARCH ALGORITHM: Heuristic search for the most likely sequence.
        
        Logic:
        - Branching: At each step, it tracks the top 'k' (beam_width) hypotheses.
        - Probability Accumulation: Sums log-probabilities to find the 'global maximum' path.
        - Termination: Stops when <EOS> is reached or max_len is exceeded.
        """
        sos_id = self.tokenizer.token_to_id("<SOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        
        # --- INITIALIZATION ---
        if "lstm" in model_tag:
            # Sequential Init: LSTMs require the initial Hidden and Cell states.
            encoder_outputs, hidden, cell = model.encoder(src)
            beams = [([sos_id], 0.0, hidden, cell)]
        else:
            # Transformer Init: Pure sequence-based, states are inferred from context.
            beams = [([sos_id], 0.0)]

        for step in range(max_len):
            new_candidates = []
            for b in beams:
                # Unpack current hypothesis.
                if "lstm" in model_tag:
                    seq, score, h, c = b
                else:
                    seq, score = b
                
                # If the hypothesis already reached <EOS>, preserve it as a candidate.
                if seq[-1] == eos_id:
                    new_candidates.append(b)
                    continue
                
                try:
                    # FORWARD PASS: Predict the probability distribution of the next token.
                    if "lstm" in model_tag:
                        input_token = torch.LongTensor([seq[-1]]).to(self.device)
                        output, next_h, next_c = model.decoder(input_token, h, c, encoder_outputs)
                    else:
                        input_tensor = torch.LongTensor([seq]).to(self.device)
                        # Transformer needs the full 'so-far' sequence to compute self-attention.
                        output = model(src, input_tensor)[:, -1, :]
                    
                    # LOG-SOFTMAX: Converts raw logits into stable log-probabilities.
                    log_probs = F.log_softmax(output, dim=-1)
                    # Expand the search space to the top 'k' tokens.
                    top_v, top_i = log_probs.topk(beam_width)
                    
                    for i in range(beam_width):
                        next_token = top_i[0, i].item()
                        new_score = score + top_v[0, i].item() # Additive scoring (log-space multiplication).
                        
                        if "lstm" in model_tag:
                            new_candidates.append((seq + [next_token], new_score, next_h, next_c))
                        else:
                            new_candidates.append((seq + [next_token], new_score))
                except Exception as e:
                    logger.error(f"Critical Search Failure at step {step}: {e}")
                    raise e
            
            # SURVIVAL OF THE FITTEST: Prune hypotheses to only the top 'k' based on scores.
            beams = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # EARLY EXIT: If all active beams ended in <EOS>, the sequence is finalized.
            if all(b[0][-1] == eos_id for b in beams):
                break
                
        return beams[0][0] # Return the highest-scoring sequence.

    def evaluate_single_checkpoint(self, checkpoint_name):
        """
        Performs a full metric audit on a single model weight file.
        
        Args:
            checkpoint_name: The .pt file to be scrutinized.
        Returns:
            Dictionary containing the 'Model ID', BLEU, and ROUGE scores.
        """
        # --- ARCHITECTURAL MAPPING ---
        if "transformer" in checkpoint_name.lower():
            model_tag = "transformer"
        elif "lstm_dotproduct" in checkpoint_name.lower():
            model_tag = "lstm_dotproduct"
        else:
            model_tag = "lstm_bahdanau"

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        print(f"\nAuditing Performance: {model_tag} | {checkpoint_name}")
        
        try:
            # RE-SYNTHESIS: Reconstruct architecture through the Factory.
            model = get_model_architecture(model_tag, self.device, vocab_size=self.vocab_size)
            
            # WEIGHT INJECTION: Loading optimized parameters into the architecture.
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval() # Vital: Disables Dropout/BatchNorm for deterministic evaluation.
            print(f"Neural Weights Loaded.")
        except Exception as e:
            print(f"Load Error [{model_tag}]: {e}")
            return {"file": checkpoint_name, "error": str(e)}
        
        # DATA INGESTION: Targeted Test-split loading.
        test_loader = get_dataloader(self.data_path, "test", self.tokenizer_path, batch_size=1, subset=self.subset_size)
        references, hypotheses = [], []

        with torch.no_grad(): # Prevent gradient memory accumulation during inference.
            try:
                for src, trg in tqdm(test_loader, desc=f"Audit {model_tag}", leave=False):
                    src = src.to(self.device)
                    # Use Beam Search for higher quality generation.
                    predicted_indices = self.beam_decode(model, src, model_tag, beam_width=3)
                    
                    # SYMBOLIC TO TEXT: Decode integer IDs back to human language.
                    pred_text = self.tokenizer.decode(predicted_indices, skip_special_tokens=True).strip()
                    real_text = self.tokenizer.decode(trg.squeeze().tolist(), skip_special_tokens=True).strip()
                    
                    hypotheses.append(pred_text.split()) # Predicted words.
                    references.append([real_text.split()]) # Ground truth (as list of lists).
            except Exception as e:
                print(f"Inference Error [{model_tag}]: {e}")
                return {"file": checkpoint_name, "error": str(e)}

        # --- NLP METRIC COMPUTATION ---
        # BLEU: Precision-based metric. Measures how many n-grams match between Pred and Real.
        bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
        
        # ROUGE-L: Recall-based metric. Measures the Longest Common Subsequence (LCS) to capture structure.
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        total_rouge = sum(scorer.score(" ".join(r[0]), " ".join(h))['rougeL'].fmeasure for h, r in zip(hypotheses, references))
        rougeL = total_rouge / len(hypotheses) if hypotheses else 0
        
        return {"file": checkpoint_name, "model": model_tag, "bleu": bleu, "rougeL": rougeL}

    def run_all(self, specific_file=None):
        """
        Orchestrates evaluation across multiple checkpoints.
        
        Args:
            specific_file: If provided, audits only one file. Otherwise, audits the whole directory.
        Returns:
            Pandas DataFrame containing a comparative leaderboard of model performance.
        """
        files = [specific_file] if specific_file else [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        if not files: return None

        print(f"\nSTART BATCH EVALUATION (Beam Search k=3)")
        for ckpt in sorted(files):
            res = self.evaluate_single_checkpoint(ckpt)
            if "error" not in res: 
                self.results.append(res)
            else:
                print(f"Skipping {ckpt} due to load/runtime failure.")
        
        return pd.DataFrame(self.results)