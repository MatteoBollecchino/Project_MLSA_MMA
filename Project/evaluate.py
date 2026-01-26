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

# Resolution paths for internal imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Logger Configuration
logger = logging.getLogger(__name__)

# --- [BATCH EVALUATOR CLASS] ---
class BatchEvaluator:
    """ Class for batch evaluation of model checkpoints. """

    # Initialization of the BatchEvaluator
    def __init__(self, device, tokenizer_path, checkpoint_dir, data_path, subset_size=200):
        """ Initializes the BatchEvaluator with necessary parameters. """

        self.device = device
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.checkpoint_dir = checkpoint_dir
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.subset_size = subset_size
        self.results = []

    # Evaluates a single checkpoint
    def evaluate_single_checkpoint(self, checkpoint_name):
        """ Evaluates a single model checkpoint and computes BLEU and ROUGE-L scores. """

        # Determine model type from filename
        model_tag = "transformer" if "transformer" in checkpoint_name.lower() else "lstm_attention"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Load model
        try:
            # Instantiate model architecture
            model = get_model_architecture(model_tag, self.device, vocab_size=self.vocab_size)
            
            # Load model state
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            
            # Evaluation of the model
            model.eval()

        except Exception as e:
            return {"file": checkpoint_name, "error": f"Load error: {str(e)}"}
        
        # Prepare test data loader
        test_loader = get_dataloader(
            self.data_path, "test", 
            self.tokenizer_path, batch_size=1, subset=self.subset_size
        )

        # Initialize references and hypotheses for evaluation metrics 
        references, hypotheses = [], []
        sos_id, eos_id = self.tokenizer.token_to_id("<SOS>"), self.tokenizer.token_to_id("<EOS>")

        # Inference loop
        with torch.no_grad():
            for src, trg in tqdm(test_loader, desc=f"Valutazione {model_tag}", leave=False):
                src = src.to(self.device)
                predicted_indices = []

                # Evaluation for LSTM with Attention
                if model_tag == "lstm_attention":
                    # Encode the source sequence
                    encoder_outputs, hidden, cell = model.encoder(src)

                    # Initialize input token
                    input_token = torch.LongTensor([sos_id]).to(self.device)

                    # Decoding loop
                    for _ in range(30):
                        # Generate next token
                        output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)

                        # Get the token with highest probability
                        top1 = output.argmax(1)

                        # Break if EOS token is generated
                        if top1.item() == eos_id: break

                        # Append predicted token and update input
                        predicted_indices.append(top1.item())
                        input_token = top1
                
                # Evaluation for Transformer
                else:
                    # Initial input sequence with SOS token
                    ys = torch.ones(1, 1).fill_(sos_id).type(torch.long).to(self.device)

                    # Decoding loop
                    for _ in range(30):
                        # Generate next token
                        out = model(src, ys)

                        # Get the token with highest probability
                        next_word = out[:, -1].argmax(1).item()

                        # Break if EOS token is generated
                        if next_word == eos_id: break

                        # Append predicted token and update input sequence
                        predicted_indices.append(next_word)
                        ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(self.device)], dim=1)

                # Decode predictions and references
                pred_text = self.tokenizer.decode(predicted_indices, skip_special_tokens=True).strip()
                real_text = self.tokenizer.decode(trg.squeeze().tolist(), skip_special_tokens=True).strip()

                # Append to lists for metric calculation
                hypotheses.append(pred_text.split()); references.append([real_text.split()])

        # Calculate BLEU and ROUGE-L scores
        bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = sum(scorer.score(" ".join(r[0]), " ".join(h))['rougeL'].fmeasure for h, r in zip(hypotheses, references)) / len(hypotheses)
        
        return {"file": checkpoint_name, "bleu": bleu, "rougeL": rougeL}

    # Runs evaluation on all checkpoints or a specific file
    def run_all(self, specific_file=None):
        """ Runs evaluation on all checkpoints or a specific checkpoint file. """
        
        files = [specific_file] if specific_file else [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        if not files: return None

        print(f"\n🚀 Analisi Metriche su {len(files)} modello/i...")
        for ckpt in sorted(files):
            res = self.evaluate_single_checkpoint(ckpt)
            if "error" not in res: self.results.append(res)
        
        # Return results as DataFrame
        return pd.DataFrame(self.results)