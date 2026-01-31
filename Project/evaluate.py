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

    def beam_decode(self, model, src, model_tag, beam_width=3, max_len=30):
        sos_id = self.tokenizer.token_to_id("<SOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        
        if "lstm" in model_tag:
            encoder_outputs, hidden, cell = model.encoder(src)
            beams = [([sos_id], 0.0, hidden, cell)]
        else:
            # TRANSFORMER BEAM INIT
            beams = [([sos_id], 0.0)]

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
                    # Se crasha qui, il problema è nella forward pass del modello
                    logger.error(f"❌ Errore durante Beam Search al passo {step}: {e}")
                    raise e
            
            beams = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(b[0][-1] == eos_id for b in beams):
                break
                
        return beams[0][0]

    def evaluate_single_checkpoint(self, checkpoint_name):
        # --- [MAPPING LOGIC] ---
        if "transformer" in checkpoint_name.lower():
            model_tag = "transformer"
        elif "lstm_dotproduct" in checkpoint_name.lower():
            model_tag = "lstm_dotproduct"
        else:
            model_tag = "lstm_bahdanau"

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        print(f"\n🧪 Testing Model: {model_tag} | File: {checkpoint_name}")
        
        try:
            # 1. ARCHITECTURE MATCHING
            model = get_model_architecture(model_tag, self.device, vocab_size=self.vocab_size)
            
            # 2. WEIGHT LOADING (Punto critico di crash)
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"✅ Checkpoint caricato con successo.")
        except Exception as e:
            print(f"❌ ERRORE CARICAMENTO [{model_tag}]: {e}")
            return {"file": checkpoint_name, "error": str(e)}
        
        test_loader = get_dataloader(self.data_path, "test", self.tokenizer_path, batch_size=1, subset=self.subset_size)
        references, hypotheses = [], []

        with torch.no_grad():
            try:
                for src, trg in tqdm(test_loader, desc=f"Audit {model_tag}", leave=False):
                    src = src.to(self.device)
                    predicted_indices = self.beam_decode(model, src, model_tag, beam_width=3)
                    
                    pred_text = self.tokenizer.decode(predicted_indices, skip_special_tokens=True).strip()
                    real_text = self.tokenizer.decode(trg.squeeze().tolist(), skip_special_tokens=True).strip()
                    
                    hypotheses.append(pred_text.split())
                    references.append([real_text.split()])
            except Exception as e:
                print(f"❌ ERRORE INFERENZA [{model_tag}]: {e}")
                return {"file": checkpoint_name, "error": str(e)}

        # Calcolo BLEU e ROUGE
        bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        total_rouge = sum(scorer.score(" ".join(r[0]), " ".join(h))['rougeL'].fmeasure for h, r in zip(hypotheses, references))
        rougeL = total_rouge / len(hypotheses) if hypotheses else 0
        
        return {"file": checkpoint_name, "model": model_tag, "bleu": bleu, "rougeL": rougeL}

    def run_all(self, specific_file=None):
        files = [specific_file] if specific_file else [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        if not files: return None

        print(f"\n🚀 START BATCH EVALUATION (Beam Search k=3)")
        for ckpt in sorted(files):
            res = self.evaluate_single_checkpoint(ckpt)
            # Aggiungiamo alla lista solo se non c'è errore per non sporcare il DataFrame
            if "error" not in res: 
                self.results.append(res)
            else:
                print(f"⚠️ Modello {ckpt} saltato causa errore.")
        
        return pd.DataFrame(self.results)