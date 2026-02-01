"""
TRANSFORMER REVISION - Version 2.6 (Unified Alignment Release)
--------------------------------------------------------------
MODIFICHE CHIRURGICHE:
1. TEACHER FORCING DETECTION: Il modello rileva se riceve una sequenza (Train/Val)
   o un singolo token (Inference). Se riceve una sequenza, applica lo shift
   indipendentemente da model.train() o model.eval().
2. DUMMY PADDING UNIVERSALE: L'output per la Loss viene sempre allineato 
   per essere processato da output[:, 1:] nel modulo train.py.
3. ATTENTION SCALING: Stabilità numerica garantita tramite:
   $$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
4. MEMORY MASKING: L'encoder non influenza il decoder sui token di padding.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, 
            batch_first=True 
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, src, trg, **kwargs):
        # 1. --- LOGICA DI ALLINEAMENTO STRUTTURALE ---
        # Verifichiamo se stiamo processando una sequenza intera (Teacher Forcing)
        # Questo accade sia in Training che nel calcolo della Val Loss in scripts/train.py
        is_teacher_forcing = trg.size(1) > 1
        
        if is_teacher_forcing:
            # Shift interno per impedire al modello di vedere il futuro
            trg_input = trg[:, :-1]
        else:
            # Decodifica autoregressiva (Audit/Inference)
            trg_input = trg
        
        # 2. GENERAZIONE MASCHERE
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (trg_input == 0)
        tgt_mask = self._generate_causal_mask(trg_input.size(1), src.device)
        
        # 3. EMBEDDING + STABILIZZAZIONE
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(trg_input) * math.sqrt(self.d_model))
        
        # 4. ESECUZIONE
        output = self.transformer(
            src_emb, tgt_emb, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Cruciale per ignorare il PAD dell'encoder
        )
        
        # 5. PROIEZIONE FINALE
        predictions = self.fc_out(output)
        
        # 6. --- RICONCILIAZIONE CON L'ORCHESTRATORE (scripts/train.py) ---
        if is_teacher_forcing:
            # Ripristiniamo la lunghezza originale tramite dummy_prefix
            # Questo permette a train.py di usare output[:, 1:] senza errori di sfasamento
            batch_size = src.size(0)
            dummy_prefix = torch.zeros(batch_size, 1, self.vocab_size).to(src.device)
            return torch.cat([dummy_prefix, predictions], dim=1)
        
        # Restituiamo le predizioni pure per la Beam Search / Greedy Search
        return predictions