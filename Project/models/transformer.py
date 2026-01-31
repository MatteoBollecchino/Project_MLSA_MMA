"""
TRANSFORMER REVISION - Version 2.1
----------------------------------
OBIETTIVO: Risolvere il collasso delle metriche senza modificare train.py.

MODIFICHE:
1. INTERNAL SHIFTING: Il modello taglia l'ultimo token di 'trg' per l'input 
   e aggiunge un padding in testa all'output per mantenere la forma [Batch, Seq_Len, Vocab].
2. SCALING FACTOR: Moltiplicazione per sqrt(d_model) per prevenire la saturazione della Softmax.
3. SINUSOIDAL POSITIONAL ENCODING: Sostituito il parametro addestrabile con segnali fissi.
4. MASKING BUG FIX: Corretta la gestione delle maschere di padding per l'encoder (memory_mask).
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
        # 1. --- INTERNAL SHIFTING LOGIC ---
        # Prendiamo tutto tranne l'ultimo token per l'input del decoder (include <SOS>)
        # trg_input: [Batch, Seq_Len - 1]
        trg_input = trg[:, :-1]
        
        # 2. GENERAZIONE MASCHERE
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (trg_input == 0)
        tgt_mask = self._generate_causal_mask(trg_input.size(1), src.device)
        
        # 3. EMBEDDING + SCALING + POSITION
        # math.sqrt(d_model) è l'iniezione di stabilità numerica mancante
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(trg_input) * math.sqrt(self.d_model))
        
        # 4. TRASFORMER EXECUTION
        output = self.transformer(
            src_emb, tgt_emb, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Impedisce di guardare il pad dell'encoder
        )
        
        # 5. PROIEZIONE FINALE
        # predictions shape: [Batch, Seq_Len - 1, Vocab]
        predictions = self.fc_out(output)
        
        # 6. --- ALIGNMENT CON train.py ---
        # Per non cambiare train.py, dobbiamo restituire una sequenza di lunghezza originale.
        # Aggiungiamo un dummy token all'inizio (che verrà scartato da train.py tramite output[:, 1:])
        batch_size = src.size(0)
        dummy_prefix = torch.zeros(batch_size, 1, self.vocab_size).to(src.device)
        
        # Ritorna [Batch, Seq_Len, Vocab]
        return torch.cat([dummy_prefix, predictions], dim=1)