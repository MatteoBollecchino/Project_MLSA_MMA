"""
TRANSFORMER REVISION - Version 2.5 (Final CTO Release)
------------------------------------------------------
MODIFICHE CHIRURGICHE APPORTATE:
1. CONDITIONAL SHIFTING: Inserito controllo self.training. Durante il training
   il modello sfasera i target internamente per impedire il 'peeking'. In audit,
   accetta la sequenza intera per permettere la generazione autoregressiva.
2. DUMMY PADDING (Alignment): Durante il training, l'output viene esteso con un 
   prefisso vuoto per mantenere la compatibilità con la riga output[:, 1:] di train.py.
3. SCALING ATTENTION: Moltiplicazione degli embedding per sqrt(d_model) per
   evitare la saturazione della Softmax: $$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
4. FIXED POSITIONAL ENCODING: Bussola geometrica sinusoidale fissa per la stabilità.
5. MEMORY MASKING: Corretto l'uso di memory_key_padding_mask per ignorare il pad dell'encoder.
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
        # 1. --- STRATEGIA DI SFASAMENTO CONDIZIONALE ---
        if self.training:
            # Durante il training, 'trg' è la sequenza completa.
            # Tagliamo l'ultimo token per l'input del decoder per non 'barare'.
            trg_input = trg[:, :-1]
        else:
            # Durante l'audit/inferenza, 'trg' è la sequenza parziale generata finora.
            # Usiamo tutto l'input fornito senza tagli.
            trg_input = trg
        
        # 2. GENERAZIONE MASCHERE PADDING E CAUSALI
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (trg_input == 0)
        tgt_mask = self._generate_causal_mask(trg_input.size(1), src.device)
        
        # 3. TRASFORMAZIONE NELLO SPAZIO LATENTE
        # Scaling con sqrt(d_model) fondamentale per mantenere i gradienti sani
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(trg_input) * math.sqrt(self.d_model))
        
        # 4. ESECUZIONE DEL CORE TRANSFORMER
        output = self.transformer(
            src_emb, tgt_emb, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Filtra il rumore dell'encoder
        )
        
        # 5. PROIEZIONE SUL VOCABOLARIO
        predictions = self.fc_out(output)
        
        # 6. --- RIFINITURA PER L'ORCHESTRATORE ---
        if self.training:
            # Per non modificare train.py, dobbiamo restituire una sequenza di lunghezza pari a 'trg'.
            # Aggiungiamo un prefisso di zeri che verrà ignorato dalla CrossEntropyLoss.
            batch_size = src.size(0)
            dummy_prefix = torch.zeros(batch_size, 1, self.vocab_size).to(src.device)
            return torch.cat([dummy_prefix, predictions], dim=1)
        else:
            # In Audit restituiamo le predizioni pure per la decodifica autoregressiva.
            return predictions