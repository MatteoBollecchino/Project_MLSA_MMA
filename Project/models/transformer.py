import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Inietta informazione sull'ordine dei token.
    Senza questo, il Transformer tratterebbe il codice come un 'bag of words'.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def _generate_causal_mask(self, sz, device):
        """Impedisce al decoder di sbirciare i token futuri (Look-ahead mask)."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=None, **kwargs):
        """
        Ingegneria delle Maschere:
        1. Causal Mask: Impedisce di vedere il futuro nel target.
        2. Padding Mask: Impedisce di dare importanza ai token <PAD> (ID 0).
        """
        # Creazione maschere di padding (True dove c'è PAD)
        src_padding_mask = (src == 0)
        tgt_padding_mask = (trg == 0)
        
        # Generazione maschera causale per il training parallelo
        tgt_mask = self._generate_causal_mask(trg.size(1), src.device)
        
        # Embedding + Scaling (Standard per Transformer stability)
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(trg) * math.sqrt(self.d_model))
        
        # Esecuzione del blocco Transformer con iniezione di tutte le maschere
        output = self.transformer(
            src_emb, 
            tgt_emb, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask # Il decoder ignora il padding dell'encoder
        )
        
        return self.fc_out(output)