import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Inietta informazione sull'ordine dei token.
    Senza questo, il Transformer vede il codice come un 'sacchetto di parole' (bag of words).
    """
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
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Embedding con scaling (Cruciale per la stabilità)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding Standard (Invece del nn.Parameter)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Core
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, 
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            batch_first=True 
        )
        
        # 4. Proiezione finale sul vocabolario
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, sz, device):
        """ Impedisce al decoder di 'guardare nel futuro' """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, src, trg, **kwargs):
        # Generazione maschere
        tgt_mask = self._generate_causal_mask(trg.size(1), src.device)
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (trg == 0)
        
        # --- [FIX CRITICO 1: SCALING] ---
        # Scalare l'embedding per sqrt(d_model) impedisce che i pesi iniziali 
        # vengano schiacciati dal positional encoding.
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(trg) * math.sqrt(self.d_model))
        
        # --- [FIX CRITICO 2: MEMORY MASK] ---
        # Aggiungiamo memory_key_padding_mask per ignorare il padding dell'encoder
        output = self.transformer(
            src_emb, 
            tgt_emb, 
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_padding_mask if 'src_padding_mask' in locals() else (src==0),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=(src == 0)
        )
        
        return self.fc_out(output)