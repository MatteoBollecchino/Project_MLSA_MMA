import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Inietta informazione sull'ordine dei token.
    Senza questo, il Transformer tratterebbe il codice Python come un set di parole a caso.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calcolo dei coefficienti seno e coseno:
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
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
            batch_first=True # Allineamento con il tuo Dataloader
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def _generate_causal_mask(self, sz):
        """Genera una maschera triangolare superiore per impedire al decoder di guardare al futuro."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=None, **kwargs):
        """
        NOTA: teacher_forcing_ratio è accettato per polimorfismo con l'LSTM, 
        ma viene ignorato dal Transformer che usa il parallelismo causale.
        """
        # 1. Embedding + Positional Encoding
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_encoder(self.embedding(trg) * math.sqrt(self.d_model))
        
        # 2. Creazione della Look-ahead mask per il training parallelo
        # Il decoder non deve 'sbirciare' i token successivi durante l'addestramento
        trg_mask = self._generate_causal_mask(trg.size(1)).to(src.device)

        # 3. Transformer Forward Pass
        # src_key_padding_mask potrebbe essere aggiunto qui per ignorare i token <PAD>
        output = self.transformer(src_emb, trg_emb, tgt_mask=trg_mask)
        
        # 4. Proiezione sul vocabolario
        return self.fc_out(output)