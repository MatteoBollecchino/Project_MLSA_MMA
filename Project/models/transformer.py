import torch
import torch.nn as nn
import math

# --- TRANSFORMER MODEL DEFINITION ---
class PositionalEncoding(nn.Module):
    """
    Injects information about the order of tokens.
    Without this, the Transformer would treat the code as a 'bag of words'.
    """
    
    # Positional Encoding Initialization
    def __init__(self, d_model, max_len=5000):
        """ Initializes the PositionalEncoding module. """
        super().__init__()

        # Creation of the matrix for positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    # Forward pass to add positional encoding to input embeddings
    def forward(self, x):
        """ Adds positional encoding to input embeddings. """

        return x + self.pe[:, :x.size(1)]


# --- SEQ2SEQ TRANSFORMER MODEL DEFINITION ---
class Seq2SeqTransformer(nn.Module):
    """ Seq2Seq Transformer Model for code generation tasks. """

    # Initialization of the Transformer model
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        """ Initializes the Seq2SeqTransformer model. """

        super().__init__()

        # Model components
        self.model_type = 'Transformer'
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer block
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        
        # Final linear layer to project to vocabulary size
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    # Causal mask generation
    def _generate_causal_mask(self, sz, device):
        """ Prevents the decoder from peeking at future tokens (Look-ahead mask). """

        # Create a square matrix of size sz x sz
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)

        # Convert boolean mask to float with -inf and 0.0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # Forward pass of the Transformer model
    def forward(self, src, trg, teacher_forcing_ratio=None, **kwargs):
        """
        Mask Engineering:
        1. Causal Mask: Prevents seeing the future in the target.
        2. Padding Mask: Prevents giving importance to <PAD> tokens (ID 0).
        """
        
        # Creation of padding masks (True where PAD)
        src_padding_mask = (src == 0)
        tgt_padding_mask = (trg == 0)
        
        # Creation of causal mask for parallel training
        tgt_mask = self._generate_causal_mask(trg.size(1), src.device)
        
        # Embedding + Scaling (Standard for Transformer stability)
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(trg) * math.sqrt(self.d_model))
        
        # Execution of the Transformer block with injection of all masks
        output = self.transformer(
            src_emb, 
            tgt_emb, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask # The decoder ignores the encoder's padding
        )
        
        return self.fc_out(output)