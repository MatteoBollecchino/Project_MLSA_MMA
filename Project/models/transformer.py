"""
================================================================================
TRANSFORMER CORE 
================================================================================
ROLE: High-Parallelism Attention-Based Sequence Transformation.

DESIGN RATIONALE:
- Temporal Order: Since Transformers lack recurrent loops, we use Sinusoidal 
  Positional Encodings to preserve token hierarchy and sequence structure.
- Interface Compatibility: Implements internal shifting and dummy padding to 
  remain "Plug & Play" with the LSTM-centric training orchestrator.
- Numerical Integrity: Uses Scaled Dot-Product Attention scaling ($1/\sqrt{d_k}$) 
  to prevent Softmax saturation during high-dimensional tensor operations.
================================================================================
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    SPATIAL ORIENTATION UNIT: Injects non-trainable sinusoidal signals.
    
    Standard Transformers are 'permutation invariant'. This module breaks that 
    symmetry by adding sine and cosine waves of varying frequencies, allowing 
    the Attention heads to distinguish between word positions.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the positional encoding matrix in log space for efficiency.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer: it stays with the model but is not updated by gradients.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Fuses the token embeddings with the spatial signal.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    """
    NEURAL ENGINE: Vanilla Transformer architecture with Unified Interface Logic.
    
    Specifically revised for Version 2.6 to handle internal data shifting, 
    making it compatible with the 'scripts/train.py' logic without code changes.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # PROJECTION LAYERS: Mapping symbols to latent attention space.
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # CORE TRANSFORMER: Built-in PyTorch implementation for optimized kernel execution.
        # batch_first=True is used to maintain consistency with the C2 dataset pipeline.
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, 
            batch_first=True 
        )
        
        # OUTPUT HEAD: Linear mapping back to vocabulary dimensions.
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, sz, device):
        """
        CAUSALITY SHIELD: Creates an upper-triangular matrix to prevent 'Peeking'.
        Ensures position 'i' can only attend to positions '<= i'.
        """
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, src, trg, **kwargs):
        """
        Unified Forward Pass.
        
        LOGIC FLOW:
        1. MODALITY DETECTION: If trg length > 1, the system is in Training/Validation.
           It then performs internal shifting (trg[:, :-1]) to prevent identity learning.
        2. MASKING: Generates padding masks and causal masks to isolate valid signals.
        3. SCALING: Scales embeddings by sqrt(d_model) to maintain gradient variance.
        4. RECONCILIATION: Concatenates a dummy prefix if in training to keep output shape
           aligned with the ground-truth target length.
        """
        # --- PHASE 1: MODALITY-BASED ALIGNMENT ---
        is_teacher_forcing = trg.size(1) > 1
        
        if is_teacher_forcing:
            # SHIFT LOGIC: Use tokens [0 to T-1] to predict [1 to T].
            trg_input = trg[:, :-1]
        else:
            # INFERENCE LOGIC: Use the provided sequence as-is (e.g., during Beam Search).
            trg_input = trg
        
        # --- PHASE 2: ATTENTION MASKING ---
        # Binary masks (True where pad index is 0) to avoid processing null data.
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (trg_input == 0)
        
        # Look-ahead mask for the autoregressive decoder.
        tgt_mask = self._generate_causal_mask(trg_input.size(1), src.device)
        
        # --- PHASE 3: LATENT TRANSFORMATION ---
        # Scaling is the 'Hidden Secret' of Transformer stability.
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(trg_input) * math.sqrt(self.d_model))
        
        # CORE EXECUTION:
        # memory_key_padding_mask ensures encoder padding does not distract the decoder.
        output = self.transformer(
            src_emb, tgt_emb, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask 
        )
        
        # --- PHASE 4: VOCABULARY PROJECTION ---
        predictions = self.fc_out(output)
        
        # --- PHASE 5: INTERFACE RECONCILIATION ---
        if is_teacher_forcing:
            # REPAIR SHAPE: The internal shift removed 1 token. 
            # We add a null prefix so train.py can safely call output[:, 1:].
            batch_size = src.size(0)
            dummy_prefix = torch.zeros(batch_size, 1, self.vocab_size).to(src.device)
            return torch.cat([dummy_prefix, predictions], dim=1)
        
        # Pure predictions for autoregressive decoding (Inference mode).
        return predictions