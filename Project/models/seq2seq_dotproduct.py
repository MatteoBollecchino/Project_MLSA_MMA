"""
================================================================================
LSTM + SCALED DOT-PRODUCT ATTENTION 
================================================================================
ROLE: Multiplicative Sequence-to-Sequence Modeling.

DESIGN RATIONALE:
- Efficiency: Replaces the MLP in Bahdanau with a direct Dot Product, reducing 
  trainable parameters and decreasing training latency.
- Mathematical Alignment: Follows the geometric approach where attention is 
  viewed as a similarity measure in a projected latent space.
- Scaling: Implements the 1/sqrt(d_k) factor to maintain stable gradients, 
  preventing the Softmax from saturating during the backward pass.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# Reusing the standard LSTM Encoder to maintain pipeline consistency and modularity.
from models.seq2seq_bahdanau import Encoder 

class DotProductAttention(nn.Module):
    """
    GEOMETRIC ALIGNMENT UNIT: Implements Multiplicative (Dot-Product) Attention.
    
    This mechanism treats the Decoder's hidden state as a 'Query' and the 
    Encoder's states as 'Keys', calculating their similarity through matrix 
    products rather than additive layers.
    """
    def __init__(self, hid_dim):
        """
        Args:
            hid_dim (int): Dimension of the Query/Key vectors.
        """
        super().__init__()
        # Linear Projections: Mapping raw hidden states into the specialized 
        # Attention Space (Standard practice in modern Transformer-style attention).
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        
        # Scaling Factor: Used to normalize the Dot Product magnitude. 
        # Prevents vanishing gradients by keeping scores within the Softmax 'sweet spot'.
        self.scale = np.sqrt(hid_dim)

    def forward(self, query, keys):
        """
        Args:
            query: Current Decoder state [batch, hid_dim].
            keys: All Encoder outputs [batch, src_len, hid_dim].
        """
        # Step 1: Project raw states into the matching Attention Space.
        # q: [batch, 1, hid_dim] | k: [batch, hid_dim, src_len] (transposed for dot product)
        q = self.w_q(query).unsqueeze(1) 
        k = self.w_k(keys).transpose(1, 2) 
        
        # Step 2: Scaled Dot Product calculation (BMM = Batch Matrix Multiplication).
        # scores = (Q * K^T) / sqrt(d)
        scores = torch.bmm(q, k) / self.scale 
        
        # Step 3: Compute Attention Weights (alphas).
        # Softmax over the source sequence length.
        alphas = F.softmax(scores, dim=-1)
        
        # Step 4: Construct the Context Vector.
        # Sum of Encoder outputs weighted by their relevance to the current Query.
        context = torch.bmm(alphas, keys) # [batch, 1, hid_dim]
        
        return alphas.squeeze(1), context

class DecoderDot(nn.Module):
    """
    FAST GENERATIVE DECODER: Integrated with Multiplicative Attention.
    Optimized for high-speed docstring generation.
    """
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = DotProductAttention(hid_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # RNN unit: Fuses the current token embedding with the global context vector.
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        # Final Projection: Translates latent features into vocabulary-sized logits.
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        Single-step autoregressive forward pass.
        """
        # input: [batch] -> embedded: [batch, 1, emb_dim]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        # QUERY SELECTION: The top-layer hidden state of the LSTM acts as the Query 
        # to find the most relevant code sections in the Encoder history.
        _, context = self.attention(hidden[-1], encoder_outputs)
        
        # FUSION: Concatenating the semantic token with the spatial code context.
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # RECURRENCE: Update the LSTM's internal state (memory).
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # LOGIT PREDICTION: Map the updated memory to the target language space.
        prediction = self.fc_out(output)
        
        return prediction.squeeze(1), hidden, cell

class Seq2SeqDotProduct(nn.Module):
    """
    SCAFFOLD: Integrates the Encoder and the Dot-Product optimized Decoder.
    """
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, device):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = DecoderDot(vocab_size, emb_dim, hid_dim, n_layers, dropout)
        self.device = device
        self.output_dim = vocab_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Full Sequence Forward Pass.
        """
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.output_dim).to(self.device)
        
        # 1. Encode source code into context vectors.
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # 2. Iterate through target sequence for word-by-word generation.
        input = trg[:, 0] # Start with <SOS>
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            
            # TEACHER FORCING: Balancing between ground truth and self-correction.
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)
            
        return outputs