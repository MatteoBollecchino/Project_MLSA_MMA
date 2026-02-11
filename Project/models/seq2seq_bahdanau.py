"""
================================================================================
LSTM + BAHDANAU ATTENTION (Additive) 
================================================================================
ROLE: Sequential Sequence-to-Sequence Modeling with Neural Alignment.

DESIGN RATIONALE:
- Hybrid Contextualization: Unlike basic LSTMs, the Decoder does not rely solely 
  on the final hidden state. It re-scans the Encoder's history at every step.
- Additive Attention: Implements a learnable alignment function using a small 
  Multi-Layer Perceptron (MLP) with tanh activation.
- Robustness: Highly effective for long source sequences (code bodies) where 
  the 'Vanishing Gradient' would otherwise erode information.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BahdanauAttention(nn.Module):
    """
    NEURAL ALIGNMENT UNIT: Implements the 'Additive' Attention mechanism.
    
    This module computes a probability distribution over the encoder's hidden 
    states to determine which parts of the source code are relevant to the 
    current word being generated in the summary.
    """

    def __init__(self, hid_dim):
        """
        Args:
            hid_dim (int): The dimensionality of the hidden states.
        """
        super().__init__()

        # Scoring Layer: Projecting the combination of Decoder hidden state 
        # and Encoder outputs into a common latent space.
        self.attn = nn.Linear(hid_dim * 2, hid_dim) # Score = W_a * [s_{t-1}; h_i] + b: Projecting the fused states into a shared latent space.

        # Weighting Vector: A learnable parameter that reduces the projection 
        # to a single scalar (energy score) per source token.
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: Previous Decoder hidden state [batch, hid_dim].
            encoder_outputs: Full history from the Encoder [batch, src_len, hid_dim].
        """
        src_len = encoder_outputs.shape[1]

        # Expand Decoder hidden state to match Encoder sequence length.
        # hidden: [batch, hid_dim] -> [batch, src_len, hid_dim] by repeating across the sequence length.
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # ENERGY CALCULATION: 
        # score = v * tanh(W * [decoder_h ; encoder_h])
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Reduce to scalar energy and remove the last dimension.
        # attention: [batch, src_len, 1] -> [batch, src_len]
        attention = self.v(energy).squeeze(2)

        # SOFTMAX: Normalize energy into probabilities (sum to 1) across the sequence length.
        return F.softmax(attention, dim=1)

class Encoder(nn.Module):
    """
    Consumes the source sequence and builds context.
    Uses a multi-layer LSTM to capture hierarchical dependencies in Python code.
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        Args:
            input_dim: Vocab size.
            emb_dim: Token embedding size.
            hid_dim: LSTM internal capacity.
            n_layers: Depth of recurrent stacking.
            dropout: Regularization probability.
        """
        super().__init__()

        # Semantic projection layer.
        # Maps discrete token IDs to continuous vectors that capture semantic relationships.
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # RECURRENT BACKBONE: Processes the sequence step-by-step.
        # batch_first=True ensures compatibility with modern PyTorch data flows.
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Transforms raw IDs into contextualized vectors.
        Returns:
            outputs: All hidden states [batch, seq_len, hid_dim].
            hidden/cell: Final state tensors [n_layers, batch, hid_dim].
        """
        # EMBEDDING: Convert token IDs to dense vectors.
        # src: [batch, seq_len] -> embedded: [batch, seq_len, emb_dim]
        embedded = self.dropout(self.embedding(src))

        # LSTM processing: outputs contains the features for EACH token.
        # hidden and cell contain the final states after processing the entire sequence.
        outputs, (hidden, cell) = self.rnn(embedded)

        return outputs, hidden, cell

class Decoder(nn.Module):
    """
    Predicts the next token in the natural language summary.
    Fuses previous tokens, recurrent memory, and the 'Attention' context.
    """

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.attention = BahdanauAttention(hid_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # The input to the RNN is a fusion of the current token and the context vector.
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        # FINAL PROJECTION: Maps concatenated features to the vocabulary space.
        # hid_dim * 2 comes from (RNN output (h) + Weighted Context).
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        Single-step decoding logic.
        """
        # hidden: [n_layers, batch, hid_dim] -> take the top layer's hidden state for attention.
        # cell is not used in attention but is passed through the RNN for state updates.
        # encoder_outputs: [batch, src_len, hid_dim]

        # input: [batch] -> reshape to [batch, 1] for sequential processing.
        input = input.unsqueeze(1)

        # EMBEDDING: Convert the current input token to its vector representation.
        # embedded: [batch, 1, emb_dim]
        embedded = self.dropout(self.embedding(input))

        # ATTENTION STEP: Find relevance of source tokens relative to current hidden state.
        # a: [batch, 1, src_len]
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)

        # CONTEXT VECTOR: Weighted sum of Encoder outputs.
        # weighted: [batch, 1, hid_dim]
        weighted = torch.bmm(a, encoder_outputs)

        # FUSION: Combine Embedding with Context to inform the RNN of the "Source Truth".
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # RECURRENCE: Update internal memory.
        # rnn_input: [batch, 1, emb_dim + hid_dim] -> output: [batch, 1, hid_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # OUTPUT LOGITS: Final prediction based on current, contextual, and recurrent information.
        # prediction: [batch, 1, output_dim] -> squeeze to [batch, output_dim]
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))

        return prediction.squeeze(1), hidden, cell

class Seq2SeqBahdanau(nn.Module):
    """
    Integrates Encoder, Attention, and Decoder into a unified pipeline.
    """

    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, device):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = Decoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
        self.device = device
        self.output_dim = vocab_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Master Forward Pass.
        
        Logic:
        1. Encode the source (Python code).
        2. Iterate over target length (Docstring).
        3. Use 'Teacher Forcing' to stabilize early training stages.
        """

        # Initialize a tensor to hold the decoder's outputs for the entire target sequence.
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.output_dim).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        # Initial token is always <SOS> (Start of Sentence).
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Pass through the attention-augmented decoder.
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output

            # TEACHER FORCING: 
            # Decides if we use the ground truth or the model's own guess for the next step.
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        # outputs: [batch, trg_len, output_dim] - ready for loss calculation against trg.
        return outputs