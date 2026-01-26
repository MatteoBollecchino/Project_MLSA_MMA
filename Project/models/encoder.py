import torch
import torch.nn as nn

# --- [ENCODER MODULE] ---
class Encoder(nn.Module):

    # Initialization of the Encoder
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        """ Initializes the Encoder with embedding and LSTM layers. """

        super().__init__()

        # Embedding layer: converts token indices to dense vectors
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # LSTM: manages temporal dependencies in the code
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        # Dropout layer: regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
    # Forward pass of the Encoder
    def forward(self, src):
        """ Forward pass through the Encoder. """

        # src: [batch_size, src_len]

        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(src))

        # Pass through the RNN
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch_size, src_len, hid_dim] -> used by Attention
        # hidden/cell: [n_layers, batch_size, hid_dim] -> compressed final state
        return outputs, hidden, cell