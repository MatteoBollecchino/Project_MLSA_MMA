import torch
import torch.nn as nn

# --- [DECODER MODULE] ---
class Decoder(nn.Module):

    # Initialization of the Decoder
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        """ Initializes the Decoder module. """
        super().__init__()
        self.output_dim = output_dim

        # Attention mechanism
        self.attention = attention

        # Layers of the Decoder
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # LSTM layer
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        # Final output layer
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    # Forward pass of the Decoder
    def forward(self, input, hidden, cell, encoder_outputs):
        """ Performs a forward pass through the Decoder. """
        
        # input: [batch_size] -> a single token (the previous one)
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        # Calculate attention weights based on the last hidden state
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)

        # Dot product to obtain the context vector: [batch_size, 1, hid_dim]
        weighted = torch.bmm(a, encoder_outputs)
        
        #Concatenate embedded token with context vector on dim 2
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Concatenate everything for the final projection on the vocabulary
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        return prediction.squeeze(1), hidden, cell