import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BahdanauAttention(nn.Module):
    """
    Implements Bahdanau (Additive) Attention Mechanism.
    """

    def __init__(self, hid_dim):
        """
        Initializes the Bahdanau Attention mechanism.
        """

        super().__init__()

        # Linear layer for attention score computation
        self.attn = nn.Linear(hid_dim * 2, hid_dim)

        # Final linear layer to produce attention weights
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Computes attention weights.
        """

        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        src_len = encoder_outputs.shape[1]

        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Compute attention scores
        attention = self.v(energy).squeeze(2)

        # Apply softmax to get attention weights
        return F.softmax(attention, dim=1)

class Encoder(nn.Module):
    """
    Encodes the source sequence.
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        Initializes the Encoder.
        """
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # Recurrent layer (LSTM)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Forward pass through the encoder.
        """

        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))

        # embedded: [batch_size, src_len, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs: [batch_size, src_len, hid_dim]
        # hidden, cell: [n_layers, batch_size, hid_dim]
        return outputs, hidden, cell

class Decoder(nn.Module):
    """
    Decodes the target sequence using attention.
    """

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        Initializes the Decoder with Bahdanau Attention.
        """
        super().__init__()

        # Output dimension (vocabulary size)
        self.output_dim = output_dim

        # Attention mechanism
        self.attention = BahdanauAttention(hid_dim)

        # Embedding layer
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # Recurrent layer (LSTM)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        # Final output layer
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        Forward pass through the decoder.
        """
        # input: [batch_size]
        # hidden, cell: [n_layers, batch_size, hid_dim]
        input = input.unsqueeze(1)

        # input: [batch_size, 1]
        # embedded: [batch_size, 1, emb_dim]
        embedded = self.dropout(self.embedding(input))

        # Calculate attention weights
        # a: [batch_size, 1, src_len]
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)

        # Compute weighted sum of encoder outputs
        # weighted: [batch_size, 1, hid_dim]
        weighted = torch.bmm(a, encoder_outputs)

        # Concatenate embedded input and weighted context
        # rnn_input: [batch_size, 1, emb_dim + hid_dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # Pass through RNN
        # output: [batch_size, 1, hid_dim]
        # hidden, cell: [n_layers, batch_size, hid_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Generate predictions
        # prediction: [batch_size, output_dim]
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))

        return prediction.squeeze(1), hidden, cell

class Seq2SeqBahdanau(nn.Module):
    """
    Sequence-to-Sequence model with Bahdanau Attention.
    """

    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, device):
        """
        Initializes the Seq2Seq model with Bahdanau Attention.
        """
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = Decoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
        self.device = device
        self.output_dim = vocab_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass through the Seq2Seq model.
        """

        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size, trg_len = trg.shape[0], trg.shape[1]

        # Prepare tensor to hold outputs
        # outputs: [batch_size, trg_len, output_dim]
        outputs = torch.zeros(batch_size, trg_len, self.output_dim).to(self.device)

        # Encode the source sequence
        # encoder_outputs: [batch_size, src_len, hid_dim]
        # hidden, cell: [n_layers, batch_size, hid_dim]
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input to the decoder is the <sos> tokens
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Decode one time step
            # output: [batch_size, output_dim]
            # hidden, cell: [n_layers, batch_size, hid_dim]
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)

            # Store output
            outputs[:, t] = output

            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token
            # If teacher forcing, use actual next token as next input
            input = trg[:, t] if teacher_force else output.argmax(1)
        return outputs