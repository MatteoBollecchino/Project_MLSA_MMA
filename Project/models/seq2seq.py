import torch
import torch.nn as nn
import random

# --- [SEQ2SEQ MODULE] ---
class Seq2Seq(nn.Module):

    # Initialization of the Seq2Seq model
    def __init__(self, encoder, decoder, device):
        """ Initializes the Seq2Seq model with encoder, decoder, and device. """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    # Forward pass of the Seq2Seq model
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """ Forward pass through the Seq2Seq model. """
        
        # src: [batch_size, src_len], trg: [batch_size, trg_len]

        # Get the batch size from the source sequence
        batch_size = src.shape[0]

        # Get the target sequence length
        trg_len = trg.shape[1]

        # Vocabulary size of the target language
        trg_vocab_size = self.decoder.output_dim
        
        # Initialize the outputs tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode the source sequence
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # The first input to the decoder is always the <SOS> token
        input = trg[:, 0]
        
        # Decode each token step by step
        for t in range(1, trg_len):

            # Get the output from the decoder
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            
            # Teacher Forcing logic: decide whether to "feed" the model with the truth
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token from our output
            top1 = output.argmax(1)

            # If teacher forcing, use actual next token as next input; else use predicted token
            input = trg[:, t] if teacher_force else top1
            
        return outputs