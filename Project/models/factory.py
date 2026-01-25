import torch
import logging
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from models.transformer import Seq2SeqTransformer # Assicurati che il file esista

# Logger Configuration
logger = logging.getLogger(__name__)

# --- [ARCHITECTURE FACTORY FUNCTION] ---
def get_model_architecture(model_type, device, vocab_size=50000, config=None):
    """
    Architecture Factory: Generates instances of models based on tags.
    Implements the Abstract Factory pattern to isolate topology from hyperparameters.
    """
    
    # --- DEFAULT CONFIGURATION (Override possible via config) ---
    # Using a dictionary to maintain orthogonality
    params = {
        "emb_dim": 256,
        "hid_dim": 512,
        "n_layers": 2,
        "dropout": 0.5,
        "nhead": 8,             # Only for Transformer
        "transformer_layers": 6 # Only for Transformer
    }
    
    # If we pass a config object (from argparse), override defaults
    if config:
        # Example: params["hid_dim"] = getattr(config, "hid_dim", params["hid_dim"])
        pass

    if model_type == "lstm_attention":
        logger.info(f"🏗️ Factory: Generating LSTM-ATTN | Vocab: {vocab_size} | Hidden: {params['hid_dim']}")
        
        # Attention mechanism
        attn = Attention(params["hid_dim"])

        # Encoder-Decoder with Attention
        enc = Encoder(vocab_size, params["emb_dim"], params["hid_dim"], params["n_layers"], params["dropout"])
        dec = Decoder(vocab_size, params["emb_dim"], params["hid_dim"], params["n_layers"], params["dropout"], attn)
        
        # Seq2Seq Model
        return Seq2Seq(enc, dec, device).to(device)

    elif model_type == "transformer":
        logger.info(f"⚡ Factory: Generating Transformer | Heads: {params['nhead']} | Layers: {params['transformer_layers']}")
        
        # Implementation of Seq2SeqTransformer based on standard PyTorch
        model = Seq2SeqTransformer(
            vocab_size=vocab_size,
            d_model=params["hid_dim"], # Often hid_dim and d_model coincide in base Transformers
            nhead=params["nhead"],
            num_layers=params["transformer_layers"],
            dropout=0.1 # Transformers prefer less aggressive dropout than LSTMs
        )
        
        # Transformer Model
        return model.to(device)

    else:
        logger.error(f"❌ Factory Failure: Tag '{model_type}' not recognized.")
        raise ValueError(f"Tag '{model_type}' not supported by the factory.")