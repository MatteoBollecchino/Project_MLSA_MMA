import logging
from models.transformer import Seq2SeqTransformer
from models.seq2seq_bahdanau import Seq2SeqBahdanau
from models.seq2seq_dotproduct import Seq2SeqDotProduct

logger = logging.getLogger(__name__)

def get_model_architecture(model_type, device, vocab_size=20000):
    """
    FACTORY ACTUATOR: Configures and instantiates the requested neural architecture.
    
    COMMON PARAMETERS:
    - emb_dim (256): Dimension of the token vector projection.
    - hid_dim (512): Capacity of the latent space (Hidden State / d_model).
    - dropout (0.1): Dropout probability for regularization.
    
    LSTM PARAMETERS (Sequential):
    - n_layers (2): Number of stacked LSTM layers.
    
    TRANSFORMER PARAMETERS (Parallel):
    - nhead (8): Number of Multi-Head Attention heads.
    - transformer_layers (6): Number of Encoder/Decoder blocks.
    """
    
    params = {
        "emb_dim": 256,
        "hid_dim": 512,
        "n_layers": 2,
        "dropout": 0.1,
        "nhead": 8,
        "transformer_layers": 6
    }

    # --- [ARCHITECTURE 1: LSTM with ADDITIVE ATTENTION (Bahdanau)] ---
    # This architecture uses a neural network (tanh layer) to calculate alignment scores.
    # It is computationally expensive but very robust for varying sequence lengths.
    if model_type == "lstm_bahdanau":
        logger.info(f"🏗️ Factory: Generating LSTM + Bahdanau (Additive Attention)")
        return Seq2SeqBahdanau(
            vocab_size=vocab_size, 
            emb_dim=params["emb_dim"], 
            hid_dim=params["hid_dim"], 
            n_layers=params["n_layers"], 
            dropout=params["dropout"], 
            device=device
        ).to(device) # Crucial: Moves the initialized model to GPU/CPU VRAM immediately.

    # --- [ARCHITECTURE 2: LSTM with MULTIPLICATIVE ATTENTION (Dot-Product)] ---
    # This architecture uses the Dot Product between vectors to calculate alignment.
    # It is faster and more memory-efficient than Bahdanau (geometric approach),
    # but theoretically slightly less expressive without scaling.
    elif model_type == "lstm_dotproduct":
        logger.info(f"🏗️ Factory: Generating LSTM + DotProduct (Geometric Attention)")
        return Seq2SeqDotProduct(
            vocab_size=vocab_size, 
            emb_dim=params["emb_dim"], 
            hid_dim=params["hid_dim"], 
            n_layers=params["n_layers"], 
            dropout=params["dropout"], 
            device=device
        ).to(device)

    # --- [ARCHITECTURE 3: TRANSFORMER (Attention Is All You Need)] ---
    # State-of-the-art architecture based entirely on Self-Attention mechanisms.
    # Unlike LSTMs, it processes the entire sequence in parallel (non-sequential),
    # allowing for massive parallelization on GPUs and better handling of long-range dependencies.
    elif model_type == "transformer":
        logger.info(f"⚡ Factory: Generating Transformer (Heads: {params['nhead']}, Layers: {params['transformer_layers']})")
        return Seq2SeqTransformer(
            vocab_size=vocab_size, 
            d_model=params["hid_dim"],  # Maps 'hid_dim' to Transformer's 'd_model'
            nhead=params["nhead"],
            num_layers=params["transformer_layers"], 
            dropout=params["dropout"]
        ).to(device)
    
    # --- [ERROR HANDLING] ---
    # Failsafe for incorrect configuration strings.
    else:
        logger.error(f"❌ Errore critico: Architettura '{model_type}' non supportata.")
        raise ValueError(f"Unknown model type: {model_type}")