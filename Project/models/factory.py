import logging
from models.transformer import Seq2SeqTransformer
from models.seq2seq_bahdanau import Seq2SeqBahdanau
from models.seq2seq_dotproduct import Seq2SeqDotProduct

logger = logging.getLogger(__name__)

def get_model_architecture(model_type, device, vocab_size=20000):
    """
    FACTORY ACTUATOR: Configura e istanzia l'architettura neurale richiesta.
    
    PARAMETRI COMUNI:
    - emb_dim (256): Dimensione della proiezione vettoriale dei token.
    - hid_dim (512): Capacità dello spazio latente (Hidden State / d_model).
    - dropout (0.1): Probabilità di dropout per la regolarizzazione.
    
    PARAMETRI LSTM (Temporali):
    - n_layers (2): Numero di strati LSTM sovrapposti.
    
    PARAMETRI TRANSFORMER (Paralleli):
    - nhead (8): Teste di Multi-Head Attention.
    - transformer_layers (6): Numero di blocchi Encoder/Decoder.
    """
    
    params = {
        "emb_dim": 256,
        "hid_dim": 512,
        "n_layers": 2,
        "dropout": 0.1,
        "nhead": 8,
        "transformer_layers": 6
    }

    if model_type == "lstm_bahdanau":
        logger.info(f"🏗️ Factory: Generating LSTM + Bahdanau (Additive Attention)")
        return Seq2SeqBahdanau(
            vocab_size=vocab_size, 
            emb_dim=params["emb_dim"], 
            hid_dim=params["hid_dim"], 
            n_layers=params["n_layers"], 
            dropout=params["dropout"], 
            device=device
        ).to(device)

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

    elif model_type == "transformer":
        logger.info(f"⚡ Factory: Generating Transformer (Heads: {params['nhead']}, Layers: {params['transformer_layers']})")
        return Seq2SeqTransformer(
            vocab_size=vocab_size, 
            d_model=params["hid_dim"], 
            nhead=params["nhead"],
            num_layers=params["transformer_layers"], 
            dropout=params["dropout"]
        ).to(device)
    
    else:
        logger.error(f"❌ Errore critico: Architettura '{model_type}' non supportata.")
        raise ValueError(f"Unknown model type: {model_type}")