import logging
import torch
from models.transformer import Seq2SeqTransformer
from models.seq2seq_bahdanau import Seq2SeqBahdanau
from models.seq2seq_dotproduct import Seq2SeqDotProduct

logger = logging.getLogger(__name__)

def get_model_architecture(model_type, device, vocab_size=20000):
    """
    FACTORY ACTUATOR v3.5 - Tailored Architectures
    ---------------------------------------------
    Parametri calibrati in base al fenotipo architettonico:
    - LSTM: Mantiene conservatorismo statistico (Dropout 0.1).
    - TRANSFORMER: Regolarizzazione aggressiva (Dropout 0.3) per rompere 
      la memorizzazione delle stringhe fisse (rote learning).
    """

    # --- CONFIGURAZIONE SPECIFICA PER LSTM (Bahdanau & DotProduct) ---
    lstm_params = {
        "emb_dim": 256,
        "hid_dim": 512,
        "n_layers": 2,
        "dropout": 0.1  # Protocollo standard
    }

    # --- CONFIGURAZIONE SPECIFICA PER TRANSFORMER (Attacco all'Overfitting) ---
    trans_params = {
        "emb_dim": 256,
        "hid_dim": 512,   # Rappresenta d_model
        "nhead": 8,
        "layers": 6,
        "dropout": 0.3,   # <--- FIX: Iniezione di entropia per forzare la generalizzazione
        "dim_feedforward": 2048
    }

    # --- BRANCH LOGICO ---

    if model_type == "lstm_bahdanau":
        logger.info(f"🏗️ Factory: Generating LSTM + Bahdanau (Conservative Dropout: {lstm_params['dropout']})")
        return Seq2SeqBahdanau(
            vocab_size=vocab_size, 
            emb_dim=lstm_params["emb_dim"], 
            hid_dim=lstm_params["hid_dim"], 
            n_layers=lstm_params["n_layers"], 
            dropout=lstm_params["dropout"], 
            device=device
        ).to(device)

    elif model_type == "lstm_dotproduct":
        logger.info(f"🏗️ Factory: Generating LSTM + DotProduct (Conservative Dropout: {lstm_params['dropout']})")
        return Seq2SeqDotProduct(
            vocab_size=vocab_size, 
            emb_dim=lstm_params["emb_dim"], 
            hid_dim=lstm_params["hid_dim"], 
            n_layers=lstm_params["n_layers"], 
            dropout=lstm_params["dropout"], 
            device=device
        ).to(device)

    elif model_type == "transformer":
        logger.info(f"⚡ Factory: Generating Transformer (Aggressive Dropout: {trans_params['dropout']})")
        return Seq2SeqTransformer(
            vocab_size=vocab_size, 
            d_model=trans_params["hid_dim"],
            nhead=trans_params["nhead"],
            num_layers=trans_params["layers"], 
            dim_feedforward=trans_params["dim_feedforward"],
            dropout=trans_params["dropout"]
        ).to(device)

    else:
        logger.error(f"❌ Errore critico: Architettura '{model_type}' non supportata.")
        raise ValueError(f"Unknown model type: {model_type}")