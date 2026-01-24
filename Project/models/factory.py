import torch
import logging
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

logger = logging.getLogger(__name__)

def get_model_architecture(model_type, device, vocab_size=10000, config=None):
    """
    Architecture Factory: Applica il principio di parsimonia.
    """
    # Iperparametri calibrati per prevenire l'overfitting su subset medi (50k-100k)
    ENC_EMB_DIM = 128   # Ridotto da 256
    DEC_EMB_DIM = 128   # Ridotto da 256
    HID_DIM = 256       # Ridotto da 512 per forzare la compressione
    N_LAYERS = 2
    ENC_DROPOUT = 0.5   # Dropout aggressivo per disturbare la memorizzazione
    DEC_DROPOUT = 0.5

    if model_type == "lstm_attention":
        logger.info(f"Factory: Generazione LSTM-ATTN (Bottleneck: {HID_DIM} units)")
        
        attn = Attention(HID_DIM)
        enc = Encoder(vocab_size, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(vocab_size, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
        
        return Seq2Seq(enc, dec, device).to(device)

    elif model_type == "transformer":
        # Placeholder per evoluzione futura
        raise NotImplementedError("Architettura Transformer in fase di progettazione.")

    else:
        raise ValueError(f"Tag '{model_type}' non riconosciuto.")