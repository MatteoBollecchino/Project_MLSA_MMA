import torch
import logging
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

logger = logging.getLogger(__name__)

def get_model_architecture(model_type, device, vocab_size=10000, config=None):
    """
    Architecture Factory: Gestisce la creazione dinamica dei modelli.
    I modelli non ancora implementati sollevano NotImplementedError per 
    permettere il salto automatico nel C2Orchestrator.
    """
    
    # Configurazione Standard (Hardware Abstraction Layer)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    if model_type == "lstm_attention":
        logger.info("Factory: Assemblaggio LSTM con Attention (Bahdanau).")
        
        # Iniezione delle Dipendenze (DI)
        attn = Attention(HID_DIM)
        enc = Encoder(vocab_size, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(vocab_size, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
        
        return Seq2Seq(enc, dec, device).to(device)

    elif model_type == "lstm_simple":
        # Placeholder per la Baseline senza meccanismo di attenzione
        logger.info("Factory: Richiesta Baseline LSTM Simple rilevata.")
        # In futuro: dec = DecoderSimple(vocab_size, ...)
        raise NotImplementedError("Architettura 'lstm_simple' non ancora integrata nel catalogo.")

    elif model_type == "transformer":
        # Placeholder per l'architettura SOTA (State-Of-The-Art)
        logger.info("Factory: Richiesta Transformer Challenger rilevata.")
        # In futuro: model = TransformerCustom(vocab_size, ...)
        raise NotImplementedError("Architettura 'transformer' attualmente in fase di R&D.")

    else:
        logger.error(f"Factory Error: Il tag '{model_type}' non corrisponde a nessuna specifica nota.")
        raise ValueError(f"Modello '{model_type}' non presente nel catalogo della Factory.")