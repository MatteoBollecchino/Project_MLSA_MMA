import torch
import logging
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from models.transformer import Seq2SeqTransformer # Assicurati che il file esista

logger = logging.getLogger(__name__)

def get_model_architecture(model_type, device, vocab_size=50000, config=None):
    """
    Architecture Factory: Genera istanze di modelli basate su tag.
    Implementa il pattern Abstract Factory per isolare la topologia dagli iperparametri.
    """
    
    # --- CONFIGURAZIONE DEFAULT (Override possibile tramite config) ---
    # Usiamo un dizionario per mantenere l'ortogonalità
    params = {
        "emb_dim": 256,
        "hid_dim": 512,
        "n_layers": 2,
        "dropout": 0.5,
        "nhead": 8,            # Solo per Transformer
        "transformer_layers": 6 # Solo per Transformer
    }
    
    # Se passiamo un oggetto config (da argparse), facciamo l'override
    if config:
        # Esempio: params["hid_dim"] = getattr(config, "hid_dim", params["hid_dim"])
        pass

    if model_type == "lstm_attention":
        logger.info(f"🏗️ Factory: Generazione LSTM-ATTN | Vocab: {vocab_size} | Hidden: {params['hid_dim']}")
        
        attn = Attention(params["hid_dim"])
        enc = Encoder(vocab_size, params["emb_dim"], params["hid_dim"], params["n_layers"], params["dropout"])
        dec = Decoder(vocab_size, params["emb_dim"], params["hid_dim"], params["n_layers"], params["dropout"], attn)
        
        return Seq2Seq(enc, dec, device).to(device)

    elif model_type == "transformer":
        logger.info(f"⚡ Factory: Generazione Transformer | Heads: {params['nhead']} | Layers: {params['transformer_layers']}")
        
        # Implementazione Seq2SeqTransformer basata su PyTorch standard
        model = Seq2SeqTransformer(
            vocab_size=vocab_size,
            d_model=params["hid_dim"], # Spesso hid_dim e d_model coincidono nei Transformer base
            nhead=params["nhead"],
            num_layers=params["transformer_layers"],
            dropout=0.1 # I Transformer preferiscono dropout meno aggressivi delle LSTM
        )
        
        return model.to(device)

    else:
        logger.error(f"❌ Factory Failure: Tag '{model_type}' non riconosciuto.")
        raise ValueError(f"Tag '{model_type}' non supportato dalla factory.")