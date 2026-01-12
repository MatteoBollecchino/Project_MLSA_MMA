# Cartella `models`

Questa cartella conterrà le definizioni di tutti i moduli PyTorch che compongono l'architettura del modello di deep learning.

L'architettura di partenza sarà un modello **Sequence-to-Sequence (Seq2Seq)** con RNN (LSTM o GRU).

## Implementazione Prevista

1.  **Encoder (`encoder.py`):**
    *   **Classe `Encoder(torch.nn.Module)`:**
    *   **Componenti:**
        *   `torch.nn.Embedding`: Per convertire gli indici dei token del codice sorgente in vettori densi (embedding).
        *   `torch.nn.LSTM` o `torch.nn.GRU`: Per processare la sequenza di embedding e produrre un "vettore di contesto" (l'ultimo stato nascosto dell'RNN).
    *   **Input:** Sequenza di token del codice sorgente.
    *   **Output:** Output dell'RNN e stati nascosti/cella (il vettore di contesto).

2.  **Decoder (`decoder.py`):**
    *   **Classe `Decoder(torch.nn.Module)`:**
    *   **Componenti:**
        *   `torch.nn.Embedding`: Per i token del riassunto.
        *   `torch.nn.LSTM` o `torch.nn.GRU`: Per generare la sequenza di output token per token.
        *   `torch.nn.Linear`: Uno strato lineare finale per mappare lo stato nascosto dell'RNN a una distribuzione di probabilità sul vocabolario dei riassunti.
        *   `torch.nn.LogSoftmax`: Per ottenere le probabilità logaritmiche.
    *   **Input:** Il vettore di contesto dall'encoder e il token precedente della sequenza generata.
    *   **Output:** La predizione del token successivo nella sequenza.

3.  **Modello Principale (`seq2seq.py`):**
    *   **Classe `Seq2Seq(torch.nn.Module)`:**
    *   **Componenti:**
        *   Un'istanza dell'Encoder.
        *   Un'istanza del Decoder.
    *   **Logica:**
        *   Orchestra il flusso dei dati: prende la sequenza di input, la passa all'encoder, usa il vettore di contesto per inizializzare il decoder.
        *   Implementa il ciclo di generazione del decoder.
        *   Gestisce il "teacher forcing": durante il training, decide se alimentare il decoder con il token predetto o con il token reale della sequenza target, per stabilizzare l'addestramento.
