# Cartella `data`

Questa cartella conterrà tutti gli script e i file relativi alla gestione del dataset.

## Implementazione Prevista

1.  **Script di Acquisizione (`download_dataset.py`):**
    *   Uno script per scaricare e scompattare il dataset (es. CodeSearchNet) nella cartella `Project/Datasets`.
    *   Dovrebbe permettere di scaricare anche solo un subset per testare rapidamente la pipeline.

2.  **Script di Preprocessing (`preprocess.py`):**
    *   **Tokenizzazione:** Questo script si occuperà di convertire sia il codice sorgente che i riassunti testuali in sequenze di token. Verranno usati tokenizer specifici (es. BPE) per ogni "linguaggio" (codice e testo).
    *   **Costruzione Vocabolario:** Creerà e salverà i vocabolari (mappe `token -> id`) per il codice e per i riassunti, includendo token speciali come `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`.

3.  **Classe `Dataset` PyTorch (`dataset.py`):**
    *   Implementerà una classe `torch.utils.data.Dataset` custom.
    *   Questa classe leggerà i dati pre-processati, li convertirà in tensori numerici e li servirà al `DataLoader`.

4.  **Funzione di Utilità per `DataLoader`:**
    *   Una funzione helper che, data una classe `Dataset`, istanzi e restituisca i `DataLoader` per training, validazione e test.
    *   Gestirà la logica per il padding dei batch (tramite l'argomento `collate_fn`) in modo che tutte le sequenze in un batch abbiano la stessa lunghezza.
