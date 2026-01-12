# Cartella `scripts`

Questa cartella conterrà gli script eseguibili per lanciare le operazioni principali del progetto: addestramento, valutazione e inferenza.

## Implementazione Prevista

1.  **Script di Addestramento (`train.py`):**
    *   **Responsabilità:** Gestire l'intero ciclo di vita dell'addestramento del modello.
    *   **Logica:**
        1.  Caricare i dati di training e validazione usando i `DataLoader` definiti in `data/`.
        2.  Inizializzare il modello (es. `Seq2Seq`), l'ottimizzatore (es. `Adam`) e la funzione di loss (`NLLLoss` o `CrossEntropyLoss` con `ignore_index` per il padding).
        3.  Eseguire il ciclo di training per un numero definito di `epoch`.
        4.  Per ogni `epoch`:
            *   Iterare sui batch di training, calcolare la loss, eseguire backpropagation e aggiornare i pesi.
            *   Eseguire un ciclo di validazione per monitorare le performance su dati non visti.
            *   Salvare i "checkpoint" del modello, specialmente quello con la migliore performance sul validation set.
        5.  (Opzionale) Loggare le metriche (es. loss, perplessità) usando `TensorBoard` o tool simili.

2.  **Script di Valutazione (`evaluate.py`):**
    *   **Responsabilità:** Valutare quantitativamente un modello addestrato sul test set.
    *   **Logica:**
        1.  Caricare il `DataLoader` del test set.
        2.  Caricare il modello migliore salvato dallo script di training.
        3.  Iterare sul test set in modalità `torch.no_grad()`.
        4.  Per ogni coppia (codice, riassunto reale), generare un riassunto predetto dal modello.
        5.  Calcolare e riportare le metriche di valutazione richieste: **BLEU** e **ROUGE**.

3.  **Script di Inferenza (`summarize.py`):**
    *   **Responsabilità:** Usare il modello addestrato per generare un riassunto di un nuovo snippet di codice fornito dall'utente.
    *   **Logica:**
        1.  Accettare uno snippet di codice come argomento da linea di comando.
        2.  Caricare il modello addestrato, i vocabolari e i tokenizer.
        3.  Applicare lo stesso preprocessing dell'addestramento al codice di input.
        4.  Eseguire il forward pass del modello (inferenza) per generare la sequenza di indici dei token del riassunto.
        5.  Decodificare la sequenza di indici in una stringa di testo leggibile.
        6.  Stampare a schermo il riassunto generato.
