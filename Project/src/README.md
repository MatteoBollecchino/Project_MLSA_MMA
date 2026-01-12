# Cartella `src`

Questa cartella è pensata per contenere il codice sorgente riutilizzabile e le logiche di base che non rientrano specificamente nelle categorie `data`, `models` o `scripts`.

Può essere vista come una libreria interna del progetto.

## Implementazione Prevista

Il contenuto di questa cartella emergerà durante lo sviluppo, ma potrebbe includere:

1.  **Funzioni di Utilità (`utils.py`):**
    *   Funzioni generiche che possono essere utili in più punti del progetto.
    *   Esempi:
        *   Funzioni per calcolare il tempo di esecuzione.
        *   Funzioni per impostare il seed per la riproducibilità degli esperimenti (`set_seed`).
        *   Classi per il logging custom.

2.  **Gestione della Configurazione (`config.py`):**
    *   Un modulo per caricare e gestire i parametri di configurazione del progetto (es. da file YAML o JSON).
    *   Questo centralizza iperparametri come `learning_rate`, `batch_size`, `hidden_size`, percorsi dei file, ecc., rendendo gli esperimenti più facili da tracciare e modificare.

3.  **Logica di Calcolo Metriche (`metrics.py`):**
    *   Wrapper o implementazioni custom per le metriche di valutazione come BLEU e ROUGE, se la logica diventa complessa e si vuole disaccoppiarla dallo script `evaluate.py`.

4.  **Componenti Condivisi del Modello:**
    *   Se si sviluppano architetture più complesse (es. Transformer), qui si potrebbero definire i blocchi base riutilizzabili, come lo strato di `Multi-Head Attention` o `Position-wise Feed-Forward Network`, se non si vogliono tenere nei file principali dei modelli.
