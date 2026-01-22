import os
import requests
import zipfile
import io

def download_codesearchnet_robust(output_dir="Project/Datasets"):
    # Mirror su Hugging Face Assets - il più stabile al momento
    url = "https://huggingface.co/datasets/code_search_net/resolve/main/data/python.zip"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Tentativo di acquisizione via Hugging Face CDN ---")
    print(f"URL: {url}")
    print("Download in corso (circa 900MB). Prendi un caffè...")

    try:
        # User-agent per evitare blocchi bot
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Usiamo BytesIO per non scrivere il file zip intero su disco prima di estrarlo
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            print("Download completato. Estrazione in corso...")
            zip_ref.extractall(output_dir)
        
        print(f"SUCCESS: Dataset estratto in {output_dir}")
        print("Struttura rilevata: " + str(os.listdir(output_dir)))

    except Exception as e:
        print(f"FALLIMENTO CRITICO: {e}")
        print("\n--- SOLUZIONE MANUALE ---")
        print(f"Se lo script fallisce, scarica manualmente da qui: {url}")
        print(f"E scompatta il contenuto in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    download_codesearchnet_robust()