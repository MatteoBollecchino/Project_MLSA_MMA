import os
import requests
import zipfile
import io

# DOWNLOAD AND EXTRACTION OF CODESEARCHNET ROBUST DATASET
def download_codesearchnet_robust(output_dir="Project/Datasets"):

    # Mirror on Hugging Face Assets - the most stable at the moment
    url = "https://huggingface.co/datasets/code_search_net/resolve/main/data/python.zip"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Attempting acquisition via Hugging Face CDN ---")
    print(f"URL: {url}")
    print("Download in progress (about 900MB). Grab a coffee...")

    try:
        # Specifying the User-agent to avoid bot blocks
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Use BytesIO to avoid writing the entire zip file to disk before extracting
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            print("Download completed. Extracting...")
            zip_ref.extractall(output_dir)
        
        print(f"SUCCESS: Dataset extracted to {output_dir}")
        print("Detected structure: " + str(os.listdir(output_dir)))

    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        print("\n--- MANUAL SOLUTION ---")
        print(f"If the script fails, manually download from here: {url}")
        print(f"And extract the contents to: {os.path.abspath(output_dir)}")

# MAIN EXECUTION
if __name__ == "__main__":
    download_codesearchnet_robust()