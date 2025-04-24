import requests
import os
import sys

HPO_JSON_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-03-03/hp.json"
DATA_DIR = "data"
HPO_FILE_PATH = os.path.join(DATA_DIR, "hp.json")

def download_hpo_json():
    """Downloads hp.json if it doesn't exist."""
    if not os.path.exists(HPO_FILE_PATH):
        print(f"Downloading {HPO_JSON_URL}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            response = requests.get(HPO_JSON_URL, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            # Show download progress
            downloaded = 0
            with open(HPO_FILE_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    f.write(chunk)
                    # Simple progress indicator
                    sys.stdout.write(f"\rDownloaded {downloaded / 1024 / 1024:.1f} MB of {total_size / 1024 / 1024:.1f} MB")
                    sys.stdout.flush()
            print(f"\nSaved HPO data to {HPO_FILE_PATH}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading HPO file: {e}")
            sys.exit(1)  # Exit if download fails
    else:
        print(f"{HPO_FILE_PATH} already exists.")

if __name__ == "__main__":
    download_hpo_json()