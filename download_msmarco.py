'''
import os
import requests
import tarfile

class MSMARCODownloader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # MSMARCO URLs
        self.urls = {
            'collection': 'https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz',
            'queries': 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz',
            'qrels': 'https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv',
            'qrels_dev': 'https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv',
        }

    def download_file(self, url, filename):
        file_path = os.path.join(self.data_dir, filename)
        if os.path.exists(file_path):
            print(f"[✓] Already downloaded: {filename}")
            return file_path

        print(f"[↓] Downloading: {filename}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"[✓] Download complete: {filename}")
            return file_path

        except requests.RequestException as e:
            print(f"[✗] Download failed for {filename}: {e}")
            return None

    def extract_tar_gz(self, file_path):
        print(f"[⇩] Extracting: {file_path}")
        try:
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.data_dir)
            print(f"[✓] Extraction complete: {file_path}")
        except tarfile.TarError as e:
            print(f"[✗] Extraction failed: {e}")

    def verify_download(self, file_path):
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        print(f"[✔] Verified: {file_path} ({size / 1024:.2f} KB)" if exists else f"[✗] Missing: {file_path}")

    def run(self):
        for name, url in self.urls.items():
            filename = os.path.basename(url)
            file_path = self.download_file(url, filename)

            if file_path and filename.endswith('.tar.gz'):
                self.extract_tar_gz(file_path)

            if file_path:
                self.verify_download(file_path)


if __name__ == '__main__':
    downloader = MSMARCODownloader()
    downloader.run()
'''

import os
from datasets import load_dataset
import json

class MSMARCODownloader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_msmarco(self):
        print("[↓] Loading MSMARCO v1.1 from Hugging Face...")
        dataset = load_dataset("ms_marco", "v1.1")

        for split in dataset:
            save_path = os.path.join(self.data_dir, f"msmarco_{split}.jsonl")
            print(f"[⇩] Saving {split} split to {save_path}...")
            with open(save_path, 'w', encoding='utf-8') as f:
                for example in dataset[split]:
                    json.dump(example, f)
                    f.write('\n')
            print(f"[✓] Saved {split} split.")

        print("[✔] All splits downloaded and saved.")

    def run(self):
        self.download_msmarco()

if __name__ == '__main__':
    downloader = MSMARCODownloader()
    downloader.run()

 
