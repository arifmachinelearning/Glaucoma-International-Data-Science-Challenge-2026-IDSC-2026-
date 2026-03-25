import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://physionet.org/files/hillel-yaffe-glaucoma-dataset/1.0.0/"
SAVE_DIR = r"C:\Users\User\Downloads\hillel_yaffe_glaucoma_dataset"

session = requests.Session()

def download_file(file_url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with session.get(file_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"Downloaded: {save_path}")

def crawl_directory(url, local_dir):
    os.makedirs(local_dir, exist_ok=True)

    response = session.get(url, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")

        if not href or href in ("../", "/"):
            continue

        full_url = urljoin(url, href)

        if href.endswith("/"):
            subfolder = os.path.join(local_dir, href.strip("/"))
            crawl_directory(full_url, subfolder)
        else:
            local_path = os.path.join(local_dir, href)
            if os.path.exists(local_path):
                print(f"Skipped: {local_path}")
            else:
                download_file(full_url, local_path)

if __name__ == "__main__":
    crawl_directory(BASE_URL, SAVE_DIR)
    print("Done.")