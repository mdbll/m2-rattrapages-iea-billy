import io
import os
import urllib.request
import zipfile

DATASET_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATASET_DIR = os.path.join(DATA_DIR, "UCI HAR Dataset")


def download_dataset():
    if os.path.exists(DATASET_DIR):
        print("Dataset already downloaded in", DATASET_DIR)
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading UCI HAR dataset (~60 MB)...")
    with urllib.request.urlopen(DATASET_URL) as response:
        outer_zip = zipfile.ZipFile(io.BytesIO(response.read()))

    inner_zip_bytes = outer_zip.read("UCI HAR Dataset.zip")
    with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zip:
        inner_zip.extractall(DATA_DIR)

    print("Done, dataset extracted in", DATASET_DIR)


if __name__ == "__main__":
    download_dataset()
