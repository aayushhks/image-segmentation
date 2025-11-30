import os
import random
import tarfile
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def download_and_extract(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    print(f"Extracting {filename}...")
    with tarfile.open(filename, "r") as tar:
        tar.extractall()


def create_splits(split=0.8):
    print("Creating train/val splits...")
    # Annotations typically extract into a folder named "annotations"
    list_path = os.path.join("annotations", "trainval.txt")

    if not os.path.exists(list_path):
        print(f"Error: {list_path} not found. Extraction might have failed.")
        return

    with open(list_path, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_idx = int(len(lines) * split)

    with open('train_list.txt', 'w') as f: f.writelines(lines[:split_idx])
    with open('val_list.txt', 'w') as f: f.writelines(lines[split_idx:])
    print(f"Splits created: {split_idx} training, {len(lines) - split_idx} validation.")


if __name__ == "__main__":
    # 1. Download Data
    # Oxford-IIIT Pet Dataset URLs
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    download_and_extract(images_url, "images.tar.gz")
    download_and_extract(annotations_url, "annotations.tar.gz")

    # 2. Create Split Files
    create_splits()