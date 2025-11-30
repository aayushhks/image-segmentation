import os
import random
import tarfile
import urllib.request
import ssl

# Fix for macOS SSL errors
ssl._create_default_https_context = ssl._create_unverified_context

VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"


def prepare_voc():
    print("Starting Pascal VOC Preparation...")

    # 1. Download
    if not os.path.exists("VOCtrainval_11-May-2012.tar"):
        print("Downloading VOC 2012 (approx 2GB)... this may take a while.")
        try:
            urllib.request.urlretrieve(VOC_URL, "VOCtrainval_11-May-2012.tar")
        except Exception as e:
            print(f"Download failed: {e}")
            return

    # 2. Extract
    if not os.path.exists("VOCdevkit"):
        print("Extracting...")
        with tarfile.open("VOCtrainval_11-May-2012.tar") as tar:
            tar.extractall()

    # 3. Create Splits
    # VOC provides its own splits in ImageSets/Segmentation
    voc_root = os.path.join("VOCdevkit", "VOC2012")
    print(f"Data located at: {voc_root}")

    # We will use standard splits provided by VOC
    train_src = os.path.join(voc_root, "ImageSets", "Segmentation", "train.txt")
    val_src = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")

    # Copy to our root as voc_train_list.txt
    with open(train_src, 'r') as f:
        lines = f.read()
    with open('voc_train_list.txt', 'w') as f:
        f.write(lines)

    with open(val_src, 'r') as f:
        lines = f.read()
    with open('voc_val_list.txt', 'w') as f:
        f.write(lines)

    print("Pascal VOC setup complete.")
    print(f"Train list: voc_train_list.txt")
    print(f"Val list:   voc_val_list.txt")


if __name__ == "__main__":
    prepare_voc()