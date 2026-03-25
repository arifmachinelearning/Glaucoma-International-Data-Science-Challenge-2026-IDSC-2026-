import os
import random
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
SOURCE_DIR = r"C:\Users\User\PycharmProjects\hillel-yaffe-glaucoma\dataset"         # folder containing GON+ and GON-
OUTPUT_DIR = r"C:\Users\User\PycharmProjects\hillel-yaffe-glaucoma\dataset_new"   # new split dataset folder

CLASSES = ["GON+", "GON-"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =========================
# CHECK RATIOS
# =========================
total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
if abs(total_ratio - 1.0) > 1e-6:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

random.seed(RANDOM_SEED)

source_dir = Path(SOURCE_DIR)
output_dir = Path(OUTPUT_DIR)

# =========================
# CREATE OUTPUT STRUCTURE
# =========================
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

# =========================
# SPLIT FUNCTION
# =========================
def split_files(file_list, train_ratio, val_ratio, test_ratio):
    random.shuffle(file_list)

    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]

    return train_files, val_files, test_files

# =========================
# PROCESS EACH CLASS
# =========================
for cls in CLASSES:
    class_dir = source_dir / cls

    if not class_dir.exists():
        print(f"[WARNING] Class folder not found: {class_dir}")
        continue

    files = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]

    if len(files) == 0:
        print(f"[WARNING] No images found in: {class_dir}")
        continue

    train_files, val_files, test_files = split_files(
        files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    for f in train_files:
        shutil.copy2(f, output_dir / "train" / cls / f.name)

    for f in val_files:
        shutil.copy2(f, output_dir / "val" / cls / f.name)

    for f in test_files:
        shutil.copy2(f, output_dir / "test" / cls / f.name)

    print(f"\nClass: {cls}")
    print(f"  Total : {len(files)}")
    print(f"  Train : {len(train_files)}")
    print(f"  Val   : {len(val_files)}")
    print(f"  Test  : {len(test_files)}")

print("\nDone. Dataset split created at:")
print(output_dir)