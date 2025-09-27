# src/preprocess.py
"""
Preprocess script:
- Reads data/raw/train.csv
- Adds absolute filepaths for images
- Shuffles and creates a train/validation split
- Saves CSVs to data/processed/
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
TRAIN_IM_DIR = os.path.join(RAW_DIR, "train_images")
TEST_IM_DIR = os.path.join(RAW_DIR, "test_images")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def prepare():
    train_csv = os.path.join(RAW_DIR, "train.csv")
    test_csv = os.path.join(RAW_DIR, "test.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"{train_csv} not found. Place train.csv in data/raw/")
    df = pd.read_csv(train_csv)  # columns: id_code, diagnosis
    # Build full filepath
    df['filepath'] = df['id_code'].apply(lambda x: os.path.join(TRAIN_IM_DIR, f"{x}.png"))
    # Some APTOS images are .png/.jpg; support both by checking existence
    def find_file(fp):
        if os.path.exists(fp):
            return fp
        # try .jpg
        fp_jpg = fp[:-4] + ".jpg"
        if os.path.exists(fp_jpg):
            return fp_jpg
        return fp  # keep original; downstream will error if missing
    df['filepath'] = df['filepath'].apply(find_file)

    # Shuffle and split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['diagnosis'])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_out = os.path.join(PROCESSED_DIR, "train_labels.csv")
    val_out = os.path.join(PROCESSED_DIR, "val_labels.csv")
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    print("Saved:", train_out, val_out)

    # Save test list if present
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        test_df['filepath'] = test_df['id_code'].apply(lambda x: os.path.join(TEST_IM_DIR, f"{x}.png"))
        test_df['filepath'] = test_df['filepath'].apply(find_file)
        test_out = os.path.join(PROCESSED_DIR, "test_files.csv")
        test_df.to_csv(test_out, index=False)
        print("Saved:", test_out)

if __name__ == "__main__":
    prepare()
