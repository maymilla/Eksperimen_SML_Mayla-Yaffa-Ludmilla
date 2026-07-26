import os
import sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_PATH = os.path.join(BASE_DIR, "..", "breast_cancer_raw", "data.csv")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")


def preprocess_data(raw_path: str = DEFAULT_RAW_PATH,
                     output_dir: str = DEFAULT_OUTPUT_DIR,
                     test_size: float = 0.2,
                     random_state: int = 42):
    """
    Melakukan preprocessing otomatis pada dataset Breast Cancer Wisconsin:
      1. Load raw data dari CSV.
      2. Validasi dasar (kolom target ada, tidak ada baris kosong total).
      3. Train-test split (stratified).
      4. Standarisasi fitur numerik dengan StandardScaler.
      5. Simpan hasil (train.csv, test.csv) ke output_dir.

    Returns
    -------
    train_df, test_df : pandas.DataFrame
        Data siap dilatih (sudah discaling, sudah termasuk kolom target).
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset tidak ditemukan di '{raw_path}'.")

    logger.info("Membaca raw data dari %s", raw_path)
    df = pd.read_csv(raw_path)

    if "target" not in df.columns:
        raise ValueError("Kolom 'target' tidak ditemukan pada dataset.")

    if df.empty:
        raise ValueError("Dataset kosong, tidak ada baris untuk diproses.")

    X = df.drop(columns=["target"])
    y = df["target"]

    logger.info("Melakukan train-test split (test_size=%.2f)", test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Melakukan standardisasi fitur dengan StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    train_df = X_train_scaled.copy()
    train_df["target"] = y_train.values

    test_df = X_test_scaled.copy()
    test_df["target"] = y_test.values

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Preprocessing selesai. Train: %s | Test: %s", train_df.shape, test_df.shape)
    logger.info("Data tersimpan di: %s , %s", train_path, test_path)

    return train_df, test_df


def main():
    try:
        preprocess_data()
    except (FileNotFoundError, ValueError) as e:
        logger.error("Preprocessing gagal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
