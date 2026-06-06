import requests
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.logger import logger
from src.utils.exception import CustomException


def download_file(url, dest_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created directory: {dest_folder}")

    # Get the file name from the URL
    filename = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)

    # Download the file
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes (e.g., 404)

        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {filename} to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        raise  # Re-raise the exception for the caller to handle if needed

    return file_path


def datasplit(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits a DataFrame into train and test sets.

    Args:
        df (pd.DataFrame): DataFrame with columns 'vector' (features) and 'sentiment' (labels).
        test_size (float): Fraction of data used for testing. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, X_test, y_train, y_test
    """
    try:
        logger.info("Starting train-test split")

        required_cols = ["vector", "sentiment"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

        # --- Convert any stringified vectors back to numeric arrays ---
        def parse_vector(v):
            if isinstance(v, str):
                try:
                    # handle cases like '[-0.62 0.13 ...]' or '[-0.62, 0.13, ...]'
                    cleaned = (
                        v.replace("\n", " ")
                        .replace("[", "")
                        .replace("]", "")
                        .replace(",", " ")
                    )
                    return np.fromstring(cleaned, sep=" ")
                except Exception:
                    # fallback: try literal_eval for list-like strings
                    import ast

                    return np.array(ast.literal_eval(v), dtype=float)
            return np.array(v, dtype=float)

        df["vector"] = df["vector"].apply(parse_vector)

        # --- features & labels ---
        X = np.vstack(df["vector"].values)
        encoder = LabelEncoder()
        y = encoder.fit_transform(df["sentiment"])

        logger.info(f"Dataset shape: X={X.shape}, y={len(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(
            f"Train-test split complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}"
        )
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.exception("Error during train-test split")
        raise CustomException(e)


# def sequence_split(df, config):
#     """
#     Tokenizes and splits dataset into train/test for sequence-based models.

#     Args:
#         df (pd.DataFrame): DataFrame containing 'clean_text' and 'sentiment'
#         config (dict): Config dictionary (from YAML)

#     Returns:
#         X_train_seq, X_test_seq, y_train, y_test, tokenizer
#     """
#     try:
#         logger.info("Starting sequence-based data preparation")

#         required_cols = ["clean_text", "sentiment"]
#         missing = [c for c in required_cols if c not in df.columns]
#         if missing:
#             raise ValueError(f"Missing required columns: {missing}")

#         data_cfg = config.get("clean_data", {})
#         text_cfg = config.get("text_preprocessing", {})

#         test_size = data_cfg.get("test_size", 0.2)
#         random_state = data_cfg.get("random_state", 42)
#         num_words = text_cfg.get("num_words", 10000)
#         max_len = text_cfg.get("max_len", 200)

#         # --- Label encoding ---
#         encoder = LabelEncoder()
#         y = encoder.fit_transform(df["sentiment"])

#         # --- Tokenization ---
#         tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
#         tokenizer.fit_on_texts(df["clean_text"])
#         sequences = tokenizer.texts_to_sequences(df["clean_text"])
#         padded_sequences = pad_sequences(
#             sequences, maxlen=max_len, padding="post", truncating="post"
#         )

#         # --- Train-test split ---
#         X_train, X_test, y_train, y_test = train_test_split(
#             padded_sequences, y, test_size=test_size, random_state=random_state
#         )

#         # --- Save tokenizer ---
#         artifacts_dir = os.path.join("artifacts", "models")
#         os.makedirs(artifacts_dir, exist_ok=True)
#         tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pkl")

#         with open(tokenizer_path, "wb") as f:
#             pickle.dump(tokenizer, f)
#         logger.info(f"Tokenizer saved at: {tokenizer_path}")

#         logger.info(
#             f"Sequence split complete: Train={X_train.shape}, Test={X_test.shape}"
#         )
#         return X_train, X_test, y_train, y_test, tokenizer

#     except Exception as e:
#         logger.exception("Error during sequence split")
#         raise CustomException(e)
