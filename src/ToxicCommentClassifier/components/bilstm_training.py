import os
import sys
import re
import urllib.request
from zipfile import ZipFile
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.ToxicCommentClassifier.logger import logger
from src.ToxicCommentClassifier.exception import CustomException
from src.ToxicCommentClassifier.entity import BiLSTMTrainingConfig

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def clean_text(text):
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip() or "unknown"

class BiLSTMTraining:
    def __init__(self, config: BiLSTMTrainingConfig):
        self.config = config

    def _build_model(self, embedding_matrix, vocab_size):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, GlobalMaxPool1D, Dense, Dropout

        model = Sequential([
            Input(shape=(self.config.max_seq_len,)),
            Embedding(
                input_dim=vocab_size,
                output_dim=self.config.embedding_dim,
                weights=[embedding_matrix],
                trainable=False
            ),
            SpatialDropout1D(0.2),
            Bidirectional(
                LSTM(
                    self.config.lstm_units,
                    return_sequences=True,
                    dropout=0.2
                )
            ),
            GlobalMaxPool1D(),
            Dense(64, activation="relu"),
            Dropout(self.config.dropout),
            Dense(6, activation="sigmoid")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(
                    multi_label=True,
                    num_labels=6,
                    name="auc"
                )
            ]
        )
        return model

    def initiate_bilstm_training(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

            # --- 1. Load and clean data ---
            logger.info(f"Loading training data from {self.config.train_data_path}")
            train_df = pd.read_csv(self.config.train_data_path)
            train_df["text_clean"] = train_df["comment_text"].map(clean_text)

            train_df["is_toxic"] = train_df[LABELS].any(axis=1).astype(int)
            train_indices, val_indices = train_test_split(
                train_df.index,
                test_size=0.10,
                random_state=42,
                stratify=train_df["is_toxic"]
            )
            train_split = train_df.loc[train_indices]
            val_split = train_df.loc[val_indices]

            # --- 2. Keras Tokenizer (word → integer mapping) ---
            logger.info(f"Fitting Keras Tokenizer (vocab_size={self.config.max_vocab_size})...")
            keras_tokenizer = Tokenizer(
                num_words=self.config.max_vocab_size,
                oov_token="<OOV>"
            )
            keras_tokenizer.fit_on_texts(train_split["text_clean"])
            
            X_train = keras_tokenizer.texts_to_sequences(train_split["text_clean"])
            X_val = keras_tokenizer.texts_to_sequences(val_split["text_clean"])

            X_train = pad_sequences(
                X_train,
                maxlen=self.config.max_seq_len,
                padding="post",
                truncating="post"
            )

            X_val = pad_sequences(
                X_val,
                maxlen=self.config.max_seq_len,
                padding="post",
                truncating="post"
            )
            
            y_train = train_split[LABELS].values.astype("float32")
            y_val = val_split[LABELS].values.astype("float32")

            # --- 3. Pre-trained Embeddings ---
            zip_file_path = "wiki-news-300d-1M.vec.zip"
            vec_file_path = "wiki-news-300d-1M.vec"
            
            if not os.path.exists(vec_file_path):
                if not os.path.exists(zip_file_path):
                    logger.info("Downloading wiki-news-300d-1M.vec.zip...")
                    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
                    urllib.request.urlretrieve(url, zip_file_path)
                    logger.info("Download complete.")
                
                logger.info("Extracting embeddings...")
                with ZipFile(zip_file_path, 'r') as zipObj:
                    zipObj.extractall()
                logger.info("Extraction complete.")

            logger.info("Loading pre-trained embeddings...")
            embeddings_index = {}
            with open(vec_file_path, encoding="utf8", errors="ignore") as f:
                next(f)  # Skip header
                for line in f:
                    values = line.rstrip().split()
                    if len(values) != 301:
                        continue
                    word = values[0]
                    vector = np.asarray(values[1:], dtype=np.float32)
                    embeddings_index[word] = vector

            logger.info(f"Loaded embeddings: {len(embeddings_index)}")
            
            embedding_matrix = np.zeros((self.config.max_vocab_size, self.config.embedding_dim))
            for word, idx in keras_tokenizer.word_index.items():
                if idx >= self.config.max_vocab_size:
                    continue
                vector = embeddings_index.get(word)
                if vector is not None:
                    embedding_matrix[idx] = vector

            # --- 4. Build and train model ---
            logger.info("Building model...")
            model = self._build_model(embedding_matrix, self.config.max_vocab_size)
            model.summary(print_fn=logger.info)

            callbacks = [
                ReduceLROnPlateau(
                    monitor="val_auc",
                    mode="max",
                    factor=0.5,
                    patience=1
                ),
                EarlyStopping(
                    monitor="val_auc", 
                    patience=2,
                    restore_best_weights=True, 
                    mode="max"
                )
            ]

            logger.info("Training model...")
            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # --- 5. Find optimal per-label thresholds on validation set ---
            logger.info("Finding optimal per-label thresholds on validation set...")
            val_probs = model.predict(X_val, batch_size=self.config.batch_size, verbose=0)
            optimal_thresholds = self._find_optimal_thresholds(y_val, val_probs)
            
            # --- 6. Save artifacts ---
            logger.info(f"Saving Keras tokenizer to {self.config.tokenizer_path}")
            joblib.dump(keras_tokenizer, self.config.tokenizer_path)

            logger.info(f"Saving BiLSTM model to {self.config.model_path}")
            model.save(str(self.config.model_path))

            thresholds_path = os.path.join(str(self.config.root_dir), "optimal_thresholds.joblib")
            logger.info(f"Saving optimal thresholds to {thresholds_path}")
            joblib.dump(optimal_thresholds, thresholds_path)

            logger.info("BiLSTM training completed successfully!")

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _find_optimal_thresholds(y_true, y_prob):
        from sklearn.metrics import f1_score as compute_f1
        thresholds = {}
        for i, label in enumerate(LABELS):
            best_f1 = 0
            best_thresh = 0.5
            for thresh in np.arange(0.10, 0.90, 0.01):
                preds = (y_prob[:, i] >= thresh).astype(int)
                f1 = compute_f1(y_true[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(round(thresh, 2))
            thresholds[label] = best_thresh
            logger.info(f"  {label}: threshold={best_thresh:.2f}, F1={best_f1:.4f}")
        return thresholds
