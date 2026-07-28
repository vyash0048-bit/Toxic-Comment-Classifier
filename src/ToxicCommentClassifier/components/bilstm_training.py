import os
import sys
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from gensim.models import FastText as FastTextModel
from src.ToxicCommentClassifier.logger import logger
from src.ToxicCommentClassifier.exception import CustomException
from src.ToxicCommentClassifier.entity import BiLSTMTrainingConfig

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(text):
    """Same cleaning function used by the TF-IDF pipeline."""
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip() or "unknown"


class BiLSTMTraining:
    def __init__(self, config: BiLSTMTrainingConfig):
        self.config = config

    def _build_model(self, embedding_matrix, vocab_size):
        """Build BiLSTM model with pretrained FastText embeddings."""
        # Import tensorflow here to avoid import overhead when not needed
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=self.config.embedding_dim,
                weights=[embedding_matrix],
                input_length=self.config.max_seq_len,
                trainable=False,
                name="fasttext_embedding"
            ),
            tf.keras.layers.SpatialDropout1D(self.config.spatial_dropout),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.config.lstm_units, return_sequences=True),
                name="bilstm"
            ),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(self.config.dropout),
            tf.keras.layers.Dense(len(LABELS), activation="sigmoid", name="output")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=["AUC"]
        )
        return model

    def initiate_bilstm_training(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            # --- 1. Load and clean data ---
            logger.info(f"Loading training data from {self.config.train_data_path}")
            train_df = pd.read_csv(self.config.train_data_path)
            train_df["text_clean"] = train_df["comment_text"].map(clean_text)

            # Stratified split (same logic as TF-IDF pipeline)
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
            logger.info("Fitting Keras Tokenizer on training text...")
            keras_tokenizer = Tokenizer(oov_token="<OOV>")
            keras_tokenizer.fit_on_texts(train_split["text_clean"])
            word_index = keras_tokenizer.word_index
            vocab_size = len(word_index) + 1  # +1 for padding token at index 0
            logger.info(f"Vocabulary size: {vocab_size}")

            # Convert text to padded integer sequences
            X_train = pad_sequences(
                keras_tokenizer.texts_to_sequences(train_split["text_clean"]),
                maxlen=self.config.max_seq_len, padding="post", truncating="post"
            )
            X_val = pad_sequences(
                keras_tokenizer.texts_to_sequences(val_split["text_clean"]),
                maxlen=self.config.max_seq_len, padding="post", truncating="post"
            )
            y_train = train_split[LABELS].values.astype(np.float32)
            y_val = val_split[LABELS].values.astype(np.float32)

            logger.info(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

            # --- 3. Train FastText on corpus ---
            logger.info("Training FastText embeddings on corpus...")
            sentences = [text.split() for text in train_split["text_clean"]]
            ft_model = FastTextModel(
                sentences,
                vector_size=self.config.embedding_dim,
                window=5,
                min_count=2,
                workers=4,
                epochs=5,
                sg=1  # Skip-gram
            )
            ft_model.save(str(self.config.fasttext_model_path))
            logger.info(f"FastText model saved to {self.config.fasttext_model_path}")

            # --- 4. Build embedding matrix ---
            logger.info("Building embedding matrix from FastText...")
            embedding_matrix = np.zeros((vocab_size, self.config.embedding_dim), dtype=np.float32)
            found, missed = 0, 0
            for word, idx in word_index.items():
                if word in ft_model.wv:
                    embedding_matrix[idx] = ft_model.wv[word]
                    found += 1
                else:
                    # FastText can generate vectors for OOV words via subword info
                    try:
                        embedding_matrix[idx] = ft_model.wv.get_vector(word)
                        found += 1
                    except KeyError:
                        missed += 1
            logger.info(f"Embedding coverage: {found}/{found + missed} words "
                        f"({100 * found / (found + missed):.1f}%)")

            # --- 5. Build and train model ---
            logger.info("Building BiLSTM model...")
            model = self._build_model(embedding_matrix, vocab_size)
            model.summary(print_fn=logger.info)

            logger.info("Training BiLSTM model...")
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=[early_stop],
                verbose=1
            )

            # --- 6. Save artifacts ---
            logger.info(f"Saving Keras tokenizer to {self.config.tokenizer_path}")
            joblib.dump(keras_tokenizer, self.config.tokenizer_path)

            logger.info(f"Saving BiLSTM model to {self.config.model_path}")
            model.save(str(self.config.model_path))

            logger.info("BiLSTM training completed successfully!")

        except Exception as e:
            raise CustomException(e, sys)
