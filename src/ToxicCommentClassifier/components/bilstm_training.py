import os
import sys
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
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


def compute_class_weights(y_train):
    """
    Compute per-label class weights to handle severe imbalance.
    Returns a dict mapping label index -> {0: weight_neg, 1: weight_pos}.
    """
    class_weights = {}
    for i, label in enumerate(LABELS):
        col = y_train[:, i]
        n_pos = col.sum()
        n_neg = len(col) - n_pos
        # Inverse frequency with smoothing, capped to avoid extreme weights
        weight_pos = min((n_neg / max(n_pos, 1)) * 0.5, 20.0)
        weight_neg = 0.5
        class_weights[i] = {0: weight_neg, 1: weight_pos}
    return class_weights


def build_sample_weights(y_train, class_weights):
    """
    Convert per-label class weights into per-sample weights.
    For multi-label: take the max weight across all positive labels for each sample.
    """
    n_samples = y_train.shape[0]
    sample_weights = np.ones(n_samples, dtype=np.float32)

    for i in range(y_train.shape[1]):
        pos_mask = y_train[:, i] == 1
        pos_weight = class_weights[i][1]
        # For samples with this label positive, take the max weight seen so far
        sample_weights[pos_mask] = np.maximum(sample_weights[pos_mask], pos_weight)

    return sample_weights


class BiLSTMTraining:
    def __init__(self, config: BiLSTMTrainingConfig):
        self.config = config

    def _build_model(self, embedding_matrix, vocab_size):
        """
        Improved BiLSTM architecture:
        - Trainable embeddings (fine-tuned from FastText)
        - SpatialDropout1D
        - Bidirectional LSTM
        - Dual pooling: GlobalMaxPool + GlobalAvgPool concatenated
        - BatchNormalization + Dense layers
        """
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(
            shape=(self.config.max_seq_len,), name="input"
        )

        # Embedding layer — trainable to fine-tune FastText vectors
        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.config.embedding_dim,
            weights=[embedding_matrix],
            input_length=self.config.max_seq_len,
            trainable=False,
            name="fasttext_embedding"
        )(input_layer)

        x = tf.keras.layers.SpatialDropout1D(self.config.spatial_dropout)(embedding)

        # Bidirectional LSTM with return_sequences for pooling
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.config.lstm_units,
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.1
            ),
            name="bilstm"
        )(x)

        # Dual pooling — captures different aspects of the sequence
        max_pool = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)
        avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        x = tf.keras.layers.Concatenate(name="concat_pool")([max_pool, avg_pool])

        # Dense head with BatchNorm
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.Dropout(self.config.dropout)(x)
        x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(self.config.dropout * 0.5)(x)

        output = tf.keras.layers.Dense(
            len(LABELS), activation="sigmoid", name="output"
        )(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(name="auc", multi_label=True),
            ]
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

            # --- 3. Compute class weights for imbalanced labels ---
            logger.info("Computing class weights for imbalanced labels...")
            cw = compute_class_weights(y_train)
            for i, label in enumerate(LABELS):
                n_pos = int(y_train[:, i].sum())
                n_total = len(y_train)
                logger.info(f"  {label}: {n_pos}/{n_total} positive "
                            f"({100*n_pos/n_total:.2f}%), weight={cw[i][1]:.2f}")
            sample_weights = build_sample_weights(y_train, cw)
            logger.info(f"Sample weights range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")

            # --- 4. Train FastText on corpus ---
            logger.info("Training FastText embeddings on corpus...")
            sentences = [text.split() for text in train_split["text_clean"]]
            ft_model = FastTextModel(
                sentences,
                vector_size=self.config.embedding_dim,
                window=5,
                min_count=2,
                workers=4,
                epochs=3,  # Reduced for faster training
                sg=1  # Skip-gram
            )
            ft_model.save(str(self.config.fasttext_model_path))
            logger.info(f"FastText model saved to {self.config.fasttext_model_path}")

            # --- 5. Build embedding matrix ---
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

            # --- 6. Build and train model ---
            logger.info("Building improved BiLSTM model...")
            model = self._build_model(embedding_matrix, vocab_size)
            model.summary(print_fn=logger.info)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", patience=3,
                    restore_best_weights=True, mode="max"
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=2, min_lr=1e-6, verbose=1
                ),
            ]

            logger.info("Training BiLSTM model with class-weighted samples...")
            model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # --- 7. Find optimal per-label thresholds on validation set ---
            logger.info("Finding optimal per-label thresholds on validation set...")
            val_probs = model.predict(X_val, batch_size=self.config.batch_size, verbose=0)
            optimal_thresholds = self._find_optimal_thresholds(y_val, val_probs)
            logger.info(f"Optimal thresholds: {optimal_thresholds}")

            # --- 8. Save artifacts ---
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
        """
        For each label, find the threshold that maximizes F1 score.
        Searches thresholds from 0.1 to 0.9 in steps of 0.01.
        """
        from sklearn.metrics import f1_score as compute_f1

        thresholds = {}
        for i, label in enumerate(LABELS):
            best_f1 = 0
            best_thresh = 0.5  # Default fallback
            for thresh in np.arange(0.10, 0.90, 0.01):
                preds = (y_prob[:, i] >= thresh).astype(int)
                f1 = compute_f1(y_true[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(round(thresh, 2))
            thresholds[label] = best_thresh
            logger.info(f"  {label}: threshold={best_thresh:.2f}, F1={best_f1:.4f}")
        return thresholds
