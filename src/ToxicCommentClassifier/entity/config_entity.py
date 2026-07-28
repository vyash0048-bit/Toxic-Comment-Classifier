from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    preprocessed_train_data_path: Path
    preprocessed_test_data_path: Path
    tokenizer_path: Path
    test_size: float
    random_state: int
    word_ngram_range: list
    word_max_features: int
    word_min_df: int
    char_ngram_range: list
    char_max_features: int
    char_min_df: int

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    solver: str
    C: float
    max_iter: int
    n_jobs: int
    verbose: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    all_params: dict

@dataclass(frozen=True)
class BiLSTMTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    tokenizer_path: Path
    fasttext_model_path: Path
    model_path: Path
    max_seq_len: int
    embedding_dim: int
    lstm_units: int
    dropout: float
    spatial_dropout: float
    batch_size: int
    epochs: int
    learning_rate: float

@dataclass(frozen=True)
class BiLSTMEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    keras_tokenizer_path: Path
    model_path: Path
    metric_file_name: Path
    all_params: dict
    max_seq_len: int
