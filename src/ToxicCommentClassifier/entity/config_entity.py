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
