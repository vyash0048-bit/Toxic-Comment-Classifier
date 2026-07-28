from src.ToxicCommentClassifier.constants import *
from src.ToxicCommentClassifier.utils.common import read_yaml, create_directories
from src.ToxicCommentClassifier.entity import (
    DataIngestionConfig, DataPreprocessingConfig, ModelTrainingConfig,
    ModelEvaluationConfig, BiLSTMTrainingConfig, BiLSTMEvaluationConfig
)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path)
        )

        return data_ingestion_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        params = self.params.DataPreprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            preprocessed_train_data_path=Path(config.preprocessed_train_data_path),
            preprocessed_test_data_path=Path(config.preprocessed_test_data_path),
            tokenizer_path=Path(config.tokenizer_path),
            test_size=params.test_size,
            random_state=params.random_state,
            word_ngram_range=params.word_ngram_range,
            word_max_features=params.word_max_features,
            word_min_df=params.word_min_df,
            char_ngram_range=params.char_ngram_range,
            char_max_features=params.char_max_features,
            char_min_df=params.char_min_df
        )

        return data_preprocessing_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training

        create_directories([config.root_dir])

        params = self.params.ModelTraining

        model_training_config = ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            model_name=config.model_name,
            solver=params.solver,
            C=params.C,
            max_iter=params.max_iter,
            n_jobs=params.n_jobs,
            verbose=params.verbose
        )

        return model_training_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            metric_file_name=Path(config.metric_file_name),
            all_params=self.params
        )

        return model_evaluation_config

    def get_bilstm_training_config(self) -> BiLSTMTrainingConfig:
        config = self.config.bilstm_training
        params = self.params.BiLSTMTraining

        create_directories([config.root_dir])

        bilstm_training_config = BiLSTMTrainingConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            tokenizer_path=Path(config.tokenizer_path),
            fasttext_model_path=Path(config.fasttext_model_path),
            model_path=Path(config.model_path),
            optimal_thresholds_path=Path(config.optimal_thresholds_path),
            max_seq_len=params.max_seq_len,
            embedding_dim=params.embedding_dim,
            lstm_units=params.lstm_units,
            dropout=params.dropout,
            spatial_dropout=params.spatial_dropout,
            batch_size=params.batch_size,
            epochs=params.epochs,
            learning_rate=params.learning_rate
        )

        return bilstm_training_config

    def get_bilstm_evaluation_config(self) -> BiLSTMEvaluationConfig:
        config = self.config.bilstm_evaluation

        create_directories([config.root_dir])

        bilstm_evaluation_config = BiLSTMEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            keras_tokenizer_path=Path(config.keras_tokenizer_path),
            model_path=Path(config.model_path),
            optimal_thresholds_path=Path(config.optimal_thresholds_path),
            metric_file_name=Path(config.metric_file_name),
            all_params=self.params,
            max_seq_len=self.params.BiLSTMTraining.max_seq_len
        )

        return bilstm_evaluation_config
