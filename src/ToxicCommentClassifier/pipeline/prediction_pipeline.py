import os
import joblib
from src.ToxicCommentClassifier.config.configuration import ConfigurationManager
from src.ToxicCommentClassifier.components.data_preprocessing import clean_text

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class PredictionPipeline:
    def __init__(self):
        config_manager = ConfigurationManager()
        
        # Extract paths using our configuration manager
        preprocessing_config = config_manager.get_data_preprocessing_config()
        evaluation_config = config_manager.get_model_evaluation_config()
        
        self.vectorizer_path = preprocessing_config.tokenizer_path
        self.model_path = evaluation_config.model_path
        
        # Load the artifacts into memory once when the class is initialized
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)

    def predict(self, text: str) -> dict:
        """
        Takes a raw string, preprocesses it, and returns a dictionary 
        of toxicity labels with their predicted probabilities.
        """
        # 1. Clean the text using the exact same function used in training
        cleaned = clean_text(text)
        
        # 2. Convert text to sparse tf-idf feature matrix
        # transform expects an iterable, so we pass it inside a list
        vectorized_text = self.vectorizer.transform([cleaned])
        
        # 3. Predict probabilities for all labels
        # predict_proba returns a 2D array, we want the first (and only) row
        probabilities = self.model.predict_proba(vectorized_text)[0]
        
        # 4. Map the probabilities to their respective labels
        prediction_result = {
            label: round(float(prob), 4) 
            for label, prob in zip(LABELS, probabilities)
        }
        
        return prediction_result

# Quick test logic if this script is run directly
if __name__ == "__main__":
    pipeline = PredictionPipeline()
    sample_text = "I love this project, it's so helpful!"
    print(f"Text: '{sample_text}'")
    print("Predictions:", pipeline.predict(sample_text))
