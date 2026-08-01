from flask import Flask, request, jsonify, render_template
import os
from src.ToxicCommentClassifier.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Initialize the pipeline globally so models are loaded only once at startup
prediction_pipeline = PredictionPipeline()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/models", methods=["GET"])
def available_models():
    """Return the list of available model types."""
    return jsonify({"models": prediction_pipeline.available_models}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")
        model_type = data.get("model", "lr")
        
        if not text.strip():
            return jsonify({"error": "Empty text provided"}), 400
        
        if model_type not in prediction_pipeline.available_models:
            return jsonify({"error": f"Model '{model_type}' is not available. Available: {prediction_pipeline.available_models}"}), 400
            
        predictions = prediction_pipeline.predict(text, model_type=model_type)
        
        return jsonify({
            "text": text,
            "model": model_type,
            "predictions": predictions
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure port 7860 is open for Hugging Face Spaces
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
