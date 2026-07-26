from flask import Flask, request, jsonify, render_template
import os
from src.ToxicCommentClassifier.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Initialize the pipeline globally so models are loaded only once at startup
prediction_pipeline = PredictionPipeline()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")
        
        if not text.strip():
            return jsonify({"error": "Empty text provided"}), 400
            
        predictions = prediction_pipeline.predict(text)
        
        return jsonify({
            "text": text,
            "predictions": predictions
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure port 5000 is open
    app.run(host="0.0.0.0", port=5000, debug=True)
