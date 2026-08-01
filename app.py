import os
import gradio as gr
import spaces
from datetime import datetime, timezone
from dotenv import load_dotenv
from src.ToxicCommentClassifier.pipeline.prediction_pipeline import PredictionPipeline
from src.ToxicCommentClassifier.logger import logger

# Load environment variables
load_dotenv()

# Initialize the pipeline
pipeline = PredictionPipeline()

# --- MongoDB connection for storing predictions ---
mongo_client = None
predictions_collection = None
db_error_message = None

try:
    mongodb_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DATABASE_NAME")
    pred_collection_name = os.getenv("PREDICTION_COLLECTION", "predictions")

    if mongodb_uri and db_name:
        from pymongo import MongoClient
        mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Verify connection
        mongo_client.admin.command("ping")
        predictions_collection = mongo_client[db_name][pred_collection_name]
        logger.info(f"MongoDB connected. Predictions will be stored in '{db_name}.{pred_collection_name}'.")
    else:
        db_error_message = "MongoDB env vars (MONGODB_URI, DATABASE_NAME) not found in Hugging Face Secrets."
        logger.warning(db_error_message)
except Exception as e:
    db_error_message = f"Connection failed: {str(e)}"
    logger.warning(f"Could not connect to MongoDB: {e}. Predictions will NOT be stored.")
    predictions_collection = None


def store_prediction(text: str, model_type: str, predictions: dict) -> str:
    """Store a prediction record in MongoDB (synchronous for debugging)."""
    if predictions_collection is None:
        return f"⚠️ Database not connected. Reason: {db_error_message}"
    try:
        predictions_collection.insert_one({
            "comment_text": text,
            "model_used": model_type,
            "predictions": predictions,
            "timestamp": datetime.now(timezone.utc),
        })
        return "✅ Saved to MongoDB successfully."
    except Exception as e:
        logger.warning(f"Failed to store prediction in MongoDB: {e}")
        return f"❌ Failed to save: {str(e)}"


@spaces.GPU
def run_model_inference(text: str, model_type: str):
    """Run the actual ML model inference on the ephemeral ZeroGPU worker."""
    return pipeline.predict(text, model_type=model_type)


def predict_toxicity(text, model_choice):
    """Main Gradio handler running on the persistent CPU container."""
    if not text.strip():
        return "Please enter some text."
        
    try:
        # map UI choice to model_type
        model_type = "lr" if model_choice == "Logistic Regression" else "bilstm"
        
        # Check if model is available
        if model_type not in pipeline.available_models:
            return f"Model '{model_choice}' is not available right now."
            
        # 1. Run inference on the ZeroGPU worker
        predictions = run_model_inference(text, model_type)

        # 2. Store prediction in MongoDB (synchronously to get the status)
        db_status = store_prediction(text, model_type, predictions)
        
        # 3. Format the output nicely
        result = "### Toxicity Analysis:\n\n"
        for label, prob in predictions.items():
            result += f"- **{label.replace('_', ' ').title()}**: {prob:.2%}\n"
            
        result += f"\n---\n**Database Status**: {db_status}"
        return result
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define the Gradio interface
with gr.Blocks(title="Toxic Comment Classifier") as demo:
    gr.Markdown("# 🛡️ Toxic Comment Classifier")
    gr.Markdown("Detect toxicity in comments using Machine Learning.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                lines=5, 
                placeholder="Enter a comment here to analyze its toxicity...",
                label="Comment Text"
            )
            
            available = ["Logistic Regression"]
            if "bilstm" in pipeline.available_models:
                available.append("BiLSTM")
                
            model_dropdown = gr.Dropdown(
                choices=available,
                value=available[0] if available else None,
                label="Select Model"
            )
            
            submit_btn = gr.Button("Analyze", variant="primary")
            
        with gr.Column():
            output_display = gr.Markdown(label="Results")
            
    submit_btn.click(
        fn=predict_toxicity,
        inputs=[text_input, model_dropdown],
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch()
