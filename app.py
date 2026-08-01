import os
import gradio as gr
import spaces
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from src.ToxicCommentClassifier.pipeline.prediction_pipeline import PredictionPipeline

# Initialize the pipeline
pipeline = PredictionPipeline()

# --- Gradio Interface ---

@spaces.GPU
def predict_toxicity(text, model_choice):
    if not text.strip():
        return "Please enter some text."
        
    try:
        # map UI choice to model_type
        model_type = "lr" if model_choice == "Logistic Regression" else "bilstm"
        
        # Check if model is available
        if model_type not in pipeline.available_models:
            return f"Model '{model_choice}' is not available right now."
            
        predictions = pipeline.predict(text, model_type=model_type)
        
        # Format the output nicely
        result = "### Toxicity Analysis:\n\n"
        for label, prob in predictions.items():
            result += f"- **{label.replace('_', ' ').title()}**: {prob:.2%}\n"
            
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

# --- FastAPI backend with Flask-style API routes ---

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the custom Flask-style HTML UI."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/models")
async def available_models():
    """Return the list of available model types."""
    return JSONResponse(content={"models": pipeline.available_models})

@app.post("/predict")
async def predict(request: Request):
    """Prediction API endpoint (mirrors flask_app.py)."""
    try:
        data = await request.json()
        text = data.get("text", "")
        model_type = data.get("model", "lr")
        
        if not text.strip():
            return JSONResponse(content={"error": "Empty text provided"}, status_code=400)
        
        if model_type not in pipeline.available_models:
            return JSONResponse(
                content={"error": f"Model '{model_type}' is not available. Available: {pipeline.available_models}"},
                status_code=400
            )
            
        predictions = pipeline.predict(text, model_type=model_type)
        
        return JSONResponse(content={
            "text": text,
            "model": model_type,
            "predictions": predictions
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Mount Gradio app at /gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
