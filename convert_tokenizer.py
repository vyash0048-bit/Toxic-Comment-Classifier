"""
One-time script to convert the Keras tokenizer (which requires TensorFlow to load)
into a plain JSON file (which needs zero dependencies to load).
Run this ONCE on your local machine, then rebuild the Docker image.
"""
import joblib
import json

tokenizer = joblib.load("artifacts/bilstm_training/keras_tokenizer.joblib")

data = {
    "word_index": tokenizer.word_index,
    "num_words": tokenizer.num_words,
    "oov_token": tokenizer.oov_token,
    "lower": tokenizer.lower,
    "char_level": tokenizer.char_level,
}

output_path = "artifacts/bilstm_training/tokenizer_config.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f)

print(f"Saved tokenizer config with {len(tokenizer.word_index)} words to {output_path}")
print(f"File size: {len(json.dumps(data)) / 1024 / 1024:.1f} MB")
