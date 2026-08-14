
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.efficientnet import preprocess_input

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_PATH, "herbify_best.keras")
CLASS_NAMES_PATH = os.path.join(BASE_PATH, "class_names.json")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Lambda):
        if hasattr(layer.function, "__globals__"):
            layer.function.__globals__["preprocess_input"] = preprocess_input

app = FastAPI(
    title="Herbify API",
    description="AI-powered herb identification API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {
        "message": "Herbify API is running!",
        "classes": len(class_names)
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": len(class_names)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()

    image = Image.open(
        __import__("io").BytesIO(image_data)
    ).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    return {
        "herb": predicted_class,
        "confidence": round(confidence, 2)
    }
