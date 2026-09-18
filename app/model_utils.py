import json
import numpy as np
import tensorflow as tf
from PIL import Image
import io

MODEL_PATH = "model/plant_disease_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


def predict_image(image_bytes):
    # Open image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize
    image = image.resize((180, 180))

    # Convert to numpy array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    # Get highest probability
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])

    predicted_class = class_names[predicted_index]

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2)
    }