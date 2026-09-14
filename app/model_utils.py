from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "plant_disease_model.keras"
CLASS_PATH = ROOT / "model" / "class_names.json"
IMG_SIZE = (180, 180)


def load_artifacts():
    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python train.py` first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names


def predict_image(image: Image.Image):
    model, class_names = load_artifacts()

    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    probability_class_1 = float(model.predict(array, verbose=0)[0][0])

    # Keras image_dataset_from_directory sorts class names alphabetically.
    # class_names[0] corresponds to probability below 0.5.
    if probability_class_1 >= 0.5:
        predicted_index = 1
        confidence = probability_class_1
    else:
        predicted_index = 0
        confidence = 1.0 - probability_class_1

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(confidence * 100, 2),
    }
