from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import io

from app.model_utils import predict_image

app = FastAPI(
    title="Plant Disease Detection API",
    description="Simple CNN-based plant leaf health classifier.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "Please upload an image file."}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        return predict_image(image)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Could not process the image: {exc}"}
