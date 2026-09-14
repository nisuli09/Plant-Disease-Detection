# 🌿 Plant Disease Detection System

A simple end-to-end computer vision project that uses a Convolutional Neural Network (CNN) to classify plant leaf images as **healthy** or **diseased**.

## Project Overview

The system follows this workflow:

```text
Leaf Image
    ↓
Image Preprocessing
    ↓
CNN Model
    ↓
Healthy / Diseased
    ↓
Confidence Score
```

## Technologies

- Python
- TensorFlow / Keras
- OpenCV/Pillow
- FastAPI
- HTML, CSS and JavaScript
- Jupyter Notebook

## Project Structure

```text
plant-disease-detection/
├── app/
│   ├── static/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── script.js
│   ├── main.py
│   └── model_utils.py
├── dataset/
│   ├── healthy/
│   └── diseased/
├── model/
├── notebooks/
│   └── training.ipynb
├── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

This project expects two classes:

```text
dataset/
├── healthy/
└── diseased/
```

Place the corresponding leaf images inside these folders.

For the CV version of this project, the Apple leaf subset of PlantVillage can be used. The PlantVillage dataset contains healthy and diseased plant leaf images and is widely used for plant-disease classification research.

Dataset/source information:
- PlantVillage project: https://plantvillage.psu.edu/
- PlantVillage dataset repository: https://github.com/spMohanty/PlantVillage-Dataset
- Dataset paper: Mohanty, Hughes & Salathé (2016), "Using deep learning for image-based plant disease detection."

**Important:** The dataset images are not included in this GitHub repository. Check the dataset's license and citation requirements before redistribution.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Make sure images are inside:

```text
dataset/
├── healthy/
└── diseased/
```

Then run:

```bash
python train.py
```

The trained model will be saved as:

```text
model/plant_disease_model.keras
```

The class names will be saved as:

```text
model/class_names.json
```

## Run the Web Application

After training:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Upload a leaf image and click **Predict**.

## Learning Goals

This project demonstrates:

- Image dataset organization
- Image preprocessing
- Data augmentation
- CNN model development
- Binary image classification
- Model training and validation
- Saving/loading a Keras model
- REST API development with FastAPI
- Connecting a frontend to an ML API

## Future Improvements

- Add individual disease classes
- Add model evaluation metrics
- Add confusion matrix
- Add prediction history
- Improve the model using transfer learning
- Deploy the API
- Add Grad-CAM visual explanations

## Disclaimer

This project is for educational and demonstration purposes. Predictions should not be treated as professional agricultural diagnosis.
