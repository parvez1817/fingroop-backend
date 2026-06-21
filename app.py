from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import json
import os
import io



app = FastAPI(title="Blood Group Predictor API")

# Match the original Flask CORS config
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://192.168.137.1:8080",
    "https://lovable.dev"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_labels = []
class_indices = {}


@app.on_event("startup")
def startup_event() -> None:
    global model, class_labels, class_indices

    base_dir = os.path.dirname(__file__)

    # Load the trained model
    model = load_model(os.path.join(base_dir, "best_blood_model_mnv2.h5"))


    # Load class labels
    labels_path = os.path.join(os.path.dirname(__file__), "class_labels.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)


    # Reverse dictionary to get labels from indices
    class_labels = [None] * len(class_indices)
    for label, index in class_indices.items():
        class_labels[index] = label


@app.get("/")
def home():
    return {
        "message": "Blood Group Predictor API is running",
        "model_loaded": True,
        "supported_blood_groups": list(class_indices.keys()),
    }


@app.post("/predict")
def predict(image: UploadFile = File(...)):
    try:
        if image is None:
            raise HTTPException(status_code=400, detail="No image part in the request")
        if image.filename == "":
            raise HTTPException(status_code=400, detail="No selected file")

        image_bytes = image.file.read()
        # Ensure bytes are not treated as text anywhere
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        pil_image = pil_image.resize((224, 224))

        arr = np.array(pil_image)
        arr = preprocess_input(arr)  # MobileNetV2 preprocessing
        arr = np.expand_dims(arr, axis=0)

        predictions = model.predict(arr)
        predicted_class = int(np.argmax(predictions[0]))
        predicted_blood_group = class_labels[predicted_class]
        confidence = float(predictions[0][predicted_class]) * 100

        return {
            "blood_group": predicted_blood_group,
            "confidence": f"{confidence:.2f}%",
            "raw_confidence": confidence,
            "all_predictions": {
                class_labels[i]: float(predictions[0][i]) * 100 for i in range(len(class_labels))
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        # Keep the error message ASCII-safe so JSON encoding never fails
        err = str(e)
        print("Error during prediction:", err)
        raise HTTPException(
            status_code=500,
            detail={"error": "Prediction failed", "details": err},
        )



@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": len(class_labels) if class_labels else 0,
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Blood Group Predictor API (FastAPI)...")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)


