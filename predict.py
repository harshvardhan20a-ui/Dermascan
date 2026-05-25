"""
predict.py — DermaScan Binary Inference
========================================
Loads the trained binary model and returns:
    - Label: "concerning" or "normal"
    - Confidence score
    - Risk level, ABCDE tips, precautions

Standalone test:
    python predict.py --image path/to/skin.jpg
"""

import os, json
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.getenv("MODEL_PATH",    os.path.join(_BASE_DIR, "model", "dermascan_final.keras"))
METADATA_PATH = os.getenv("METADATA_PATH", os.path.join(_BASE_DIR, "model", "metadata.json"))

_model    = None
_metadata = None

def _load():
    global _model, _metadata
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'.\n"
                "Run  python train1.py  first."
            )
        print("Loading DermaScan model...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
        print("[OK] Model loaded")
    return _model, _metadata


# ── Clinical content ───────────────────────────────────────────────────────────
PRECAUTIONS = {
    "concerning": [
        "See a dermatologist as soon as possible — within 24–48 hours if rapidly changing.",
        "Do NOT scratch, pick, or apply any creams or home remedies to the lesion.",
        "Photograph the lesion now to document its current appearance.",
        "Avoid direct sun exposure on the affected area until evaluated.",
        "Ask your doctor about dermoscopy or a biopsy to confirm the diagnosis.",
    ],
    "normal": [
        "Monitor the area monthly for any changes in size, color, or texture.",
        "See a doctor promptly if it grows, bleeds, itches, or changes color.",
        "Apply SPF 50+ broad-spectrum sunscreen on all exposed skin daily.",
        "Perform a full-body skin self-examination every month.",
        "Schedule an annual skin check with a dermatologist.",
    ],
}

URGENCY = {
    "concerning": "See a dermatologist within 24–48 hours",
    "normal":     "Monitor at home — see a doctor if anything changes",
}

ABCDE = {
    "concerning": {
        "asymmetry": "The lesion may be asymmetric — two halves don't match",
        "border":    "Edges may be irregular, ragged, notched, or blurred",
        "color":     "Multiple or uneven shades present (brown, black, red, white)",
        "diameter":  "May be larger than 6mm or growing",
        "evolution": "Changing in size, shape, or color — needs evaluation",
    },
    "normal": {
        "asymmetry": "Appears relatively symmetric",
        "border":    "Smooth, round, and well-defined edges",
        "color":     "Uniform color throughout",
        "diameter":  "Usually stable and smaller than 6mm",
        "evolution": "No significant recent changes observed",
    },
}


def preprocess(image_bytes, img_size):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(image_bytes, location="India", symptoms="None"):
    model, meta = _load()
    img_size = meta["img_size"]

    tensor = preprocess(image_bytes, img_size)
    raw    = float(model.predict(tensor, verbose=0)[0][0])   # sigmoid output

    # sigmoid → probability of "concerning" (index 0, first alphabetically)
    # Keras assigns label 0 = "concerning", 1 = "normal" (alphabetical)
    # sigmoid output = P(class=1) = P(normal)
    prob_normal     = raw
    prob_concerning = 1.0 - raw

    if prob_concerning >= 0.5:
        label      = "concerning"
        confidence = round(prob_concerning * 100, 1)
    else:
        label      = "normal"
        confidence = round(prob_normal * 100, 1)

    risk_level = "high" if label == "concerning" else "low"

    analysis_points = [
        f"Classification: {label.upper()} ({confidence}% confidence)",
        f"Concerning probability: {round(prob_concerning*100,1)}%",
        f"Normal probability    : {round(prob_normal*100,1)}%",
        f"Symptoms reported: {symptoms}",
    ]

    summary = (
        f"The image shows features that appear {'potentially concerning' if label=='concerning' else 'normal/benign'} "
        f"({confidence}% confidence). "
        + ("Please consult a dermatologist promptly." if label == "concerning"
           else "Continue monitoring and practice sun protection.")
    )

    return {
        "condition":      "Concerning Lesion" if label == "concerning" else "Normal / Benign",
        "riskLevel":      risk_level,
        "confidence":     confidence,
        "label":          label,
        "summary":        summary,
        "analysisPoints": analysis_points,
        "top3": [
            {"class": "Concerning", "probability": round(prob_concerning * 100, 1)},
            {"class": "Normal",     "probability": round(prob_normal     * 100, 1)},
        ],
        "abcde":       ABCDE[label],
        "precautions": PRECAUTIONS[label],
        "urgency":     URGENCY[label],
        "doctors":     _get_doctors(location, risk_level),
    }


def _get_doctors(location, risk_level):
    return [
        {"name": "Dr. Priya Sharma",  "specialty": "Dermatologist",
         "hospital": f"City Skin Clinic, {location}", "distance": "2.3 km",
         "rating": 4.7, "phone": "+91-98765-43210", "urgent": risk_level == "high"},
        {"name": "Dr. Rajesh Kumar",  "specialty": "Skin & Cosmetic Surgeon",
         "hospital": f"Apollo Hospital, {location}",  "distance": "4.1 km",
         "rating": 4.5, "phone": "+91-98765-11111", "urgent": False},
        {"name": "Dr. Anita Verma",   "specialty": "Oncologist",
         "hospital": f"AIIMS, {location}",            "distance": "6.8 km",
         "rating": 4.9, "phone": "+91-98765-22222", "urgent": False},
    ]


if __name__ == "__main__":
    import argparse, json as _json
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    required=True)
    parser.add_argument("--location", default="Delhi")
    parser.add_argument("--symptoms", default="None")
    args = parser.parse_args()
    with open(args.image, "rb") as f:
        print(_json.dumps(predict(f.read(), args.location, args.symptoms), indent=2))
