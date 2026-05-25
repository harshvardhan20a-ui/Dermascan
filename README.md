# DermaScan AI — Skin Cancer Prediction Web App

A production-style Flask web application for skin disease prediction using
EfficientNetB3 trained on the ISIC dataset.

---

## Project Structure

```
dermascan_ml/
|
+-- app.py                          <- Flask backend (routes, serves UI)
+-- train1.py                       <- ML training script (EfficientNetB3)
+-- predict.py                      <- Inference engine (called by app.py)
+-- requirements.txt                <- Python dependencies
+-- .env.example                    <- Rename to .env to configure paths
|
+-- dataset_utils/
|   +-- prepare_dataset.py          <- Converts ISIC CSV to folder structure
|
+-- templates/
|   +-- index.html                  <- Jinja2 HTML frontend
|
+-- static/
|   +-- css/style.css               <- All styles
|   +-- js/main.js                  <- Frontend logic
|
+-- model/                          <- Auto-created after training
|   +-- dermascan_final.keras       <- Saved trained model
|   +-- metadata.json               <- Class names, img_size, etc.
|   +-- training_curves.png         <- Accuracy + AUC plots
|
+-- dataset/                        <- Your ISIC images go here
    +-- train/
    |   +-- melanoma/
    |   +-- nevus/
    |   +-- basal_cell_carcinoma/
    |   +-- squamous_cell_carcinoma/
    |   +-- actinic_keratosis/
    |   +-- dermatofibroma/
    |   +-- seborrheic_keratosis/
    |   +-- vascular_lesion/
    +-- val/
        +-- (same 8 class folders)
```

---

## Setup & Run (Step by Step)

### Step 1 — Install dependencies

```cmd
pip install -r requirements.txt
```

### Step 2 — Prepare ISIC dataset

Download from: https://challenge.isic-archive.com/data/
You need:
- ISIC_2019_Training_Input/  (folder of .jpg images)
- ISIC_2019_Training_GroundTruth.csv

Then run:
```cmd
python dataset_utils/prepare_dataset.py ^
    --images_dir C:\path\to\ISIC_2019_Training_Input ^
    --csv_path   C:\path\to\ISIC_2019_Training_GroundTruth.csv ^
    --output_dir ./dataset ^
    --val_split  0.2
```

### Step 3 — Train the model

```cmd
python train1.py
```

Options:
```cmd
python train1.py --epochs 30 --batch_size 16
```

This will create:
- model/dermascan_final.keras
- model/metadata.json
- model/training_curves.png

Training time: ~2-4 hours on GPU, ~10+ hours on CPU.

### Step 4 — Start the web server

```cmd
python app.py
```

### Step 5 — Open in browser

```
http://localhost:5000
```

---

## How It Works

```
User uploads skin photo
        |
        v
Flask /analyze route (app.py)
        |
        v
predict.py -> loads model -> preprocesses image (300x300)
        |
        v
EfficientNetB3 inference -> softmax probabilities
        |
        v
Risk classification (high/medium/low)
ABCDE criteria lookup
Precautions and urgency
Doctor suggestions
        |
        v
JSON response -> frontend renders results
```

---

## Classes Predicted

| Class                   | Risk Level |
|-------------------------|------------|
| Melanoma                | HIGH       |
| Basal Cell Carcinoma    | HIGH       |
| Squamous Cell Carcinoma | HIGH       |
| Actinic Keratosis       | MEDIUM     |
| Seborrheic Keratosis    | LOW        |
| Benign Nevus (Mole)     | LOW        |
| Dermatofibroma          | LOW        |
| Vascular Lesion         | LOW        |

---

## Test Single Image (without running web server)

```cmd
python predict.py --image path\to\skin.jpg --location "Delhi" --symptoms "Itching"
```

---

## Deploying to Production

```cmd
pip install gunicorn
gunicorn -w 4 app:app
```

For Windows, use waitress instead:
```cmd
pip install waitress
waitress-serve --port=5000 app:app
```

---

## Medical Disclaimer

This application is for informational and educational purposes only.
It is not a substitute for professional medical diagnosis.
Always consult a qualified dermatologist for any skin concerns.
