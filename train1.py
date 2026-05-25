"""
train1.py — DermaScan Binary Classifier
=======================================
Trains EfficientNetB3 to classify skin lesions as:
    concerning  (melanoma, BCC, SCC, actinic keratosis)
    normal      (nevus, seborrheic keratosis, dermatofibroma, vascular lesion)

Dataset structure expected:
    dataset/
    ├── train/
    │   ├── concerning/
    │   └── normal/
    └── val/
        ├── concerning/
        └── normal/

Run:
    python train1.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# ── Config ─────────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(_BASE_DIR, "model")

IMG_SIZE    = 300
BATCH_SIZE  = 32
EPOCHS      = 3
LR_INITIAL  = 1e-3
LR_FINETUNE = 1e-5
AUTOTUNE    = tf.data.AUTOTUNE

CLASS_NAMES = ["concerning", "normal"]   # alphabetical = Keras default order


# ── Data Pipeline ──────────────────────────────────────────────────────────────
def build_dataset(split):
    path = os.path.join(DATA_DIR, split)
    ds = keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="binary",          # single sigmoid output
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=42,
    )
    aug = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])
    def preprocess(images, labels):
        if split == "train":
            images = aug(images, training=True)
        return images, labels
    return ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model():
    base = EfficientNetB3(include_top=False, weights="imagenet",
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)   # binary output

    return keras.Model(inputs, outputs, name="dermascan_binary"), base


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_history(h1, h2):
    acc  = h1.history["accuracy"]     + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    p2   = len(h1.history["accuracy"])
    epcs = range(1, len(acc) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epcs, acc,  label="Train Accuracy")
    plt.plot(epcs, vacc, label="Val Accuracy")
    plt.axvline(p2, color="gray", linestyle="--", alpha=0.6, label="Fine-tune start")
    plt.title("Training Accuracy"); plt.legend(); plt.xlabel("Epoch")
    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"[OK] Curves saved → {out}")


# ── Train ──────────────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[1/5] Loading datasets...")
    train_ds = build_dataset("train")
    val_ds   = build_dataset("val")

    print("\n[2/5] Computing class weights...")
    counts = np.zeros(2)
    for _, labels in train_ds:
        for lbl in labels.numpy().flatten():
            counts[int(round(lbl))] += 1
    total = counts.sum()
    class_weights = {
        0: total / (2 * counts[0]) if counts[0] > 0 else 1.0,
        1: total / (2 * counts[1]) if counts[1] > 0 else 1.0,
    }
    print(f"       concerning (0): {class_weights[0]:.2f}")
    print(f"       normal     (1): {class_weights[1]:.2f}")

    print("\n[3/5] Building model...")
    model, base = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(LR_INITIAL),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # Phase 1 — frozen base
    print("\n[4/5] Phase 1 — training head (base frozen)...")
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS // 2,
        class_weight=class_weights,
        callbacks=[
            ModelCheckpoint(os.path.join(MODEL_DIR, "best_phase1.keras"),
                            monitor="val_auc", mode="max", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
            CSVLogger(os.path.join(MODEL_DIR, "phase1_log.csv")),
        ],
    )

    # Phase 2 — fine-tune top 20% of EfficientNet
    print("\n[5/5] Phase 2 — fine-tuning top layers...")
    base.trainable = True
    freeze_until = int(len(base.layers) * 0.8)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    print(f"       Fine-tuning {len(base.layers)-freeze_until}/{len(base.layers)} layers")

    model.compile(
        optimizer=keras.optimizers.Adam(LR_FINETUNE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, initial_epoch=EPOCHS // 2,
        class_weight=class_weights,
        callbacks=[
            ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"),
                            monitor="val_auc", mode="max", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_auc", patience=7, mode="max", restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-8, verbose=1),
            CSVLogger(os.path.join(MODEL_DIR, "phase2_log.csv")),
        ],
    )

    # Save
    final_path = os.path.join(MODEL_DIR, "dermascan_final.keras")
    model.save(final_path)

    metadata = {
        "class_names":  CLASS_NAMES,
        "img_size":     IMG_SIZE,
        "num_classes":  2,
        "mode":         "binary",
        "model_file":   "dermascan_final.keras",
        "concerning_idx": 0,   # CLASS_NAMES[0] = "concerning"
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    plot_history(h1, h2)

    print("\n── Final Evaluation ──")
    loss, acc, auc = model.evaluate(val_ds)
    print(f"   Loss     : {loss:.4f}")
    print(f"   Accuracy : {acc*100:.1f}%")
    print(f"   AUC      : {auc:.4f}")
    print(f"\n[OK] Model    → {final_path}")
    print(f"[OK] Metadata → {MODEL_DIR}/metadata.json")
    print("\n     Next → python app.py")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPUs detected: {len(gpus)}")
    train()
