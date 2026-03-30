# ============================================================
# train_model.py
# PURPOSE: Train a MobileNetV2 model to classify
#          "mask" vs "no mask" face images.
# RUN THIS ONCE to generate mask_detector.model
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Step 1: Load images from dataset/ folder ──────────────────
DATASET_PATH = r"C:\Users\medeh\Desktop\face_mask_detector\data"
CATEGORIES   = ["with_mask", "without_mask"]
IMG_SIZE     = 224   # MobileNetV2 expects 224x224

print("[INFO] Loading images...")

data   = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        # Load and resize image
        image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        image = img_to_array(image)
        image = preprocess_input(image)   # normalize for MobileNetV2
        data.append(image)
        labels.append(category)

# ── Step 2: Encode labels (with_mask=1, without_mask=0) ───────
lb     = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data   = np.array(data, dtype="float32")

# Train/test split (80% train, 20% test)
(X_train, X_test, y_train, y_test) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42)

# ── Step 3: Data augmentation (helps prevent overfitting) ─────
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ── Step 4: Build model (MobileNetV2 + custom head) ───────────
print("[INFO] Building model...")

# Load MobileNetV2 WITHOUT the top classification layers
base_model = MobileNetV2(
    weights="imagenet",      # use ImageNet pre-trained weights
    include_top=False,       # remove the original classifier
    input_tensor=Input(shape=(224, 224, 3))
)

# Freeze base model layers (we only train our custom head)
for layer in base_model.layers:
    layer.trainable = False

# Add our own classification head
head = base_model.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten(name="flatten")(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)   # 2 classes

model = Model(inputs=base_model.input, outputs=head)

# ── Step 5: Compile & train ────────────────────────────────────
INIT_LR  = 1e-4
EPOCHS   = 20
BS       = 32

print("[INFO] Training model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(
    aug.flow(X_train, y_train, batch_size=BS),
    steps_per_epoch=len(X_train) // BS,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BS,
    epochs=EPOCHS
)

# ── Step 6: Evaluate & save ────────────────────────────────────
print("[INFO] Evaluating model...")
pred = model.predict(X_test, batch_size=BS)
pred = np.argmax(pred, axis=1)
print(classification_report(np.argmax(y_test, axis=1), pred,
                             target_names=lb.classes_))

print("[INFO] Saving model to mask_detector.model ...")
model.save("mask_detector.h5", save_format="h5")

# ── Plot training accuracy/loss ────────────────────────────────
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"],     label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.plot(history.history["accuracy"],     label="Train acc")
plt.plot(history.history["val_accuracy"], label="Val acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")
print("[INFO] Training complete! Plot saved as training_plot.png")