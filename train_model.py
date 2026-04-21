# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# # --- 1. Settings & Hyperparameters ---
# DATASET_DIR = "dataset"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 5
# LEARNING_RATE = 1e-4

# print("[INFO] Preparing dataset...")

# # --- 2. Data Loading & Preprocessing ---
# # ImageDataGenerator handles loading, resizing, normalizing, and splitting
# datagen = ImageDataGenerator(
#     rescale=1./255,          # Normalize pixel values to [0, 1]
#     validation_split=0.2     # Split 20% of data for validation/testing
# )

# # Training Data
# train_generator = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,    # Resize images to 224x224
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="training"        # Set as training data
# )

# # Validation Data
# val_generator = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="validation"      # Set as validation data
# )

# # --- 3. Build Model (MobileNetV2 + Custom Head) ---
# print("[INFO] Building model...")

# # Load pre-trained MobileNetV2 model (without the top classification layers)
# base_model = MobileNetV2(weights="imagenet", include_top=False,
#                          input_tensor=Input(shape=(224, 224, 3)))

# # Freeze base model layers so their weights aren't changed during training
# for layer in base_model.layers:
#     layer.trainable = False

# # Add a custom classification head on top
# head_model = base_model.output
# head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
# head_model = Flatten(name="flatten")(head_model)
# head_model = Dense(128, activation="relu")(head_model)
# head_model = Dropout(0.5)(head_model)
# head_model = Dense(2, activation="softmax")(head_model) # 2 classes: with mask / without mask

# # Create the final model
# model = Model(inputs=base_model.input, outputs=head_model)

# # --- 4. Compile Model ---
# print("[INFO] Compiling model...")
# optimizer = Adam(learning_rate=LEARNING_RATE)
# model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# # --- 5. Train Model ---
# print("[INFO] Training model for {} epochs...".format(EPOCHS))
# history = model.fit(
#     train_generator,
#     steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
#     validation_data=val_generator,
#     validation_steps=max(1, val_generator.samples // BATCH_SIZE),
#     epochs=EPOCHS
# )

# # --- 6. Save Model ---
# print("[INFO] Saving model to mask_detector.h5...")
# model.save("mask_detector.h5")
# print("[INFO] Training complete and model saved!")


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------------- SETTINGS ----------------
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4

print("[INFO] Preparing dataset...")

# ---------------- DATA AUGMENTATION ----------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Training Data
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Validation Data
val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ---------------- BUILD MODEL ----------------
print("[INFO] Building model...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten()(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

# ---------------- COMPILE ----------------
print("[INFO] Compiling model...")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- TRAIN ----------------
print(f"[INFO] Training model for {EPOCHS} epochs...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ---------------- SAVE MODEL ----------------
print("[INFO] Saving model...")

model.save("mask_detector.h5")

# ---------------- EVALUATION ----------------
print("[INFO] Evaluating model...")

val_generator.reset()
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Mask", "No Mask"],
            yticklabels=["Mask", "No Mask"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# ---------------- ACCURACY GRAPH ----------------
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_graph.png")
plt.show()

# ---------------- LOSS GRAPH ----------------
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_graph.png")
plt.show()

print("[INFO] Training complete.")