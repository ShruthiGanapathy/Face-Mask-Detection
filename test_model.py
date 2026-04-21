import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("mask_detector.h5")

# IMPORTANT: adjust order if needed
labels = ["with_mask", "without_mask"]

# Folder containing test images
folder_path = "testData"

# Allowed formats
valid_ext = (".jpg", ".jpeg", ".png")

for file in os.listdir(folder_path):
    if file.lower().endswith(valid_ext):

        img_path = os.path.join(folder_path, file)

        # Load image
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        result = labels[class_index]

        print(f"{file} --> {result} ({confidence*100:.2f}%)")

        # Show image
        plt.imshow(image.load_img(img_path))
        plt.title(f"{result} ({confidence*100:.2f}%)")
        plt.axis("off")
        plt.show()