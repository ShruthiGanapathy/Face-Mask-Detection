import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained mask detection model from the local file.
# Make sure mask_detector.h5 is in the same folder as this script.
MODEL_PATH = "mask_detector.h5"

# Load Haar Cascade for face detection from OpenCV's built-in data.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load the face detector and the mask detector model.
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
mask_detector = load_model(MODEL_PATH)

# Define labels and colors for display.
LABELS = {0: ("Mask", (0, 255, 0)), 1: ("No Mask", (0, 0, 255))}

# Start video capture from the default webcam (index 0).
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam. Make sure a webcam is connected.")
    exit(1)

print("Press 'Q' to quit.")

while True:
    # Read a frame from the webcam.
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    # Convert the frame to grayscale for face detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame.
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Crop the face region from the frame.
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face to 224x224 pixels as expected by the model.
        face_resized = cv2.resize(face_roi, (224, 224))

        # Normalize pixel values to the range [0, 1].
        face_normalized = face_resized.astype("float32") / 255.0

        # Expand dimensions so the model receives a batch of one image.
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict mask or no mask.
        predictions = mask_detector.predict(face_input)
        confidence = float(np.max(predictions))
        label_index = int(np.argmax(predictions))
        label_text, label_color = LABELS[label_index]

        # Prepare the display text with confidence percentage.
        text = f"{label_text}: {confidence * 100:.2f}%"

        # Draw a rectangle around the face and put the label text above it.
        cv2.rectangle(frame, (x, y), (x + w, y + h), label_color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    # Show the output frame with annotations.
    cv2.imshow("Face Mask Detection", frame)

    # Break the loop when the user presses 'Q' or 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows.
video_capture.release()
cv2.destroyAllWindows()