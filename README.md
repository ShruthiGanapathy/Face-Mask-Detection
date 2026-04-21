# Face Mask Detection in Public Places Using Deep Learning and Computer Vision

## Project Overview
This project is an automated **Face Mask Detection System** developed using **Deep Learning** and **Computer Vision** techniques. It detects whether a person is wearing a face mask or not in real-time using a webcam.

The system uses the **MobileNetV2** pretrained Convolutional Neural Network model for classification and **OpenCV** for face detection and live video processing.

---

## Features
- Detects faces in real-time using webcam
- Predicts **Mask** or **No Mask**
- Displays confidence percentage
- Real-time bounding box around detected face
- Lightweight and fast model using MobileNetV2
- Easy to run on standard systems

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Algorithm Used
### MobileNetV2 (Transfer Learning)
MobileNetV2 is a lightweight Convolutional Neural Network optimized for image classification tasks. It was used as a pretrained model and fine-tuned for mask detection.

### Haar Cascade Classifier
Used for detecting human faces in real-time webcam frames.

---

## Dataset
The model was trained using a face mask dataset containing two classes:

- With Mask
- Without Mask

Dataset Source: Kaggle

---

## Model Performance
- Validation Accuracy: **98.4%**
- Precision: **99%**
- Recall: **99%**
- F1 Score: **99%**

---

## Project Structure

```text
FaceMaskProject/
│── train_model.py
│── detect_mask.py
│── mask_detector.h5
│── accuracy_graph.png
│── loss_graph.png
│── confusion_matrix.png
│── README.md
