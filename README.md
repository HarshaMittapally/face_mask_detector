#  Face Mask Detection using Computer Vision

##  Project Overview

This project is a **real-time face mask detection system** built using **Computer Vision and Deep Learning**. It detects whether a person is wearing a mask or not using a webcam feed.

The system uses a **pre-trained face detector** combined with a **MobileNetV2-based deep learning model** to classify faces into:

*  With Mask
*  Without Mask

This solution can be used in public places like offices, airports, and schools to enforce safety measures.

---

##  Features

*  Real-time detection using webcam
*  Deep learning-based classification
*  Lightweight and fast (MobileNetV2)
*  Accurate mask/no-mask prediction
*  Easy to run and deploy

---

## Tech Stack

* Python
* OpenCV
* TensorFlow / Keras
* NumPy

---

##  Project Structure

```
face_mask_detector/
│── detector.py              # Real-time detection script
│── train.py                 # Model training script
│── mask_detector.h5        # Trained model
│── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│── dataset/
│   ├── with_mask/
│   └── without_mask/
│── requirements.txt
│── README.md
```

---

## Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/HarshaMittapally/face_mask_detector.git
cd face_mask_detector
```

### 2️⃣ Create virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

##  Model Training

Run the training script:
```
python train.py
```

👉 This will:

* Load dataset
* Train MobileNetV2 model
* Save model as `mask_detector.h5`

---

## 🎥 Run Detection

Run the real-time detector:
```
python detector.py
```
👉 Output:

* Webcam opens
* Faces detected
* Labels shown:

  * Mask
  * No Mask

Press **Q** to exit.

---

##  Working Principle

1. Face detection using OpenCV DNN
2. Extract face ROI
3. Preprocess image (resize + normalize)
4. Predict using trained CNN model
5. Display result on screen

Face mask detection systems automate monitoring in real-time environments where manual checking is difficult ([Scribd][1])

---

##  Model Details

* Architecture: MobileNetV2 (Transfer Learning)
* Input Size: 224x224
* Output: Binary classification (Mask / No Mask)
* Loss Function: Binary Crossentropy
* Optimizer: Adam

---

##  Output Example

* Green box → Mask detected
* Red box → No mask detected

---

##  Applications

* Public safety monitoring
* Office/college entry systems
* Airports & railway stations
* Smart surveillance systems

---

## Limitations

* Performance depends on lighting conditions
* May fail on partially covered faces
* Requires good dataset for high accuracy

---

## Future Improvements

* Add face recognition
* Deploy as web/mobile app
* Improve accuracy using larger datasets
* Add alert system (sound/email)

---
