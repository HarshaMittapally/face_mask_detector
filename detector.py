# ============================================================
# detector.py
# PURPOSE: Use the trained model + webcam to detect
#          face masks in real time.
# RUN: python detector.py
# ============================================================

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ── Step 1: Load models ────────────────────────────────────────
print("[INFO] Loading face detector model...")
prototxt_path = "face_detector/deploy.prototxt"
weights_path  = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

print("[INFO] Loading mask detector model...")
mask_net = load_model("mask_detector.h5")

# ── Step 2: Helper function ────────────────────────────────────
def detect_and_predict_mask(frame, face_net, mask_net):
    """
    Given a video frame:
    1. Detect all faces using face_net
    2. For each face, predict mask/no-mask using mask_net
    Returns: list of face bounding boxes and predictions
    """
    (h, w) = frame.shape[:2]

    # Create a blob from the image (required by OpenCV DNN)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces     = []
    locations = []
    preds     = []

    # Loop over each detected face
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections (below 50% confidence)
        if confidence > 0.5:
            # Compute bounding box coordinates
            box  = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clamp to frame boundaries
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX,   endY)   = (min(w - 1, endX), min(h - 1, endY))

            # Crop face, convert to RGB, resize to 224x224
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)   # normalize

            faces.append(face)
            locations.append((startX, startY, endX, endY))

    # Predict all faces at once (batch prediction = faster)
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return locations, preds


# ── Step 3: Start webcam ───────────────────────────────────────
print("[INFO] Starting webcam... Press 'q' to quit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    (locations, preds) = detect_and_predict_mask(frame, face_net, mask_net)

    # Draw results on frame
    for (box, pred) in zip(locations, preds):
        (startX, startY, endX, endY) = box
        (mask_prob, no_mask_prob)    = pred

        # Choose label and color based on prediction
        if mask_prob > no_mask_prob:
            label = "Mask"
            color = (0, 200, 0)        # Green = safe
            confidence = mask_prob
        else:
            label = "No Mask"
            color = (0, 0, 220)        # Red = warning
            confidence = no_mask_prob

        # Display label with confidence percentage
        display_text = f"{label}: {confidence * 100:.1f}%"

        # Draw bounding box and label
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.rectangle(frame, (startX, startY - 30),
                      (endX, startY), color, -1)           # filled label bg
        cv2.putText(frame, display_text,
                    (startX + 5, startY - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Show the output frame
    cv2.imshow("Face Mask Detector  |  Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Detector stopped.")