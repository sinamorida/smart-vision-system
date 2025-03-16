import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import os
import sqlite3
import insightface

# =====================
# Configuration Settings
# =====================

IMG_SIZE = 112  # InsightFace models typically use 112x112 images
FACE_THRESHOLD = 0.65
GPU_MEM_LIMIT = 1536  # MB
DATABASE_FILE = "face_database.db"
CAPTURE_DIR = "detected_faces"
MIN_CAPTURE_INTERVAL = 5  # seconds

# =====================
# Initialize Models
# =====================

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_LIMIT)]
        )
    except RuntimeError as e:
        print(f"GPU Config Error: {e}")

# Face Detection Model
face_detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Face Embedding Model
face_model = insightface.app.FaceAnalysis()
face_model.prepare(ctx_id=0, nms=0.4)

# Image Classification Model
classification_model = tf.keras.applications.ResNet50V2(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)

# =====================
# Database Functions
# =====================
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces
                (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB)''')
    conn.commit()
    conn.close()

def save_to_db(name, embedding):
    """Save face embedding to database"""
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)",
             (name, embedding.tobytes()))
    conn.commit()
    conn.close()

def load_from_db():
    """Load all face embeddings from database"""
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM faces")
    data = {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in c.fetchall()}
    conn.close()
    return data

# =====================
# Image Processing
# =====================
def process_face(image):
    """Preprocess face image for embedding"""
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return (image / 127.5) - 1.0

def get_embedding(image):
    """Generate face embedding"""
    faces = face_model.get(image)
    if faces:
        return faces[0].embedding
    return None

def classify_image(image):
    """Classify image using ResNet50V2"""
    img = cv2.resize(image, (224, 224))
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    preds = classification_model.predict(np.expand_dims(img, 0))
    return tf.keras.applications.resnet_v2.decode_predictions(preds, top=3)[0]

def similarity_search(query_img, threshold):
    """Search for similar faces in database"""
    embedding = get_embedding(query_img)
    if embedding is None:
        return []

    database = load_from_db()
    results = []
    for name, db_emb in database.items():
        similarity = cosine_similarity([embedding], [db_emb])[0][0]
        if similarity >= threshold:
            results.append((name, similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)

def similarity_search_from_image(image_path, thresh):
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.9:
            box = detections[0,0,i,3:7] * np.array([image.shape[1], image.shape[0]]*2)
            faces.append(box.astype("int"))

    for (x, y, w, h) in faces:
        face_img = image[y:h, x:x+w]

    return similarity_search(face_img, thresh)

def webcam_face_identifier():
    init_db()
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    db = load_from_db()
    last_capture = {}
    register_mode = False
    embeddings = []
    current_name = ""
    known_embs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector.setInput(blob)
        detections = face_detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.9:
                box = detections[0,0,i,3:7] * np.array([frame.shape[1], frame.shape[0]]*2)
                faces.append(box.astype("int"))

        for (x, y, w, h) in faces:
            face_img = frame[y:h, x:x+w]

            if register_mode:
                embeddings.append(get_embedding(face_img))
                cv2.putText(frame, f"Captured: {len(embeddings)}/50", (x+5,y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                emb = get_embedding(face_img)
                if emb is None:
                    continue
                best_match = ("Unknown", 0.0)
                for name, db_emb in db.items():
                    similarity = cosine_similarity([emb], [db_emb])[0][0]
                    if similarity > best_match[1]:
                        best_match = (name, similarity)

                color = (0,255,0) if best_match[1] > FACE_THRESHOLD else (0,0,255)
                cv2.rectangle(frame, (x,y), (w,h), color, 2)
                cv2.putText(frame, f"{best_match[0]} ({best_match[1]:.0%})",
                           (x+5,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        db = load_from_db()

        cv2.imshow('Smart Vision System', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            current_name = input("Enter name: ")
            register_mode = True
        elif key == ord(' ') and register_mode and len(embeddings) >= 50:
            avg_emb = np.mean(embeddings, axis=0)
            save_to_db(current_name, avg_emb)
            db = load_from_db()
            register_mode = False
            embeddings = []

    cap.release()
    cv2.destroyAllWindows()

def object_detection_from_webcam():
    model_path = 'yolo11m.pt'
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = box.cls[0]
                confidence = box.conf[0]

                if confidence > 0.5:
                    label = model.names[int(class_id)]
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def object_detection_from_image(image_path):
    model_path = "yolo11m.pt"
    model = YOLO(model_path)

    img = cv2.imread(image_path)

    results = model.predict(img)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = box.cls[0]
            confidence = box.conf[0]

            if confidence > 0.5:
                label = model.names[int(class_id)]
                color = (0, 255, 0)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
