import json
import numpy as np
import face_recognition

FACE_MAP_FILE = "known_faces.json"

def load_embeddings():
    with open(FACE_MAP_FILE, "r") as f:
        face_map = json.load(f)

    known_embeddings = []
    known_names = []

    for name, path in face_map.items():
        encodings = np.load(path)
        for enc in encodings:
            known_embeddings.append(enc)
            known_names.append(name)

    return known_embeddings, known_names

def find_match(image_path, threshold=0.55):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return "No face found", 0.0

    unknown_enc = encodings[0]
    known_embeddings, known_names = load_embeddings()

    distances = face_recognition.face_distance(known_embeddings, unknown_enc)
    best_idx = np.argmin(distances)
    confidence = 1 - distances[best_idx]
    name = known_names[best_idx]

    if confidence >= threshold:
        return name, confidence
    else:
        return "Unknown", confidence
