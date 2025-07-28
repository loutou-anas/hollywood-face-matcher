import os
import json
import numpy as np
import face_recognition

ENCODING_DIR = "encodings"
FACE_DIR = "hollywood_faces"
FACE_MAP_FILE = "known_faces.json"

def process_faces():
    os.makedirs(ENCODING_DIR, exist_ok=True)
    face_map = {}

    for person_name in os.listdir(FACE_DIR):
        person_path = os.path.join(FACE_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        print(f"🔍 Processing: {person_name}")
        person_encodings = []

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            try:
                print(f"📷 Image: {image_name}")
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    person_encodings.append(encodings[0])
                    print(f"✅ Found {len(encodings)} face(s)")
                else:
                    print(f"⚠️ No face found in {image_name}")
            except Exception as e:
                print(f"❌ Error with {image_name}: {e}")

        if person_encodings:
            encoding_path = os.path.join(ENCODING_DIR, f"{person_name}.npy")
            np.save(encoding_path, np.array(person_encodings))
            face_map[person_name] = encoding_path
            print(f"💾 Saved encodings to {encoding_path}")

    # Save known_faces.json
    with open(FACE_MAP_FILE, "w") as f:
        json.dump(face_map, f, indent=4)
    print(f"\n📘 Face mapping saved to {FACE_MAP_FILE}")

if __name__ == "__main__":
    print(f"🧭 Scanning directory: {os.path.abspath(FACE_DIR)}")
    process_faces()
