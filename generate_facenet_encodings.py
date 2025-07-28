import os
import numpy as np
from embedding_utils import load_image, get_embedding

faces_dir = "hollywood_faces"
encodings_dir = "encodings"

if not os.path.exists(encodings_dir):
    os.makedirs(encodings_dir)

for celeb_name in os.listdir(faces_dir):
    celeb_folder = os.path.join(faces_dir, celeb_name)
    if os.path.isdir(celeb_folder):
        embeddings = []

        for image_file in os.listdir(celeb_folder):
            if image_file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(celeb_folder, image_file)
                try:
                    img_pil = load_image(image_path)
                    emb = get_embedding(img_pil)
                    embeddings.append(emb)
                except Exception as e:
                    print(f"❌ Skipped {image_file}: {e}")

        if embeddings:
            mean_embedding = np.mean(np.vstack(embeddings), axis=0)
            mean_embedding /= np.linalg.norm(mean_embedding)
            np.save(os.path.join(encodings_dir, f"{celeb_name}.npy"), mean_embedding)
            print(f"✅ Saved average embedding for {celeb_name}")
        else:
            print(f"⚠️ No valid images for {celeb_name}")
