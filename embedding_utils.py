import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0, device=device)

def load_image(image_path):
    """
    Open image and convert to RGB (handles grayscale and RGBA PNGs)
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def get_embedding(image_pil):
    """
    Detects face, crops, resizes, and generates 512D embedding
    """
    face_tensor = mtcnn(image_pil)
    if face_tensor is None:
        raise ValueError("No face detected")
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.cpu().numpy()

def compare_embeddings(input_embedding, known_embeddings, threshold=0.35):
    best_match = "Unknown"
    best_score = -1
    for name, emb in known_embeddings.items():
        score = cosine_similarity(input_embedding, emb.reshape(1, -1))[0][0]
        if score > best_score:
            best_score = score
            best_match = name
    if best_score < threshold:
        return "Unknown", best_score
    return best_match, best_score

def load_known_embeddings(encoding_dir="encodings"):
    embeddings = {}
    for file in os.listdir(encoding_dir):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            path = os.path.join(encoding_dir, file)
            embeddings[name] = np.load(path)
    return embeddings
