import os
from PIL import Image
import cv2
import numpy as np

def clean_and_reencode_images(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        rgb_img = img.convert("RGB")
                        img_np = np.array(rgb_img)

                        # Save using OpenCV to fully clean metadata
                        cv2.imwrite(file_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                        print(f"✅ Re-encoded: {file_path}")
                except Exception as e:
                    print(f"❌ Failed to clean {file_path}: {e}")

if __name__ == "__main__":
    clean_and_reencode_images("hollywood_faces")
