# 🎭 Hollywood Celebrity Look-Alike App

This is a Streamlit-based web application that uses face recognition to match your photo with a Hollywood celebrity.

---

## 🚀 Features

- Upload your photo via the web interface
- Matches your face with a celebrity from the dataset
- Uses facial embeddings generated via `face_recognition` and `dlib`
- Simple and interactive UI using Streamlit

---

## 🧠 Tech Stack

- Python 3.11
- Streamlit
- face_recognition
- NumPy
- OpenCV
- Pillow
- scikit-learn

---

## 🗂️ Project Structure

```
.
├── app.py                        # Main Streamlit app
├── embedding_utils.py           # Embedding and matching logic
├── generate_facenet_encodings.py# Script to generate encodings
├── encodings/                   # Precomputed .npy embeddings
├── hollywood_faces/             # Celebrity image folders
├── utils/                       # Helper scripts (embedder, cleaner, etc.)
│   ├── embedder.py
│   ├── finder.py
│   ├── image_cleaner.py
│   └── scraper.py
├── known_faces.json             # Mapping of celebrity names to encodings
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/loutou-anas/hollywood-face-matcher.git
cd hollywood-face-matcher

# 2. Set up virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📌 Notes

- To add more celebrities, place their images in `hollywood_faces/<Name>/` and regenerate encodings using `utils/embedder.py`.
- `image_cleaner.py` is useful to sanitize non-RGB or unsupported image formats.
- `scraper.py` is optional; it can help you populate your dataset with Google Image search via `icrawler`.

---

## 📸 Sample Screenshot

![screenshot](./demo/example_output.png)

