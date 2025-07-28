import streamlit as st
import os
import tempfile
from PIL import Image
from embedding_utils import get_embedding, compare_embeddings, load_known_embeddings

st.set_page_config(page_title="Hollywood Celebrity Look-Alike App", layout="centered")

st.markdown("# 🎭 Hollywood Celebrity Look-Alike App")
st.markdown("Upload your photo and we'll tell you which Hollywood celebrity you look like!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        img_path = tmp_file.name

    # Display uploaded image
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    st.image(image, caption=uploaded_file.name, width=200)

    with st.spinner("🔍 Analyzing face..."):
        try:
            input_embedding = get_embedding(image)
            known_embeddings = load_known_embeddings("encodings")
            match_name, confidence = compare_embeddings(input_embedding, known_embeddings)

            st.success(f"✅ Matched with: **{match_name}** ({confidence * 100:.2f}% confidence)")

            if match_name != "Unknown":
                celeb_img_path = None
                for root, _, files in os.walk("hollywood_faces"):
                    for file in files:
                        if match_name in file:
                            celeb_img_path = os.path.join(root, file)
                            break
                    if celeb_img_path:
                        break

                if celeb_img_path:
                    st.image(Image.open(celeb_img_path), caption=f"{match_name}", width=200)
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
