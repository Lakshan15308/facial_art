import streamlit as st
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw
from io import BytesIO

st.set_page_config(
    page_title="Facial Expression Dot Art",
    page_icon="🎨",
    layout="centered",
)

st.title("🎨 Facial Expression Dot Art")
st.write(
    "Upload a portrait to create a dot-art sketch "
    "with highlighted eyes, eyebrows, nose, lips, and outlines."
)


def make_dot_art(image, spacing=4, max_dot=5):
    image = image.convert("RGB")
    image.thumbnail((450, 600))

    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Preserve edges while reducing image noise.
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # Find strong edges.
    edges = cv2.Canny(gray, 35, 90)
    edges = cv2.dilate(
        edges, np.ones((2, 2), np.uint8), iterations=1
    )

    height, width = gray.shape
    result = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(result)

    random.seed(10)

    # Add dots based on image darkness.
    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            darkness = 255 - int(gray[y, x])

            if darkness < 35:
                continue

            size = 1 + (darkness / 255) * (max_dot - 1)

            # Use smaller dots along edges for sharper features.
            if edges[y, x] > 0:
                size = min(size, 1.5)

            size *= random.uniform(0.8, 1.2)
            radius = max(0.5, size / 2)

            draw.ellipse(
                (x - radius, y - radius,
                 x + radius, y + radius),
                fill="black",
            )

    # Add extra fine dots along edges.
    ys, xs = np.where(edges > 0)
    for x, y in zip(xs, ys):
        if random.random() < 0.45:
            draw.ellipse(
                (x, y, x + 1, y + 1),
                fill="black",
            )

    return result


uploaded = st.file_uploader(
    "Upload your photo",
    type=["jpg", "jpeg", "png"],
)

spacing = st.slider(
    "Dot density",
    min_value=2,
    max_value=8,
    value=4,
    help="Lower values create more dots and more detail.",
)

max_dot = st.slider(
    "Maximum dot size",
    min_value=2,
    max_value=7,
    value=5,
)

if uploaded is not None:
    try:
        original = Image.open(uploaded).convert("RGB")
        st.subheader("Original photo")
        st.image(original, use_container_width=True)

        if st.button("Generate Dot-Art Portrait", type="primary"):
            with st.spinner("Creating your portrait..."):
                result = make_dot_art(original, spacing, max_dot)
                st.session_state["dot_art"] = result

    except Exception as error:
        st.error(f"Unable to process this image: {error}")

if "dot_art" in st.session_state:
    result = st.session_state["dot_art"]

    st.subheader("Your dot-art portrait")
    st.image(result, use_container_width=True)

    buffer = BytesIO()
    result.save(buffer, format="PNG")

    st.download_button(
        "Download Dot-Art Portrait",
        data=buffer.getvalue(),
        file_name="facial_dot_art.png",
        mime="image/png",
    )

st.caption(
    "Facial details are emphasized through image-edge detection. "
    "This is not a dedicated facial-expression recognition model."
)