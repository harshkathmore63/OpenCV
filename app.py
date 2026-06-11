import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="OpenCV Image Studio", layout="wide")
st.title("OpenCV Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def read_image(file):
    img = Image.open(file)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

operations = [
    "Original Image",
    "Grayscale",
    "Resize",
    "Crop",
    "Rotate",
    "Flip",
    "Gaussian Blur",
    "Median Blur",
    "Sharpen",
    "Canny Edge Detection",
    "Binary Threshold",
    "Adaptive Threshold",
    "Bitwise AND",
    "Drawing Shapes",
    "Add Text on Image",
    "Face, Eye & Smile Detection"
]

selected_op = st.selectbox("Select an OpenCV Operation", operations)

if uploaded_file:
    original = read_image(uploaded_file)
    processed = original.copy()
    h, w = original.shape[:2]

    preview = original.copy()

    if selected_op == "Resize":
        new_w = st.sidebar.slider("Width", 50, w * 2, min(400, w))
        new_h = st.sidebar.slider("Height", 50, h * 2, min(400, h))
        processed = cv2.resize(original, (new_w, new_h))

    elif selected_op == "Crop":
        x = st.sidebar.slider("X", 0, w - 10, 50)
        y = st.sidebar.slider("Y", 0, h - 10, 50)
        cw = st.sidebar.slider("Crop Width", 10, w - x, min(200, w - x))
        ch = st.sidebar.slider("Crop Height", 10, h - y, min(200, h - y))

        preview = original.copy()
        cv2.rectangle(preview, (x, y), (x + cw, y + ch), (0, 255, 0), 3)
        processed = original[y:y + ch, x:x + cw]

    elif selected_op == "Rotate":
        angle = st.sidebar.slider("Rotation Angle (degrees)", -180, 180, 0)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        processed = cv2.warpAffine(original, matrix, (w, h))

    elif selected_op == "Flip":
        flip_type = st.sidebar.selectbox(
            "Flip Type",
            ["Horizontal", "Vertical", "Both"]
        )
        if flip_type == "Horizontal":
            processed = cv2.flip(original, 1)
        elif flip_type == "Vertical":
            processed = cv2.flip(original, 0)
        else:
            processed = cv2.flip(original, -1)

    elif selected_op == "Grayscale":
        processed = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    elif selected_op == "Gaussian Blur":
        processed = cv2.GaussianBlur(original, (9, 9), 0)

    elif selected_op == "Median Blur":
        processed = cv2.medianBlur(original, 7)

    elif selected_op == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed = cv2.filter2D(original, -1, kernel)

    elif selected_op == "Canny Edge Detection":
        processed = cv2.Canny(original, 100, 200)

    elif selected_op == "Binary Threshold":
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        _, processed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    elif selected_op == "Adaptive Threshold":
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    elif selected_op == "Bitwise AND":
        processed = cv2.bitwise_and(original, original)

    elif selected_op == "Drawing Shapes":
        shape_type = st.sidebar.selectbox(
        "Select Shape",
        ["Line", "Circle"])

        processed = original.copy()

        if shape_type == "Line":
            x1 = st.sidebar.slider("Start X", 0, w, 50)
            y1 = st.sidebar.slider("Start Y", 0, h, 50)
            x2 = st.sidebar.slider("End X", 0, w, w - 50)
            y2 = st.sidebar.slider("End Y", 0, h, h - 50)
            thickness = st.sidebar.slider("Thickness", 1, 10, 3)

            cv2.line(
                processed,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness
            )

        elif shape_type == "Circle":
            cx = st.sidebar.slider("Center X", 0, w, w // 2)
            cy = st.sidebar.slider("Center Y", 0, h, h // 2)
            radius = st.sidebar.slider("Radius", 10, min(w, h) // 2, 50)
            thickness = st.sidebar.slider("Thickness", 1, 10, 3)

            cv2.circle(
                processed,
                (cx, cy),
                radius,
                (255, 0, 0),
                thickness)

    elif selected_op == "Add Text on Image":
        processed = original.copy()

        text = st.sidebar.text_input("Enter Text", "Hello OpenCV")
        x = st.sidebar.slider("Text X", 0, w, 50)
        y = st.sidebar.slider("Text Y", 0, h, 50)
        font_scale = st.sidebar.slider("Font Size", 0.5, 3.0, 1.0)
        thickness = st.sidebar.slider("Thickness", 1, 5, 2)

        cv2.putText(
            processed,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
            )

    elif selected_op == "Face, Eye & Smile Detection":
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(processed, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            roi_gray = gray[fy:fy + fh, fx:fx + fw]
            roi_color = processed[fy:fy + fh, fx:fx + fw]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        img_to_show = preview if selected_op == "Crop" else original
        st.image(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        if len(processed.shape) == 2:
            st.image(processed, use_container_width=True)
        else:
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

    buf = io.BytesIO()
    if len(processed.shape) == 2:
        final_img = processed
    else:
        final_img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    Image.fromarray(final_img).save(buf, format="PNG")

    st.download_button(
        "Download Processed Image",
        data=buf.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )
