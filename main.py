#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="Artificial Intelligence 2 - Machine Project",
    page_icon="assets/icon.png",  # change if you have a custom icon
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################
# Initialize page_selection
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # default page

def set_page_selection(page):
    st.session_state.page_selection = page

#######################
# Sidebar
with st.sidebar:
    st.title('Artificial Intelligence 2\nMachine Project')

    st.subheader("Pages")
    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'
    if st.button("Model Predictions", use_container_width=True, on_click=set_page_selection, args=('predictions',)):
        st.session_state.page_selection = 'predictions'
    if st.button("References", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = 'conclusion'

    st.subheader("Members")
    st.markdown("1. Tan, Gabriel Christian D.\n2. Novesteras, Aaron Gabriel L.\n3. Zerda, Thomas Kaden K.\n4. Brown, Ian Miguel A.")

    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of training a Brain Tumor Predictor using the Brain Tumor dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification/data)")
    st.markdown("📔 [Google Colab Notebook](https://colab.research.google.com/drive/1-lZr_QGnVoqgKUNHxGWbknkzz7m-hjWp?usp=sharing)")
    st.markdown("🗄️ [GitHub Repository](https://github.com/OrnnlyFans/AI2-Machine-Project.git)")
#######################
# Pages
# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.write("""
    This project aims to detect and classify brain tumors using YOLOv12.
    The model was trained on the Kaggle Brain Tumor dataset, which contains MRI scans categorized by tumor type.
    This page serves as an overview of the system's goals, approach, and context.
    """)

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset Overview")
    st.write("""
    The dataset contains MRI images labeled under categories such as Meningioma, Glioma, and Pituitary.
    It is used to train YOLO for segmentation and detection tasks.
    """)
    st.info("You can add sample images or charts here to visualize the dataset distribution.")

# Model Predictions Page
elif st.session_state.page_selection == "predictions":
    st.header("🧠 Model Prediction (YOLO)")

    from ultralytics import YOLO
    import cv2
    import numpy as np
    from PIL import Image

    # Load YOLO model once
    @st.cache_resource
    def load_model():
        model_path = "model/best.pt"
        return YOLO(model_path)

    model = load_model()

    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)[:, :, ::-1].copy()  # RGB → BGR for OpenCV

        # YOLO prediction
        results = model.predict(source=image_array, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        r = results[0]
        names = model.names
        detections_text = []

        # Bounding boxes and labels
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes, scores, class_ids):
                label = f"{names[cls_id]} ({score:.2f})"
                detections_text.append(label)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_array, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Segmentation mask overlay
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]))
                colored_mask = np.zeros_like(image_array)
                colored_mask[:, :, 2] = mask  # red overlay
                image_array = cv2.addWeighted(image_array, 1.0, colored_mask, 0.5, 0)

        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Detection Result", use_container_width=True)


        if detections_text:
            st.subheader("🧩 Detected Tumor Types")
            for det in detections_text:
                st.write(f"- {det}")
        else:
            st.warning("No tumor detected.")

# References Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 References")
    st.markdown("""
    **References**
    - Kaggle: Brain Tumor Dataset (Segmentation and Classification)
    - Ultralytics YOLOv12 Documentation
    - Streamlit API Reference

    **Acknowledgments**
    - Our professors and peers for continuous support during model training and testing.
    """)

