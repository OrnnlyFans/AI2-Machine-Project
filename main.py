#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="Artifial  Intelligence 2\nMachine Project", #  Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Artificial Intelligence 2\nMachine Project')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'
    if st.button("Model Predictions", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'predictions'
    if st.button(" References", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Tan, Gabriel Christian D.\n2. Novesteras, Aaron Gabriel L.\n3. Zerda, Thomas Kaden K.\n4. Brown, Ian Miguel A.")
    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of training a Brain Tumor Predictor  using the Brain Tumor dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification/data?fbclid=IwY2xjawNJgSBleHRuA2FlbQIxMQABHuCpx4SyVx1eEZAbeaPquphNQXlw0SGfGkwTdx5NO5vPfRcSs2YK2bpv2MIY_aem_a2vrfIut0uWRjd5gu1TkGg)")
    st.markdown("📔 [Google Colab Notebook](https://colab.research.google.com/drive/1-lZr_QGnVoqgKUNHxGWbknkzz7m-hjWp?usp=sharing)")
    st.markdown("🗄️ [GitHub Repository](https://github.com/OrnnlyFans/AI2-Machine-Project.git)")
#######################
# Data

# Load data
#dataset = pd.read_csv("data/IRIS.csv")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("Brain Tumor Dataset")
    st.write("")

    # Your content for your DATASET page goes here
# Dataset Page
elif st.session_state.page_selection == "predictions":
    st.header("📊 Model Prediction (YOLO)")
# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
#elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here