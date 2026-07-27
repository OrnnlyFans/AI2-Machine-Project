App is hosted here : https://ai2thetaspace-llc-braintumor.streamlit.app/


Project Description
Abstract
This project implements a custom YOLOv12 model to automatically detect and classify brain tumors from MRI scans.
The model was trained on a curated dataset consisting of three primary tumor types: Glioma, Meningioma, and Pituitary.
Through optimized preprocessing, data augmentation, and GPU-accelerated training on Google Colab, the model achieved strong detection accuracy and balanced precision–recall performance.
Results show that YOLOv12 effectively identifies complex medical features with robust spatial consistency, demonstrating its suitability for real-world medical image analysis.

Keywords: Brain tumor detection, YOLOv12, deep learning, MRI classification, object detection, image segmentation, medical imaging, computer vision.

I. Introduction
Brain tumor detection is a critical challenge in medical imaging. Manual MRI interpretation is often subjective, time-consuming, and prone to human error.
With the rise of deep learning, object detection models such as YOLO (You Only Look Once) provide automated and consistent analysis of complex medical images.

In this study, a custom-trained YOLOv12 model was developed to classify and localize tumors across three categories: Glioma, Meningioma, and Pituitary.
The model’s transformer-based backbone enhances feature extraction and spatial awareness, leading to improved accuracy in both bounding box detection and segmentation tasks.

The implementation of such automated systems aims to assist radiologists in diagnostic workflows, enabling faster and more reliable tumor identification compared to traditional methods.

Objectives
The objectives of this project are to:

Develop a YOLOv12-based model for brain tumor detection and classification from MRI scans.
Apply preprocessing and augmentation strategies to enhance model generalization.
Evaluate performance using detection and segmentation metrics (Precision, Recall, mAP50, mAP50–95).
Analyze results to assess YOLOv12’s effectiveness in real-world medical imaging applications.
