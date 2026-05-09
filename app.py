
import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import shutil
# --- Streamlit UI ---
st.set_page_config(
    layout="wide",
    page_title="GastroVision AI",
    page_icon="🩺"
)


# --- Configuration ---
# Ensure the model path is correct relative to the app.py when deployed
MODEL_PATH = 'ultimate_best_model_stage3.pth'
CLASSES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
IMAGE_SIZE = (300, 300)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Image Transformations (Validation Transform) ---
val_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# --- Model and GradCAM Loading ---
@st.cache_resource
def load_model_and_cam():
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(CLASSES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}. Make sure it's in the same directory as app.py.")
        st.stop()
    model.eval()
    model.to(device)

    target_layer = model.conv_head # EfficientNet last layer
    cam = GradCAM(model=model, target_layers=[target_layer])
    return model, cam

model, cam = load_model_and_cam()

# --- Prediction and GradCAM Function ---
def get_prediction_and_gradcam(image_pil):
    input_tensor = val_tf(image_pil).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_label_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label_name = CLASSES[predicted_label_idx]
        predicted_probability = probabilities[0, predicted_label_idx].item()

    # Generate GradCAM heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    # Prepare image for visualization (resize to match input to model for consistent heatmap overlay)
    img_np = np.array(image_pil.resize(IMAGE_SIZE)) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    return predicted_label_name, predicted_probability, img_np, visualization


# --- Sidebar ---
st.sidebar.title("🩺 GastroVision AI")
st.sidebar.markdown("""
### EfficientNet + GradCAM

AI-powered gastrointestinal disease classification using the Kvasir dataset.

### Model Details
- Backbone: EfficientNet-B3
- Classes: 8
- Accuracy: ~93%
- Explainability: GradCAM
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "This project demonstrates deep learning-based gastrointestinal disease classification with explainable AI visualization."
)

# --- Main Title ---
st.title("🩺 GastroVision AI Dashboard")
st.markdown(
    "### AI-based Gastrointestinal Disease Classification using EfficientNet and GradCAM"
)

# --- Dataset Information ---
st.markdown("---")
st.subheader("📊 About the Kvasir Dataset")

col1, col2 = st.columns(2)

with col1:
    st.write("""
The Kvasir dataset is a medical image dataset containing gastrointestinal tract images captured during endoscopy procedures.

### Dataset Features
- 8 GI disease classes
- Real clinical endoscopy images
- Used widely in medical AI research
- Supports explainable AI studies
""")

with col2:
    st.write("""
### Classes
- Dyed Lifted Polyps
- Dyed Resection Margins
- Esophagitis
- Normal Cecum
- Normal Pylorus
- Normal Z-Line
- Polyps
- Ulcerative Colitis
""")

# --- Sample Images Section ---
st.markdown("---")
st.subheader("🖼️ Sample Dataset Images")

sample_images_dir_app = 'sample_images'

if os.path.exists(sample_images_dir_app):
    sample_files = [
        f for f in os.listdir(sample_images_dir_app)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if sample_files:
        cols = st.columns(min(4, len(sample_files)))

        for idx, sample_file in enumerate(sample_files[:4]):
            sample_path = os.path.join(sample_images_dir_app, sample_file)

            try:
                sample_img = Image.open(sample_path)
                cols[idx].image(sample_img, caption=sample_file, use_column_width=True)
            except:
                pass

# --- Upload Section ---
st.markdown("---")
st.subheader("📤 Upload Endoscopy Image")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# --- Prediction Logic ---
if uploaded_file is not None:

    image_pil = Image.open(uploaded_file).convert('RGB')

    st.markdown("---")
    st.subheader("🖼️ Uploaded Image")

    st.image(image_pil, width=350)

    with st.spinner("Analyzing image using EfficientNet-B3..."):

        predicted_label, probability, original_img_np, gradcam_viz = get_prediction_and_gradcam(image_pil)

    st.markdown("---")
    st.subheader("📌 Prediction Results")

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Predicted Disease",
        predicted_label
    )

    metric_col2.metric(
        "Confidence",
        f"{probability:.2%}"
    )

    st.markdown("---")
    st.subheader("🔥 GradCAM Explainability")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original_img_np,
            caption='Original Image',
            use_column_width=True
        )

    with col2:
        st.image(
            gradcam_viz,
            caption='GradCAM Heatmap',
            use_column_width=True
        )

    st.success("Analysis Complete!")

# --- Footer ---
st.markdown("---")

st.markdown("""
### 📚 Project Workflow
1. Upload Endoscopy Image  
2. EfficientNet-B3 Performs Classification  
3. GradCAM Generates Explainability Heatmap  
4. Final Disease Prediction Displayed  

---

### 👨‍💻 Research Project
Developed using Transfer Learning and Explainable AI for gastrointestinal disease classification.
""")
