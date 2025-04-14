import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import densenet121
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# Define the 8 skin disease classes
classes = [
    'FU-nail-fungus', 'FU-ringworm', 'VI-shingles', 'BA-impetigo',
    'FU-athlete-foot', 'VI-chickenpox', 'PA-cutaneous-larva-migrans', 'BA-cellulitis'
]

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Load a pre-trained DenseNet121 model modified for 8 skin disease classes."""
    model = densenet121(pretrained=True)
    # Modify the classifier to output 8 classes instead of ImageNet's 1000
    model.classifier = nn.Linear(model.classifier.in_features, 8)
    model = model.to('cpu')  # Ensure model runs on CPU
    return model

# Initialize the model
model = load_model()

# Define the target layer for Grad-CAM (last convolutional layer in DenseNet121)
target_layers = [model.features.denseblock4.denselayer16.conv2]

# --- Streamlit App Interface ---
st.title("Skin Disease Classification with XAI Heatmap")
st.write(
    """
    Upload a skin image to classify it into one of 8 skin disease classes using a deep learning model and explainable AI.
    The app processes the image, predicts the skin disease, and highlights the regions influencing the prediction with a heatmap.
    """
)

# File uploader for skin images
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "png", "jpeg"],
    help="Upload a skin image in JPG, PNG, or JPEG format."
)

# Process the uploaded image and display results
if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file).convert('RGB')  # Keep as RGB for skin diseases
    img_resized = img.resize((224, 224))  # Resize to model input size (224x224)
    img_array = np.array(img_resized) / 255.0  # Normalize to [0,1]

    # Normalize with ImageNet mean and std for DenseNet121
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_normalized = (img_array - np.array(mean)[np.newaxis, np.newaxis, :]) / np.array(std)[np.newaxis, np.newaxis, :]

    # Convert to tensor for model input
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float().unsqueeze(0)  # Shape: (1, 3, 224, 224)
    img_tensor = img_tensor.to('cpu')  # Ensure tensor is on CPU

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # Apply softmax to get probabilities

    # Determine the predicted class and its probability
    predicted_class = np.argmax(probs)
    predicted_prob = probs[predicted_class]

    # Generate Grad-CAM heatmap
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_class)]  # Target the predicted class
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    # Ensure grayscale_cam is 2D
    if len(grayscale_cam.shape) == 3:
        grayscale_cam = grayscale_cam[0]  # Shape: (height, width)
    elif len(grayscale_cam.shape) != 2:
        raise ValueError(f"Unexpected shape for grayscale_cam: {grayscale_cam.shape}")

    # Overlay heatmap on the original image
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    # --- Display Results ---
    st.subheader("Analysis Results")
    st.image(img_resized, caption="Processed Input Image (224x224)", use_column_width=True)
    st.image(visualization, caption="Grad-CAM Heatmap Highlighting Regions Influencing the Prediction", use_column_width=True)
    st.write(f"**Predicted Skin Disease:** {classes[predicted_class]}")
    st.write(f"**Probability:** {predicted_prob*100:.2f}%")

# --- Sidebar Information ---
st.sidebar.title("About the Application")
st.sidebar.write(
    """
    **Model:** This app uses a DenseNet121 model, assumed to be fine-tuned on a dataset of skin disease images covering 8 classes:
    'FU-nail-fungus', 'FU-ringworm', 'VI-shingles', 'BA-impetigo', 'FU-athlete-foot', 'VI-chickenpox', 
    'PA-cutaneous-larva-migrans', 'BA-cellulitis'.

    **XAI Method:** Grad-CAM (Gradient-weighted Class Activation Mapping) generates a heatmap to explain the model's predictions 
    by highlighting influential image regions.

    **Purpose:** Demonstrates AI-assisted medical imaging analysis with interpretable results.
    """
)
st.sidebar.warning(
    """
    **Disclaimer:** This is a demonstration app. The model is pre-trained on ImageNet and not fine-tuned on the specific skin disease dataset,
    so predictions are not accurate. Consult a healthcare professional for medical advice.
    """
)

# --- Footer ---
st.write("---")
st.write("Developed under the guidance of Dr. Rama Parvathy L by Saksham Anand and Vasu Arya at VIT Chennai, 2025.")