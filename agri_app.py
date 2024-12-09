import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

# Title of the application
st.title("Rice Leaf Disease Detection")

# Load the trained model
@st.cache_resource
def load_model():
    class RiceLeafDiseaseMobileNet(nn.Module):
        def __init__(self, num_classes=3):
            super(RiceLeafDiseaseMobileNet, self).__init__()
            self.features = models.mobilenet_v2(pretrained=True)
            self.features.classifier[1] = nn.Linear(self.features.classifier[1].in_features, num_classes)

        def forward(self, x):
            return self.features(x)

    # Initialize model and load state_dict
    model = RiceLeafDiseaseMobileNet()
    model.load_state_dict(torch.load("best_rice_leaf_mobilenetnet.pth", map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

# Disease labels
disease_labels = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Leaf Smut"
}

# Treatment suggestions based on disease
treatment_suggestions = {
    "Bacterial Leaf Blight": (
        "For Bacterial Leaf Blight (BLB):\n\n"
        "• Apply 0.1% Agricin three times at 10-day intervals, starting from the jointing stage, to control BLB and improve rice production."
    ),
    "Brown Spot": (
        "For Brown Spot:\n\n"
        "• Seed treatment with Bavistin (1g/kg of seeds).\n"
        "• Drenching of Bavistin (1g/litre of water) as basal application.\n"
        "• Spraying of Bavistin (1g/litre of water) three times at an interval of 15 days, starting from 30 days after sowing.\n"
        "• Seed treatment with Bavistin (1g/kg of seeds) + Drenching of Bavistin (1g/litre of water) as basal application + Spraying of Bavistin (1g/litre of water) three times at an interval of 15 days, starting from 30 days after sowing.\n"
        "• Seed treatment with Emissan-6 (2g/kg of seeds)."
    ),
    "Leaf Smut": (
        "For Leaf Smut:\n\n"
        "• Use Amistar Top fungicide (200 ml per acre or 1 ml per litre of water)."
        "• Use Bcontrol fungicide (500 ml per acre or 2.5 ml per litre of water)."
        "• Use Roko Fungicide (100 – 200 gm per acre or 0.5 gm per litre of water)."
        "• Use Custodia Fungicide (300 ml per acre or 1 – 1.5 ml per litre of water)."
    )
}

# Function to preprocess the image
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image of a rice leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    input_image = preprocess_image(uploaded_file)

    with torch.no_grad():
        predictions = model(input_image)
        predicted_label = torch.argmax(predictions, dim=1).item()
        predicted_disease = disease_labels[predicted_label]

    st.subheader(f"Predicted Disease: {predicted_disease}")
    #st.text(f"Confidence Scores: {predictions.numpy()}")

    # Display treatment suggestions based on predicted disease
    st.subheader("Treatment Suggestions:")
    st.text(treatment_suggestions[predicted_disease])
