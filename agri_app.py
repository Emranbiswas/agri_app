import streamlit as st
import numpy as np


# Title of the application
st.title("Rice Leaf Disease Detection")

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model("rice_disease_detection_mobilenet.h5")  # Update with your saved model path
    return model

cnn_model = load_trained_model()

# Disease labels
disease_labels = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Leaf Smut"
}

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image of a rice leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Process the image
    def preprocess_image(image_file):
        img = image.load_img(image_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize
        return img_array

    input_image = preprocess_image(uploaded_file)
    
    # Predict the disease
    predictions = cnn_model.predict(input_image)
    predicted_label = np.argmax(predictions, axis=1)[0]
    predicted_disease = disease_labels[predicted_label]
    
    # Display the result
    st.subheader(f"Predicted Disease: {predicted_disease}")
