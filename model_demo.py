import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as F
from transformers import SegformerForSemanticSegmentation
#from transformers import SamModel, SamProcessor
import requests

# Load your trained model
def load_model():
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model.to("cuda")  # Move the model to GPU
    model.eval()  # Set the model to evaluation mode
    return model

# Perform semantic segmentation
def perform_segmentation(model, image):
    input_tensor = F.to_tensor(image).unsqueeze(0).to("cuda")  # Move input to GPU
    with torch.no_grad():
        output = model(input_tensor)
    segmented_image = ...  # Process the output to obtain segmented image
    return segmented_image

def main():
    st.title("Auto Semantic Segmentation")
    st.write("Upload an image for semantic segmentation")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        model = load_model()

        if st.button("Segment Image"):
            with Image.open(uploaded_image) as image:
                segmented_image = perform_segmentation(model, image)
                st.image(segmented_image, caption="Segmented Image", use_column_width=True)

if __name__ == "__main__":
    main()