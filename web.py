import streamlit as st
from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation, AutoFeatureExtractor
import requests
from sklearn import feature_extraction
#from image_processing_segformer import SegformerImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sam(uploaded_image):
    processor =  AutoFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)

    #url = input("Enter image url: ")
    image = Image.open(requests.get(uploaded_image, stream=True).raw)


    inputs = feature_extraction(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    return logits

def process_image(uploaded_image):
    # Process the uploaded image using your model or any other logic
    image = Image.open(uploaded_image)
    # Perform some processing on the image
    # For example, you could run some analysis using a machine learning model
    processed_image = image  # Placeholder for processed image
    return processed_image

def main():
    st.title("Meta-SAM")
    st.write("Upload, Process, and Display Image")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Process Image"):
            processed_image = process_image(uploaded_image)
            st.image(processed_image, caption="Processed Image", use_column_width=True)
            
            if st.button("Download Processed Image"):
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                processed_image.save("processed_image.jpg")  # Save the processed image
                st.success("Processed image saved!")

if __name__ == "__main__":
    main()
