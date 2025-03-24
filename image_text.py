import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import warnings

# Suppress Google auth warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth")

def main():
    st.title("Image Text Extraction App")
    
    # Option 1: URL input
    image_url = st.text_input(
        "Enter image URL:", 
        value="https://templates.invoicehome.com/receipt-template-us-neat-750px.png"
    )
    
    # Option 2: File upload
    uploaded_file = st.file_uploader("Or upload an image:", type=["png", "jpg", "jpeg"])
    
    if image_url or uploaded_file:
        try:
            if uploaded_file:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_container_width=True)
            else:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Downloaded Image", use_container_width=True)
            
            #
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()