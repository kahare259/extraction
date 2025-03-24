import streamlit as st
import requests
from PIL import Image
import io
import pytesseract  # OCR engine
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure Tesseract path (if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Text Extraction Pro", page_icon="🔍")
st.title("🔍 Text Extraction from Images")

def extract_text_from_image(image_source, source_type="url"):
    try:
        if source_type == "url":
            response = requests.get(image_source)
            img = Image.open(io.BytesIO(response.content))
        else:  # uploaded file
            img = Image.open(io.BytesIO(image_source))
        
        # Convert to grayscale for better OCR
        img = img.convert('L')
        
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text
        
    except Exception as e:
        st.error(f"Extraction error: {str(e)}")
        return None

def analyze_with_ai(text):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    message = HumanMessage(content=f"Extract key information from this receipt:\n\n{text}")
    response = llm.invoke([message])
    return response.content

# Main UI
input_method = st.radio("Input Method", ("Image URL", "Upload Image"), index=0)

if input_method == "Image URL":
    url = st.text_input("Enter image URL:", value="https://templates.invoicehome.com/receipt-template-us-neat-750px.png")
    if st.button("Extract Text"):
        with st.spinner("Processing image..."):
            extracted_text = extract_text_from_image(url)
            if extracted_text:
                st.subheader("Extracted Text")
                st.text(extracted_text)
                
                st.subheader("Structured Data")
                analysis = analyze_with_ai(extracted_text)
                st.write(analysis)
else:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded_file and st.button("Extract Text"):
        with st.spinner("Processing image..."):
            extracted_text = extract_text_from_image(uploaded_file.read(), "file")
            if extracted_text:
                st.subheader("Extracted Text")
                st.text(extracted_text)
                
                st.subheader("Structured Data")
                analysis = analyze_with_ai(extracted_text)
                st.write(analysis)

# Instructions
with st.expander("Setup Instructions"):
    st.markdown("""
    1. **Install Tesseract OCR**:
       - Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
       - Mac: `brew install tesseract`
       - Linux: `sudo apt install tesseract-ocr`
    
    2. **Install Python packages**:
       ```bash
       pip install pytesseract pillow requests streamlit langchain-google-genai
       ```
    
    3. **For better accuracy**:
       - Use clear, high-resolution images
       - Crop to the relevant text area
    """)