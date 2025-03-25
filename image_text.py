import streamlit as st
import requests
from PIL import Image
import io
import pytesseract  # OCR engine
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# Set page config
st.set_page_config(page_title="Text Extraction Pro", page_icon="🔍")
st.title("🔍 Text Extraction from Images")

def extract_text_from_image(image_source, source_type="url"):
    """Extract text from image using Tesseract OCR"""
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
        st.error(f"⚠️ Extraction error: {str(e)}")
        return None

def analyze_with_ai(text):
    """Analyze extracted text using Gemini Pro"""
    try:
        # Initialize with your Google API key
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            google_api_key=st.secrets["google"]["AIzaSyCbfDu9kFG-efA-kLEeSbxXKDd0e-Dj9CA"] 
           
        )
        
        message = HumanMessage(
            content=f"Extract and structure key information from this receipt:\n\n{text}"
            "\nReturn as markdown with headings for each data type."
        )
        
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        st.error(f"🤖 AI analysis failed: {str(e)}")
        return None

# --- Main UI ---
input_method = st.radio("Input Method", ("Image URL", "Upload Image"), index=0)

if input_method == "Image URL":
    url = st.text_input(
        "Enter image URL:",
        value="https://templates.invoicehome.com/receipt-template-us-neat-750px.png"
    )
    if st.button("Extract Text"):
        with st.spinner("🔍 Processing image..."):
            extracted_text = extract_text_from_image(url)
            
            if extracted_text:
                st.subheader("📜 Extracted Text")
                st.text(extracted_text)
                
                st.subheader("🧠 Structured Data")
                analysis = analyze_with_ai(extracted_text)
                if analysis:
                    st.markdown(analysis)
else:
    uploaded_file = st.file_uploader(
        "Upload image", 
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False
    )
    if uploaded_file and st.button("Extract Text"):
        with st.spinner("🔍 Processing image..."):
            extracted_text = extract_text_from_image(uploaded_file.read(), "file")
            
            if extracted_text:
                st.subheader("📜 Extracted Text")
                st.text(extracted_text)
                
                st.subheader("🧠 Structured Data")
                analysis = analyze_with_ai(extracted_text)
                if analysis:
                    st.markdown(analysis)

# 