import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def main():
    st.title('🖼️ Image to Text Converter')
    st.markdown("*Please Upload an image to convert it into text.*")


   

    st.sidebar.title('📷 Image to Text Converter')
    st.sidebar.markdown("""
    This tool allows you to convert images into text. 
    Upload an image, and the tool will process it to extract the text content.
    """)

    # Add larger banana emoji
    st.sidebar.markdown("<h1 style='text-align: center; color: #F9A825; font-size: 150px;'>🖼️</h1>", unsafe_allow_html=True)

    # Use configure to set the api key
    genai.configure(api_key=os.getenv("Google_api_key"))

    # Set up the model
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    model = genai.GenerativeModel(model_name="gemini-pro-vision",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    user_question = st.text_input("Ask a question about the image:")

    if uploaded_file is not None and user_question:
        if st.button('Submit'):
            image_parts = [{"mime_type": "image/png", "data": uploaded_file.getvalue()}]
            prompt_parts = [user_question + "\n", image_parts[0]]
            response = model.generate_content(prompt_parts)
            st.write(response.text)

if __name__ == "__main__":
    main()
