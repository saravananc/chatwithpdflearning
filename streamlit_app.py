import streamlit as st
from PyPDF2 import PdfReader  # Updated import
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client function
def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

# Function to read and extract text from a PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)  # Use PdfReader directly
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to process large text into manageable chunks and summarize
def process_text(client, text):
    max_chunk_size = 4000
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    processed_chunks = []
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Please summarize the following text:\n\n{chunk}"}
            ]
        )
        processed_chunks.append(response.choices[0].message.content)
    
    return " ".join(processed_chunks)

# Function to ask a question based on processed text
def ask_question(client, processed_text, question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
            {"role": "user", "content": f"Context: {processed_text}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# Streamlit app
st.title("PDF Question Answering App")

# Step 1: Input field to allow the user to provide their OpenAI API key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Ensure the key is stored in session state
if api_key:
    st.session_state["api_key"] = api_key
    st.success("API key saved successfully!")

# Step 2: File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Ensure API key is present and a file is uploaded
if "api_key" in st.session_state and uploaded_file is not None:
    # Initialize OpenAI client with the user's API key
    client = initialize_openai_client(st.session_state["api_key"])
    
    # Process the PDF
    text = read_pdf(uploaded_file)
    processed_text = process_text(client, text)
    st.success("PDF processed successfully!")
    
    # Step 3: Allow user to ask a question about the PDF
    question = st.text_input("Ask a question about the PDF:")
    if question:
        answer = ask_question(client, processed_text, question)
        st.write("Answer:", answer)
else:
    if uploaded_file is None:
        st.warning("Please upload a PDF file.")
    if "api_key" not in st.session_state:
        st.warning("Please enter your OpenAI API key.")
