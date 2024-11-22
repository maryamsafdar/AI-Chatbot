import streamlit as st
import requests
import os

st.set_page_config(page_title="AI Q&A Assistant", page_icon="🤖", layout="wide")

# Ensure the temp directory exists
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Styles for the app
st.markdown(
    """
    <style>
    .main-title {
        color: #4CAF50;
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 18px;
        text-align: center;
        margin-bottom: 40px;
        color: #555555;
    }
    .sidebar-header {
        font-size: 20px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 10px;
    }
    .file-upload {
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Backend API base URL
API_BASE_URL = "http://localhost:8000"

# Function to upload PDF to the backend
def upload_pdf(file):
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload_pdf/",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
        )
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading PDF: {e}")
        return None

# Function to ask a question from the backend
def ask_question(question):
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask_question/",
            json={"question": question}, 
        )
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error asking question: {e}")
        return None

# Chat Display Function
def display_chat():
    for message in st.session_state.get("history", []):
        role = message["role"]
        if role == "user":
            st.chat_message("user").markdown(f"**You:** {message['content']}")
        elif role == "assistant":
            st.chat_message("assistant").markdown(f"**AI:** {message['content']}")

# Main App
def main():
    # Title and Subtitle
    st.markdown('<div class="main-title">🤖 AI Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your PDF documents and ask questions instantly!</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">📂 Upload PDF Files</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        with st.spinner("Uploading and processing your file..."):
            result = upload_pdf(uploaded_file)
            if result:
                st.sidebar.success(result.get("message", "File uploaded successfully."))
            else:
                st.sidebar.error("Failed to upload the file. Please try again.")

    if st.sidebar.button("🔄 Reset Conversation"):
        st.session_state["history"] = []

    # Chat Section
    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_query = st.chat_input("💬 Type your question here...")
    if user_query:
        st.session_state["history"].append({"role": "user", "content": user_query})
        with st.spinner("Generating response..."):
            response = ask_question(user_query)
            if response and "answer" in response:
                st.session_state["history"].append({"role": "assistant", "content": response["answer"]})
            else:
                st.session_state["history"].append(
                    {"role": "assistant", "content": "Sorry, I couldn't process your question. Please try again!"}
                )
        display_chat()

if __name__ == "__main__":
    main()

