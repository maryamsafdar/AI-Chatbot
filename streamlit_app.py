# import streamlit as st
# import requests

# # Streamlit Page Configuration
# st.set_page_config(page_title="AI Q&A Assistant", page_icon="🤖")

# # FastAPI server URL
# API_URL = "http://127.0.0.1:8000"

# st.title("🤖 AI Q&A Assistant")
# st.subheader("Upload PDFs and ask questions!")

# # Sidebar for file upload
# st.sidebar.header("Upload PDF Files")
# uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# if uploaded_file:
#     with st.spinner("Uploading and indexing your PDF..."):
#         response = requests.post(f"{API_URL}/upload-pdf/", files={"file": uploaded_file})
#         if response.status_code == 200:
#             st.sidebar.success("PDF successfully indexed!")
#         else:
#             st.sidebar.error("Failed to index PDF.")

# # Chat Input
# user_query = st.text_input("Ask your question:")
# if user_query:
#     with st.spinner("Fetching your answer..."):
#         response = requests.post(f"{API_URL}/ask-question/", json={"query": user_query})
#         if response.status_code == 200:
#             answer = response.json()["answer"]
#             context = response.json()["context"]
#             st.markdown(f"**Answer:** {answer}")
#             st.markdown(f"**Context:** {context}")
#         else:
#             st.error("Failed to fetch the answer.")

# import streamlit as st
# import requests

# import os
# st.set_page_config(page_title="AI Q&A Assistant", page_icon="🤖", layout="centered")

# # API base URL
# API_BASE_URL = "http://127.0.0.1:8000"

# # Ensure the temp directory exists
# temp_dir = "temp"
# if not os.path.exists(temp_dir):
#     os.makedirs(temp_dir)

# # Upload PDF to backend
# def upload_pdf(file):
#     try:
#         response = requests.post(
#             f"{API_BASE_URL}/upload_pdf/",
#             files={"file": (file.name, file.getvalue(), "application/pdf")},
#         )
#         return response.json()
#     except Exception as e:
#         st.error(f"Error uploading PDF: {e}")
#         return None

# # Ask a question
# def ask_question(question):
#     try:
#         response = requests.post(
#             f"{API_BASE_URL}/ask_question/",
#             json={"question": question},
#         )
#         return response.json()
#     except Exception as e:
#         st.error(f"Error asking question: {e}")
#         return None

# # UI Components
# st.title("🤖 AI Q&A Assistant")
# st.subheader("Upload your PDF documents and ask questions!")

# # Sidebar for PDF upload
# uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
# if uploaded_file:
#     with st.spinner("Uploading PDF..."):
#         result = upload_pdf(uploaded_file)
#         if result:
#             st.sidebar.success(result.get("message", "File uploaded successfully."))

# # Input and response section
# user_query = st.text_input("Type your question here...")
# if user_query:
#     with st.spinner("Fetching answer..."):
#         response = ask_question(user_query)
#         if response:
#             st.write(f"**Question:** {response.get('question')}")
#             st.write(f"**Answer:** {response.get('answer')}")

# import streamlit as st
# import requests

# # Streamlit Page Configuration
# st.set_page_config(page_title="AI Q&A Assistant", page_icon="🤖")

# # FastAPI server URL
# API_URL = "http://127.0.0.1:8000"

# st.title("🤖 AI Q&A Assistant")
# st.subheader("Upload PDFs and ask questions!")

# # Sidebar for file upload
# st.sidebar.header("Upload PDF Files")
# uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# if uploaded_file:
#     with st.spinner("Uploading and indexing your PDF..."):
#         response = requests.post(f"{API_URL}/upload-pdf/", files={"file": uploaded_file})
#         if response.status_code == 200:
#             st.sidebar.success("PDF successfully indexed!")
#         else:
#             st.sidebar.error("Failed to index PDF.")

# # Chat Input
# user_query = st.text_input("Ask your question:")
# if user_query:
#     with st.spinner("Fetching your answer..."):
#         response = requests.post(f"{API_URL}/ask-question/", json={"query": user_query})
#         if response.status_code == 200:
#             answer = response.json()["answer"]
#             context = response.json()["context"]
#             st.markdown(f"**Answer:** {answer}")
#             st.markdown(f"**Context:** {context}")
#         else:
#             st.error("Failed to fetch the answer.")

import streamlit as st
import requests

import os
st.set_page_config(page_title="AI Q&A Assistant", page_icon="🤖", layout="centered")

# API base URL
API_BASE_URL = "http://localhost:8000"

# Ensure the temp directory exists
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Upload PDF to backend
def upload_pdf(file):
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload_pdf/",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
        )
        return response.json()
    except Exception as e:
        st.error(f"Error uploading PDF: {e}")
        return None
    
# http://localhost:8000/ask_question/?question=hi%20ai%20how%20are%20you%3F%20what%20are%20you%20doing%20these%20day
# Ask a question
def ask_question(question):
    try:
        res = requests.post(f"http://host.docker.internal:8000/ask_question/?question={question}")
        return res.json()
    except Exception as e:
        st.error(f"Error asking question: {e}")
        return None
    

def test():
    res = requests.get("http://host.docker.internal:8000/test-endpoint/")
    st.write(res.json())

# def main():

#     # UI Components
#     st.title("🤖 AI Q&A Assistant")
#     st.subheader("Upload your PDF documents and ask questions!")

#     # Sidebar for PDF upload
#     uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
#     if uploaded_file:
#         with st.spinner("Uploading PDF..."):
#             result = upload_pdf(uploaded_file)
#             if result:
#                 st.sidebar.success(result.get("message", "File uploaded successfully."))

#     # Input and response section
#     user_query = st.text_input("Type your question here...")
#     if user_query:
#         with st.spinner("Fetching answer..."):
#             response = ask_question(user_query)
#             if response:
#                 st.write(f"**Question:** {response.get('question')}")
#                 st.write(f"**Answer:** {response.get('answer')}")

# if __name__ == "__main__":
#     main()

test()


user_query = st.text_input("Type your question here...")
if user_query:
    with st.spinner("Fetching answer..."):
        response = ask_question(question=user_query)
        if response:
            st.write(f"**Question:** {response.get('question')}")
            st.write(f"**Answer:** {response.get('answer')}") 