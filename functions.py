import os
import logging
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT, PINECONE_INDEX_NAME
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

#Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_API_ENVIRONMENT
        )
    )

# Connect to the Pinecone index
index = pinecone_client.Index(PINECONE_INDEX_NAME)


# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

# Load PDF into VectorDB
def load_pdf_to_vector_db(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        vector_store.add_documents(documents)
        return vector_store
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return None

# QA Function
def qa_function(query):
    qa_prompt_template = """
    You are an assistant for answering questions.

    If context is provided below, use it to answer the question concisely.

    Context:
    {context}

    Question: {input}
    """
    qa_prompt = PromptTemplate.from_template(qa_prompt_template)
    qa_chain = create_stuff_documents_chain(ChatOpenAI(api_key=OPENAI_API_KEY), qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=qa_chain)
    result = retrieval_chain.invoke({"input": query})
    return result["answer"], result.get("context", [])
