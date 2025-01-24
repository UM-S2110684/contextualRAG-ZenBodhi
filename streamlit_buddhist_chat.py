import streamlit as st
import os
import json
import uuid
import clipboard
from datetime import datetime
from glob import glob
from langchain.schema import Document
from langchain.document_loaders import JSONLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Streamlit App Configuration
st.set_page_config(
    page_title="ZenBodhi Chat",
    page_icon="☸️",
    layout="centered"
)

# Copy button handler
def on_copy_click(text):
    try:
        clipboard.copy(text)
        st.success('Text copied successfully!')
    except Exception as e:
        st.error(f"Failed to copy text: {str(e)}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Feedback logging function
def log_feedback(feedback_type, question, response):
    try:
        log_entry = {
            "timestamp": str(datetime.now()),
            "feedback": feedback_type,
            "question": question,
            "response": response
        }
        # Ensure directory exists
        os.makedirs("logs", exist_ok=True)
        # Write to log file
        with open("logs/feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        st.error(f"Error logging feedback: {str(e)}")
        return False

# Buddhist-themed CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5dc;
        color: #4a4a4a;
    }
    .stTextInput > div > div > input {
        background-color: #fff8dc;
        border-radius: 20px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #8b4513;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
    }
    .stMarkdown {
        font-family: 'Georgia', serif;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #e6e6fa;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f0e68c;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #FFFFFF;
        color: #4a4a4a;
        padding: 10px;
        border-top: 1px solid #ddd;
        z-index: 1000;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .footer-links {
        display: flex;
        gap: 20px;
        margin-left: 20px;
    }
    .footer a {
        color: #8b4513;
        text-decoration: none;
        display: flex;
        align-items: center;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .footer img {
        height: 20px;
        margin-right: 5px;
    }
    .footer-copyright {
        margin-right: 20px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
st.sidebar.title("Configuration")

# API Configuration

# Initialize API provider in session state
if "api_provider" not in st.session_state:
    st.session_state.api_provider = "OpenRouter"

# Select API provider
api_provider = st.sidebar.selectbox(
    "Select API Provider",
    ["OpenRouter", "DeepSeek", "Ollama"],
    index=["OpenRouter", "DeepSeek", "Ollama"].index(st.session_state.api_provider)
)
st.session_state.api_provider = api_provider

# API Key Validation Functions
def validate_openrouter_key(key):
    return key.startswith("sk-or-") 

def validate_deepseek_key(key):
    return key.startswith("sk-") 

def validate_ollama_key(key):
    return True  # Ollama doesn't require key validation

# Get API key based on provider
api_key_label = {
    "OpenRouter": "OpenRouter API Key",
    "DeepSeek": "DeepSeek API Key",
    "Ollama": "Ollama API Key (press Enter to skip)"
}[api_provider]

api_key = st.sidebar.text_input(f"Enter your {api_key_label}:", type="password")

# Validate and handle API key changes
if api_key:
    # Validate key format
    validators = {
        "OpenRouter": validate_openrouter_key,
        "DeepSeek": validate_deepseek_key,
        "Ollama": validate_ollama_key
    }
    
    if validators[api_provider](api_key):
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.need_refresh = True  # Flag to refresh chat model
            st.sidebar.success("API key updated successfully!")
        else:
            st.sidebar.info("Using current API key")
    else:
        st.sidebar.error(f"Invalid {api_key_label} format")
        st.stop()
elif st.session_state.api_key:
    st.sidebar.info("Using previously saved API key")
else:
    st.sidebar.warning(f"Please enter your {api_key_label} to continue")
    st.stop()


# Footer Section
st.markdown("""
<div class="footer">
    <div class="footer-links">
        <a href="https://github.com/UM-S2110684/contextualRAG-ZenBodhi" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub">
            GitHub
        </a>
        <a href="https://github.com/UM-S2110684/contextualRAG-ZenBodhi/blob/main/README.md" target="_blank">
            <img src="https://img.icons8.com/?size=100&id=v0YYnU84T2c4&format=png&color=000000" alt="Documentation">
            Documentation
        </a>
        <a href="mailto:contact@zenbodhi.com">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Mail_%28iOS%29.svg" alt="Contact">
            Contact
        </a>
    </div>
    <div class="footer-copyright">
        © 2025 ZenBodhi Chatbot. All rights reserved.
    </div>
</div>
""", unsafe_allow_html=True)

# Load documents and setup RAG system
@st.cache_resource
def initialize_rag_system():
    # Load and process documents
    loader = JSONLoader(file_path='./rag_docs/buddhism_wikidata.jsonl',
                        jq_schema='.',
                        text_content=False,
                        json_lines=True)
    wiki_docs = loader.load()
    
    # Process documents
    wiki_docs_processed = []
    for doc in wiki_docs:
        doc = json.loads(doc.page_content)
        metadata = {
            "title": doc['title'],
            "id": doc['id'],
            "source": "Wikipedia",
            "page": 1
        }
        data = ' '.join(doc['paragraphs'])
        wiki_docs_processed.append(Document(page_content=data, metadata=metadata))

    # Loading back from the JSON file
    def load_chunks_from_json(file_path):
        from langchain.schema import Document
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in data]

    # Loading chunks
    paper_docs = load_chunks_from_json("contextual_chunks.json")

    # Combine all documents
    total_docs = wiki_docs_processed + paper_docs

    # Initialize embeddings and vector DB
    hf_embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chroma_db = Chroma.from_documents(documents=total_docs,
                                    collection_name='my_context_db',
                                    embedding=hf_embed_model,
                                    persist_directory="./my_context_db")

    # Setup retrievers
    similarity_retriever = chroma_db.as_retriever(search_type="similarity",
                                                search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(documents=total_docs, k=5)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, similarity_retriever],
        weights=[0.5, 0.5]
    )
    
    # Setup reranker
    reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    reranker_compressor = CrossEncoderReranker(model=reranker, top_n=5)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker_compressor,
        base_retriever=ensemble_retriever
    )

    # Initialize chat model
    chatgpt = get_chat_model(st.session_state.api_provider, st.session_state.api_key)

    # Setup RAG chain
    rag_prompt = """You are an assistant who is an expert in Buddhist teachings.
                    Answer the following question using only the following pieces of retrieved context.
                    If the answer is not in the context, do not make up answers, just say that you don't know.
                    Keep the answer detailed and well formatted based on the information from the context.

                    Question:
                    {question}

                    Context:
                    {context}

                    Answer:
                """
    rag_prompt_template = ChatPromptTemplate.from_template(rag_prompt)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_rag_chain = (
        {
            "context": (final_retriever | format_docs),
            "question": RunnablePassthrough()
        }
        | rag_prompt_template
        | chatgpt
    )

    return qa_rag_chain

# Function to get chat model based on provider
def get_chat_model(api_provider, api_key):
    if api_provider == "OpenRouter":
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model="anthropic/claude-3.5-sonnet"
        )
    elif api_provider == "DeepSeek":
        return ChatOpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key,
            model="deepseek-chat"
        )
    else:  # Ollama
        return ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Ollama doesn't require an API key
            model="llama2"  # Default model, can be changed
        )

# Store current provider/key in session state
if "current_provider" not in st.session_state:
    st.session_state.current_provider = api_provider
if "current_api_key" not in st.session_state:
    st.session_state.current_api_key = st.session_state.api_key

# Check if provider or key changed
if (st.session_state.current_provider != api_provider or 
    st.session_state.current_api_key != st.session_state.api_key):
    st.session_state.current_provider = api_provider
    st.session_state.current_api_key = st.session_state.api_key
    st.session_state.chat_model = get_chat_model(api_provider, st.session_state.api_key)
    st.sidebar.success("API configuration updated!")
    # Clear cache to force RAG system reinitialization
    initialize_rag_system.clear()

# Initialize chat model
chatgpt = get_chat_model(st.session_state.api_provider, st.session_state.api_key)

# Initialize the RAG system
qa_rag_chain = initialize_rag_system()

# PDF Upload Section
st.sidebar.header("Add Knowledge")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents", 
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    total_chunks = 0
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_path = f"temp_{uuid.uuid4().hex}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process PDF
        loader = PyMuPDFLoader(temp_path)
        pdf_docs = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        pdf_chunks = text_splitter.split_documents(pdf_docs)
        total_chunks += len(pdf_chunks)
        
        # Process chunks with metadata
        processed_chunks = []
        for chunk in pdf_chunks:
            chunk_metadata = chunk.metadata
            chunk_metadata_upd = {
                'id': str(uuid.uuid4()),
                'page': chunk_metadata['page'],
                'source': uploaded_file.name,
                'title': uploaded_file.name,
            }
            processed_chunks.append({
                "page_content": chunk.page_content,
                "metadata": chunk_metadata_upd
            })
        
        # Load existing chunks
        try:
            with open("contextual_chunks.json", "r") as f:
                existing_chunks = json.load(f)
        except FileNotFoundError:
            existing_chunks = []
        
        # Add new chunks
        existing_chunks.extend(processed_chunks)
        
        # Save updated chunks
        with open("contextual_chunks.json", "w") as f:
            json.dump(existing_chunks, f, indent=4)
        
        # Clean up
        os.remove(temp_path)
    
    st.sidebar.success(f"Added {total_chunks} chunks from {len(uploaded_files)} files")
    # Clear cache to force RAG system reinitialization
    initialize_rag_system.clear()

# Chat interface
st.title("☸️ ZenBodhi Chat")
st.caption("Ask questions about Buddhist teachings and philosophy")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question about Buddhism..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Contemplating your question..."):
            try:
                response = qa_rag_chain.invoke(prompt)
                st.markdown(response.content)
                st.button("📋", on_click=on_copy_click, args=(response.content,))
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
                # Store response in session state
                st.session_state.last_response = response.content
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")