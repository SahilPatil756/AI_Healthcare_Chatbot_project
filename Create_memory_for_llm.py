from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("vectorstore/db_faiss", exist_ok=True)

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
    try:
        loader = DirectoryLoader(data,
                                glob='*.pdf',
                                loader_cls=PyPDFLoader)
        
        documents = loader.load()
        print(f"Loaded {len(documents)} document pages")
        return documents
    except Exception as e:
        print(f"Error loading PDF files: {e}")
        return []

# Check if PDF files exist
pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
if not pdf_files:
    print("Warning: No PDF files found in 'data/' directory.")
    print("Please add some PDF files to the 'data/' folder.")
    
    # Create a sample text file instead for testing
    sample_text = """
    This is a sample document about artificial intelligence. 
    AI is transforming various industries including healthcare, finance, and education.
    Machine learning is a subset of AI that focuses on algorithms learning from data.
    """
    
    with open("data/sample.txt", "w") as f:
        f.write(sample_text)
    print("Created sample.txt for testing purposes.")
    
    # Use TextLoader instead
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader("data/sample.txt")
    documents = loader.load()
else:
    print(f"Found PDF files: {pdf_files}")
    documents = load_pdf_files(data=DATA_PATH)

if not documents:
    print("No documents loaded. Exiting.")
    exit()

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Created {len(text_chunks)} text chunks")
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)

# Step 3: Create Vector Embeddings 
def get_embedding_model():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded successfully")
        return embedding_model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        exit()

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

try:
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS vector store created and saved to {DB_FAISS_PATH}")
    print("Success! Your documents have been processed and stored.")
    
    # Test loading the database
    test_db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("Database loading test successful!")
    
except Exception as e:
    print(f"Error creating FAISS vector store: {e}")