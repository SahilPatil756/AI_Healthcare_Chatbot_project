import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import sys

# Step 1: Simple text-based Q&A system
class SimpleQASystem:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def search_documents(self, query, k=3):
        """Search for relevant documents"""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)
    
    def extract_keywords(self, text):
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def find_best_answer(self, query, documents):
        """Find the best answer from documents"""
        if not documents:
            return "No relevant information found in the documents."
        
        query_keywords = self.extract_keywords(query)
        best_score = 0
        best_doc = None
        
        for doc in documents:
            doc_keywords = self.extract_keywords(doc.page_content)
            # Simple scoring based on keyword overlap
            score = len(set(query_keywords) & set(doc_keywords))
            if score > best_score:
                best_score = score
                best_doc = doc
        
        if best_doc:
            return best_doc.page_content
        else:
            return documents[0].page_content  # Fallback to first document
    
    def answer_question(self, query):
        """Answer a question using document search"""
        print(f"Searching for: {query}")
        
        # Search for relevant documents
        docs = self.search_documents(query, k=3)
        
        if not docs:
            return "I couldn't find any relevant information in the documents to answer your question."
        
        # Find the best answer
        answer = self.find_best_answer(query, docs)
        
        return answer, docs

# Step 2: Load FAISS database
def load_vectorstore():
    """Load the FAISS vector store"""
    DB_FAISS_PATH = "vectorstore/db_faiss"
    
    try:
        print("Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("Loading FAISS vector store...")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded successfully")
        return db
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        print("Please make sure you have run Create_memory_for_llm.py first")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return None

# Step 3: Interactive Q&A
def interactive_qa():
    """Run interactive Q&A session"""
    print("🤖 Simple Document Q&A System")
    print("=" * 40)
    
    # Load vector store
    print("Loading vector store...")
    db = load_vectorstore()
    
    if db is None:
        print("Failed to load vector store. Exiting.")
        return
    
    # Create Q&A system
    qa_system = SimpleQASystem(db)
    
    print("\n✅ System ready! Ask questions about your documents.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user input
            user_query = input("❓ Your question: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_query:
                print("Please enter a question.")
                continue
            
            # Get answer
            answer, source_docs = qa_system.answer_question(user_query)
            
            # Display results
            print("\n" + "="*50)
            print("📝 ANSWER:")
            print(answer)
            print("\n" + "="*50)
            print("📚 SOURCES:")
            for i, doc in enumerate(source_docs, 1):
                print(f"\nSource {i}:")
                print(f"Content: {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"Metadata: {doc.metadata}")
            print("="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.\n")

# Step 4: Main execution
def main():
    """Main function"""
    print("🚀 Starting Document Q&A System...")
    
    # Check if vector store exists
    if not os.path.exists("vectorstore/db_faiss"):
        print("❌ Vector store not found!")
        print("Please run Create_memory_for_llm.py first to create the vector store.")
        return
    
    # Start interactive session
    interactive_qa()

if __name__ == "__main__":
    main()
