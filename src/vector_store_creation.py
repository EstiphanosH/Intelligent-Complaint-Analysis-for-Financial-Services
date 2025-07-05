# src/vector_store_creation.py
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import time
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path('../data/processed/filtered_complaints.csv')
VECTOR_STORE_PATH = Path('../vector_store/chroma_db')
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512  # Number of words per chunk
CHUNK_OVERLAP = 64  # Overlap between chunks

def load_data():
    """Load processed complaint data"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}. Run preprocessing first.")
    
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} processed complaints")
    return df

def clean_text(text):
    """Basic text cleaning function"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Move start position with overlap
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
            
        # Break if we're not progressing
        if start >= end:
            break
    
    return chunks

def process_data(df):
    """Process data into chunks with metadata"""
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        # Get clean text
        text = row.get('clean_narrative', '')
        if not text:
            text = clean_text(row.get('Consumer complaint narrative', ''))
        
        # Skip empty texts
        if not text.strip():
            continue
        
        # Create chunks
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                'complaint_id': row.get('Complaint ID', f'id_{idx}'),
                'product': row['Product'],
                'source': 'cfpb',
                'chunk_index': i
            })
            ids.append(f"doc_{idx}_chunk_{i}")
    
    logger.info(f"Created {len(documents)} chunks from {len(df)} narratives")
    return documents, metadatas, ids

def create_vector_store(documents, metadatas, ids):
    """Create and persist ChromaDB vector store"""
    # Create output directory
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize Chroma client
    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_PATH),
        settings=Settings(allow_reset=True)
    )
    
    # Create embedding function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="complaints",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents in batches
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)//batch_size)+1}")
    
    logger.info(f"Vector store created with {collection.count()} items")
    return collection

def test_retrieval(collection):
    """Test retrieval performance"""
    test_queries = [
        "BNPL late fees",
        "Credit card fraud",
        "Loan application denied",
        "Savings account withdrawal issues"
    ]
    
    logger.info("\nRetrieval Performance Test:")
    for query in test_queries:
        start = time.time()
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        latency = time.time() - start
        
        logger.info(f"Query: '{query}' | Latency: {latency:.4f}s")
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['complaint_id']
            logger.info(f"  Result {i+1}: {doc[:70]}... (Source: {source})")

def main():
    logger.info("Starting vector store creation...")
    
    # Step 1: Load data
    try:
        df = load_data()
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        return
    
    # Step 2: Process data into chunks
    documents, metadatas, ids = process_data(df)
    
    # Step 3: Create vector store
    collection = create_vector_store(documents, metadatas, ids)
    
    # Step 4: Test retrieval
    test_retrieval(collection)
    
    logger.info(f"Vector store persisted to {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    main()