import gradio as gr
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import time

# Configuration
DATA_PATH = Path('../data/processed/filtered_complaints.csv')
VECTOR_STORE_PATH = Path('../vector_store/chroma_db')
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K = 5

def load_or_create_vector_store():
    """Initialize or create ChromaDB vector store"""
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
    
    # Create collection if it doesn't exist
    try:
        collection = client.get_collection("complaints")
        print("Loaded existing vector store")
    except ValueError:
        print("Creating new vector store...")
        # Load and prepare data
        df = pd.read_csv(DATA_PATH)
        documents = df['clean_narrative'].tolist()
        metadata = df[['Product', 'Complaint ID']].to_dict('records')
        ids = [f"id_{i}" for i in range(len(documents))]
        
        # Create embedding function
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Create collection
        collection = client.create_collection(
            name="complaints",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents in batches
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadata[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
            print(f"Added batch {i//batch_size + 1}/{(len(documents)//batch_size)+1}")
    
    return collection

def search_complaints(query, k=DEFAULT_K, product_filter=None):
    """Search vector store for relevant complaints"""
    start_time = time.time()
    
    # Build filters
    filters = {}
    if product_filter and product_filter != "All":
        filters = {"Product": product_filter}
    
    # Perform search
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=filters
    )
    
    # Process results
    response = []
    for i in range(min(k, len(results['documents'][0]))):
        doc = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        response.append({
            "product": metadata['Product'],
            "complaint_id": metadata['Complaint ID'],
            "distance": f"{distance:.4f}",
            "text": doc[:500] + "..." if len(doc) > 500 else doc
        })
    
    latency = time.time() - start_time
    return response, f"Search took {latency:.2f} seconds | {len(response)} results"

# Initialize vector store
collection = load_or_create_vector_store()

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🔍 CrediTrust Complaint Explorer")
    gr.Markdown("Search customer complaints using semantic search")
    
    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(label="Search Complaints", 
                              placeholder="e.g., 'BNPL late fees' or 'credit card fraud'")
            
            with gr.Row():
                k_slider = gr.Slider(1, 10, value=DEFAULT_K, label="Number of Results")
                product_filter = gr.Dropdown(
                    ["All", "Credit card", "Personal loan", "Buy Now, Pay Later (BNPL)", 
                     "Savings account", "Money transfer"],
                    value="All",
                    label="Filter by Product"
                )
                search_btn = gr.Button("Search", variant="primary")
            
            latency = gr.Label(value="")
        
        with gr.Column(scale=2):
            gr.Markdown("### ℹ️ About This Tool")
            gr.Markdown("""
            - Searches 50,000+ financial complaints
            - Powered by ChromaDB vector database
            - Uses `all-MiniLM-L6-v2` embeddings
            - Filters by product category
            """)
    
    results = gr.DataFrame(
        headers=["Product", "Complaint ID", "Similarity", "Complaint Excerpt"],
        datatype=["str", "str", "str", "str"],
        col_count=(4, "fixed")
    )
    
    # Example queries
    gr.Examples(
        examples=[
            ["Unauthorized BNPL charges"],
            ["Credit card payment processing delays"],
            ["Savings account withdrawal issues"],
            ["Money transfer failed transactions"]
        ],
        inputs=query
    )
    
    # Event handling
    search_btn.click(
        fn=search_complaints,
        inputs=[query, k_slider, product_filter],
        outputs=[results, latency]
    )

if __name__ == "__main__":
    demo.launch(server_port=7861, share=False)