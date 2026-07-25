import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 
 
from dagster import asset, Definitions 
import pandas as pd 
import numpy as np 
from pathlib import Path 
 
DATA_FILE = "data/processed/filtered_complaints.csv" 
 
@asset 
def raw_complaints(): 
    df = pd.read_csv(DATA_FILE) 
    return df.to_dict('records') 
 
@asset 
def prepared_complaints(raw_complaints): 
    df = pd.DataFrame(raw_complaints) 
    df['cleaned_narrative'] = df['Consumer complaint narrative'] 
    return df.to_dict('records') 
 
@asset 
def chunked_complaints(prepared_complaints): 
    df = pd.DataFrame(prepared_complaints) 
    chunks = [] 
    for idx, row in df.iterrows(): 
        chunks.append({ 
            'complaint_id': idx, 
            'chunk_index': 0, 
            'total_chunks': 1, 
            'chunk_text': row['cleaned_narrative'], 
            'chunk_length': len(row['cleaned_narrative']), 
            'product_category': row['Product'] 
        }) 
    return chunks 
 
@asset 
def complaint_embeddings(chunked_complaints): 
    from sentence_transformers import SentenceTransformer 
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    texts = [c['chunk_text'] for c in chunked_complaints] 
    embeddings = model.encode(texts, normalize_embeddings=True) 
    return embeddings.tolist() 
 
@asset 
def vector_index(chunked_complaints, complaint_embeddings): 
    import faiss 
    embeddings = np.array(complaint_embeddings).astype(np.float32) 
    index = faiss.IndexFlatIP(embeddings.shape[1]) 
    index.add(embeddings) 
    Path("vector_store").mkdir(exist_ok=True) 
    faiss.write_index(index, "vector_store/index.faiss") 
    import pickle 
    with open("vector_store/chunks.pkl", "wb") as f: 
        pickle.dump(chunked_complaints, f) 
    return {"total_vectors": index.ntotal} 
 
@asset 
def answer_query(vector_index): 
    import faiss 
    import pickle 
    from sentence_transformers import SentenceTransformer 
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    index = faiss.read_index("vector_store/index.faiss") 
    with open("vector_store/chunks.pkl", "rb") as f: 
        chunks = pickle.load(f) 
    query = "Why are customers unhappy with credit cards?" 
    query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32) 
    distances, indices = index.search(query_emb, 5) 
    results = [] 
    for i, idx in enumerate(indices[0]): 
            results.append({ 
                'score': float(distances[0][i]), 
                'product': chunks[idx]['product_category'], 
                'text': chunks[idx]['chunk_text'] 
            }) 
    return {"query": query, "results": results} 
 
defs = Definitions(assets=[raw_complaints, prepared_complaints, chunked_complaints, complaint_embeddings, vector_index, answer_query]) 
