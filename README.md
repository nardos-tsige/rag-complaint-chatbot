# RAG Complaint Chatbot

RAG-powered chatbot for analyzing CFPB financial complaints.

## Quick Start

```bash
pip install -r requirements.txt
python app.py

## Project Structure
data/ - Raw and processed data

notebooks/ - Jupyter notebooks for EDA, chunking, RAG, UI

src/ - Core Python modules

tests/ - Unit tests

vector_store/ - FAISS index and embeddings

reports/ - Interim and final reports

Tasks
Task	                             Description	Status
1	EDA & Preprocessing	                      ✅
2	Chunking, Embeddings, Vector Store	      ✅
3	RAG Pipeline & Evaluation	              ✅
4	Interactive UI (Gradio)	                  ✅

Run Tests
```bash
pytest tests/ -v

License
MIT
