# Intelligent Complaint Analysis for Financial Services

## Introduction
Problem: CrediTrust receives thousands of complaints daily. Need to extract insights faster.

## Solution
Built RAG-powered chatbot that answers questions from complaint data.

## Technical Choices
- **Data**: CFPB complaints (4 products: Credit Card, Personal Loan, Savings, Money Transfer)
- **Chunking**: 500 characters, 50 overlap
- **Embeddings**: all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **LLM**: Flan-T5-base

## System Evaluation
[Paste evaluation table with scores]

## UI Showcase
[Screenshots]

## Challenges & Learnings
- Large parquet file caused memory issues → used smaller sample
- LLM integration required API alternatives
- Chunk overlap critical for context preservation

## Future Improvements
- Use larger LLM (Mistral-7B)
- Add more products
- Deploy as web app

## Conclusion
RAG system reduces complaint analysis time from days to minutes.