# RAG Complaint Chatbot for Financial Services

A Retrieval-Augmented Generation system that enables instant natural language querying of customer complaints from financial services.

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Solution Overview](#solution-overview)
- [Key Results](#key-results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Dashboard Features](#dashboard-features)
- [Dagster Orchestration](#dagster-orchestration)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Overview

This project builds an intelligent complaint analysis system for financial services companies. The system transforms raw, unstructured complaint data into actionable insights using Retrieval-Augmented Generation (RAG). Users can ask plain-English questions about customer complaints and receive synthesized, evidence-backed answers in seconds.

The system processes complaint data from the Consumer Financial Protection Bureau (CFPB) dataset and supports querying across four financial product categories: **Credit Cards**, **Personal Loans**, **Savings Accounts**, and **Money Transfers**.

## Business Problem

Financial institutions receive thousands of customer complaints monthly. Product managers spend hours manually reading complaints to identify trends. Support teams are overwhelmed by volume. Compliance teams react slowly to emerging risks. Executives lack visibility into emerging pain points.

The RAG-powered complaint analyzer addresses these challenges by enabling instant natural language querying of complaint data, reducing insight discovery time from days to minutes and empowering non-technical teams to self-serve analytics.

## Solution Overview

The system uses a RAG architecture that combines semantic search with generative AI. Complaint narratives are chunked, embedded using sentence-transformers, and indexed in a FAISS vector store. When a user submits a question, the system retrieves the most relevant complaint chunks and uses a Flan-T5-base language model to generate an evidence-based answer with source attribution.

Key components include:

- Data preprocessing pipeline for cleaning and filtering complaints
- Text chunking with configurable chunk size and overlap
- Embedding generation using `all-MiniLM-L6-v2`
- FAISS vector store for similarity search
- Flan-T5-base for answer generation
- Gradio dashboard for interactive querying
- Dagster orchestration for automated pipeline scheduling

## Key Results

| Metric | Value |
|---|---|
| Total Complaints Processed | 464,000+ |
| Target Products | 4 |
| Chunks Created | 1.37 million |
| Embedding Dimension | 384 |
| Vector Store Size | 50,000+ vectors |
| Average Quality Score | 4.29/5 |

**Quality scores by question:**

| Question | Score |
|---|---|
| Why are customers unhappy with credit cards? | 5/5 |
| What are the main issues with money transfers? | 4/5 |
| How do customers feel about personal loans? | 5/5 |
| What complaints about savings accounts appear most often? | 3/5 |
| Are there recurring problems with hidden fees? | 4/5 |
| What are the most common complaints about credit cards? | 5/5 |
| How do customers describe their experience with money transfers? | 4/5 |

## Quick Start

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- Git

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/nardos-tsige/rag-complaint-chatbot.git
cd rag-complaint-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download the Dataset

Download the CFPB complaint dataset and place it in the `data/raw/` directory:

```bash
mkdir -p data/raw
# Download complaints.csv from CFPB website or use provided file
# Place it in data/raw/complaints.csv
```

### Run the Dashboard

```bash
python app.py
```

The dashboard will be available at `http://127.0.0.1:7860`.

### Run Dagster Pipeline

```bash
dagster dev -m src.dagster
```

The Dagster UI will be available at `http://127.0.0.1:3000`.

## Project Structure

```
rag-complaint-chatbot/
│
├── data/
│   ├── raw/
│   │   └── complaints.csv
│   └── processed/
│       ├── filtered_complaints.csv
│       ├── stratified_sample.csv
│       ├── evaluation_table.csv
│       └── evaluation_summary.csv
│
├── vector_store/
│   ├── index.faiss
│   ├── chunks.pkl
│   └── embeddings.npy
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_chunking_embedding.ipynb
│   ├── 03_rag_pipeline.ipynb
│   └── 04_interactive_ui.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── utils.py
│   └── dagster/
│       └── __init__.py
│
├── tests/
│   └── test_rag_pipeline.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app.py
├── app_enhanced.py
├── requirements.txt
├── workspace.yaml
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

## Technical Details

### Data Source

The project uses the Consumer Financial Protection Bureau (CFPB) complaint dataset, which contains real customer complaints across multiple financial products. Each record includes a short issue label, a free-text consumer narrative, product and company metadata, and submission date.

### Data Preprocessing

Complaints are filtered to four target products: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers. Text narratives are cleaned by lowercasing, removing boilerplate phrases, removing special characters, and normalizing whitespace. Narratives are filtered to a length between 10 and 5000 characters.

### Chunking

Text is chunked into overlapping segments of 500 characters with a 50 character overlap. The chunking strategy preserves sentence boundaries when possible to maintain semantic coherence.

### Embeddings

The `all-MiniLM-L6-v2` sentence-transformer model generates 384-dimensional embeddings for each chunk. Embeddings are normalized for cosine similarity search.

### Vector Store

FAISS (Facebook AI Similarity Search) is used to index embeddings and perform efficient similarity search. The index type is `IndexFlatIP` for inner product similarity on normalized vectors.

### Language Model

Flan-T5-base is used for answer generation. The model receives retrieved chunks as context and generates a concise, evidence-based answer to the user's question.

### Dashboard

The Gradio dashboard provides an interactive interface for querying the system. Features include:

- Text input for questions
- K-value slider to control number of retrieved sources
- Answer display area
- Source attribution with scores and products
- Clear button to reset the conversation

### Orchestration

Dagster is used to orchestrate the data pipeline. The pipeline includes assets for loading data, preprocessing, chunking, embedding generation, vector store indexing, and query answering. The pipeline is scheduled to run daily.

## Dashboard Features

The enhanced dashboard includes:

- Gradient header with company branding
- Product badges for four financial products
- Question input with placeholder text
- K-value slider (1-10 sources)
- Answer display with formatted text
- Source attribution with product and score
- Stats row showing key metrics
- Responsive design with professional colors
- Hover effects on buttons and badges

## Dagster Orchestration

The Dagster pipeline includes the following assets:

| Asset | Description |
|---|---|
| `raw_complaints` | Load complaint data from CSV |
| `prepared_complaints` | Filter and preprocess complaints |
| `chunked_complaints` | Split narratives into chunks |
| `complaint_embeddings` | Generate embeddings for chunks |
| `vector_index` | Build FAISS index and save to disk |
| `answer_query` | Answer a question using RAG pipeline |

The pipeline is scheduled to run daily at 2:00 AM.

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration. The pipeline runs on every push and pull request to the main branch.

**Steps:**

1. Lint code with flake8
2. Run unit tests with pytest
3. Check code formatting with black
4. Build Docker image
5. Scan for vulnerabilities

## Future Improvements

- Fine-tune the LLM on financial complaint data
- Stream complaints from live data sources
- Implement RAGAS evaluation framework
- Add support for Amharic and other local languages
- Deploy dashboard as a mobile-friendly web app
- Allow users to customize prompt templates
- Collect user feedback to improve answers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

**Nardos Tsige**

- GitHub: [nardos-tsige](https://github.com/nardos-tsige)
- Project: [rag-complaint-chatbot](https://github.com/nardos-tsige/rag-complaint-chatbot)

## Acknowledgments

- Consumer Financial Protection Bureau (CFPB) for the complaint dataset
- Hugging Face for Flan-T5 and Sentence-Transformers
- FAISS for vector similarity search
- Gradio for the interactive dashboard framework
- Dagster for pipeline orchestration
- 10 Academy for project guidance and support
