import gradio as gr
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

index = faiss.read_index('vector_store/index.faiss')
with open('vector_store/chunks.pkl', 'rb') as f:
    chunked_data = pickle.load(f)

embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
if device == 'cuda':
    model = model.to('cuda')

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if device == 'cuda':
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def retrieve(query, k=5):
    query_emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_emb, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunked_data):
            results.append({
                'score': float(distances[0][i]),
                'product': chunked_data[idx]['product_category'],
                'text': chunked_data[idx]['chunk_text']
            })
    return results

def rag_answer(question, k=5):
    retrieved = retrieve(question, k)
    if not retrieved:
        return "No relevant information found.", []
    
    context = "\n\n".join([f"Excerpt {i+1}: {r['text']}" for i, r in enumerate(retrieved)])
    
    prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""
    
    answer = generate_text(prompt)
    return answer, retrieved

def respond(question, k=5):
    if not question or question.strip() == "":
        return "Please enter a question.", ""
    
    answer, retrieved = rag_answer(question, k)
    
    sources_text = ""
    if retrieved:
        sources_text = "\n\n" + "="*50 + "\n"
        sources_text += "SOURCES:\n"
        sources_text += "="*50 + "\n"
        for i, r in enumerate(retrieved):
            sources_text += f"\n[{i+1}] Score: {r['score']:.4f}\n"
            sources_text += f"Product: {r['product']}\n"
            sources_text += f"Text: {r['text'][:300]}\n"
    
    return answer, sources_text

with gr.Blocks(title="CrediTrust Complaint Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # CrediTrust Complaint Analyzer
    ### Ask questions about customer complaints across financial products
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., Why are customers unhappy with credit cards?",
                lines=2
            )
            
            with gr.Row():
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of sources to retrieve (k)"
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
        
        with gr.Column(scale=3):
            answer_output = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False
            )
            sources_output = gr.Textbox(
                label="Sources",
                lines=10,
                interactive=False
            )
    
    gr.Markdown("""
    ---
    **Tip:** The answer is generated from customer complaint data. Sources show the exact complaint excerpts used.
    """)
    
    submit_btn.click(
        fn=respond,
        inputs=[question_input, k_slider],
        outputs=[answer_output, sources_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", ""),
        inputs=[],
        outputs=[question_input, answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=False)