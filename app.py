
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
        return "no relevant information found.", []

    context = "\n\n".join([f"excerpt {i+1}: {r['text']}" for i, r in enumerate(retrieved)])

    prompt = f"""you are a financial analyst assistant for creditrust. your task is to answer questions about customer complaints. use the following retrieved complaint excerpts to formulate your answer. if the context doesn't contain the answer, state that you don't have enough information.

context:
{context}

question: {question}

answer:"""

    answer = generate_text(prompt)
    return answer, retrieved

def respond(question, k=5):
    if not question or question.strip() == "":
        return "please enter a question.", ""

    answer, retrieved = rag_answer(question, k)

    sources_text = ""
    if retrieved:
        sources_text = "\n\n" + "="*50 + "\n"
        sources_text += "sources:\n"
        sources_text += "="*50 + "\n"
        for i, r in enumerate(retrieved):
            sources_text += f"\n[{i+1}] score: {r['score']:.4f}\n"
            sources_text += f"product: {r['product']}\n"
            sources_text += f"text: {r['text'][:300]}\n"

    return answer, sources_text

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
    radius_size=gr.themes.sizes.radius_lg,
    text_size=gr.themes.sizes.text_lg,
)

custom_css = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    padding: 2rem !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
}
.header-section {
    background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 50%, #3182ce 100%) !important;
    padding: 2rem !important;
    border-radius: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 15px 40px rgba(26, 54, 93, 0.3) !important;
    text-align: center !important;
}
.header-title {
    color: #ffffff !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
}
.header-subtitle {
    color: #bee3f8 !important;
    font-size: 1.2rem !important;
    font-weight: 400 !important;
    margin-top: 0.25rem !important;
}
.badge-container {
    display: flex !important;
    justify-content: center !important;
    gap: 0.75rem !important;
    flex-wrap: wrap !important;
    margin: 1rem 0 0.5rem 0 !important;
}
.badge {
    padding: 0.5rem 1.2rem !important;
    border-radius: 9999px !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.025em !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    transition: all 0.3s !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}
.badge:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25) !important;
}
.badge-credit { background: linear-gradient(135deg, #3182ce, #2b6cb0) !important; }
.badge-loan { background: linear-gradient(135deg, #38a169, #2f855a) !important; }
.badge-savings { background: linear-gradient(135deg, #805ad5, #6b46c1) !important; }
.badge-transfer { background: linear-gradient(135deg, #d69e2e, #b7791f) !important; }
.main-card {
    background: rgba(255,255,255,0.92) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 1.5rem !important;
    padding: 2rem !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.08) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}
.label-icon {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #1a202c !important;
    margin-bottom: 0.5rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}
.label-icon .icon {
    font-size: 1.4rem !important;
}
.question-box {
    background: #f7fafc !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 1rem !important;
    padding: 0.75rem !important;
    font-size: 1.05rem !important;
    transition: all 0.3s !important;
}
.question-box:focus {
    border-color: #3182ce !important;
    box-shadow: 0 0 0 4px rgba(49,130,206,0.15) !important;
}
.answer-box {
    background: linear-gradient(135deg, #ebf8ff 0%, #e0edff 100%) !important;
    border: 2px solid #bee3f8 !important;
    border-radius: 1rem !important;
    min-height: 150px !important;
    padding: 1rem !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
    color: #1a202c !important;
}
.sources-box {
    background: linear-gradient(135deg, #f0fff4 0%, #e6f7ed 100%) !important;
    border: 2px solid #c6f6d5 !important;
    border-radius: 1rem !important;
    min-height: 150px !important;
    padding: 1rem !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    color: #2d3748 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.btn-primary {
    background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 0.75rem !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(49,130,206,0.35) !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.025em !important;
}
.btn-primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(49,130,206,0.45) !important;
}
.btn-clear {
    background: #e2e8f0 !important;
    color: #2d3748 !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 0.75rem !important;
    transition: all 0.3s !important;
    font-size: 1.05rem !important;
}
.btn-clear:hover {
    background: #cbd5e0 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
}
.slider-label {
    color: #2d3748 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
.slider-label input {
    accent-color: #3182ce !important;
}
.stats-row {
    display: flex !important;
    justify-content: space-around !important;
    flex-wrap: wrap !important;
    gap: 1rem !important;
    background: #f7fafc !important;
    border-radius: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    margin: 0.75rem 0 0.25rem 0 !important;
    border: 1px solid #e2e8f0 !important;
}
.stats-row span {
    color: #2d3748 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
}
.stat-value {
    color: #3182ce !important;
    font-weight: 700 !important;
}
.footer {
    color: #718096 !important;
    font-size: 0.9rem !important;
    text-align: center !important;
    border-top: 2px solid #e2e8f0 !important;
    padding-top: 1.5rem !important;
    margin-top: 1.5rem !important;
}
.footer strong {
    color: #2d3748 !important;
}
"""

with gr.Blocks(title="creditrust complaint analyzer", theme=custom_theme, css=custom_css, fill_height=False) as demo:
    gr.Markdown("""
    <div class="header-section">
        <h1 class="header-title">🏛️ creditrust financial</h1>
        <p class="header-subtitle">🔍 intelligent complaint analysis system · powered by rag</p>
    </div>
    """)

    gr.Markdown("""
    <div class="badge-container">
        <span class="badge badge-credit">💳 credit cards</span>
        <span class="badge badge-loan">💰 personal loans</span>
        <span class="badge badge-savings">🏦 savings accounts</span>
        <span class="badge badge-transfer">📤 money transfers</span>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("""
            <div class="label-icon">
                <span class="icon">❓</span> ask a question
                <span style="font-weight:400;color:#718096;font-size:0.85rem;">about customer complaints</span>
            </div>
            """)
            question_input = gr.Textbox(
                placeholder="e.g., why are customers unhappy with credit cards?",
                lines=3,
                elem_classes="question-box",
                container=False
            )

            with gr.Row():
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="📚 number of sources to retrieve",
                    elem_classes="slider-label",
                    scale=2
                )

            with gr.Row():
                submit_btn = gr.Button("🚀 analyze", variant="primary", size="lg", elem_classes="btn-primary", scale=1)
                clear_btn = gr.Button("🗑️ clear", variant="secondary", size="lg", elem_classes="btn-clear", scale=1)

            gr.Markdown("""
            <div class="stats-row">
                <span>📊 <span class="stat-value">4</span> products</span>
                <span>📝 <span class="stat-value">10,000</span> complaints</span>
                <span>🔢 <span class="stat-value">384</span> dim embeddings</span>
                <span>🧠 <span class="stat-value">flan-t5-base</span> llm</span>
                <span>⚡ <span class="stat-value">faiss</span> search</span>
            </div>
            """)

        with gr.Column(scale=3, min_width=500):
            gr.Markdown("""
            <div class="label-icon">
                <span class="icon">📋</span> analysis result
            </div>
            """)
            answer_output = gr.Textbox(
                label="",
                lines=6,
                interactive=False,
                elem_classes="answer-box",
                container=False
            )

            gr.Markdown("""
            <div class="label-icon" style="margin-top:0.75rem;">
                <span class="icon">📎</span> supporting sources
            </div>
            """)
            sources_output = gr.Textbox(
                label="",
                lines=8,
                interactive=False,
                elem_classes="sources-box",
                container=False
            )

    gr.Markdown("""
    <div class="footer">
        <strong>🏛️ creditrust financial</strong> · intelligent complaint analysis · retrieval augmented generation
        <br>
        <span style="color:#a0aec0;">all-minilm-l6-v2 · flan-t5-base · faiss · gradio</span>
    </div>
    """)

    submit_btn.click(fn=respond, inputs=[question_input, k_slider], outputs=[answer_output, sources_output])
    clear_btn.click(fn=lambda: ("", "", ""), inputs=[], outputs=[question_input, answer_output, sources_output])

if __name__ == "__main__":
    demo.launch(share=False, debug=False, server_name="0.0.0.0", server_port=7860)
