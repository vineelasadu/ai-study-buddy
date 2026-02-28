import streamlit as st
from transformers import pipeline
import re
import torch

# ---------------- Page Config ----------------
st.set_page_config(page_title="AI Study Buddy", page_icon="📘", layout="centered")

st.title("📘 AI-Powered Study Buddy")
st.markdown("Transform your notes into summaries, simple explanations, or quizzes.")

text = st.text_area("Paste your study notes or a complex topic here:", height=250, placeholder="e.g., Photosynthesis is the process by which green plants...")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    # Automatically detect GPU
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text2text-generation",   
        model="google/flan-t5-base",
        device=device
    )

model = load_model()

# ---------------- Helper Function ----------------
def generate_response(prompt, max_tokens=300, temp=0.3):
    output = model(
        prompt, 
        max_new_tokens=max_tokens, 
        temperature=temp, 
        do_sample=True if temp > 0.1 else False,
        top_p=0.9
    )
    return output[0]["generated_text"].strip()

# ---------------- Summarize ----------------
if st.button("📄 Summarize"):
    if text.strip():
        with st.spinner("Creating bullet points..."):
            prompt = f"Summarize the following text into a few clear bullet points: {text}"
            content = generate_response(prompt)
            
            st.subheader("📄 Summary")
            # Split by common bullet markers or newlines
            points = re.split(r'\n|•|\*', content)
            for p in points:
                if len(p.strip()) > 5:
                    st.success(f"• {p.strip()}")
    else:
        st.warning("Please enter some text first.")

# ---------------- Explain Simply ----------------
if st.button("🧠 Explain Simply"):
    if text.strip():
        with st.spinner("Simplifying concepts..."):
            # Using the "explain like I'm 5" approach for better results
            prompt = f"Explain this concept to a child in simple terms: {text}"
            content = generate_response(prompt, max_tokens=400, temp=0.5)
            
            st.subheader("🧠 Simple Explanation")
            st.info(content)
    else:
        st.warning("Please enter some text first.")

# ---------------- Question Generator ----------------
if st.button("📝 Generate Questions"):
    if text.strip():
        with st.spinner("Thinking of questions..."):
            prompt = f"Generate 3 study questions based on this text: {text}"
            content = generate_response(prompt, temp=0.7)
            
            st.subheader("📝 Study Questions")
            # Cleaner way to find questions without strict Q1: format
            questions = re.split(r'\d[\.\)\:]', content)
            count = 1
            for q in questions:
                if "?" in q:
                    st.write(f"**{count}.** {q.strip()}")
                    count += 1
    else:
        st.warning("Please enter some text first.")

# ---------------- Flashcards ----------------
if st.button("🃏 Flashcards"):
    if text.strip():
        with st.spinner("Preparing flashcards..."):
            prompt = f"What are the 5 most important facts in this text? {text}"
            content = generate_response(prompt, temp=0.4)
            
            st.subheader("🃏 Flashcards (Click to expand)")
            # Split by sentences or lines
            facts = re.split(r'\n|\. ', content)
            fact_count = 0
            for f in facts:
                if len(f.strip()) > 10 and fact_count < 5:
                    fact_count += 1
                    with st.expander(f"Fact {fact_count}"):
                        st.write(f.strip() + (". " if not f.strip().endswith('.') else ""))
    else:
        st.warning("Please enter some text first.")