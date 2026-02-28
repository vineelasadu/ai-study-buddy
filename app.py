import streamlit as st
from transformers import pipeline
import re

# ---------------- Page Config ----------------
st.set_page_config(page_title="AI Study Buddy", page_icon="📘")

st.title("📘 AI-Powered Study Buddy")
st.write("Enter your study notes or topic below:")

text = st.text_area("Your Text Here", height=220)

# ---------------- Load Model (ONCE) ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

model = load_model()
# ---------------- Generation Settings ----------------
GEN_KWARGS = {
    "max_new_tokens": 300,
    "temperature": 0.3,
    "do_sample": False
}
# ---------------- Summarize ----------------
if st.button("📄 Summarize"):
    if text.strip():
        with st.spinner("Summarizing..."):
            prompt = f"""
Summarize the following text in exactly 3 bullet points.
Each bullet point must be one clear sentence.
Do not add a title or introduction.

Text:
{text}
"""
            output = model(prompt, **GEN_KWARGS)

        st.subheader("📄 Summary")
        content = output[0]["generated_text"].strip()

        # Try to extract bullet-style lines
        bullets = re.findall(r"(?:-|\•)\s*(.*)", content)

        if len(bullets) >= 2:
            for bullet in bullets[:3]:
                st.success(f"• {bullet.strip()}")
        else:
            # Fallback: split model output into sentences (still dynamic)
            sentences = re.split(r"\.\s+", content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 25][:3]

            if sentences:
                st.info("Formatted summary:")
                for s in sentences:
                    st.success(f"• {s}")
            else:
                st.warning("Summary could not be generated clearly. Please try again.")
                st.code(content)

    else:
        st.warning("Please enter text.")
# ---------------- Explain Simply ----------------
if st.button("🧠 Explain Simply"):
    if text.strip():
        with st.spinner("Explaining simply..."):
            prompt = f"""
Explain the following text in simple words.
Write at least 4 complete sentences.
Avoid headings, labels, or short answers.

Text:
{text}
"""
            output = model(prompt, **GEN_KWARGS)
            content = output[0]["generated_text"].strip()

        st.subheader("🧠 Simple Explanation")

        # If explanation is too short, retry with stronger instruction
        if len(content.split()) < 30:
            with st.spinner("Refining explanation..."):
                retry_prompt = f"""
Explain the following topic clearly for a beginner.
Write a short paragraph with simple language.
Do not return just the topic name.

Text:
{text}
"""
                retry_output = model(retry_prompt, **GEN_KWARGS)
                content = retry_output[0]["generated_text"].strip()

        # Final validation
        if len(content.split()) >= 30:
            st.info(content)
        else:
            st.warning("Explanation could not be generated clearly.")
            st.code(content)

    else:
        st.warning("Please enter text.")
# ---------------- Question Generator ----------------
if st.button("📝 Generate Questions"):
    if text.strip():
        with st.spinner("Generating questions..."):
            prompt = f"""
Generate study questions from the text below.

Fill ALL 3 slots with one clear question each.
Each line MUST start with Q1, Q2, Q3.
Do NOT include answers or explanations.

Format EXACTLY like this:
Q1: question?
Q2: question?
Q3: question?

Text:
{text}
"""
            output = model(
                prompt,
                max_length=256,
                temperature=0.7,
                do_sample=True
            )

        st.subheader("📝 Study Questions")

        content = output[0]["generated_text"].strip()

        # Extract Q1, Q2, Q3 safely
        questions = re.findall(r"Q\d:\s*(.*?\?)", content)

        if len(questions) == 3:
            for i, q in enumerate(questions, 1):
                st.success(f"Q{i}. {q}")
        else:
            st.warning("Model output was incomplete. Retrying once...")

            # --- ONE automatic retry ---
            retry = model(prompt, max_length=256, temperature=0.9, do_sample=True)
            retry_content = retry[0]["generated_text"]

            questions = re.findall(r"Q\d:\s*(.*?\?)", retry_content)

            if len(questions) >= 2:
                for i, q in enumerate(questions[:3], 1):
                    st.success(f"Q{i}. {q}")
            else:
                st.error("Could not generate 3 questions. Showing raw output.")
                st.code(content)

    else:
        st.warning("Please enter text.")
# ---------------- Flashcards ----------------
if st.button("🃏 Flashcards"):
    if text.strip():
        with st.spinner("Creating flashcards..."):
            prompt = f"""
Extract exactly 5 important learning points from the text.
Each point should be a clear, complete sentence.
Do not add a title or numbering.

Text:
{text}
"""
            output = model(prompt, **GEN_KWARGS)

        st.subheader("🃏 Flashcards")

        content = output[0]["generated_text"].strip()

        # Strategy:
        # 1. First try to split by line breaks
        lines = [l.strip() for l in content.split("\n") if len(l.strip()) > 25]

        # 2. If not enough lines, chunk the text into sentence-sized pieces
        if len(lines) < 3:
            lines = re.findall(r".{40,200}", content)

        # Limit to 5 flashcards
        points = lines[:5]

        if points:
            for i, point in enumerate(points, 1):
                with st.expander(f"📘 Card {i}"):
                    st.write(point.strip())
        else:
            st.warning("Flashcards could not be generated clearly. Please try again.")
            st.code(content)
    else:
        st.warning("Please enter text.")