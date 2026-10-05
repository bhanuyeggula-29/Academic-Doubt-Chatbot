import streamlit as st
from pypdf import PdfReader
import math
import re
from collections import Counter

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="AI Academic Doubt Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Academic Doubt-Solving Chatbot")
st.write("📚 Upload your academic PDF and ask questions from it.")

# --------------------------------
# Chat History
# --------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------
# Text Tokenization
# --------------------------------

def tokenize(text):
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())


# --------------------------------
# TF-IDF Calculation
# --------------------------------

def calculate_tfidf(documents):

    tokenized_docs = [tokenize(doc) for doc in documents]

    total_documents = len(tokenized_docs)

    vocabulary = set()

    for words in tokenized_docs:
        vocabulary.update(words)

    tfidf_vectors = []

    for words in tokenized_docs:

        word_count = Counter(words)

        total_words = len(words)

        vector = {}

        for word in vocabulary:

            # Term Frequency
            tf = (
                word_count[word] / total_words
                if total_words > 0
                else 0
            )

            # Document Frequency
            document_frequency = sum(
                1 for doc in tokenized_docs
                if word in doc
            )

            # Inverse Document Frequency
            idf = math.log(
                (total_documents + 1)
                / (document_frequency + 1)
            ) + 1

            # TF-IDF
            vector[word] = tf * idf

        tfidf_vectors.append(vector)

    return tfidf_vectors


# --------------------------------
# Cosine Similarity
# --------------------------------

def cosine_similarity(vector1, vector2):

    words = set(vector1.keys()) | set(vector2.keys())

    dot_product = sum(
        vector1.get(word, 0)
        * vector2.get(word, 0)
        for word in words
    )

    magnitude1 = math.sqrt(
        sum(value ** 2 for value in vector1.values())
    )

    magnitude2 = math.sqrt(
        sum(value ** 2 for value in vector2.values())
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# --------------------------------
# PDF Upload
# --------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload your academic PDF",
    type=["pdf"]
)

chunks = []
chunk_pages = []
pdf_text = ""

if uploaded_file is not None:

    reader = PdfReader(uploaded_file)

    # Read every page
    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pdf_text += text + "\n"

            # Create chunks
            chunk_size = 500

            for i in range(
                0,
                len(text),
                chunk_size
            ):

                chunk = text[
                    i:i + chunk_size
                ].strip()

                if chunk:

                    chunks.append(chunk)

                    chunk_pages.append(
                        page_number
                    )

    if pdf_text.strip():

        st.success(
            "✅ PDF uploaded and processed successfully!"
        )

        st.write(
            "📄 File:",
            uploaded_file.name
        )

        st.write(
            "📑 Pages:",
            len(reader.pages)
        )

        st.write(
            "🧩 Text chunks:",
            len(chunks)
        )

    else:

        st.error(
            "❌ Could not extract text from this PDF."
        )


# --------------------------------
# Question Section
# --------------------------------

st.subheader("💬 Ask your Doubt")

question = st.text_input(
    "Enter your academic question:",
    placeholder="Example: What is Java?"
)


# --------------------------------
# Ask Button
# --------------------------------

if st.button("🔍 Ask"):

    if not question:

        st.warning(
            "Please enter a question."
        )

    elif not pdf_text:

        st.warning(
            "Please upload a readable PDF first."
        )

    elif not chunks:

        st.warning(
            "No readable content found in the PDF."
        )

    else:

        # Combine PDF chunks and question
        documents = chunks + [question]

        # TF-IDF
        tfidf_vectors = calculate_tfidf(
            documents
        )

        question_vector = tfidf_vectors[-1]

        chunk_vectors = tfidf_vectors[:-1]

        # Calculate similarity
        similarities = []

        for chunk_vector in chunk_vectors:

            score = cosine_similarity(
                question_vector,
                chunk_vector
            )

            similarities.append(score)

        # Best matching chunk
        best_index = similarities.index(
            max(similarities)
        )

        best_score = similarities[
            best_index
        ]

        # Minimum similarity threshold
        if best_score >= 0.10:

            answer = chunks[best_index]

            source_page = chunk_pages[
                best_index
            ]

            source_file = uploaded_file.name

        else:

            answer = (
                "Sorry, relevant information "
                "was not found in the uploaded PDF."
            )

            source_page = None

            source_file = None


        # Save chat
        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "score": best_score,
                "page": source_page,
                "source": source_file
            }
        )


# --------------------------------
# Clear Chat
# --------------------------------

if st.button("🗑️ Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()


# --------------------------------
# Chat History
# --------------------------------

st.subheader("💬 Chat History")

if st.session_state.chat_history:

    for chat in st.session_state.chat_history:

        st.markdown("### 👤 You")

        st.write(
            chat["question"]
        )

        st.markdown("### 🤖 Chatbot")

        st.info(
            chat["answer"]
        )

        # Source + Page + Score
        if chat["page"] is not None:

            st.caption(
                f"📚 Source: {chat['source']} | "
                f"📑 Page: {chat['page']} | "
                f"🎯 Match Score: "
                f"{chat['score'] * 100:.1f}%"
            )

        else:

            st.caption(
                "❌ No relevant source found "
                "in the PDF"
            )

else:

    st.write(
        "No questions asked yet."
    )


# --------------------------------
# Footer
# --------------------------------

st.divider()

st.caption(
    "🎓 AI Academic Doubt-Solving Chatbot | "
    "TF-IDF Based PDF Retrieval"
)