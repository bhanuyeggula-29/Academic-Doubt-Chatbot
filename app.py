from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
import math
import re
from collections import Counter

app = Flask(__name__)

chunks = []
chunk_pages = []
pdf_name = ""


def tokenize(text):
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())


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
            tf = word_count[word] / total_words if total_words > 0 else 0

            document_frequency = sum(
                1 for doc in tokenized_docs if word in doc
            )

            idf = math.log(
                (total_documents + 1) /
                (document_frequency + 1)
            ) + 1

            vector[word] = tf * idf

        tfidf_vectors.append(vector)

    return tfidf_vectors


def cosine_similarity(vector1, vector2):

    words = set(vector1.keys()) | set(vector2.keys())

    dot_product = sum(
        vector1.get(word, 0) * vector2.get(word, 0)
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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():

    global chunks
    global chunk_pages
    global pdf_name

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "message": "No PDF file selected."
        })

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "message": "Please select a PDF."
        })

    try:

        reader = PdfReader(file)

        chunks = []
        chunk_pages = []
        pdf_name = file.filename

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text:

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

        if not chunks:

            return jsonify({
                "success": False,
                "message":
                "Could not extract readable text from this PDF."
            })

        return jsonify({

            "success": True,

            "message":
            "PDF uploaded successfully!",

            "file": pdf_name,

            "pages":
            len(reader.pages),

            "chunks":
            len(chunks)
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })


@app.route("/ask", methods=["POST"])
def ask_question():

    if not chunks:

        return jsonify({
            "success": False,
            "message":
            "Please upload a PDF first."
        })

    data = request.get_json()

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:

        return jsonify({
            "success": False,
            "message":
            "Please enter a question."
        })

    documents = chunks + [question]

    tfidf_vectors = calculate_tfidf(
        documents
    )

    question_vector = tfidf_vectors[-1]

    chunk_vectors = tfidf_vectors[:-1]

    similarities = []

    for chunk_vector in chunk_vectors:

        score = cosine_similarity(
            question_vector,
            chunk_vector
        )

        similarities.append(score)

    best_index = similarities.index(
        max(similarities)
    )

    best_score = similarities[
        best_index
    ]

    if best_score >= 0.10:

        answer = chunks[best_index]

        page = chunk_pages[
            best_index
        ]

        return jsonify({

            "success": True,

            "answer": answer,

            "source": pdf_name,

            "page": page,

            "score":
            round(
                best_score * 100,
                1
            )
        })

    else:

        return jsonify({

            "success": True,

            "answer":
            "Sorry, relevant information was not found in the uploaded PDF.",

            "source": None,

            "page": None,

            "score": 0
        })


if __name__ == "__main__":
    app.run(debug=True)