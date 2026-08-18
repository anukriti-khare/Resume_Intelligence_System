import os
import pickle

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================
# LOAD CNN CLASSIFIER
# =========================

cnn_model = load_model("resume_classifier_cnn.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# =========================
# LOAD FAISS + METADATA
# =========================

faiss_index = faiss.read_index("resume_faiss.index")

resume_df = pd.read_pickle("resume_metadata.pkl")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# LOAD GROQ
# =========================

client = None

if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)


# =========================
# 1. RESUME CLASSIFICATION
# =========================

def classify_resume(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=300
    )

    prediction = cnn_model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_index = np.argmax(prediction, axis=1)[0]

    predicted_label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(
        np.max(prediction) * 100
    )

    return predicted_label, confidence


# =========================
# 2. FAISS CANDIDATE SEARCH
# =========================

def search_resumes(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    results = resume_df.iloc[
        indices[0]
    ].copy()

    results["similarity_score"] = distances[0]

    return results


# =========================
# 3. RAG QUESTION ANSWERING
# =========================

def ask_resume_rag(question, top_k=3):

    if client is None:
        return "Groq API key not found.", None

    results = search_resumes(
        question,
        top_k
    )

    context = ""

    for i, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

        context += f"""
RESUME {i}
Filename: {row['filename']}
Category: {row['label']}

{row['text'][:2500]}

----------------------------
"""

    prompt = f"""
You are a resume analysis assistant.

Answer the user's question ONLY using the resume context below.
Do not invent information.
If the answer is not present in the resumes, say so.

RESUME CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return answer, results