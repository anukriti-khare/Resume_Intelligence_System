import streamlit as st

from utils import extract_text_from_pdf

from main import (
    classify_resume,
    search_resumes,
    ask_resume_rag
)


st.set_page_config(
    page_title="Resume Intelligence System",
    page_icon="📄",
    layout="wide"
)


st.title("📄 Resume Intelligence System")

st.write(
    "Classify resumes, search candidates semantically, "
    "and ask AI questions about relevant candidates."
)


tab1, tab2, tab3 = st.tabs([
    "Resume Classifier",
    "Candidate Search",
    "Ask AI"
])


# =====================================
# TAB 1: CNN RESUME CLASSIFICATION
# =====================================

with tab1:

    st.header("Resume Classification")

    uploaded_file = st.file_uploader(
        "Upload a resume PDF",
        type=["pdf"],
        key="classifier_upload"
    )

    if uploaded_file is not None:

        if st.button("Classify Resume"):

            with st.spinner("Analyzing resume..."):

                text = extract_text_from_pdf(
                    uploaded_file
                )

                if text.strip():

                    label, confidence = classify_resume(
                        text
                    )

                    st.success(
                        f"Predicted Category: {label}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                else:
                    st.error(
                        "Could not extract text from this PDF."
                    )


# =====================================
# TAB 2: FAISS SEARCH
# =====================================

with tab2:

    st.header("Semantic Candidate Search")

    query = st.text_area(
        "Describe the candidate you are looking for",
        placeholder=(
            "Example: Candidate with Python, machine learning "
            "and data analysis experience"
        )
    )

    if st.button("Search Candidates"):

        if query.strip():

            with st.spinner("Searching resumes..."):

                results = search_resumes(
                    query,
                    top_k=5
                )

                for rank, (_, row) in enumerate(
                    results.iterrows(),
                    start=1
                ):

                    with st.expander(
                        f"#{rank} — {row['filename']} | "
                        f"{row['label']} | "
                        f"Score: {row['similarity_score']:.3f}"
                    ):

                        st.write(
                            row["text"][:1500]
                        )

        else:
            st.warning(
                "Please enter a search query."
            )


# =====================================
# TAB 3: RAG + GROQ
# =====================================

with tab3:

    st.header("Ask AI About Candidates")

    question = st.text_area(
        "Ask a question about the resume database",
        placeholder=(
            "Example: Which candidate has the strongest "
            "technical and programming skill set?"
        )
    )

    if st.button("Ask AI"):

        if question.strip():

            with st.spinner(
                "Retrieving resumes and generating answer..."
            ):

                answer, rag_results = ask_resume_rag(
                    question,
                    top_k=3
                )

                st.subheader("Answer")

                st.write(answer)

                if rag_results is not None:

                    st.subheader(
                        "Retrieved Resumes Used"
                    )

                    for rank, (_, row) in enumerate(
                        rag_results.iterrows(),
                        start=1
                    ):

                        st.write(
                            f"**#{rank}: {row['filename']}** "
                            f"({row['label']}) — "
                            f"Score: "
                            f"{row['similarity_score']:.3f}"
                        )

        else:
            st.warning(
                "Please enter a question."
            )