import os
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥"
)

st.title("🎥 YouTube Chatbot")
st.write("Ask questions about any YouTube video transcript.")


# -----------------------------
# Inputs
# -----------------------------
video_id = st.text_input(
    "Enter YouTube Video ID",
    placeholder="m7v7KLttM3k"
)

question = st.text_input(
    "Ask a question about the video"
)


# -----------------------------
# Helper Function
# -----------------------------
def format_docs(retrieved_docs):
    return "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


# -----------------------------
# Build RAG Pipeline
# -----------------------------
def create_chain(video_id):

    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        chunk.text
        for chunk in transcript_list
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents(
        [transcript]
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient,
just say you don't know.

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = (
        parallel_chain
        | prompt
        | llm
        | parser
    )

    return main_chain


# -----------------------------
# Ask Question
# -----------------------------
if st.button("Get Answer"):

    if not video_id:
        st.warning("Please enter a YouTube Video ID.")
        st.stop()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    try:

        with st.spinner("Processing transcript..."):

            chain = create_chain(video_id)

            answer = chain.invoke(question)

        st.success("Answer")
        st.write(answer)

    except TranscriptsDisabled:
        st.error(
            "Transcript is disabled for this video."
        )

    except Exception as e:
        st.error(str(e))
