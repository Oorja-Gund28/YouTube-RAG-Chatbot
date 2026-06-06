import os
import re
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


# -----------------------------------
# Load API Key
# -----------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        st.error("Google API Key not found.")
        st.stop()


# -----------------------------------
# Streamlit Page
# -----------------------------------
st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥",
)

st.title("🎥 YouTube Chatbot")

st.markdown(
    """
Hey! Ask me questions about any YouTube video.

**Paste the full YouTube URL**.
"""
)


# -----------------------------------
# Inputs
# -----------------------------------
youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

question = st.text_input(
    "Ask a question about the video"
)


# -----------------------------------
# Helpers
# -----------------------------------
def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def format_docs(retrieved_docs):
    return "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


# -----------------------------------
# Cached RAG Pipeline
# -----------------------------------
@st.cache_resource(show_spinner=False)
def create_chain(video_id):

    transcript_list = YouTubeTranscriptApi().fetch(
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

Answer ONLY from the transcript context.

If the answer is not present in the context,
say "I don't know."

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel(
        {
            "context": retriever
            | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    parser = StrOutputParser()

    chain = (
        parallel_chain
        | prompt
        | llm
        | parser
    )

    return chain


# -----------------------------------
# Run
# -----------------------------------
if st.button("Get Answer"):

    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
        st.stop()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()

    try:

        with st.spinner("Processing transcript..."):

            chain = create_chain(video_id)

            answer = chain.invoke(question)

        st.success("Answer")
        st.write(answer)

    except TranscriptsDisabled:
        st.error(
            "This video does not have transcripts enabled."
        )

    except Exception as e:

        if "YouTube is blocking requests" in str(e):

            st.error(
                """
YouTube is currently blocking transcript access from the hosting server.

Try:
1. Another video
2. A video with manually created captions
3. Running the app locally
"""
            )

        else:
            st.error(str(e))
