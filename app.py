import os
import re
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)

from pytube import YouTube

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


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
<style>

.block-container{
    padding-top:2rem;
    max-width:1100px;
}

.hero{
    text-align:center;
    padding:1rem;
    margin-bottom:1rem;
}

.hero-title{
    font-size:3rem;
    font-weight:700;
}

.hero-subtitle{
    color:#b0b0b0;
    font-size:1.1rem;
}

.answer-box{
    background:#1f2937;
    padding:20px;
    border-radius:15px;
    border-left:6px solid #ff4b4b;
    margin-top:10px;
}

.feature-card{
    background:#111827;
    padding:15px;
    border-radius:12px;
    text-align:center;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:14px;
}

</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD API KEY
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        st.error("Google API Key not found.")
        st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Architecture")

    st.markdown(
        """
### Workflow

1. Extract Transcript
2. Chunk Text
3. Generate Embeddings
4. Store in FAISS
5. Retrieve Relevant Chunks
6. Gemini Generates Answer

---

### Tech Stack

- Gemini 2.5 Flash
- Gemini Embeddings
- LangChain
- FAISS
- Streamlit
        """
    )

# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(
    """
<div class='hero'>
    <div class='hero-title'>🎥 YouTube RAG Chatbot</div>
    <div class='hero-subtitle'>
        Chat with any YouTube video using Retrieval-Augmented Generation (RAG)
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# FEATURE METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("LLM", "Gemini 2.5 Flash")

with col2:
    st.metric("Vector Store", "FAISS")

with col3:
    st.metric("Framework", "LangChain")

st.divider()

# --------------------------------------------------
# INPUTS
# --------------------------------------------------

youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

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


# --------------------------------------------------
# VIDEO DETAILS
# --------------------------------------------------

video_id = None

if youtube_url:

    video_id = extract_video_id(youtube_url)

    if video_id:

        try:

            yt = YouTube(youtube_url)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(
                    yt.thumbnail_url,
                    use_container_width=True
                )

            with col2:

                st.subheader(yt.title)

                st.write(f"📺 Channel: **{yt.author}**")

                st.write(
                    f"👀 Views: {yt.views:,}"
                )

        except:
            pass

# --------------------------------------------------
# SUGGESTED QUESTIONS
# --------------------------------------------------

st.markdown("### Suggested Questions")

c1, c2, c3 = st.columns(3)

suggested_question = None

with c1:
    if st.button("📝 Summarize Video"):
        suggested_question = "Provide a concise summary of the video."

with c2:
    if st.button("🎯 Key Takeaways"):
        suggested_question = "What are the key takeaways from this video?"

with c3:
    if st.button("📌 Important Facts"):
        suggested_question = "List the most important facts discussed in the video."

question = st.text_input(
    "Ask a question",
    value=suggested_question if suggested_question else ""
)

# --------------------------------------------------
# RAG PIPELINE
# --------------------------------------------------

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
        search_kwargs={"k":4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY using the transcript context.

If the answer is not available,
respond with:

"I don't know."

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel(
        {
            "context":
            retriever
            | RunnableLambda(format_docs),

            "question":
            RunnablePassthrough(),
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

# --------------------------------------------------
# ASK BUTTON
# --------------------------------------------------

if st.button("🚀 Get Answer", use_container_width=True):

    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
        st.stop()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()

    try:

        with st.spinner(
            "🧠 Processing transcript and generating answer..."
        ):

            chain = create_chain(video_id)

            answer = chain.invoke(question)

        st.markdown("### Conversation")

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.markdown(answer)

    except TranscriptsDisabled:

        st.error(
            "This video does not have transcripts enabled."
        )

    except Exception as e:

        if "YouTube is blocking requests" in str(e):

            st.error(
                """
YouTube is currently blocking transcript access.

Try:
- Another video
- A video with captions
- Running locally
"""
            )

        else:
            st.error(str(e))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
<div class='footer'>
Built by Oorja Gund • Gemini • LangChain • FAISS • Streamlit
</div>
""",
    unsafe_allow_html=True,
)
