import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile, os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("📄 RAG Conversational AI Chatbot")

# ── Secrets ──────────────────────────────────────────────
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── Session state ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None

# ── Sidebar: PDF upload ───────────────────────────────────
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Ingesting documents..."):
            docs = []
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    loader = PyPDFLoader(tmp.name)
                    docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma.from_documents(chunks, embeddings)
            st.session_state.vectorstore = vectorstore

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=False,
            )
        st.success(f"✅ Processed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

# ── Chat UI ───────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something about your documents..."):
    if st.session_state.chain is None:
        st.warning("Please upload and process documents first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # LangChain expects list of (human, ai) tuples
                history_tuples = [
                    (st.session_state.chat_history[i]["content"],
                     st.session_state.chat_history[i+1]["content"])
                    for i in range(0, len(st.session_state.chat_history)-1, 2)
                ]
                result = st.session_state.chain({
                    "question": prompt,
                    "chat_history": history_tuples
                })
                answer = result["answer"]
                st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
