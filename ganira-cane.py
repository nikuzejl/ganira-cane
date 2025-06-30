import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import tempfile
import os

# needed for streamlit cloud
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()


st.title("💬 Conversational Bot Powered by Multi-File Context")

uploaded_files = st.file_uploader(
    "Upload .txt or .pdf files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

@st.cache_resource(show_spinner="Processing files...")
def setup_chain_from_files(uploaded_files):
    all_docs = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if suffix == ".txt":
            loader = TextLoader(tmp_file_path, encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            continue

        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    vector_db = Chroma.from_documents(chunks, embedding=embeddings)

    retriever = vector_db.as_retriever()
    llm = ChatOpenAI(temperature=0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    return chain

qa_chain = None
if uploaded_files:
    qa_chain = setup_chain_from_files(uploaded_files)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for user_msg, ai_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(ai_msg)

user_input = st.chat_input("Ask a question about the uploaded documents")

if user_input and qa_chain:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        result = qa_chain({"question": user_input})
        answer = result["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append((user_input, answer))
elif user_input and not qa_chain:
    st.warning("Please upload at least one .txt or .pdf file first.")
