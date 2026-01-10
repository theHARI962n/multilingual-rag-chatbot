from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import MessagesState

# -------------------- UI --------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot")

uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

# -------------------- Session State --------------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# -------------------- LLM --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -------------------- LangGraph --------------------
if "graph_app" not in st.session_state:

    SYSTEM_MESSAGE = SystemMessage(
        content=(
            "You are a helpful assistant.\n"
            "You may receive document context.\n"
            "Use it ONLY if relevant.\n"
            "If you don't know, say you don't know.\n"
            "Keep answers concise."
        )
    )

    def call_model(state: MessagesState):
        messages = state["messages"]

        # Ensure system prompt exists ONCE
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SYSTEM_MESSAGE] + messages

        response = llm.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()

    st.session_state.graph_app = workflow.compile(
        checkpointer=memory
    )


# -------------------- PDF Functions --------------------
def load_documents(pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf.read())
        path = tmp.name

    return PyPDFLoader(path).load()

def build_vector_db(docs):
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

# -------------------- Build Knowledge Base --------------------
if st.button("📥 Build Knowledge Base"):
    if not uploaded_pdf:
        st.warning("Upload a PDF first")
    else:
        with st.spinner("Processing PDF..."):
            docs = load_documents(uploaded_pdf)
            st.session_state.vectordb = build_vector_db(docs)
            st.success("Knowledge base ready ✅")

# -------------------- Chat UI --------------------
st.divider()
st.subheader("💬 Chat")

for msg in st.session_state.ui_messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask a question")

if question and st.session_state.vectordb:

    # ---- UI display (user)
    st.session_state.ui_messages.append(
        {"role": "user", "content": question}
    )
    st.chat_message("user").write(question)

    # ---- RAG Retrieval (ephemeral)
    docs = st.session_state.vectordb.similarity_search(question, k=3)

    context_blocks = []
    citations = []

    for d in docs:
        context_blocks.append(d.page_content)
        citations.append(
            f"- Page {d.metadata.get('page', 'N/A')}"
        )

    context_text = "\n\n".join(context_blocks)

    rag_prompt = f"""
Use the following document context if relevant.

Context:
{context_text}

Question:
{question}

If the context is not relevant, answer from conversation memory.
"""

    user_message = HumanMessage(content=rag_prompt)

    # ---- Assistant
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            result = st.session_state.graph_app.invoke(
                {"messages": [user_message]},
                config={"configurable": {"thread_id": "chat"}}
            )

            answer = result["messages"][-1].content
            st.write(answer)

            if context_blocks:
                st.markdown("**📚 Citations:**")
                for c in citations:
                    st.markdown(c)

    # ---- Save assistant reply
    st.session_state.ui_messages.append(
        {"role": "assistant", "content": answer}
    )
