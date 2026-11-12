import os
import re
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.utilities import PythonREPL

st.set_page_config(page_title="Universal Curriculum Assistant", layout="wide")
st.title("📚 Unified Offline Curriculum Assistant")
st.caption("Answers syllabus questions and mathematically computes algorithms/formulas extracted from your documents.")

DB_DIR = "./curriculum_chroma_db"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

embeddings = OllamaEmbeddings(model="nomic-embed-text")

if "vector_store" not in st.session_state:
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        try:
            st.session_state.vector_store = Chroma(
                persist_directory=DB_DIR, 
                embedding_function=embeddings
            )
        except Exception:
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# --- SIDEBAR: Ingestion Engine ---
with st.sidebar:
    st.header("Curriculum Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Syllabi, Notes, or Lab Manuals (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Process & Index Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Parsing and vectorizing documents..."):
                all_docs = []
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    try:
                        loader = PyPDFLoader(tmp_path)
                        all_docs.extend(loader.load())
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                clean_docs = [d for d in all_docs if d.page_content and d.page_content.strip()]

                if not clean_docs:
                    st.error("No readable text found in the PDF. Scanned images require OCR preprocessing.")
                else:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                    chunks = splitter.split_documents(clean_docs)
                    valid_chunks = [c for c in chunks if c.page_content.strip()]

                    st.session_state.vector_store = Chroma.from_documents(
                        documents=valid_chunks,
                        embedding=embeddings,
                        persist_directory=DB_DIR
                    )
                    st.success(f"Indexed {len(valid_chunks)} chunks successfully!")

    if st.session_state.vector_store is not None:
        st.info("Curriculum index loaded and ready.")
    else:
        st.warning("No curriculum indexed yet.")

# --- EXECUTION ENGINE ---
def run_repl(code: str) -> str:
    repl = PythonREPL()
    try:
        output = repl.run(code)
        return output.strip() if output else "Execution finished with no printed output."
    except Exception as err:
        return f"Execution Error: {str(err)}"

# --- CHAT DISPLAY ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question, request an explanation, or supply data to calculate...")

if user_query:
    if st.session_state.vector_store is None:
        st.error("Please upload and index your curriculum PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 1. Semantic Retrieval
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        docs = retriever.invoke(user_query)
        context = "\n\n".join(f"[Page {d.metadata.get('page', 0) + 1}]: {d.page_content}" for d in docs) if docs else ""

        # 2. Strict Unified Reasoning Prompt
        master_prompt = ChatPromptTemplate.from_template("""
You are an offline academic curriculum assistant. You must resolve the student's request using ONLY the provided syllabus context.

Syllabus Excerpts:
{context}

User Request:
{question}

Execution Rules:
1. OUTSIDE CURRICULUM:
   If the topic, formula, or calculation requested is NOT grounded in the excerpts, output EXACTLY:
   "This topic is outside the prescribed curriculum."

2. MATHEMATICAL / ALGORITHMIC / DATA REQUESTS:
   If the user asks to solve, calculate, compute, or run a matrix/numerical procedure (e.g., Row Echelon Form, Newton-Raphson, Simpson's Rule, numerical iterations) using a method detailed in the text:
   - Formulate Python code implementing the exact steps or formula prescribed in the text.
   - The code must be self-contained (using standard libraries like math or numpy), assign variables, execute the logic, and print intermediate and final outputs clearly.
   - Enclose the code strictly inside:
     ```python
     <executable code>
     ```
   - Before or after the code block, state the formula/rule retrieved from the syllabus and cite the page number.

3. THEORETICAL / CONCEPTUAL REQUESTS:
   If the user asks an explanatory or conceptual question:
   - Provide a direct, concise, and structured answer using only the provided facts.
   - Cite the page numbers referenced.
   - Do not generate code blocks for purely theoretical questions.
""")

        llm = ChatOllama(model="llama3.2", temperature=0)
        unified_chain = master_prompt | llm | StrOutputParser()

        with st.chat_message("assistant"):
            with st.spinner("Consulting curriculum materials..."):
                response = unified_chain.invoke({"context": context, "question": user_query})

                # Check if Python execution is needed
                if "```python" in response:
                    code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
                    if code_match:
                        raw_code = code_match.group(1).strip()
                        explanation_part = response.replace(code_match.group(0), "").strip()

                        if explanation_part:
                            st.markdown(explanation_part)

                        with st.expander("Generated Implementation Code", expanded=False):
                            st.code(raw_code, language="python")

                        execution_output = run_repl(raw_code)
                        st.markdown(f"**Computed Output:**\n```\n{execution_output}\n```")
                        
                        full_saved_response = f"{explanation_part}\n\n```python\n{raw_code}\n```\n\n**Output:**\n```\n{execution_output}\n```"
                    else:
                        st.markdown(response)
                        full_saved_response = response
                else:
                    st.markdown(response)
                    full_saved_response = response

        st.session_state.chat_history.append({"role": "assistant", "content": full_saved_response})