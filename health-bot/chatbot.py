# chatbot.py

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
try:
    from langchain_community.chat_models import ChatOllama  # optional local LLM
except Exception:  # pragma: no cover
    ChatOllama = None
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
import os
from pathlib import Path
from dotenv import load_dotenv
import time
from translator import detect_language, normalize_user_lang, translate_text
import chroma_manager

# Load your HUGGING_FACE_HUB_TOKEN from the .env file
load_dotenv()

# ----------------------------
# LLM SETUP (unchanged base)
# ----------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hf")  # "hf" or "ollama"
if LLM_PROVIDER == "ollama" and ChatOllama is not None:
    # Use local Ollama for speed and zero-cost
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama:latest")
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
else:
    # Use a smaller, faster open-access instruction model on HF
    repo_id = os.getenv("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    base_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=128,
        temperature=0.2,
        huggingfacehub_api_token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
    )
    llm = ChatHuggingFace(llm=base_llm)

# ----------------------------
# RAG: Documents, Embeddings, Vector Store
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "vecstore"
INDEX_DIR.mkdir(exist_ok=True)

def load_documents():
    docs = []
    if not DATA_DIR.exists():
        # fall through to starter corpus below
        pass

    # Load PDFs
    for pdf_path in DATA_DIR.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs.extend(loader.load())
        except Exception:
            pass

    # Load TXT (including .txt.txt)
    for txt_path in list(DATA_DIR.glob("**/*.txt")) + list(DATA_DIR.glob("**/*.txt.txt")):
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs.extend(loader.load())
        except Exception:
            pass

    # Load CSV (concatenate rows to text)
    for csv_path in DATA_DIR.glob("**/*.csv"):
        try:
            loader = CSVLoader(str(csv_path))
            docs.extend(loader.load())
        except Exception:
            pass

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    if not docs:
        # Starter health corpus (concise, safe guidance). Sources are general public health references.
        starter_texts = [
            ("Malaria basics", "Malaria is caused by Plasmodium parasites transmitted by Anopheles mosquitoes. Common symptoms include fever (often cyclical), chills, sweating, headache, nausea, and fatigue. Severe warning signs: confusion, breathing difficulty, persistent vomiting, jaundice, bleeding, or seizures. Prevention: insecticide-treated nets, repellents, and prompt testing if fever occurs after travel to endemic areas. Treatment depends on species and local resistance patterns; seek medical care promptly."),
            ("Dengue basics", "Dengue is a viral illness spread by Aedes mosquitoes. Symptoms: high fever, severe headache, pain behind the eyes, muscle and joint pains, rash, and mild bleeding (nose/gums). Warning signs (seek urgent care): severe abdominal pain, persistent vomiting, bleeding, lethargy, restlessness, or sudden drop in platelets. Hydration is key. Avoid NSAIDs like ibuprofen unless advised by a clinician due to bleeding risk."),
            ("Typhoid basics", "Typhoid fever is a bacterial infection (Salmonella Typhi) spread via contaminated food or water. Symptoms: sustained fever, abdominal pain, constipation or diarrhea, headache, weakness, and sometimes rash. Prevention: safe water, hand hygiene, proper food handling, and vaccination for travelers to high-risk areas. Requires medical evaluation and appropriate antibiotics."),
            ("Tuberculosis basics", "TB is a bacterial infection (Mycobacterium tuberculosis) spread through airborne droplets. Symptoms: cough lasting >2 weeks, fever, night sweats, weight loss, fatigue, and sometimes coughing up blood. Diagnosis involves sputum testing and chest imaging. Treatment uses combination antibiotics for several months; adherence is essential."),
            ("COVID-19 basics", "COVID-19 is a respiratory viral illness. Symptoms: fever, cough, sore throat, loss of taste or smell, fatigue, and breathing difficulty. Prevention: vaccination, masks in high-risk settings, hand hygiene, and ventilation. Seek urgent care for trouble breathing, persistent chest pain, confusion, or bluish lips/face."),
            ("Diabetes type 2 basics", "Type 2 diabetes involves insulin resistance and elevated blood glucose. Symptoms can include increased thirst, frequent urination, fatigue, and blurred vision. Management: balanced diet (emphasize vegetables, fiber, lean proteins), regular physical activity, weight management, glucose monitoring, and medications as prescribed. Watch for warning signs of hypo/hyperglycemia and complications (foot ulcers, vision changes)."),
            ("Hypertension basics", "Hypertension is persistently elevated blood pressure. Often asymptomatic, but can cause headaches or dizziness. Lifestyle measures: reduce salt intake, maintain healthy weight, exercise regularly, avoid tobacco, and limit alcohol. Medication is often needed and must be taken consistently. Seek urgent care for severe headache, chest pain, shortness of breath, or neurological symptoms.")
        ]
        starter_docs = [Document(page_content=txt, metadata={"source": title}) for title, txt in starter_texts]
        return splitter.split_documents(starter_docs)

    return splitter.split_documents(docs)

def build_or_load_vectorstore():
    index_path = INDEX_DIR / "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if index_path.exists():
        try:
            return FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            pass

    # Build fresh index
    docs = load_documents()
    if not docs:
        # Create an empty store to avoid crashes; retrieval will yield no context
        return FAISS.from_texts([""], embedding=embeddings)

    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(str(index_path))
    return vs

# Initialize / load vector store and retriever once
VECTORSTORE = build_or_load_vectorstore()
# Use fewer, higher-quality chunks to improve speed and reduce prompt size
RETRIEVER = VECTORSTORE.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.2},
)

def refresh_vectorstore() -> int:
    """Rebuild the FAISS index from the data/ directory and refresh the retriever.

    Returns the number of text chunks indexed.
    """
    global VECTORSTORE, RETRIEVER
    # Recreate documents and index, then persist
    docs = load_documents()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not docs:
        VECTORSTORE = FAISS.from_texts([""], embedding=embeddings)
    else:
        VECTORSTORE = FAISS.from_documents(docs, embedding=embeddings)
    # Save to disk for reuse across restarts
    index_path = (Path(__file__).resolve().parent / "vecstore" / "faiss_index")
    VECTORSTORE.save_local(str(index_path))
    # Refresh retriever
    RETRIEVER = VECTORSTORE.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.2})
    return len(docs)

# ----------------------------
# Prompt with context for RAG
# ----------------------------
system_prompt = (
    "You are Curebot, a helpful and conversational AI Health Assistant. "
    "Your goal is to provide clear, accurate information in natural, easy-to-read paragraphs. "
    "CRITICAL RULE 1: If a user asks a question but does not provide all the necessary information for a full answer (for example, asking for their BMI without giving their height), you MUST ask a clarifying question to get the missing information. DO NOT invent or assume data. "
    "CRITICAL RULE 2: Write your answers in plain, natural sentences. You are FORBIDDEN from using markdown formatting like '##' for headings or '*' for bullet points. "
    "CRITICAL RULE 3: You MUST respond in one of three languages ONLY: English, Hindi, or Gujarati, matching the user's language. "
    "CRITICAL RULE 4: Answer ONLY using the provided context. If the context is missing or insufficient, say that you don't know and, if appropriate, provide general safety guidance or suggest consulting a professional. Do NOT fabricate facts. "
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + " Use the following context to answer the user question. If the context is not relevant, say you don't know.\n\nContext:\n{context}"),
    ("human", "{question}")
])

# Compose the RAG chain: retrieve -> prompt -> llm -> parse
def _format_docs(docs):
    parts = []
    for d in docs or []:
        src = d.metadata.get("source") if hasattr(d, "metadata") else None
        if src:
            parts.append(f"Source: {src}\n{getattr(d, 'page_content', str(d))}")
        else:
            parts.append(getattr(d, 'page_content', str(d)))
    return "\n\n".join(parts)

def _build_rag_answer(context: str, question: str) -> str:
    chain = rag_prompt | llm | StrOutputParser()
    # retry once on transient errors
    for attempt in range(2):
        try:
            return chain.invoke({"context": context, "question": question})
        except Exception:
            if attempt == 0:
                time.sleep(0.8)
                continue
            raise

# Public function used by the FastAPI endpoint
def get_llm_response(query: str, user_id: int | None = None):
    # Detect and normalize user language to one of: en, hi, gu
    detected = detect_language(query)
    user_lang = normalize_user_lang(detected)

    # Translate query to English for retrieval if needed
    internal_query = query if user_lang == "en" else translate_text(query, target_lang="en")

    # If it's just a greeting or very short, respond without RAG to avoid unnecessary "no info" messages
    simple = query.strip().lower()
    if simple in {"hi", "hello", "hey", "hii", "heyy"} or len(simple.split()) <= 2:
        greet_en = (
            "Hello! I'm your health assistant. You can ask about symptoms, prevention, and when to seek care. "
            "You may also upload your health documents so I can answer more precisely. How can I help you today?"
        )
        return translate_text(greet_en, target_lang=user_lang) if user_lang != "en" else greet_en

    # Build combined context: global FAISS + per-user Chroma (if available)
    faiss_docs = []
    try:
        faiss_docs = RETRIEVER.get_relevant_documents(internal_query)
    except Exception:
        pass
    user_snippets = []
    if user_id is not None:
        try:
            user_snippets = chroma_manager.query_user_texts(user_id, internal_query, k=4)
        except Exception:
            user_snippets = []
    context_parts = []
    if faiss_docs:
        context_parts.append(_format_docs(faiss_docs))
    if user_snippets:
        context_parts.append("\n\n".join([f"UserDoc: {t}" for t in user_snippets]))

    if not context_parts:
        msg_en = (
            "I don't have enough reliable information in my knowledge base to answer this. "
            "Please try rephrasing, or upload relevant health documents and run reindex."
        )
        return translate_text(msg_en, target_lang=user_lang) if user_lang != "en" else msg_en

    context_str = "\n\n".join(context_parts)

    # Main call with fallback layers
    try:
        answer_en = _build_rag_answer(context_str, internal_query)
    except Exception:
        # Fallback: direct LLM without retrieval, with one retry
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        chain = fallback_prompt | llm | StrOutputParser()
        for attempt in range(2):
            try:
                answer_en = chain.invoke({"question": internal_query})
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.8)
                    continue
                answer_en = "The service is currently busy. Please try again in a moment."

    # Translate back to user's language if needed
    if user_lang != "en":
        return translate_text(answer_en, target_lang=user_lang)
    return answer_en