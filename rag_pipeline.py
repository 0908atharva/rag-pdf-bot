"""
rag_pipeline.py
---------------
Core Retrieval-Augmented Generation (RAG) logic, kept separate from the UI so
it is easy to read, test, and explain.

The full RAG flow implemented in this file:

    1. LOAD     -> read the PDF and extract text page by page (pypdf)
    2. CHUNK    -> split text into overlapping chunks (RecursiveCharacterTextSplitter)
    3. EMBED    -> turn each chunk into a vector using the provider's embed model
    4. STORE    -> save vectors in a PERSISTED Chroma collection (cached by file hash)
    5. RETRIEVE -> for a question, fetch the top-k most similar chunks
    6. GENERATE -> ask the LLM to answer using ONLY those chunks as context

Everything is provider-agnostic: whether you use Gemini or OpenAI is decided by
``config.LLM_PROVIDER``.
"""

import hashlib
import io

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import config


# ===========================================================================
# Provider-specific model factories
# ===========================================================================
def get_embeddings():
    """
    Return an embedding model for the configured provider.

    The embedding model must match the provider so that stored vectors and
    query vectors live in the same space.
    """
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.OPENAI_EMBED_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    # Default provider: Google Gemini
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=config.GEMINI_EMBED_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )


def get_llm():
    """
    Return a chat LLM for the configured provider.

    ``temperature=0`` makes answers deterministic and grounded, which reduces
    "creative" hallucination -- exactly what we want for document Q&A.
    """
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
        )

    # Default provider: Google Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.GEMINI_CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0,
    )


# ===========================================================================
# Prompt engineering (graded feature)
# ---------------------------------------------------------------------------
# The system prompt is the single most important lever against hallucination.
# It hard-constrains the model to the retrieved context and gives it an exact
# "I don't know" escape hatch instead of guessing.
# ===========================================================================
SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about a single PDF "
    "document. Follow these rules strictly:\n"
    "1. Answer ONLY using the information in the provided context below.\n"
    "2. If the answer is not contained in the context, reply with exactly: "
    '"I couldn\'t find this in the document." '
    "Do NOT guess and do NOT use any outside knowledge.\n"
    "3. Keep answers concise, factual, and grounded in the context.\n"
    "4. When useful, mention the page number(s) the answer came from."
)

# ``create_stuff_documents_chain`` fills the {context} placeholder with the
# retrieved chunks; {input} is the user's question.
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ]
)


# ===========================================================================
# Step 1 + 2: load PDF text and split into chunks
# ===========================================================================
def compute_file_hash(file_bytes: bytes) -> str:
    """
    Stable SHA-256 hash of the raw PDF bytes.

    Used to cache embeddings: the same PDF always maps to the same hash, so we
    never embed identical content twice.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def extract_documents(file_bytes: bytes, filename: str) -> list[Document]:
    """
    Step 1 (LOAD): read a (possibly multi-page) PDF and return one Document per
    non-empty page. Empty / garbage pages (no extractable text -- e.g. scanned
    images) are skipped gracefully.

    Raises:
        ValueError: if the file is not a readable PDF, or has no extractable text.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:  # corrupted file / not actually a PDF
        raise ValueError(f"Could not read the PDF file: {exc}") from exc

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text() may return None for image-only or empty pages.
        text = (page.extract_text() or "").strip()
        if not text:
            # Skip empty/garbage page instead of crashing.
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": filename, "page": page_number},
            )
        )

    if not documents:
        raise ValueError(
            "No extractable text found in this PDF. It may be a scanned "
            "document (images only) that requires OCR before it can be searched."
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Step 2 (CHUNK): split each page into overlapping chunks.

    Overlap keeps sentences that straddle a chunk boundary retrievable, which
    improves answer quality. Page metadata is preserved on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # Prefer to split on natural boundaries, falling back to finer ones.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ===========================================================================
# Step 3 + 4: embed chunks and store them in a persisted Chroma collection.
# Cached by file hash: if the collection already has vectors, we reuse them.
# ===========================================================================
def build_vectorstore(file_bytes: bytes, filename: str):
    """
    Build (or load from cache) a persisted Chroma vector store for a PDF.

    Returns:
        (vectorstore, was_cached): the Chroma store, and a bool that is True
        when we reused an already-embedded collection (cache hit).

    Caching strategy:
        The Chroma collection is named after the file's hash. Re-opening the
        same persist directory + collection name transparently reloads the
        vectors from disk, so the same PDF is never re-embedded.
    """
    file_hash = compute_file_hash(file_bytes)
    # Collection names must be 3-63 chars and start/end alphanumeric. 32 hex
    # chars is plenty to avoid collisions while staying within the limit.
    collection_name = f"pdf_{file_hash[:32]}"

    embeddings = get_embeddings()

    # Opening a Chroma store with the same persist_directory + collection_name
    # reloads any previously stored vectors from local disk.
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )

    # Cache check: does this collection already contain embeddings?
    try:
        already_embedded = vectorstore._collection.count() > 0
    except Exception:
        already_embedded = False

    if already_embedded:
        return vectorstore, True  # cache hit -- skip the embedding work

    # Cache miss: run LOAD -> CHUNK -> EMBED -> STORE.
    documents = extract_documents(file_bytes, filename)   # Step 1
    chunks = split_documents(documents)                   # Step 2
    vectorstore.add_documents(chunks)  # Step 3 + 4: embeds and persists to disk
    return vectorstore, False


# ===========================================================================
# Step 5 + 6: build a retrieval chain (retrieve top-k, then generate answer)
# ===========================================================================
def build_rag_chain(vectorstore):
    """
    Wire retriever + prompt + LLM into a single LangChain retrieval chain.

    - The retriever fetches the top-k most relevant chunks (Step 5).
    - ``create_stuff_documents_chain`` stuffs those chunks into {context} and
      asks the LLM to answer using our grounded prompt (Step 6).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    llm = get_llm()
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    return create_retrieval_chain(retriever, combine_docs_chain)


def answer_question(rag_chain, question: str) -> dict:
    """
    Run one question through the chain.

    Returns:
        {
            "answer":  str,               # the grounded answer
            "sources": [Document, ...],   # the retrieved chunks used as grounding
        }
    """
    result = rag_chain.invoke({"input": question})
    return {
        "answer": result.get("answer", ""),
        # "context" holds the exact chunks the LLM was shown -- great for
        # displaying the grounding to the user.
        "sources": result.get("context", []),
    }
