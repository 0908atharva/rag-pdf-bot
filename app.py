"""
app.py
------
Streamlit web UI for the RAG PDF bot.

This file is presentation only. All RAG logic lives in ``rag_pipeline.py`` and is
called through the exact same functions as before — the backend is untouched:
    - build_vectorstore()  -> load, chunk, embed, store (cached by file hash)
    - build_rag_chain()    -> retriever + prompt + LLM
    - answer_question()    -> returns {"answer", "sources"}

What the user sees:
  1. A polished custom-themed header.
  2. A PDF upload widget with a friendly empty state.
  3. A chat-style Q&A with distinct user/bot bubbles + avatars.
  4. An expandable "Sources" section with a clean card per chunk (page badge).
  5. A tidy sidebar: provider config, "How it works", clear-chat, and footer.
"""

import html

import streamlit as st

import config
from rag_pipeline import (
    answer_question,
    build_rag_chain,
    build_vectorstore,
    compute_file_hash,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — modern dark theme, chat bubbles, source cards, sidebar polish.
# (Purely cosmetic. `:has()` is used to distinguish user vs bot bubbles.)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
      --accent:#6366f1; --accent-2:#a5a8fb;
      --accent-soft:rgba(99,102,241,0.14);
      --card:#171d2b; --card-2:#1b2233;
      --border:rgba(255,255,255,0.08);
      --text-dim:#94a0b8;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"] {
      font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Comfortable, centered content column */
    .block-container { padding-top:2.2rem; padding-bottom:6rem; max-width:880px; }

    /* ---- App header ---- */
    .app-header{ display:flex; align-items:center; gap:16px; margin-bottom:1.4rem; }
    .app-badge{
      font-size:30px; width:60px; height:60px; flex:none;
      display:flex; align-items:center; justify-content:center;
      background:linear-gradient(135deg, #6366f1, #4f46e5);
      border-radius:16px; box-shadow:0 8px 24px rgba(99,102,241,0.35);
    }
    .app-title{ font-size:1.9rem; font-weight:700; margin:0; letter-spacing:-0.02em; }
    .app-tag{ color:var(--text-dim); margin:.2rem 0 0; font-size:.95rem; }

    /* ---- Chat messages ---- */
    [data-testid="stChatMessage"]{
      border-radius:16px; padding:.5rem 1.1rem; margin-bottom:.7rem;
      border:1px solid var(--border); background:var(--card);
      box-shadow:0 1px 2px rgba(0,0,0,0.2);
    }
    /* User bubble = accent-tinted */
    [data-testid="stChatMessage"]:has(.msg-user){
      background:var(--accent-soft); border-color:rgba(99,102,241,0.35);
    }
    /* Bot bubble = neutral card */
    [data-testid="stChatMessage"]:has(.msg-assistant){ background:var(--card); }
    .msg-role{ display:none; }  /* hidden marker used only for :has() targeting */

    /* ---- Source cards (inside the expander) ---- */
    .source-card{
      background:var(--card-2); border:1px solid var(--border);
      border-left:3px solid var(--accent); border-radius:12px;
      padding:.7rem .9rem; margin:.55rem 0;
    }
    .source-head{ display:flex; justify-content:space-between; align-items:center; margin-bottom:.45rem; }
    .chunk-label{ font-weight:600; font-size:.72rem; color:var(--text-dim);
      text-transform:uppercase; letter-spacing:.05em; }
    .page-badge{ background:var(--accent-soft); color:var(--accent-2);
      font-size:.72rem; font-weight:600; padding:.15rem .6rem; border-radius:999px;
      border:1px solid rgba(99,102,241,0.4); }
    .source-body{ font-size:.85rem; color:#c8cede; white-space:pre-wrap;
      line-height:1.55; max-height:220px; overflow:auto; }

    /* ---- Empty state ---- */
    .empty-state{ text-align:center; padding:2.6rem 1rem; border:1px dashed var(--border);
      border-radius:18px; background:rgba(255,255,255,0.02); margin-top:1.2rem; }
    .empty-icon{ font-size:44px; margin-bottom:.3rem; }
    .empty-state h3{ margin:.2rem 0; font-weight:600; }
    .empty-state p{ color:var(--text-dim); max-width:430px; margin:.35rem auto 0; font-size:.92rem; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"]{ border-right:1px solid var(--border); }
    .side-title{ font-size:.72rem; text-transform:uppercase; letter-spacing:.09em;
      color:var(--text-dim); font-weight:700; margin:1.2rem 0 .55rem; }
    .cfg-row{ display:flex; justify-content:space-between; gap:10px; padding:.4rem .65rem;
      background:var(--card); border:1px solid var(--border); border-radius:10px;
      margin-bottom:.4rem; font-size:.8rem; align-items:center; }
    .cfg-row .k{ color:var(--text-dim); }
    .cfg-row .v{ color:#fff; font-weight:600; font-family:ui-monospace, monospace; font-size:.76rem; }
    .step{ display:flex; gap:.6rem; align-items:flex-start; margin-bottom:.6rem; }
    .step .ic{ font-size:.95rem; width:28px; height:28px; flex:none; display:flex;
      align-items:center; justify-content:center; background:var(--accent-soft);
      border-radius:9px; border:1px solid rgba(99,102,241,.3); }
    .step .t{ font-size:.84rem; font-weight:600; line-height:1.15; }
    .step .d{ font-size:.74rem; color:var(--text-dim); }
    .side-footer{ margin-top:1.5rem; padding-top:1rem; border-top:1px solid var(--border);
      font-size:.8rem; color:var(--text-dim); }
    .side-links{ margin-top:.35rem; }
    .side-links a{ color:var(--accent-2); text-decoration:none; }
    .side-links a:hover{ text-decoration:underline; }

    /* ---- Buttons ---- */
    .stButton>button{ border-radius:10px; border:1px solid var(--border); font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state (persists across Streamlit reruns within a browser session)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []       # chat history: list of message dicts
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None    # the active retrieval chain
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None    # hash of the currently-indexed PDF
if "file_name" not in st.session_state:
    st.session_state.file_name = None    # name of the currently-indexed PDF


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
      <div class="app-badge">📄</div>
      <div>
        <h1 class="app-title">Chat with your PDF</h1>
        <p class="app-tag">Ask questions in plain English — every answer stays grounded in your document, with sources shown.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: configuration, "How it works", session controls, footer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-title">⚙️ Configuration</div>', unsafe_allow_html=True)

    if config.LLM_PROVIDER == "openai":
        chat_model, embed_model = config.OPENAI_CHAT_MODEL, config.OPENAI_EMBED_MODEL
    else:
        chat_model, embed_model = config.GEMINI_CHAT_MODEL, config.GEMINI_EMBED_MODEL

    st.markdown(
        f"""
        <div class="cfg-row"><span class="k">Provider</span><span class="v">{config.LLM_PROVIDER}</span></div>
        <div class="cfg-row"><span class="k">Chat model</span><span class="v">{chat_model}</span></div>
        <div class="cfg-row"><span class="k">Embeddings</span><span class="v">{embed_model}</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Switch providers with `LLM_PROVIDER=gemini|openai` in `.env`, then restart.")

    # ---- How it works ----
    st.markdown('<div class="side-title">🧭 How it works</div>', unsafe_allow_html=True)
    steps = [
        ("📤", "Upload", "Drop in a PDF document"),
        ("✂️", "Chunk", "Split into overlapping passages"),
        ("🧠", "Embed", "Vectorize & store in Chroma"),
        ("🔍", "Retrieve", "Fetch the top-4 relevant chunks"),
        ("💬", "Answer", "LLM replies using only those chunks"),
    ]
    steps_html = "".join(
        f'<div class="step"><div class="ic">{ic}</div>'
        f'<div><div class="t">{title}</div><div class="d">{desc}</div></div></div>'
        for ic, title, desc in steps
    )
    st.markdown(steps_html, unsafe_allow_html=True)

    # ---- Session controls: Clear chat (does not drop the indexed PDF) ----
    if st.session_state.messages:
        st.markdown('<div class="side-title">🧹 Session</div>', unsafe_allow_html=True)
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ---- Footer ----
    # NOTE: replace the GitHub/LinkedIn hrefs below with your real profile URLs.
    st.markdown(
        """
        <div class="side-footer">
          Built by <strong style="color:#e7e9f2;">Atharva Somwanshi</strong>
          <div class="side-links">
            <a href="mailto:atharvasomwanshi121@gmail.com">Email</a> ·
            <a href="https://github.com/" target="_blank">GitHub</a> ·
            <a href="https://www.linkedin.com/" target="_blank">LinkedIn</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Fail fast with a clear message if the API key / provider is misconfigured.
# ---------------------------------------------------------------------------
try:
    config.validate_config()
except config.ConfigError as exc:
    st.error(f"⚠️ Configuration error: {exc}")
    st.stop()


# ---------------------------------------------------------------------------
# Helper: render retrieved source chunks as clean cards inside an expander
# ---------------------------------------------------------------------------
def render_sources(sources):
    """Show the retrieved chunks so the grounding is visible and auditable."""
    if not sources:
        return
    with st.expander(f"📚 Sources · {len(sources)} chunks used to ground this answer"):
        for i, doc in enumerate(sources, start=1):
            page = doc.metadata.get("page", "?")
            # Escape the chunk text (it is raw document text, not trusted HTML).
            body = html.escape(doc.page_content)
            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-head">
                    <span class="chunk-label">Chunk {i}</span>
                    <span class="page-badge">Page {page}</span>
                  </div>
                  <div class="source-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Emoji avatars for the chat.
USER_AVATAR = "🧑"
BOT_AVATAR = "🤖"


# ---------------------------------------------------------------------------
# PDF upload + indexing
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = compute_file_hash(file_bytes)

    # Only (re)build the index when a genuinely NEW file is uploaded.
    if file_hash != st.session_state.file_hash:
        with st.spinner("Processing PDF: extracting → chunking → embedding…"):
            try:
                vectorstore, was_cached = build_vectorstore(
                    file_bytes, uploaded_file.name
                )
                st.session_state.rag_chain = build_rag_chain(vectorstore)
                st.session_state.file_hash = file_hash
                st.session_state.file_name = uploaded_file.name
                st.session_state.messages = []  # fresh chat for the new document
            except ValueError as exc:
                # Expected, user-friendly errors (bad/empty PDF).
                st.error(f"❌ {exc}")
                st.stop()
            except Exception as exc:
                # Unexpected errors (network, API key rejected, quota, etc.).
                st.error(f"❌ Failed to process the PDF: {exc}")
                st.stop()

        if was_cached:
            st.success(f"✅ “{uploaded_file.name}” loaded from cache — ask away!")
        else:
            st.success(f"✅ “{uploaded_file.name}” indexed successfully — ask away!")


# ---------------------------------------------------------------------------
# Main area: empty state, or chat history + input
# ---------------------------------------------------------------------------
if st.session_state.rag_chain is None:
    # Friendly empty state shown before any PDF is uploaded.
    st.markdown(
        """
        <div class="empty-state">
          <div class="empty-icon">🗂️</div>
          <h3>No document yet</h3>
          <p>Upload a PDF above to start a grounded conversation. Your questions
          will be answered strictly from its contents — with the exact source
          passages shown under every answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Small context caption showing which document is active.
    if st.session_state.file_name:
        st.caption(f"💬 Chatting with **{st.session_state.file_name}**")

    # Replay existing chat history (each answer keeps its own sources).
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            # Hidden marker lets CSS style user vs bot bubbles differently.
            st.markdown(
                f'<span class="msg-role msg-{message["role"]}"></span>',
                unsafe_allow_html=True,
            )
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    # Chat input -> retrieve + generate
    question = st.chat_input("Ask a question about your PDF…")
    if question:
        # 1) Echo the user's question and store it in history.
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown('<span class="msg-role msg-user"></span>', unsafe_allow_html=True)
            st.markdown(question)

        # 2) Generate a grounded answer.
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown('<span class="msg-role msg-assistant"></span>', unsafe_allow_html=True)
            with st.spinner("🔎 Searching your document…"):
                try:
                    result = answer_question(st.session_state.rag_chain, question)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as exc:
                    answer = f"❌ Error while answering: {exc}"
                    sources = []
            st.markdown(answer)
            render_sources(sources)

        # 3) Store the assistant turn (with its sources) in history.
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
