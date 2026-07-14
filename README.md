# 📄 RAG PDF Bot

Upload a PDF and ask questions about it in natural language. Every answer is
**grounded strictly in the PDF's content** — if the answer isn't in the
document, the bot says so instead of making something up.

Built with **LangChain + ChromaDB + Streamlit**, and works with **Google Gemini
(default)** or **OpenAI** via a single environment variable.

-------

## ✨ Features

- **PDF upload** with multi-page support; empty/scanned pages are skipped gracefully.
- **Persistent vector store** (ChromaDB on local disk) so embeddings survive restarts.
- **Embedding cache by file hash** — the same PDF is never embedded twice.
- **Chat-style Q&A** that remembers your conversation history.
- **Visible grounding**: every answer has an expandable "Sources" section showing
  the exact chunks (with page numbers) used to build it.
- **Anti-hallucination prompt** that forces answers to come only from the document.
- **One-variable provider switch**: `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai`.

---

## 🗂 Project structure

```
rag-pdf-bot/
├── app.py             # Streamlit UI (upload + chat + sources)
├── rag_pipeline.py    # Core RAG logic: load → chunk → embed → store → retrieve → answer
├── config.py          # Loads API keys + provider choice from .env
├── requirements.txt   # Pinned, compatible dependencies
├── .env.example       # Template for your secrets (copy to .env)
├── .gitignore         # Keeps .env and the Chroma DB out of git
└── README.md
```

---

## 🚀 Setup

> **Recommended Python: 3.10 – 3.12.** These have prebuilt wheels for
> `chromadb` and its native dependencies. Very new interpreters (e.g. 3.13/3.14)
> may not have wheels yet and can fail to install — see *Troubleshooting*.

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
cd rag-pdf-bot
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Copy the example env file and fill in your key:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```
**macOS / Linux:**
```bash
cp .env.example .env
```

Then edit `.env`:
```dotenv
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-real-key-here
```
- Gemini key: https://aistudio.google.com/app/apikey
- To use OpenAI instead, set `LLM_PROVIDER=openai` and fill in `OPENAI_API_KEY`.

### 5. Run the app
```bash
streamlit run app.py
```
Your browser opens at `http://localhost:8501`. Upload a PDF and start asking questions.

---

## 🧠 How it works

This is a classic **Retrieval-Augmented Generation (RAG)** pipeline. Instead of
asking the LLM to answer from memory (which invites hallucination), we retrieve
the most relevant passages from *your* PDF and make the model answer using only
those.

```
        ┌─────────┐   ┌────────┐   ┌────────┐   ┌────────────────┐
Upload →│  LOAD   │ → │ CHUNK  │ → │ EMBED  │ → │ STORE (Chroma) │   (once per PDF)
  PDF   └─────────┘   └────────┘   └────────┘   └────────────────┘
                                                        │
Question ───────────────────────────────────────►  RETRIEVE top-k=4
                                                        │
                                                    GENERATE (LLM) → grounded answer + sources
```

1. **Load** — `pypdf` extracts text page by page. Pages with no extractable
   text (blank or scanned images) are skipped.
2. **Chunk** — `RecursiveCharacterTextSplitter` splits the text into
   ~1000-character chunks with ~150-character overlap so context isn't lost at
   boundaries. Each chunk keeps its page number.
3. **Embed** — each chunk is converted to a vector using the provider's current
   recommended embedding model (`gemini-embedding-001` for Gemini,
   `text-embedding-3-small` for OpenAI).
4. **Store** — vectors are saved in a **persisted Chroma collection** named after
   the PDF's SHA-256 hash. Re-uploading the same PDF reuses the existing
   collection instead of re-embedding (the **cache**).
5. **Retrieve** — for each question, Chroma returns the **top-4** most similar
   chunks.
6. **Generate** — those chunks are passed as context to the LLM through a
   LangChain retrieval chain, which produces the answer.

### 🛡 Prompt engineering (how hallucination is reduced)

The single biggest lever against hallucination is the **system prompt** (see
`SYSTEM_PROMPT` in `rag_pipeline.py`). It instructs the model to:

- **Answer only from the provided context** — no outside knowledge.
- **Say `"I couldn't find this in the document."`** when the answer isn't in the
  retrieved chunks, instead of guessing.
- **Stay concise and grounded**, optionally citing page numbers.

Combined with `temperature=0` (deterministic, non-"creative" decoding) and
showing the retrieved **sources** under every answer, the user can always verify
that a response is actually backed by the document.

---

## 🔧 Configuration reference

All settings are read from `.env` (see `.env.example`):

| Variable             | Default                         | Purpose                                  |
| -------------------- | ------------------------------- | ---------------------------------------- |
| `LLM_PROVIDER`       | `gemini`                        | `gemini` or `openai`                     |
| `GOOGLE_API_KEY`     | —                               | Required when provider is `gemini`       |
| `OPENAI_API_KEY`     | —                               | Required when provider is `openai`       |
| `GEMINI_CHAT_MODEL`  | `gemini-2.5-flash`              | Gemini chat model                        |
| `GEMINI_EMBED_MODEL` | `models/gemini-embedding-001`   | Gemini embedding model                   |
| `OPENAI_CHAT_MODEL`  | `gpt-4o-mini`                   | OpenAI chat model                        |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small`        | OpenAI embedding model                   |
| `CHROMA_DIR`         | `chroma_db`                     | Where the vector store is persisted      |

---

## 🩹 Troubleshooting

- **`Configuration error: GOOGLE_API_KEY is missing`** — you haven't created
  `.env` or the key line is blank. Copy `.env.example` to `.env` and add your key.
- **`pip install` fails building `chromadb` / `onnxruntime`** — you're likely on
  a very new Python. Use Python 3.10–3.12 in your virtual environment.
- **"No extractable text found in this PDF"** — the PDF is scanned images with no
  text layer. Run it through an OCR tool first, then re-upload.
- **Answers are always "I couldn't find this in the document"** — the info may
  genuinely not be in the PDF, or the PDF didn't extract cleanly. Check the
  "Sources" expander to see what was retrieved.
- **Switched providers but nothing changed** — restart the app after editing
  `.env`. Note the two providers use different embedding spaces, so a PDF indexed
  under one provider is re-embedded when you switch.

---

## 📝 Notes

- API keys are **only** read from `.env` — nothing is hardcoded, and `.env` plus
  the `chroma_db/` folder are git-ignored.
- The vector store persists between runs, so previously indexed PDFs load
  instantly from cache.
