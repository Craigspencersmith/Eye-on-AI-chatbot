# Eye on AI — Podcast Chatbot

RAG chatbot over 300+ episodes of the [Eye on AI](https://www.eye-on.ai/) podcast. Ask questions about guests, topics, AI research, and more — answers are grounded in actual episode transcripts.

## Stack

- **Backend:** Python / FastAPI
- **Vector DB:** ChromaDB (self-hosted)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLM:** OpenAI GPT-4o (configurable — supports Anthropic too)
- **Frontend:** Embedded chat widget
- **Drive Sync:** Auto-pulls new/updated transcripts from Google Drive

## Quick Start

### 1. Set up credentials

```bash
# Copy the example env file
cp .env.example .env

# Edit .env — at minimum set:
#   OPENAI_API_KEY=sk-...
```

Place your Google service account JSON in `credentials/service-account.json`.

### 2. Run with Docker

```bash
docker compose up -d
```

The app starts at `http://localhost:8000`. On first launch, it syncs all transcripts from Drive and indexes them (this takes a few minutes for 300+ episodes).

### 3. Run without Docker

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat widget UI |
| `/api/chat` | POST | Send a question, get an answer with sources |
| `/api/sync` | POST | Manually trigger Drive sync |
| `/api/status` | GET | Check indexing status |

### Chat request

```json
POST /api/chat
{
  "question": "What episodes discuss transformer architectures?",
  "conversation_history": []
}
```

### Chat response

```json
{
  "answer": "Several episodes cover transformer architectures...",
  "sources": ["episode_145", "episode_203", "episode_287"]
}
```

## Configuration

All config via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for embeddings + chat |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o` | Model for chat completions |
| `DRIVE_FOLDER_ID` | — | Google Drive folder with transcripts |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `SYNC_INTERVAL` | `3600` | Auto-sync interval (seconds) |
| `TOP_K` | `10` | Number of chunks retrieved per query |

## Embedding as a Widget

Add to any page:

```html
<iframe src="https://your-server:8000" width="400" height="600" frameborder="0"></iframe>
```

## Architecture

```
Google Drive (transcripts)
    ↓ periodic sync
Drive Sync → Chunker → OpenAI Embeddings → ChromaDB
                                                ↑
User Question → Embed → Vector Search ──────────┘
                              ↓
                    Retrieved Chunks + Question
                              ↓
                         LLM (GPT-4o)
                              ↓
                         Answer + Sources
```
