# Eye on AI Podcast Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about AI research and applications using transcripts from 300+ episodes of the [Eye on AI](https://www.eye-on.ai/) podcast.

## Features

- **Intelligent Q&A** — Ask questions about AI topics discussed across hundreds of podcast episodes
- **Source citations** — Every answer includes references to specific episodes
- **Auto-sync** — Automatically detects and ingests new transcripts added to Google Drive
- **Embeddable** — Drop the chat widget into any website via iframe
- **Dual LLM support** — Works with both OpenAI and Anthropic models
- **Self-hosted vector DB** — Uses ChromaDB locally, no external vector database needed
- **Docker-ready** — Single `docker-compose up` deployment

## Architecture

```
Google Drive (transcripts) → Ingestion Pipeline → ChromaDB (vectors)
                                                        ↓
User question → Embedding → Similarity Search → LLM → Answer + Sources
```

| Component    | Technology                    |
| ------------ | ----------------------------- |
| Backend      | Python / FastAPI              |
| Vector DB    | ChromaDB (local, persistent)  |
| Embeddings   | OpenAI text-embedding-3-small |
| LLM          | OpenAI GPT-4o or Anthropic Claude |
| Frontend     | Vanilla HTML/CSS/JS           |
| Deployment   | Docker / docker-compose       |

## Quick Start

### Prerequisites

- Docker and docker-compose
- A Google Cloud service account with access to the Drive folder ([setup guide](#google-drive-setup))
- An OpenAI API key (for embeddings + chat)
- Optionally, an Anthropic API key

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/eye-on-ai-chatbot.git
cd eye-on-ai-chatbot
```

### 2. Add credentials

Place your Google service account JSON file as `credentials.json` in the project root:

```bash
cp /path/to/your/downloaded-key.json credentials.json
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:
- `OPENAI_API_KEY` — your OpenAI API key
- `DRIVE_FOLDER_ID` — your Google Drive folder ID (already set to the default)
- `LLM_PROVIDER` — `openai` (default) or `anthropic`
- `LLM_MODEL` — e.g., `gpt-4o`, `claude-sonnet-4-20250514`

### 4. Build and start

```bash
docker-compose up -d --build
```

### 5. Run initial ingestion

The first time, you need to ingest all the transcripts:

```bash
# Full ingestion (all documents)
docker-compose exec chatbot python ingest.py --full
```

This will:
1. Connect to Google Drive and list all Google Docs in the folder
2. Export each document as plain text
3. Chunk the text with overlap (optimized for dense technical content)
4. Generate embeddings via OpenAI
5. Store everything in ChromaDB

**Note:** First ingestion of 300+ episodes will take several minutes and use OpenAI embedding API credits (~$0.50-2.00 depending on transcript lengths).

### 6. Open the chatbot

Visit `http://localhost:8000` in your browser. You're ready to chat!

## Syncing New Episodes

When new transcripts are added to the Google Drive folder:

```bash
# Incremental sync — only processes new/modified documents
docker-compose exec chatbot python sync.py

# Dry run — see what would be synced without doing it
docker-compose exec chatbot python sync.py --dry-run

# Full re-sync — reprocess everything
docker-compose exec chatbot python sync.py --full
```

### Automatic sync via cron

Add a cron job to sync automatically (e.g., daily at 3 AM):

```bash
crontab -e
```

```
0 3 * * * cd /path/to/eye-on-ai-chatbot && docker-compose exec -T chatbot python sync.py >> /var/log/eyeonai-sync.log 2>&1
```

## Embedding in a Website

### Option 1: iframe

```html
<iframe
  src="https://your-server.com"
  width="100%"
  height="600"
  style="border: none; border-radius: 12px;"
></iframe>
```

### Option 2: Custom API URL

If hosting the frontend separately, set the API URL before loading `chat.js`:

```html
<script>
  window.EYEONAI_API_URL = 'https://your-api-server.com';
</script>
<script src="/path/to/chat.js"></script>
```

## API Endpoints

| Method | Path      | Description                      |
| ------ | --------- | -------------------------------- |
| GET    | `/`       | Serves the chat UI               |
| GET    | `/health` | Health check + collection stats  |
| GET    | `/stats`  | Collection statistics            |
| POST   | `/chat`   | Send a message, get a response   |

### POST /chat

**Request:**
```json
{
  "message": "What do guests say about AI safety?",
  "conversation_id": "optional-uuid-for-multi-turn"
}
```

**Response:**
```json
{
  "response": "Several guests have discussed AI safety...",
  "sources": [
    {
      "episode": "Episode 142 - Stuart Russell on AI Alignment",
      "snippet": "The key challenge with AI safety is..."
    }
  ],
  "conversation_id": "uuid"
}
```

## Project Structure

```
eye-on-ai-chatbot/
├── server.py           # FastAPI server + chat endpoint
├── ingest.py           # Full ingestion pipeline (CLI)
├── sync.py             # Incremental sync script (CLI)
├── config.py           # Configuration from environment
├── drive_client.py     # Google Drive API client
├── chunker.py          # Text chunking with overlap
├── embeddings.py       # OpenAI embedding generation
├── vector_store.py     # ChromaDB operations
├── llm.py              # LLM abstraction (OpenAI / Anthropic)
├── static/
│   ├── index.html      # Chat UI
│   ├── chat.css        # Styles
│   └── chat.js         # Client-side logic
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Google Drive Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project (or use an existing one)
3. Enable the **Google Drive API** and **Google Docs API**
4. Go to **IAM & Admin → Service Accounts** → Create a service account
5. Create a JSON key and download it
6. Share your Google Drive folder with the service account email (Viewer access)
7. Place the JSON key as `credentials.json` in the project root

## Configuration Reference

All settings are in `.env` (see `.env.example`):

| Variable               | Default             | Description                           |
| ---------------------- | ------------------- | ------------------------------------- |
| `LLM_PROVIDER`         | `openai`            | `openai` or `anthropic`              |
| `LLM_MODEL`            | `gpt-4o`            | Model name                           |
| `OPENAI_API_KEY`       | —                   | Required for embeddings + OpenAI chat |
| `ANTHROPIC_API_KEY`    | —                   | Required if using Anthropic           |
| `GOOGLE_CREDENTIALS_PATH` | `credentials.json` | Path to service account key        |
| `DRIVE_FOLDER_ID`      | (preset)            | Google Drive folder ID               |
| `CHROMA_PERSIST_DIR`   | `./chroma_data`     | ChromaDB storage directory           |
| `CHUNK_SIZE`           | `1000`              | Tokens per chunk                     |
| `CHUNK_OVERLAP`        | `200`               | Overlap tokens between chunks        |
| `TOP_K`                | `8`                 | Chunks retrieved per query           |
| `PORT`                 | `8000`              | Server port                          |
| `CORS_ORIGINS`         | `*`                 | Allowed CORS origins                 |

## Running Without Docker

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up .env
cp .env.example .env
# Edit .env with your keys

# Run ingestion
python ingest.py --full

# Start the server
python server.py
```

## License

MIT
