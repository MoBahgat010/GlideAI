# Enterprise Multi-Modal Agentic RAG Platform

> **A production-ready microservices Multi-Modal Agentic Retrieval-Augmented Generation (RAG) platform powered by Triton GPU inference acceleration, Rev.ai speech transcription, hybrid vector retrieval, and Google Workspace automation.**

---

## Executive Summary

Enterprise RAG is an end-to-end, microservices-driven intelligent assistant platform. It enables enterprise teams to index complex multimodal knowledge (PDFs with layout bounding boxes, tables, diagrams, and video/audio transcripts) and interact with it through real-time streaming conversational agents.

Inference is decoupled from the web application layer using **NVIDIA Triton Inference Server**, ensuring high-throughput GPU model execution. Video and audio recordings are transcribed into timestamped segments via **Rev.ai Speech Transcription**. Retrieval utilizes **Weaviate Hybrid Search** (dense embeddings + sparse BM25) combined with cross-encoder reranking. The agent orchestrator supports autonomous tool use, human-in-the-loop email dispatch via the **Google Gmail API**, and in-browser interactive citation spotlighting.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        React SPA Frontend (Vite)                       │
│     • Real-time SSE Token Streaming     • Interactive Citation Pills   │
│     • PDF & Media Spotlight Modal      • Human-in-the-Loop Dialogs    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / SSE / REST
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend Server                         │
│     • JWT Auth & Token Rotation        • Session & File Management     │
│     • LangGraph ReAct Orchestrator     • Celery Ingestion Triggers     │
│     • PII & Safety Middlewares         • Google OAuth & Gmail API      │
└──────────────┬────────────────────┬────────────────────┬───────────────┘
               │                    │                    │
      gRPC / HTTP (8001/8000)       │                    │
               ▼                    │                    │
┌──────────────────────────────┐    │                    │
│ NVIDIA Triton Inference Server│   │                    │
│  • Bi-Encoder Embeddings     │    │                    │
│  • Cross-Encoder Reranker    │    │                    │
│  • Layout OCR & Geometry     │    │                    │
└──────────────────────────────┘    │                    │
                                    ▼                    ▼
                    ┌──────────────────────┐  ┌──────────────────────┐
                    │ Weaviate Vector DB   │  │   Cloudinary CDN /   │
                    │  • Hybrid BM25+Dense │  │   Object Storage     │
                    │  • Session Filtering │  │  • Document PDFs     │
                    │  • Multi-Modal Graph │  │  • Media & BBox Crops│
                    └──────────────────────┘  └──────────────────────┘
                                    ▲                    ▲
                                    │                    │
                    ┌───────────────┴────────────────────┴───────┐
                    │            Storage & State Layer           │
                    │  • MongoDB: Persistent Sessions & History  │
                    │  • Redis: Working Memory & Celery Broker   │
                    │  • Celery Workers: Async Chunking & Parsing│
                    │  • Rev.ai: Speech-to-Text Transcription    │
                    └────────────────────────────────────────────┘
```

---

## Key Capabilities & Technology Stack

### 1. NVIDIA Triton Inference Server & Rev.ai Speech Transcription
- **Decoupled GPU Inference (Triton)**: Machine learning models run inside dedicated GPU containers rather than within the FastAPI web process, communicating over binary **gRPC (Port 8001)** and **HTTP (Port 8000)**.
- **Bi-Encoder Ingestion Model**: Generates normalized dense vector embeddings for multimodal document passages.
- **Cross-Encoder Reranker**: Performs fine-grained relevance re-scoring across hybrid search candidates.
- **Rev.ai Speech Transcription**: Accurately transcribes uploaded audio and video media files into timestamp-grounded transcript passages.
- **Layout Analyzers**: Extracts coordinate bounding boxes and document geometry.
- **Resource Management**: Dynamic batching and strict VRAM quota allocation per execution pipeline.

### 2. Hybrid Vector Retrieval (Weaviate)
- **Hybrid Search**: Combines BM25 keyword matching with dense cosine vector similarity (`alpha = 0.5`).
- **Two-Stage Retrieval**:
  1. Top-$K$ candidate retrieval from Weaviate filtered by `session_id`.
  2. Neural cross-encoder reranking to produce the final top-$N$ most relevant chunks.
- **Linked Multimodal Context**: Associated tables, charts, and image crops are linked to text passages and returned alongside retrieval results.

### 3. Object Storage & Media Processing (Cloudinary)
- **Cloud Object Storage**: PDF documents, diagrams, extracted images, and audio/video files are securely uploaded to **Cloudinary CDN** and mapped to unique asset URLs.
- **Local Fallback**: Full support for local disk storage when running in offline or private enterprise environments.
- **PDF Layout Extraction**: Uses **OpenDataLoader PDF** (local JVM process) to preserve table structures, visual bounding boxes, and document geometry.

### 4. Database, State & Cache Layer
- **MongoDB**: Persists user credentials, session hierarchies, uploaded file catalogs, and turn-by-turn chat history with complete citation metadata.
- **Redis**: Serves as the high-speed working memory for the agent graph and task message broker for Celery workers.
- **Celery**: Executes background asynchronous ingestion tasks (chunking, layout parsing, embedding extraction, and index upsert).

### 5. Google Workspace Integration & Human-in-the-Loop (HiTL)
- **Google OAuth 2.0**: Secure authentication granting scoped access to Gmail.
- **Gmail Search & Read Tools**:
  - `fetch_user_emails`: Queries user inbox with Gmail search syntax (`is:unread`, `from:`, `newer_than:`).
  - `get_email_details`: Fetches full email body and attachment metadata.
- **Interactive Write Operations (`send_email`)**:
  - Automatically triggers a **Human-in-the-Loop (HiTL) Approval Card** in the UI.
  - The user can review, edit the recipient/subject/body, and explicitly confirm or reject the email dispatch before any action is executed.

### 6. Streaming Agent & Interactive Citations
- **Server-Sent Events (SSE)**: Streams AI response tokens in real-time.
- **Zero Tool Output Leakage**: Intermediate tool executions (e.g. raw JSON dumps, database queries) are intercepted and kept out of the user chat stream.
- **NotebookLM-Style Citation Badges**: Inline citations (`[1]`, `[2]`) render as interactive badges with hover previews that open the PDF Spotlight Viewer modal at the exact bounding box and page.

---

## Running with Docker

The GPU inference tier is containerized with Docker and Docker Compose using the NVIDIA Container Toolkit.

### Prerequisites
- [Docker Engine](https://docs.docker.com/engine/install/) (v24.0+)
- [Docker Compose](https://docs.docker.com/compose/) (v2.0+)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU acceleration)
- NVIDIA GPU with CUDA support

### Step 1: Clone Repository & Configure Environment
```bash
git clone https://github.com/your-org/Enterprise_RAG.git
cd Enterprise_RAG

# Copy sample environment configuration
cp .env.example .env
```

Ensure the following variables are defined in your `.env` file:
```dotenv
# GPU & Triton Server
TRITON_HTTP_URL=http://localhost:8000
TRITON_GRPC_URL=localhost:8001
HF_TOKEN=your_huggingface_token
HF_CACHE_DIR=~/.cache/huggingface

# Databases
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=enterprise-rag
REDIS_URL=redis://localhost:6000/0

# Weaviate Vector Database
WEAVIATE_REST_ENDPOINT=your_weaviate_endpoint
WEAVIATE_API_KEY=your_weaviate_key
INDEX_NAME=EnterpriseDocuments

# Cloudinary Object Storage
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret

# LLM Orchestrator
BASE_URL=https://api.openai.com/v1
API_KEY=your_llm_api_key
SUMMARIZER=gpt-4o-mini
LVLM=gpt-4o

# Google OAuth & Gmail
GOOGLE_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8080/api/auth/google/callback
GMAIL_REDIRECT_URI=http://localhost:8080/api/gmail/callback
FRONTEND_URL=http://localhost:5173
```

### Step 2: Start Triton Inference Server via Docker
```bash
# Launch Triton GPU Inference container
docker compose -f triton_server/docker-compose.yml up --build -d
```

Verify Triton health:
```bash
curl -f http://localhost:8000/v2/health/ready
# Returns HTTP 200 OK when models are loaded
```

---

## Local Development & Services Setup

### Step 1: Start Redis and MongoDB
```bash
# Start Redis on port 6000
redis-server --port 6000 --daemonize yes

# Ensure MongoDB is running on port 27017
sudo systemctl start mongod
```

### Step 2: Set Up Python Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run Backend Services
You can run the individual services or use the `run.py` launcher:

```bash
# Terminal 1: Start Triton Server (Docker)
python3 run.py triton

# Terminal 2: Start Celery Ingestion Worker
python3 run.py celery

# Terminal 3: Start FastAPI Application Server
python3 run.py fastapi
```

### Step 4: Run the React UI
```bash
cd UI
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Project Structure

```text
Enterprise_RAG/
├── server/                          # FastAPI application backend
│   ├── config.py                    # Environment and runtime configurations
│   ├── main.py                      # Application entrypoint & CORS middleware
│   ├── RAG_Pipeline/                # Multi-modal RAG processing pipelines
│   │   ├── ingestion/               # Chunking, layout parsing & embedding
│   │   ├── retrieval/               # Hybrid search & cross-encoder reranking
│   │   └── storage/                 # Weaviate vector database client
│   └── src/
│       ├── auth/                    # JWT authentication & password hashing
│       ├── db/                      # MongoDB and Redis connection managers
│       ├── interrupts/              # LangGraph Human-in-the-Loop middlewares
│       ├── jobs/                    # Celery asynchronous ingestion tasks
│       ├── models/                  # Pydantic schemas and database models
│       ├── routers/                 # REST API endpoints (Chat, Auth, Ingest, Sessions)
│       └── services/                # Agent workflow, Gmail tools, Google OAuth
├── triton_server/                   # NVIDIA Triton GPU inference server
│   ├── models/                      # Triton model repository (Bi-Encoder, Reranker)
│   ├── scripts/                     # Automated model download utilities
│   ├── docker-compose.yml           # Development Docker Compose specification
│   ├── docker-compose.prod.yml      # Production Docker Compose specification
│   └── Dockerfile                   # Custom Triton container definition
├── UI/                              # Modern React SPA frontend
│   ├── src/
│   │   ├── App.jsx                  # Main chat, viewer modal, and session layout
│   │   ├── App.css                  # Design system & dark-mode styling
│   │   ├── pages/                   # Auth, Landing, and Session management views
│   │   └── components/              # Modals, drawers, and interactive controls
│   └── vite.config.js               # Vite bundler configuration
├── run.py                           # CLI launcher for server, triton, and celery
└── requirements.txt                 # Backend Python package requirements
```

---

## REST API Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Readiness probe (checks API & Triton connectivity) |
| `POST` | `/api/auth/register` | Register new user account |
| `POST` | `/api/auth/login` | Login and receive access + refresh JWT pair |
| `POST` | `/api/sessions` | Create a new isolated RAG session |
| `GET` | `/api/sessions` | List active sessions for authenticated user |
| `GET` | `/api/sessions/{session_id}` | Retrieve session history and attached files |
| `POST` | `/api/sessions/{session_id}/ask` | SSE streaming endpoint for agentic RAG chat |
| `POST` | `/api/sessions/{session_id}/approve` | Resume agent after HiTL approval decision |
| `POST` | `/api/ingest/chunk` | Upload and trigger asynchronous document ingestion |
| `GET` | `/api/ingest/status/{task_id}` | Poll Celery ingestion job progress |
| `GET` | `/api/auth/google/login` | Initiate Google OAuth flow |
| `GET` | `/api/gmail/messages` | Search and preview user Gmail emails |

---

## License & Notices

- **Application Code**: Licensed under the MIT License.
- **NVIDIA Triton Inference Server**: Subject to upstream NVIDIA software license agreements.
- **Third-Party Libraries**: Subject to their respective Apache 2.0, MIT, and BSD licenses.
