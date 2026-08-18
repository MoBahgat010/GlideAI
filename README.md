# Enterprise Production RAG

Enterprise Production RAG is a practical microservices retrieval-augmented generation system for teams that need answers from mixed content in one place. The FastAPI service connects to the Triton GPU server over gRPC, PDFs are parsed locally with OpenDataLoader PDF in a Java/JVM-backed process, and the system is built to handle documents, images, tables, diagrams, and media transcripts with a clean separation between the API layer and the inference layer.

## What this project does

The system helps users ingest content, store it, search it, and ask questions in a structured way. It is designed for:

- enterprise knowledge search
- internal document assistants
- multimodal search across text and visual content
- session-based chat over private knowledge bases
- scalable GPU-backed retrieval and reranking

## Architecture

This project uses a simple microservices setup with gRPC between the API and GPU services:

- FastAPI server: handles authentication, ingestion, chat, sessions, and API routing
- Triton server: runs GPU inference for embedding and reranking and serves requests over gRPC
- MongoDB: stores session and application data
- Redis: manages working memory and short-lived chat state

This split keeps the FastAPI service light while Triton handles model execution on the GPU side.

## Multimodal support

The platform is built for more than plain text. It can work with:

- PDF content
- images and visual references
- tables and structured chunks
- diagrams and linked content
- audio or media transcripts

This makes it useful when the important answer is not only in text, but also in supporting visual or media material.

## PDF parsing

PDFs are handled locally with OpenDataLoader PDF. That is the better choice for this project because it keeps parsing inside the stack, avoids sending files to an external parsing service, and preserves layout details such as tables, bounding boxes, and embedded images. It fits the rest of the multimodal pipeline better than a plain text-only PDF reader.

## Memory management

The system includes session memory so conversations stay consistent within a user session. Redis is used for working memory and chat history, while MongoDB keeps longer-lived session records and file metadata. This gives the assistant context without depending only on the current request.

## Main uses

Use this project when you need:

- a private enterprise search layer over mixed internal content
- a chat experience that can cite relevant source chunks
- retrieval that can handle both text and image-linked content
- GPU-backed inference without running models inside the web server
- a production-style RAG stack that can grow with more data and users

## Repository layout

```text
Enterprise_RAG/
├── server/            FastAPI app, RAG pipeline, auth, chat, ingest, storage
├── triton_server/     Triton model repository and GPU inference service
├── UI/                Frontend application
├── output/            Generated outputs and extracted assets
└── run.py             Local entry point
```

## Getting started

The project is split into separate services, so start by setting up the backend and GPU service together.

1. Install the Python dependencies for the server.
2. Start MongoDB and Redis.
3. Build and run the Triton server.
4. Start the FastAPI server.
5. Open the UI and upload content for ingestion.

For exact commands and environment values, check the README files inside [server/](server/README.md) and [triton_server/](triton_server/README.md).

## Health check

The FastAPI server exposes a readiness endpoint at `/api/health`.

## Third-party licenses and notices

This project depends on software from other vendors and open-source communities. The most visible upstream licenses and notices include:

- FastAPI: MIT License
- NVIDIA Triton Inference Server: upstream NVIDIA license terms apply
- Apache-licensed components: Apache License 2.0 where used in the stack
- MongoDB ecosystem components: use their respective MongoDB or driver licenses

Please review the upstream projects for the exact license text before redistribution or commercial use.

## Notes

- The project is focused on production-style RAG workflows.
- The GPU work stays outside the FastAPI process.
- The system is designed to support multimodal enterprise search without adding unnecessary complexity.
