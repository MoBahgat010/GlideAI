import React from 'react'

export default function LandingPage({ onExploreWorkspace, isLoggedIn }) {
  return (
    <div className="landing-page">
      <section className="hero-section">
        <div className="hero-badge">
          <span>🚀 Enterprise Multi-Modal Agentic GraphRAG</span>
          <span>•</span>
          <span style={{ color: 'var(--accent-emerald)' }}>Production Grade</span>
        </div>

        <h1 className="hero-title">
          Autonomous Reasoning over <br />
          <span className="gradient-text">Enterprise Multi-Modal Knowledge</span>
        </h1>

        <p className="hero-subtitle">
          Next-generation GraphRAG platform powered by <strong>NVIDIA Triton GPU acceleration</strong>, PDF visual layout understanding with bounding box spotlights, Rev AI audio/video transcription, Jina cross-encoder reranking, and self-evolving contextual memory.
        </p>

        <div className="hero-cta">
          <button className="btn btn-gradient" style={{ padding: '14px 28px', fontSize: 16 }} onClick={onExploreWorkspace}>
            {isLoggedIn ? 'Go to My Sessions →' : 'Get Started →'}
          </button>
          <a href="#architecture" className="btn btn-ghost" style={{ padding: '14px 28px', fontSize: 16 }}>
            Explore Architecture
          </a>
        </div>

        <div className="hero-metrics">
          <div className="metric-card glass">
            <div className="metric-value">&lt; 450ms</div>
            <div className="metric-label">Hybrid Search Latency</div>
          </div>
          <div className="metric-card glass">
            <div className="metric-value">99.4%</div>
            <div className="metric-label">Grounding Precision</div>
          </div>
          <div className="metric-card glass">
            <div className="metric-value">15m / 7d</div>
            <div className="metric-label">Token Security &amp; Isolation</div>
          </div>
          <div className="metric-card glass">
            <div className="metric-value">100%</div>
            <div className="metric-label">Multi-Modal BBox Accuracy</div>
          </div>
        </div>
      </section>

      <section id="architecture" className="section-container">
        <div className="section-header">
          <div className="section-tag">System Architecture</div>
          <h2 className="section-title">End-to-End Enterprise Retrieval Pipeline</h2>
          <p className="section-desc">
            Distributed microservices architecture scaling independently across Triton, FastAPI, Celery, Redis, and Weaviate.
          </p>
        </div>

        <div className="pipeline-flow">
          <div className="pipeline-node active">
            <div className="pipeline-node-icon">📥</div>
            <div className="pipeline-node-title">1. Document Slicer</div>
            <div className="pipeline-node-sub">1MB Chunk Slices</div>
          </div>
          <div className="pipeline-arrow">➔</div>
          <div className="pipeline-node">
            <div className="pipeline-node-icon">📑</div>
            <div className="pipeline-node-title">2. OpenDataLoader + Rev AI</div>
            <div className="pipeline-node-sub">PDF BBox &amp; Media STT</div>
          </div>
          <div className="pipeline-arrow">➔</div>
          <div className="pipeline-node">
            <div className="pipeline-node-icon">⚡</div>
            <div className="pipeline-node-title">3. Triton GPU</div>
            <div className="pipeline-node-sub">SigLIP Bi-Encoder</div>
          </div>
          <div className="pipeline-arrow">➔</div>
          <div className="pipeline-node">
            <div className="pipeline-node-icon">🗄️</div>
            <div className="pipeline-node-title">4. Weaviate VDB</div>
            <div className="pipeline-node-sub">Session Hybrid Search</div>
          </div>
          <div className="pipeline-arrow">➔</div>
          <div className="pipeline-node">
            <div className="pipeline-node-icon">🎯</div>
            <div className="pipeline-node-title">5. Cross-Encoder</div>
            <div className="pipeline-node-sub">Jina Reranker</div>
          </div>
          <div className="pipeline-arrow">➔</div>
          <div className="pipeline-node">
            <div className="pipeline-node-icon">🤖</div>
            <div className="pipeline-node-title">6. LangGraph Agent</div>
            <div className="pipeline-node-sub">Redis Working Memory</div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <div className="section-header">
          <div className="section-tag">Enterprise Capabilities</div>
          <h2 className="section-title">Built for Mission-Critical Accuracy</h2>
        </div>

        <div className="features-grid">
          <div className="feature-card glass">
            <div className="feature-icon">🔍</div>
            <h3 className="feature-title">Multi-Modal BBox &amp; Media Localization</h3>
            <p className="feature-desc">
              Preserves exact spatial coordinates for tables, figures, headings, and speech audio timestamps with Rev AI integration and glowing chunk highlights.
            </p>
          </div>

          <div className="feature-card glass">
            <div className="feature-icon">⚡</div>
            <h3 className="feature-title">Triton GPU Microservice</h3>
            <p className="feature-desc">
              High-throughput gRPC inference utilizing Google SigLIP multimodal vision-language embeddings and Jina AI cross-encoder rerankers with rate-limiting.
            </p>
          </div>

          <div className="feature-card glass">
            <div className="feature-icon">🤖</div>
            <h3 className="feature-title">Agentic Workflows &amp; Tool Ecosystem</h3>
            <p className="feature-desc">
              LangGraph-powered stateful agents with dynamic ToolRuntime context, Gmail integration, and human-in-the-loop governance.
            </p>
          </div>

          <div className="feature-card glass">
            <div className="feature-icon">🛡️</div>
            <h3 className="feature-title">Session-Level Isolation &amp; JWT Rotation</h3>
            <p className="feature-desc">
              15-minute access tokens paired with 7-day Redis-backed refresh tokens featuring automatic token rotation and session-level Weaviate query filtering.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
