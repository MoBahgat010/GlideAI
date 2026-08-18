import React, { useState, useEffect, useCallback, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'
import './App.css'
import LandingPage from './pages/LandingPage'
import AuthPage from './pages/AuthPage'
import SessionsPage from './pages/SessionsPage'
import GmailStatusBar from './components/GmailStatusBar'
import HiTLBanner from './components/HiTLBanner'
import MemoriesModal from './components/MemoriesModal'

// ── Constants & Helpers ────────────────────────────────────────────────────────
const CHUNK_SIZE = 1024 * 1024 // 1 MB
const POLL_INTERVAL_MS = 1500

function formatBytes(b) {
  if (!b || b <= 0) return '0 B'
  if (b < 1024) return `${b} B`
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1048576).toFixed(1)} MB`
}

function genId() {
  return crypto.randomUUID()
}

// ── Resilient Authenticated Fetch with Automatic Refresh Rotation ──────────
let refreshPromise = null

async function requestTokenRefresh() {
  const currentRefreshToken = localStorage.getItem('graphrag_refresh_token') || ''
  if (!currentRefreshToken) return null

  if (!refreshPromise) {
    refreshPromise = (async () => {
      try {
        const refreshRes = await fetch('/api/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: currentRefreshToken }),
        })
        if (refreshRes.ok) {
          const tokenData = await refreshRes.json()
          localStorage.setItem('graphrag_access_token', tokenData.access_token)
          localStorage.setItem('graphrag_refresh_token', tokenData.refresh_token)
          if (tokenData.username) localStorage.setItem('graphrag_username', tokenData.username)
          window.dispatchEvent(new Event('auth_token_refreshed'))
          return tokenData.access_token
        } else {
          localStorage.removeItem('graphrag_access_token')
          localStorage.removeItem('graphrag_refresh_token')
          localStorage.removeItem('graphrag_username')
          window.dispatchEvent(new Event('auth_logout'))
          return null
        }
      } catch (e) {
        console.error('Failed to auto-refresh token:', e)
        return null
      } finally {
        refreshPromise = null
      }
    })()
  }
  return refreshPromise
}

async function authFetch(url, options = {}) {
  let token = localStorage.getItem('graphrag_access_token') || ''

  const headers = { ...(options.headers || {}) }
  if (token && !headers['Authorization']) headers['Authorization'] = `Bearer ${token}`

  let res = await fetch(url, { ...options, headers })

  if (res.status === 401) {
    const refreshToken = localStorage.getItem('graphrag_refresh_token') || ''
    if (refreshToken) {
      const newToken = await requestTokenRefresh()
      if (newToken) {
        headers['Authorization'] = `Bearer ${newToken}`
        res = await fetch(url, { ...options, headers })
      }
    }
  }
  return res
}

// ── Chunked File Upload API ───────────────────────────────────────────────────
async function uploadFileInChunks(file, uploadId, sessionId, onProgress, token) {
  if (!sessionId || sessionId === 'default') {
    throw new Error('Please select or create a valid session before uploading files.')
  }
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE)
  const headers = {}
  if (token) headers['Authorization'] = `Bearer ${token}`

  for (let i = 0; i < totalChunks; i++) {
    const slice = file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE)
    const form = new FormData()
    form.append('file', slice, file.name)
    form.append('upload_id', uploadId)
    form.append('chunk_index', String(i))
    form.append('total_chunks', String(totalChunks))
    form.append('filename', file.name)
    form.append('session_id', sessionId)

    const res = await authFetch('/api/ingest/chunk', { method: 'POST', body: form, headers })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `Upload slice ${i} failed: ${res.statusText}`)
    }
    onProgress(((i + 1) / totalChunks) * 0.5)
  }

  const finalizeForm = new FormData()
  finalizeForm.append('upload_id', uploadId)
  finalizeForm.append('filename', file.name)
  finalizeForm.append('total_chunks', String(totalChunks))
  finalizeForm.append('session_id', sessionId)

  const res = await authFetch('/api/ingest/finalize', { method: 'POST', body: finalizeForm, headers })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Finalize failed: ${res.statusText}`)
  }
  const data = await res.json()
  return data.task_id
}

// ── In-App PDF & Bounding Box Spotlight Viewer Modal ─────────────────────────
function BBoxModal({ isOpen, onClose, citation, activeSessionId }) {
  const [activeTab, setActiveTab] = useState('split')
  const [currentPage, setCurrentPage] = useState(1)

  useEffect(() => {
    if (citation) {
      setActiveTab('split')
      setCurrentPage(citation.page || 1)
    }
  }, [citation, isOpen])

  if (!isOpen || !citation) return null

  const bbox = citation.bbox || [-29.718, -29.758, 535.734, 533.773]
  const fileName = citation.file_name || citation.filename || 'document.pdf'
  const type = citation.type || 'text'
  const score = citation.score != null ? Number(citation.score).toFixed(4) : 'N/A'
  const startTime = citation.start_time
  const endTime = citation.end_time
  const isTranscript = type === 'transcript' || startTime != null
  const isPdf = fileName.toLowerCase().endsWith('.pdf')
  const isMedia = fileName.toLowerCase().match(/\.(mp4|webm|mp3|wav|ogg|mov|mkv|aac|flac)$/) || isTranscript
  const sessionId = citation.session_id || activeSessionId || 'default'

  // Direct Cloudinary file_url
  const baseFileUrl = citation.file_url || citation.url || ''
  const pageJumpUrl = `${baseFileUrl}#page=${currentPage}`
  const timeJumpUrl = `${baseFileUrl}#t=${startTime || 0}`

  // Precise BBox Coordinate Mapping (Standard PDF Points: ~595 x 842 pt A4 / 612 x 792 pt Letter)
  const [rawX0 = 0, rawY0 = 0, rawX1 = 535, rawY1 = 533] = Array.isArray(bbox) ? bbox : [0, 0, 535, 533]
  const docWidth = 595.28
  const docHeight = 841.89

  // Clamp and handle negative margin offsets
  const minX = Math.max(0, Math.min(rawX0, rawX1))
  const maxX = Math.max(rawX0, rawX1)
  const minY = Math.max(0, Math.min(rawY0, rawY1))
  const maxY = Math.max(rawY0, rawY1)

  const leftPct = Math.min(Math.max((minX / docWidth) * 100, 2), 85)
  const topPct = Math.min(Math.max((minY / docHeight) * 100, 2), 85)
  const widthPct = Math.min(Math.max(((maxX - minX) / docWidth) * 100, 15), 96 - leftPct)
  const heightPct = Math.min(Math.max(((maxY - minY) / docHeight) * 100, 8), 96 - topPct)

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog glass" style={{ maxWidth: 1050, width: '92vw', height: '88vh', display: 'flex', flexDirection: 'column' }} onClick={e => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-head" style={{ padding: '14px 20px', borderBottom: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 22 }}>{isTranscript ? '🎬' : '📑'}</span>
            <div>
              <div className="modal-title" style={{ fontSize: '1.05rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: 8 }}>
                <span>{fileName}</span>
                <a
                  href={baseFileUrl}
                  target="_blank"
                  rel="noreferrer"
                  style={{ fontSize: 12, color: 'var(--accent-cyan)', textDecoration: 'none', background: 'rgba(6, 182, 212, 0.1)', padding: '2px 8px', borderRadius: 4, border: '1px solid rgba(6, 182, 212, 0.3)' }}
                >
                  Open in Tab ↗
                </a>
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                {isTranscript
                  ? `Media Playback Segment (${startTime?.toFixed(1) || 0}s - ${endTime?.toFixed(1) || 0}s)`
                  : `Page ${currentPage} • Highlighted Chunk BBox: [${Array.isArray(bbox) ? bbox.map(n => typeof n === 'number' ? n.toFixed(1) : n).join(', ') : bbox}]`}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {/* View Mode Switcher */}
            {isPdf && (
              <div className="nav-tabs" style={{ padding: 2 }}>
                <button
                  className={`nav-btn ${activeTab === 'split' ? 'active' : ''}`}
                  onClick={() => setActiveTab('split')}
                  style={{ padding: '4px 10px', fontSize: 12 }}
                >
                  📑 Split Spotlight
                </button>
                <button
                  className={`nav-btn ${activeTab === 'pdf' ? 'active' : ''}`}
                  onClick={() => setActiveTab('pdf')}
                  style={{ padding: '4px 10px', fontSize: 12 }}
                >
                  🔍 Live PDF
                </button>
                <button
                  className={`nav-btn ${activeTab === 'text' ? 'active' : ''}`}
                  onClick={() => setActiveTab('text')}
                  style={{ padding: '4px 10px', fontSize: 12 }}
                >
                  📝 Raw Text
                </button>
              </div>
            )}
            <button className="modal-close-btn" onClick={onClose}>✕</button>
          </div>
        </div>

        {/* Modal Content Body */}
        <div className="modal-body" style={{ flex: 1, overflow: 'hidden', padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Split Mode: Live PDF on Left, BBox Spotlight & Text on Right */}
          {activeTab === 'split' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 14, flex: 1, minHeight: 0 }}>
              {/* Left Column: Embedded PDF Viewer at exact page */}
              <div style={{ background: '#0b0f19', borderRadius: 8, border: '1px solid var(--border-subtle)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                <div style={{ padding: '6px 12px', background: 'rgba(255,255,255,0.03)', borderBottom: '1px solid var(--border-subtle)', fontSize: 12, color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>Live PDF Document (Page {currentPage})</span>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button className="btn btn-ghost" style={{ padding: '2px 8px', fontSize: 11 }} onClick={() => setCurrentPage(p => Math.max(1, p - 1))}>◀ Prev</button>
                    <button className="btn btn-ghost" style={{ padding: '2px 8px', fontSize: 11 }} onClick={() => setCurrentPage(p => p + 1)}>Next ▶</button>
                  </div>
                </div>
                <iframe
                  src={pageJumpUrl}
                  style={{ width: '100%', flex: 1, border: 'none', background: '#ffffff' }}
                  title="PDF Viewer"
                />
              </div>

              {/* Right Column: Visual Bounding Box Spotlight & Grounded Text */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto', minHeight: 0 }}>
                {/* Visual BBox Coordinate Spotlight */}
                <div style={{ background: '#0b0f19', border: '1px solid var(--border-subtle)', borderRadius: 8, padding: 12, position: 'relative' }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
                    <span>📍 Chunk BBox Spotlight on Page {currentPage}</span>
                    <span style={{ color: 'var(--accent-emerald)', fontSize: 11 }}>Rerank Score: {score}</span>
                  </div>

                  <div style={{ position: 'relative', width: '100%', height: 220, background: '#131b2e', borderRadius: 6, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.08)' }}>
                    {/* Simulated Document Page Canvas */}
                    <div style={{ position: 'absolute', inset: 0, padding: 16, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', opacity: 0.25, userSelect: 'none' }}>
                      <div style={{ fontSize: 36, marginBottom: 4 }}>📄</div>
                      <div style={{ fontSize: 11, color: '#ffffff' }}>{fileName} (p. {currentPage})</div>
                    </div>

                    {/* Precise Glowing Yellow Bounding Box Highlight Overlay */}
                    <div
                      style={{
                        position: 'absolute',
                        top: `${topPct}%`,
                        left: `${leftPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                        background: 'rgba(254, 240, 138, 0.42)',
                        border: '2.5px solid #eab308',
                        borderRadius: 4,
                        boxShadow: '0 0 20px rgba(234, 179, 8, 0.8), inset 0 0 10px rgba(250, 204, 21, 0.3)',
                        pointerEvents: 'none',
                        zIndex: 10,
                      }}
                      title={`BBox: [${Array.isArray(bbox) ? bbox.join(', ') : bbox}]`}
                    >
                      <span style={{ position: 'absolute', top: -20, left: 0, background: '#eab308', color: '#000000', fontSize: 10, fontWeight: 800, padding: '1px 6px', borderRadius: 3, whiteSpace: 'nowrap' }}>
                        Highlighted Chunk
                      </span>
                    </div>
                  </div>
                </div>

                {/* Grounded Text Excerpt */}
                <div style={{ flex: 1, background: 'var(--bg-tertiary)', borderRadius: 8, padding: 12, border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column' }}>
                  <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>
                    Grounded Passage Content:
                  </div>
                  <div style={{ fontSize: 13, lineHeight: 1.6, color: '#cbd5e1', overflowY: 'auto', flex: 1 }}>
                    {citation.text || citation.chunk_text || 'No text snippet available.'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Full PDF Mode */}
          {activeTab === 'pdf' && (
            <div style={{ flex: 1, background: '#ffffff', borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
              <iframe
                src={pageJumpUrl}
                style={{ width: '100%', height: '100%', border: 'none' }}
                title="Full PDF Viewer"
              />
            </div>
          )}

          {/* Text Metadata Mode */}
          {activeTab === 'text' && (
            <div style={{ overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="bbox-meta-grid">
                <div className="bbox-meta-item">
                  <div className="bbox-meta-title">Source Document</div>
                  <div className="bbox-meta-val">{fileName}</div>
                </div>
                <div className="bbox-meta-item">
                  <div className="bbox-meta-title">Page & Type</div>
                  <div className="bbox-meta-val">Page {currentPage} • <span style={{ color: 'var(--accent-cyan)' }}>{type}</span></div>
                </div>
                <div className="bbox-meta-item">
                  <div className="bbox-meta-title">BBox Coordinates</div>
                  <div className="bbox-meta-val">[ {Array.isArray(bbox) ? bbox.map(n => typeof n === 'number' ? n.toFixed(2) : n).join(', ') : bbox} ]</div>
                </div>
                <div className="bbox-meta-item">
                  <div className="bbox-meta-title">Rerank Score</div>
                  <div className="bbox-meta-val" style={{ color: 'var(--accent-emerald)' }}>{score}</div>
                </div>
              </div>

              <div style={{ background: 'var(--bg-tertiary)', padding: 14, borderRadius: 8, border: '1px solid var(--border-subtle)', flex: 1 }}>
                <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 8 }}>Full Extracted Text:</div>
                <div style={{ fontSize: 13.5, lineHeight: 1.65, color: '#e2e8f0' }}>
                  {citation.text || citation.chunk_text || 'No text snippet available.'}
                </div>
              </div>
            </div>
          )}

          {/* Media Player for Transcripts */}
          {isMedia && (
            <div style={{ background: '#0b0f19', borderRadius: 8, padding: 16, textAlign: 'center' }}>
              <video controls src={timeJumpUrl} style={{ width: '100%', maxHeight: 280, borderRadius: 6 }} />
              <div style={{ fontSize: 13, color: 'var(--accent-cyan)', marginTop: 10 }}>
                ⏱ Playing Media from Timestamp: {startTime != null ? `${startTime.toFixed(1)}s` : '0.0s'}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── NotebookLM-Style Interactive Citation Pill & Floating Popover ─────────────
function CitationPill({ num, cite, activeSessionId, onOpenBBox }) {
  const [isOpen, setIsOpen] = useState(false)
  const hideTimerRef = useRef(null)

  if (!cite) return <span>[{num}]</span>

  const fileName = cite.file_name || cite.filename || 'Source Document'
  const page = cite.page || 1
  const isTranscript = cite.type === 'transcript' || cite.start_time != null
  const text = cite.text || cite.chunk_text || 'Referenced grounding passage from enterprise knowledge base.'
  const hasFileUrl = Boolean(cite.file_url || cite.url)
  const fileUrl = cite.file_url || cite.url

  const handleMouseEnter = () => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current)
      hideTimerRef.current = null
    }
    setIsOpen(true)
  }

  const handleMouseLeave = () => {
    hideTimerRef.current = setTimeout(() => {
      setIsOpen(false)
    }, 250)
  }

  useEffect(() => {
    return () => {
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current)
    }
  }, [])

  return (
    <span
      className="citation-pill-container"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <span
        className={`citation-num-badge ${isOpen ? 'active' : ''}`}
        onClick={(e) => {
          e.stopPropagation()
          onOpenBBox && onOpenBBox(cite)
        }}
        title={`Citation [${num}]: ${fileName} (Click to open in document viewer)`}
      >
        {num}
      </span>

      {isOpen && (
        <div
          className="citation-popover-card glass"
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="citation-popover-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
              <span style={{ fontSize: 13 }}>{isTranscript ? '🎬' : '📄'}</span>
              <span className="citation-popover-filename" title={fileName}>{fileName}</span>
            </div>
            <span className="citation-popover-tag">
              {isTranscript ? `⏱ ${cite?.start_time?.toFixed(1) || 0}s` : `p. ${page}`}
            </span>
          </div>

          <div className="citation-popover-body">
            {text}
          </div>

          <div className="citation-popover-footer">
            <button
              className="citation-popover-btn"
              onClick={() => {
                setIsOpen(false)
                onOpenBBox && onOpenBBox(cite)
              }}
            >
              📄 View in Document
            </button>
            {hasFileUrl && (
              <a
                href={`${fileUrl}#page=${page}`}
                target="_blank"
                rel="noreferrer"
                className="citation-popover-link"
              >
                Open source ↗
              </a>
            )}
          </div>
        </div>
      )}
    </span>
  )
}

// ── Code Block with Copy Button and Syntax Highlighting ───────────────────────
function CodeBlock({ inline, className, children, ...props }) {
  const match = /language-(\w+)/.exec(className || '')
  const language = match ? match[1] : ''
  const [copied, setCopied] = useState(false)

  const codeText = String(children).replace(/\n$/, '')

  const handleCopy = (e) => {
    e.stopPropagation()
    navigator.clipboard.writeText(codeText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (!inline && (match || codeText.includes('\n') || className)) {
    return (
      <div className="code-block-wrapper">
        <div className="code-block-header">
          <span className="code-lang-tag">{language || 'code'}</span>
          <button
            type="button"
            onClick={handleCopy}
            className="code-copy-btn"
            title="Copy code to clipboard"
          >
            {copied ? '✓ Copied' : '📋 Copy'}
          </button>
        </div>
        <pre className="code-block-pre">
          <code className={className} {...props}>
            {children}
          </code>
        </pre>
      </div>
    )
  }

  return (
    <code className="inline-code" {...props}>
      {children}
    </code>
  )
}

// ── Recursive Citation Text Processor for React Markdown ─────────────────────
function processCitationText(children, citations, onCitationClick, activeSessionId) {
  if (children === null || children === undefined) return children

  if (typeof children === 'string') {
    const citationRegex = /(\[\d+(?:\s*,\s*\d+)*\]|\[(?:Doc|Media|Page|Citation)[^\]]*\])/g
    const parts = children.split(citationRegex)
    if (parts.length === 1) return children

    return parts.map((part, idx) => {
      if (!part) return null

      // Match [1], [2], [1, 2]
      const numMatch = part.match(/^\[([\d\s,]+)\]$/)
      if (numMatch) {
        const rawNums = numMatch[1].split(',').map(s => s.trim()).filter(s => /^\d+$/.test(s))
        const validPills = rawNums
          .map(n => {
            const citeIndex = parseInt(n, 10)
            const cite = citations?.find(c => c.index === citeIndex) || (citations && citations[citeIndex - 1])
            return { num: n, cite }
          })
          .filter(item => item.cite != null)

        if (validPills.length === 0) {
          return part
        }

        return (
          <span key={`pill-group-${idx}`} className="citation-group-inline">
            {validPills.map(({ num, cite }, nIdx) => (
              <CitationPill
                key={`cite-pill-${idx}-${nIdx}`}
                num={num}
                cite={cite}
                activeSessionId={activeSessionId}
                onOpenBBox={onCitationClick}
              />
            ))}
          </span>
        )
      }

      // Match [Doc: file.pdf | Page: 2]
      if (part.startsWith('[') && part.endsWith(']')) {
        const citeText = part.slice(1, -1).trim()
        const pageMatch = citeText.match(/Page:\s*(\d+)/i) || citeText.match(/p\.\s*(\d+)/i)
        const pageNum = pageMatch ? pageMatch[1] : null
        const fileMatch = citeText.match(/(?:Doc|Media):\s*([^\s\|]+)/i)
        const fileName = fileMatch ? fileMatch[1] : null

        return (
          <span
            key={`pill-doc-${idx}`}
            className="citation-pill citation-pill-inline"
            onClick={() => onCitationClick && onCitationClick({ file_name: fileName || citeText, page: pageNum ? parseInt(pageNum, 10) : 1, text: `Reference: ${citeText}` })}
            title="Click to inspect chunk in document"
          >
            🔍 {citeText}
          </span>
        )
      }

      return part
    })
  }

  if (Array.isArray(children)) {
    return React.Children.map(children, (child) => {
      if (typeof child === 'string') {
        return processCitationText(child, citations, onCitationClick, activeSessionId)
      }
      if (React.isValidElement(child) && child.props && child.props.children) {
        return React.cloneElement(child, {
          ...child.props,
          children: processCitationText(child.props.children, citations, onCitationClick, activeSessionId),
        })
      }
      return child
    })
  }

  if (React.isValidElement(children) && children.props && children.props.children) {
    return React.cloneElement(children, {
      ...children.props,
      children: processCitationText(children.props.children, citations, onCitationClick, activeSessionId),
    })
  }

  return children
}

// ── Markdown Renderer Component with Remark GFM & Rehype Highlight ──────────
function MarkdownRenderer({ content, citations, onCitationClick, activeSessionId }) {
  if (!content) return null

  const customComponents = {
    p: ({ children, ...props }) => (
      <p {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</p>
    ),
    li: ({ children, ...props }) => (
      <li {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</li>
    ),
    h1: ({ children, ...props }) => (
      <h1 {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</h1>
    ),
    h2: ({ children, ...props }) => (
      <h2 {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</h2>
    ),
    h3: ({ children, ...props }) => (
      <h3 {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</h3>
    ),
    h4: ({ children, ...props }) => (
      <h4 {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</h4>
    ),
    blockquote: ({ children, ...props }) => (
      <blockquote {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</blockquote>
    ),
    strong: ({ children, ...props }) => (
      <strong {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</strong>
    ),
    em: ({ children, ...props }) => (
      <em {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</em>
    ),
    td: ({ children, ...props }) => (
      <td {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</td>
    ),
    th: ({ children, ...props }) => (
      <th {...props}>{processCitationText(children, citations, onCitationClick, activeSessionId)}</th>
    ),
    a: ({ href, children, ...props }) => (
      <a href={href} target="_blank" rel="noreferrer" className="markdown-link" {...props}>
        {children} ↗
      </a>
    ),
    code: CodeBlock,
    table: ({ children, ...props }) => (
      <div className="table-responsive-wrapper">
        <table className="markdown-table" {...props}>
          {children}
        </table>
      </div>
    ),
  }

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={customComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

// ── Left Panel: Session Uploaded Files & Upload Dropzone ───────────────────────
function SessionFilesPanel({ token, activeSessionId, files, onFileDeleted, onFileUploadSuccess, onOpenFile }) {
  const [uploadList, setUploadList] = useState([])
  const [dragOver, setDragOver] = useState(false)
  const pollersRef = useRef({})

  const startPolling = useCallback((uploadId, taskId) => {
    const iv = setInterval(async () => {
      try {
        const res = await fetch(`/api/ingest/status/${taskId}`)
        const data = await res.json()

        setUploadList(prev => prev.map(f => f.uploadId !== uploadId ? f : {
          ...f,
          status: data.state === 'SUCCESS' ? 'done' : data.state === 'FAILURE' ? 'error' : 'running',
          stage: data.stage,
          message: data.message,
          pct: 0.5 + (data.pct || 0) * 0.5,
        }))

        if (data.state === 'SUCCESS' || data.state === 'FAILURE') {
          clearInterval(iv)
          delete pollersRef.current[uploadId]
          if (data.state === 'SUCCESS') {
            onFileUploadSuccess && onFileUploadSuccess()
          }
        }
      } catch (err) {
        console.error('Polling status error:', err)
      }
    }, POLL_INTERVAL_MS)
    pollersRef.current[uploadId] = iv
  }, [onFileUploadSuccess])

  const handleFiles = useCallback(async (selectedFileList) => {
    const fileArray = Array.from(selectedFileList)
    if (!fileArray.length) return
    if (!activeSessionId) {
      alert('Please create or select a session first before uploading files.')
      return
    }

    for (const file of fileArray) {
      const uploadId = genId()
      const entry = {
        uploadId,
        name: file.name,
        size: file.size,
        status: 'running',
        stage: 'UPLOADING',
        message: 'Slicing & uploading chunks…',
        pct: 0.1,
      }
      setUploadList(prev => [entry, ...prev])

      try {
        const taskId = await uploadFileInChunks(file, uploadId, activeSessionId, pct => {
          setUploadList(prev => prev.map(f => f.uploadId !== uploadId ? f : {
            ...f,
            pct,
            message: `Uploading: ${Math.round(pct * 100)}%`,
          }))
        }, token)

        setUploadList(prev => prev.map(f => f.uploadId !== uploadId ? f : {
          ...f,
          stage: 'PARSING',
          message: 'Indexing PDF layout & media STT…',
          pct: 0.5,
        }))
        startPolling(uploadId, taskId)
      } catch (err) {
        setUploadList(prev => prev.map(f => f.uploadId !== uploadId ? f : { ...f, status: 'error', message: err.message }))
      }
    }
  }, [activeSessionId, startPolling, token])

  const handleDelete = async (filename) => {
    if (!activeSessionId || !confirm(`Delete file "${filename}" from this session?`)) return
    try {
      const res = await authFetch(`/api/sessions/${activeSessionId}/files/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      })
      if (res.ok) {
        onFileDeleted && onFileDeleted(filename)
      }
    } catch (err) {
      alert(`Failed to delete file: ${err.message}`)
    }
  }

  return (
    <div className="session-files-panel glass">
      <div className="files-panel-header">
        <div className="files-panel-title">
          <span>📁 Session Documents</span>
          <span className="files-panel-count">{(files || []).length} files</span>
        </div>
      </div>

      {/* Files List Scroll Area */}
      <div className="files-list-scroll">
        {(!files || files.length === 0) && (
          <div style={{ textAlign: 'center', padding: '30px 10px', color: 'var(--text-muted)', fontSize: 13 }}>
            <div style={{ fontSize: 28, marginBottom: 6 }}>📄</div>
            <p>No documents uploaded yet for this session.</p>
            <p style={{ fontSize: 11.5, marginTop: 4, opacity: 0.8 }}>Use the dropzone below to add PDF or audio/video files.</p>
          </div>
        )}

        {(files || []).map((file, idx) => {
          const fn = file.filename || file.name || `file_${idx}`
          const isPdf = fn.toLowerCase().endsWith('.pdf')
          const isMedia = fn.toLowerCase().match(/\.(mp4|webm|mp3|wav|ogg|mov|mkv|aac|flac)$/)
          const fileIcon = isPdf ? '📑' : isMedia ? '🎬' : '📄'

          return (
            <div key={fn || idx} className="session-file-item">
              <div className="session-file-info">
                <span className="session-file-icon">{fileIcon}</span>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="session-file-name" title={fn}>{fn}</div>
                  <div className="session-file-meta">
                    {formatBytes(file.size)} • {file.file_type || (isPdf ? 'pdf' : isMedia ? 'media' : 'doc')}
                  </div>
                </div>
              </div>

              <div className="session-file-actions">
                <button
                  className="file-action-btn"
                  onClick={() => onOpenFile && onOpenFile({ ...file, file_name: fn, file_url: file.file_url || file.url, url: file.file_url || file.url, page: 1, session_id: activeSessionId })}
                  title="View / Spotlight file"
                >
                  📍 View
                </button>
                <button
                  className="file-action-btn delete"
                  onClick={() => handleDelete(fn)}
                  title="Delete file"
                >
                  ✕
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {/* Compact Drag & Drop Upload Component */}
      <div
        className={`compact-dropzone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => {
          e.preventDefault()
          setDragOver(false)
          handleFiles(e.dataTransfer.files)
        }}
        onClick={() => document.getElementById('session-file-picker').click()}
      >
        <input
          id="session-file-picker"
          type="file"
          multiple
          accept=".pdf,.mp4,.mp3,.wav,.mov,.mkv,.aac,.flac"
          style={{ display: 'none' }}
          onChange={e => {
            handleFiles(e.target.files)
            e.target.value = ''
          }}
        />
        <div className="compact-dropzone-icon">⚡</div>
        <div className="compact-dropzone-text">+ Upload to Session</div>
        <div className="compact-dropzone-sub">Drop PDF, MP4, MP3 to index</div>
      </div>

      {/* Live Upload & Indexing Progress */}
      {uploadList.length > 0 && (
        <div className="upload-progress-list">
          {uploadList.slice(0, 3).map(item => (
            <div key={item.uploadId} className="upload-progress-item">
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#e2e8f0', fontSize: 11 }}>
                <span style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.name}</span>
                <span style={{ color: item.status === 'done' ? 'var(--accent-emerald)' : item.status === 'error' ? 'var(--accent-rose)' : 'var(--accent-cyan)' }}>
                  {item.status === 'done' ? '✓ Ready' : item.status === 'error' ? '✗ Error' : item.stage}
                </span>
              </div>
              <div className="upload-progress-bar-bg">
                <div className="upload-progress-bar-fill" style={{ width: `${Math.round((item.pct || 0.1) * 100)}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Agentic Chat Workspace (Right Panel + Split Layout) ───────────────────────
function ChatWorkspace({
  token,
  activeSessionId,
  activeSession,
  sessions,
  onSessionTitleUpdated,
  onSessionChange,
  onCreateSession,
  onEndSession,
  onViewMemories,
  onHiTLRequired,
  resumeStream,
}) {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [streaming, setStreaming] = useState(false)
  const [isTitleStreaming, setIsTitleStreaming] = useState(false)
  const [selectedCitation, setSelectedCitation] = useState(null)
  const [sessionFiles, setSessionFiles] = useState([])
  const abortRef = useRef(null)

  useEffect(() => {
    if (!resumeStream) return
    let active = true
    const readResumeStream = async () => {
      try {
        setStreaming(true)
        const reader = resumeStream.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        while (active) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })
          const parts = buf.split('\n\n')
          buf = parts.pop()
          for (const part of parts) {
            if (!part.startsWith('data:')) continue
            try {
              const msg = JSON.parse(part.slice(5).trim())
              if (msg.type === 'token') {
                setMessages(prev => {
                  const lastIdx = prev.length - 1
                  if (lastIdx < 0) return prev
                  return prev.map((m, i) => i !== lastIdx ? m : {
                    ...m,
                    content: m.content + msg.content,
                    isReasoning: false,
                  })
                })
              } else if (msg.type === 'error') {
                setMessages(prev => {
                  const lastIdx = prev.length - 1
                  if (lastIdx < 0) return prev
                  return prev.map((m, i) => i !== lastIdx ? m : {
                    ...m,
                    content: m.content + `\n\n**Error:** ${msg.content}`,
                    isReasoning: false,
                  })
                })
              }
            } catch { }
          }
        }
      } catch (err) {
        console.error('Error reading resume stream:', err)
      } finally {
        setStreaming(false)
      }
    }
    readResumeStream()
    return () => { active = false }
  }, [resumeStream])

  const loadSessionDetails = useCallback(async () => {
    if (!activeSessionId) {
      setMessages([])
      setSessionFiles([])
      return
    }
    try {
      const res = await authFetch(`/api/sessions/${activeSessionId}`)
      if (res.ok) {
        const data = await res.json()
        if (data.history) {
          setMessages(data.history.map((m, idx) => ({
            id: m.id || `hist_${idx}_${m.role}`,
            role: m.role,
            content: m.content,
            thoughts: m.thoughts || [],
            steps: m.steps || [],
            citations: m.citations || [],
          })))
        } else {
          setMessages([])
        }
        if (data.files) {
          setSessionFiles(data.files)
        } else {
          setSessionFiles([])
        }
      }
    } catch (err) {
      console.error('Failed to load session details:', err)
    }
  }, [activeSessionId])

  useEffect(() => {
    loadSessionDetails()
  }, [loadSessionDetails])

  // Stream Session Title concurrently using Summarizer model
  const triggerTitleStreaming = useCallback(async (promptText) => {
    if (!activeSessionId) return
    setIsTitleStreaming(true)

    try {
      const res = await authFetch(`/api/sessions/${activeSessionId}/generate-title`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: promptText }),
      })

      if (!res.ok) return

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      let accumulatedTitle = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })

        const parts = buf.split('\n\n')
        buf = parts.pop()

        for (const part of parts) {
          if (!part.startsWith('data:')) continue
          try {
            const data = JSON.parse(part.slice(5).trim())
            if (data.type === 'title_token') {
              accumulatedTitle += data.content
              onSessionTitleUpdated && onSessionTitleUpdated(activeSessionId, accumulatedTitle)
            } else if (data.type === 'title_done') {
              onSessionTitleUpdated && onSessionTitleUpdated(activeSessionId, data.title)
            }
          } catch { }
        }
      }
    } catch (err) {
      console.error('Title generation stream error:', err)
    } finally {
      setIsTitleStreaming(false)
    }
  }, [activeSessionId, onSessionTitleUpdated, token])

  const handleAsk = useCallback(async () => {
    if (!query.trim() || streaming) return
    const userQuery = query.trim()
    setQuery('')
    setStreaming(true)

    if (!activeSessionId) {
      alert('Please create or select an active session before sending messages.')
      return
    }

    const isFirstPrompt = messages.length === 0 || (activeSession && (activeSession.title === 'New RAG Session' || activeSession.title.startsWith('Session ')))

    // If first prompt, trigger concurrent title streaming
    if (isFirstPrompt) {
      triggerTitleStreaming(userQuery)
    }

    const assistantId = genId()
    setMessages(prev => [
      ...prev,
      { role: 'user', content: userQuery },
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        thoughts: [],
        steps: [],
        citations: [],
        isReasoning: true,
        reasoningCollapsed: false,
      }
    ])

    try {
      const ctrl = new AbortController()
      abortRef.current = ctrl

      const endpoint = `/api/sessions/${activeSessionId}/ask`
      const headers = { 'Content-Type': 'application/json' }

      const res = await authFetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({ query: userQuery }),
        signal: ctrl.signal,
      })

      if (!res.ok) throw new Error(`Server error (${res.status}): ${res.statusText}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })

        const parts = buf.split('\n\n')
        buf = parts.pop()

        for (const part of parts) {
          if (!part.startsWith('data:')) continue
          try {
            const msg = JSON.parse(part.slice(5).trim())

            if (msg.type === 'token') {
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : {
                ...m,
                content: m.content + msg.content,
                isReasoning: false,
              }))
            } else if (msg.type === 'step') {
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : {
                ...m,
                steps: [...m.steps, `Executing Tool: ${msg.name} (input: ${msg.input})`],
              }))
            } else if (msg.type === 'step_result') {
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : {
                ...m,
                steps: [...m.steps, `Result: ${msg.output}`],
              }))
            } else if (msg.type === 'thought') {
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : {
                ...m,
                thoughts: [...m.thoughts, msg.content],
              }))
            } else if (msg.type === 'sources') {
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : {
                ...m,
                citations: msg.citations || [],
              }))
            } else if (msg.type === 'approval_required') {
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : {
                ...m,
                isReasoning: false,
              }))
              onHiTLRequired && onHiTLRequired(msg)
            } else if (msg.type === 'error') {
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : {
                ...m,
                content: m.content + `\n\n**Error:** ${msg.content}`,
                isReasoning: false,
              }))
            }
          } catch { }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages(prev => prev.map(m => m.id !== assistantId ? m : {
          ...m,
          content: m.content + `\n\n**Error:** ${err.message}`,
          isReasoning: false,
        }))
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }, [activeSession, activeSessionId, messages.length, query, streaming, token, triggerTitleStreaming])

  const toggleReasoning = (msgId) => {
    setMessages(prev => prev.map(m => m.id !== msgId ? m : {
      ...m,
      reasoningCollapsed: !m.reasoningCollapsed,
    }))
  }

  const currentTitle = activeSession?.title || 'Active Session'

  return (
    <div className="split-workspace">
      {/* Left Panel: Uploaded Files & Upload Dropzone */}
      <SessionFilesPanel
        token={token}
        activeSessionId={activeSessionId}
        files={sessionFiles}
        onFileDeleted={() => loadSessionDetails()}
        onFileUploadSuccess={() => loadSessionDetails()}
        onOpenFile={(cite) => setSelectedCitation({ ...cite, session_id: activeSessionId })}
      />

      {/* Right Panel: Interactive Agent Chat */}
      <div className="chat-panel-container glass">
        {/* Session Header */}
        <div className="chat-panel-header">
          <div className="session-title-wrap">
            <span style={{ fontSize: 18 }}>💬</span>
            <div className="session-title-text" title={currentTitle}>{currentTitle}</div>
            {isTitleStreaming && (
              <span className="title-streaming-badge">
                <span className="pulse-spinner" style={{ width: 10, height: 10 }} />
                <span>Summarizing Title…</span>
              </span>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button className="btn btn-ghost" style={{ padding: '5px 12px', fontSize: 12.5 }} onClick={onCreateSession}>
              + New Session
            </button>
            <button className="btn btn-ghost" style={{ padding: '5px 12px', fontSize: 12.5 }} onClick={onViewMemories}>
              🧠 Memories
            </button>
            <button className="btn btn-ghost" style={{ padding: '5px 12px', fontSize: 12.5, color: 'var(--accent-rose)' }} onClick={onEndSession}>
              ⏹ End
            </button>
          </div>
        </div>

        {/* Chat Message Scroll Window */}
        <div className="chat-panel-body">
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: 42, marginBottom: 12 }}>🤖</div>
              <h4 style={{ color: '#ffffff', fontSize: '1.25rem', marginBottom: 8 }}>Enterprise Agent Grounded in Your Session Documents</h4>
              <p style={{ fontSize: 14, maxWidth: 500, margin: '0 auto', lineHeight: 1.6 }}>
                Ask any question about your uploaded documents and audio/video media. Real-time reasoning traces, inline citations, and bounding box spotlights will stream live.
              </p>
            </div>
          )}

          {messages.map((m, idx) => (
            <div key={m.id || idx} className={`message-card ${m.role}`}>
              <div className="message-header">
                <span className="message-role">{m.role === 'user' ? '👤 You' : '🤖 Enterprise GraphRAG Agent'}</span>
              </div>

              {/* Faded, Collapsible Thought Section */}
              {m.role === 'assistant' && (m.steps?.length > 0 || m.thoughts?.length > 0 || m.isReasoning) && (
                <div className="reasoning-accordion">
                  <div className="reasoning-header" onClick={() => toggleReasoning(m.id)}>
                    <div className="reasoning-title">
                      {m.isReasoning ? <span className="pulse-spinner" /> : <span>🧠</span>}
                      <span>{m.isReasoning ? 'Agent is searching knowledge & reasoning…' : `Reasoning Trace (${(m.steps?.length || 0) + (m.thoughts?.length || 0)} steps)`}</span>
                    </div>
                    <span style={{ fontSize: 11 }}>{m.reasoningCollapsed ? '▶ Expand' : '▼ Collapse'}</span>
                  </div>

                  {!m.reasoningCollapsed && (
                    <div className="reasoning-content">
                      {m.thoughts?.map((th, i) => (
                        <div key={`th-${i}`} style={{ color: '#94a3b8' }}>• Thought: {th}</div>
                      ))}
                      {m.steps?.map((st, i) => (
                        <div key={`st-${i}`} className="step-pill">⚙ {st}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Real Answer Text Rendered with Inline Citations and Markdown */}
              {m.content && (
                <div className="answer-body">
                  <MarkdownRenderer
                    content={m.content}
                    citations={m.citations}
                    activeSessionId={activeSessionId}
                    onCitationClick={(citeItem) => {
                      if (typeof citeItem === 'object' && citeItem !== null) {
                        setSelectedCitation({ ...citeItem, session_id: activeSessionId })
                      }
                    }}
                  />
                </div>
              )}

              {/* Source Document Citations Section at Bottom of Message */}
              {m.citations && m.citations.length > 0 && (
                <div className="citations-section">
                  <div className="citations-label">Source Document Citations (Click to view in document):</div>
                  <div className="citations-grid">
                    {m.citations.map((c, i) => {
                      const isTrans = c.type === 'transcript' || c.start_time != null
                      return (
                        <div
                          key={c.custom_id || i}
                          className="citation-pill citation-card-bottom"
                          onClick={() => setSelectedCitation({ ...c, session_id: activeSessionId })}
                          title={`Click to view in document (Page ${c.page || 1})`}
                        >
                          <span className="citation-badge">[{c.index || (i + 1)}]</span>
                          <span className="citation-doc-name">{isTrans ? '🎬' : '📄'} {c.file_name}</span>
                          <span style={{ color: 'var(--accent-cyan)' }}>
                            {isTrans ? `⏱ ${c.start_time?.toFixed(1)}s` : `p. ${c.page || 1}`}
                          </span>
                          {c.score != null && <span style={{ color: 'var(--accent-emerald)', fontSize: 11 }}>{(c.score).toFixed(2)}</span>}
                          <span className="citation-view-action">📄 View in Document</span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Chat Input Bar */}
        <div className="chat-input-bar">
          <textarea
            className="chat-input"
            rows={2}
            placeholder={activeSessionId ? "Ask anything about this session's uploaded documents and media transcripts…" : "Type your question..."}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleAsk()
              }
            }}
            disabled={streaming}
          />
          <button
            className="btn btn-primary"
            onClick={handleAsk}
            disabled={streaming || !query.trim()}
          >
            {streaming ? '⏹ Stop' : 'Ask Agent'}
          </button>
        </div>
      </div>

      <BBoxModal
        isOpen={!!selectedCitation}
        citation={selectedCitation}
        activeSessionId={activeSessionId}
        onClose={() => setSelectedCitation(null)}
      />
    </div>
  )
}

// ── Sessions History Tab View ─────────────────────────────────────────────────
function SessionsHistoryView({ sessions, activeSessionId, onSelectSession, onCreateSession, onDeleteSession }) {
  return (
    <div className="sessions-history-container">
      <div className="sessions-history-header">
        <div>
          <h2 className="sessions-history-title">🗂 Conversation Sessions History</h2>
          <p className="sessions-history-desc">
            All your conversations stored in MongoDB. Click any session to continue the chat with its isolated documents and memory.
          </p>
        </div>
        <button className="btn btn-gradient" onClick={onCreateSession}>
          + Create New Session
        </button>
      </div>

      {sessions.length === 0 ? (
        <div className="glass" style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: 36, marginBottom: 12 }}>📂</div>
          <h3 style={{ color: '#ffffff', marginBottom: 6 }}>No Sessions Found</h3>
          <p style={{ marginBottom: 16 }}>Start your first conversation session to chat with the agent and upload documents.</p>
          <button className="btn btn-gradient" onClick={onCreateSession}>Create Session</button>
        </div>
      ) : (
        <div className="sessions-grid">
          {sessions.map(s => {
            const isActive = s.session_id === activeSessionId
            const createdStr = s.created_at ? new Date(s.created_at).toLocaleString() : 'Recent'
            const fileCount = s.files?.length || 0

            return (
              <div
                key={s.session_id}
                className={`session-card glass ${isActive ? 'active' : ''}`}
                onClick={() => onSelectSession(s.session_id)}
              >
                <div>
                  <div className="session-card-header">
                    <div className="session-card-title">{s.title || 'Conversation Session'}</div>
                    <span className={`status-pill ${s.status === 'active' ? 'success' : ''}`}>
                      {s.status}
                    </span>
                  </div>
                  <div className="session-card-date">Created: {createdStr}</div>
                </div>

                <div className="session-card-files">
                  <span>📑</span>
                  <span>{fileCount} {fileCount === 1 ? 'file' : 'files'} attached</span>
                </div>

                <div className="session-card-actions">
                  <button
                    className="btn btn-gradient"
                    style={{ padding: '6px 14px', fontSize: 12.5 }}
                    onClick={(e) => {
                      e.stopPropagation()
                      onSelectSession(s.session_id)
                    }}
                  >
                    Continue Chat →
                  </button>
                  <button
                    className="file-action-btn delete"
                    style={{ padding: '6px 10px' }}
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteSession && onDeleteSession(s.session_id)
                    }}
                    title="Delete Session"
                  >
                    🗑 Delete
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Ingestion Studio Component ─────────────────────────────────────────────────
function IngestionStudio({ token, sessions, activeSessionId, onSessionChange }) {
  const [files, setFiles] = useState([])
  const [dragOver, setDragOver] = useState(false)
  const [logs, setLogs] = useState([])
  const pollersRef = useRef({})

  const addLog = useCallback((msg, type = 'info') => {
    setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), msg, type }])
  }, [])

  const startPolling = useCallback((uploadId, taskId) => {
    const iv = setInterval(async () => {
      try {
        const res = await fetch(`/api/ingest/status/${taskId}`)
        const data = await res.json()

        setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : {
          ...f,
          status: data.state === 'SUCCESS' ? 'done' : data.state === 'FAILURE' ? 'error' : 'running',
          stage: data.stage,
          message: data.message,
          pct: 0.5 + (data.pct || 0) * 0.5,
        }))

        addLog(`[${data.stage || 'PROGRESS'}] ${data.message}`, data.state === 'SUCCESS' ? 'success' : 'info')

        if (data.state === 'SUCCESS' || data.state === 'FAILURE') {
          clearInterval(iv)
          delete pollersRef.current[uploadId]
        }
      } catch (err) {
        addLog(`Polling status error: ${err.message}`, 'error')
      }
    }, POLL_INTERVAL_MS)
    pollersRef.current[uploadId] = iv
  }, [addLog])

  const handleFiles = useCallback(async (selectedFileList) => {
    const fileArray = Array.from(selectedFileList)
    if (!fileArray.length) return
    if (!activeSessionId) {
      alert('Please select or create a session first.')
      return
    }

    addLog(`Initiated multi-file upload (${fileArray.length} files) for session: ${activeSessionId}`, 'info')

    for (const file of fileArray) {
      const uploadId = genId()
      const entry = {
        uploadId,
        name: file.name,
        size: file.size,
        status: 'running',
        stage: 'UPLOADING',
        message: 'Slicing & uploading chunks…',
        pct: 0.1,
      }
      setFiles(prev => [entry, ...prev])

      try {
        const taskId = await uploadFileInChunks(file, uploadId, activeSessionId, pct => {
          setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : {
            ...f,
            pct,
            message: `Uploading chunk slices: ${Math.round(pct * 100)}%`,
          }))
        }, token)

        addLog(`Upload complete for ${file.name}. Ingestion task dispatched: ${taskId}`, 'success')
        setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : {
          ...f,
          stage: 'PARSING',
          message: 'PDF layout parsing & Rev AI media transcription…',
          pct: 0.5,
        }))
        startPolling(uploadId, taskId)
      } catch (err) {
        addLog(`Failed to upload ${file.name}: ${err.message}`, 'error')
        setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : { ...f, status: 'error', message: err.message }))
      }
    }
  }, [activeSessionId, addLog, startPolling, token])

  return (
    <div className="workspace-container">
      {/* Session destination selector */}
      {sessions && sessions.length > 0 && (
        <div className="session-bar glass" style={{ marginBottom: 4 }}>
          <div className="session-selector">
            <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)' }}>DESTINATION SESSION:</span>
            <select
              className="session-select"
              value={activeSessionId}
              onChange={e => onSessionChange(e.target.value)}
            >
              {sessions.map(s => (
                <option key={s.session_id} value={s.session_id}>
                  {s.title} ({s.status})
                </option>
              ))}
            </select>
          </div>
          <span style={{ fontSize: 12, color: 'var(--accent-cyan)' }}>
            ⚡ Files will be placed into the session folder and isolated in Weaviate.
          </span>
        </div>
      )}

      <div
        className={`upload-dropzone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => {
          e.preventDefault()
          setDragOver(false)
          handleFiles(e.dataTransfer.files)
        }}
        onClick={() => document.getElementById('studio-file-picker').click()}
      >
        <input
          id="studio-file-picker"
          type="file"
          multiple
          accept=".pdf,.mp4,.mp3,.wav,.mov,.mkv,.aac,.flac"
          style={{ display: 'none' }}
          onChange={e => {
            handleFiles(e.target.files)
            e.target.value = ''
          }}
        />
        <div className="upload-icon">📁</div>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.4rem', marginBottom: 6 }}>
          Drop Multiple PDF Documents & Media Audio/Video Files
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>
          Upload multiple files simultaneously. PDFs are parsed with table clusters and bounding boxes, while video/audio files are transcribed and indexed into Triton & Weaviate.
        </p>
      </div>

      {files.length > 0 && (
        <div className="file-cards-list">
          {files.map(item => (
            <div key={item.uploadId} className="file-card-item glass">
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <span style={{ fontSize: 24 }}>{item.name.endsWith('.pdf') ? '📄' : '🎬'}</span>
                <div>
                  <div style={{ fontWeight: 600 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    {formatBytes(item.size)} • {item.message}
                  </div>
                </div>
              </div>
              <span className={`status-pill ${item.status === 'done' ? 'success' : ''}`}>
                {item.status === 'done' ? '✓ Indexed' : item.status === 'error' ? '✗ Failed' : `${item.stage || 'Processing'}`}
              </span>
            </div>
          ))}
        </div>
      )}

      {logs.length > 0 && (
        <div style={{ background: '#0d1117', padding: 18, borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', fontFamily: 'var(--font-mono)', fontSize: 12, maxHeight: 220, overflowY: 'auto' }}>
          {logs.map((l, i) => (
            <div key={i} style={{ color: l.type === 'error' ? 'var(--accent-rose)' : l.type === 'success' ? 'var(--accent-emerald)' : 'var(--text-secondary)', marginBottom: 4 }}>
              <span style={{ color: 'var(--text-muted)' }}>[{l.time}]</span> {l.msg}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Auth Modal ─────────────────────────────────────────────────────────────────
function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  if (!isOpen) return null

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      if (mode === 'register') {
        const res = await fetch('/api/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, email, password }),
        })
        const text = await res.text()
        let data = {}
        try { data = JSON.parse(text) } catch { data = { detail: text } }
        if (!res.ok) throw new Error(data.detail || 'Registration failed')
        setMode('login')
      }

      const form = new URLSearchParams()
      form.append('username', username)
      form.append('password', password)

      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: form,
      })
      const text = await res.text()
      let data = {}
      try { data = JSON.parse(text) } catch { data = { detail: text } }
      if (!res.ok) throw new Error(data.detail || 'Authentication failed')

      localStorage.setItem('graphrag_access_token', data.access_token)
      localStorage.setItem('graphrag_refresh_token', data.refresh_token)
      localStorage.setItem('graphrag_username', data.username)

      onAuthSuccess(data.access_token, data.refresh_token, data.username)
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog glass" style={{ maxWidth: 440 }} onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">{mode === 'login' ? '🔐 Enterprise Sign In' : '✨ Create Enterprise Account'}</div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        <form onSubmit={handleSubmit} className="modal-body">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div>
              <label style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 4, display: 'block' }}>Username</label>
              <input
                className="chat-input"
                style={{ width: '100%' }}
                type="text"
                required
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder="e.g. enterprise_admin"
              />
            </div>

            {mode === 'register' && (
              <div>
                <label style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 4, display: 'block' }}>Email</label>
                <input
                  className="chat-input"
                  style={{ width: '100%' }}
                  type="email"
                  required
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="name@enterprise.com"
                />
              </div>
            )}

            <div>
              <label style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 4, display: 'block' }}>Password</label>
              <input
                className="chat-input"
                style={{ width: '100%' }}
                type="password"
                required
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
              />
            </div>

            {error && (
              <div style={{ color: 'var(--accent-rose)', fontSize: 13, background: 'rgba(244, 63, 94, 0.1)', padding: '8px 12px', borderRadius: 6 }}>
                ⚠ {error}
              </div>
            )}

            <button className="btn btn-gradient" type="submit" disabled={loading} style={{ marginTop: 8 }}>
              {loading ? 'Authenticating…' : mode === 'login' ? 'Sign In' : 'Register Account'}
            </button>

            <div style={{ fontSize: 13, textAlign: 'center', color: 'var(--text-muted)', marginTop: 6 }}>
              {mode === 'login' ? (
                <span>Need an account? <a href="#register" onClick={() => setMode('register')}>Register now</a></span>
              ) : (
                <span>Already registered? <a href="#login" onClick={() => setMode('login')}>Sign in</a></span>
              )}
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}

// ── Main App Root ─────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView] = useState(() => {
    const savedToken = localStorage.getItem('graphrag_access_token')
    return savedToken ? 'workspace' : 'auth'
  })
  const [token, setToken] = useState(() => localStorage.getItem('graphrag_access_token') || '')
  const [refreshToken, setRefreshToken] = useState(() => localStorage.getItem('graphrag_refresh_token') || '')
  const [username, setUsername] = useState(() => localStorage.getItem('graphrag_username') || '')
  const [memoryModalOpen, setMemoryModalOpen] = useState(false)
  const [memoryData, setMemoryData] = useState(null)
  // HiTL state: holds the pending approval_required event payload
  const [hitlPending, setHitlPending] = useState(null)
  const [resumeStream, setResumeStream] = useState(null)

  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState(() => localStorage.getItem('graphrag_active_session') || '')

  const selectSession = useCallback((sid) => {
    setActiveSessionId(sid || '')
    if (sid) localStorage.setItem('graphrag_active_session', sid)
    else localStorage.removeItem('graphrag_active_session')
  }, [])

  useEffect(() => {
    const onRefreshed = () => {
      setToken(localStorage.getItem('graphrag_access_token') || '')
      setRefreshToken(localStorage.getItem('graphrag_refresh_token') || '')
      setUsername(localStorage.getItem('graphrag_username') || '')
    }
    const onLogout = () => {
      setToken('')
      setRefreshToken('')
      setUsername('')
      setView('auth')
    }
    window.addEventListener('auth_token_refreshed', onRefreshed)
    window.addEventListener('auth_logout', onLogout)
    return () => {
      window.removeEventListener('auth_token_refreshed', onRefreshed)
      window.removeEventListener('auth_logout', onLogout)
    }
  }, [])

  const loadSessions = useCallback(async () => {
    try {
      const res = await authFetch('/api/sessions')
      if (res.ok) {
        const data = await res.json()
        setSessions(data)
        if (data.length > 0) {
          setActiveSessionId(prev => {
            const exists = data.some(s => (s.id || s.session_id) === prev)
            const chosen = (exists && prev) ? prev : (data[0].id || data[0].session_id)
            localStorage.setItem('graphrag_active_session', chosen)
            return chosen
          })
        } else {
          // Auto-create initial session for new user
          const createRes = await authFetch('/api/sessions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: 'New RAG Session' }),
          })
          if (createRes.ok) {
            const newSess = await createRes.json()
            const sid = newSess.id || newSess.session_id
            setSessions([newSess])
            selectSession(sid)
          }
        }
      }
    } catch (err) {
      console.error('Error fetching sessions:', err)
    }
  }, [selectSession])

  useEffect(() => {
    if (token) loadSessions()
  }, [token, loadSessions])

  const handleCreateSession = async () => {
    try {
      const res = await authFetch('/api/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title: `New RAG Session` }),
      })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || 'Failed to create session')
      }
      const newSess = await res.json()
      setSessions(prev => [newSess, ...prev])
      selectSession(newSess.id || newSess.session_id)
      setView('workspace')
    } catch (err) {
      alert(err.message)
    }
  }

  const handleDeleteSession = async (sessionId) => {
    if (!confirm('Are you sure you want to delete this session? All files and chat history will be removed.')) return
    try {
      const res = await authFetch(`/api/sessions/${sessionId}`, {
        method: 'DELETE',
      })
      if (res.ok) {
        setSessions(prev => prev.filter(s => (s.id || s.session_id) !== sessionId))
        if (activeSessionId === sessionId) {
          const remaining = sessions.filter(s => (s.id || s.session_id) !== sessionId)
          if (remaining.length > 0) {
            selectSession(remaining[0].id || remaining[0].session_id)
          } else {
            selectSession('')
          }
        }
      }
    } catch (err) {
      alert(`Failed to delete session: ${err.message}`)
    }
  }

  const handleEndSession = async () => {
    if (!activeSessionId) return
    try {
      const res = await authFetch(`/api/sessions/${activeSessionId}/end`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error('Failed to end session')
      const data = await res.json()
      alert(`Session closed: ${data.message}`)
      setSessions(prev => prev.map(s => (s.id || s.session_id) !== activeSessionId ? s : { ...s, status: 'ending' }))
    } catch (err) {
      alert(err.message)
    }
  }

  const handleViewMemories = async () => {
    if (!activeSessionId) return
    try {
      const res = await authFetch(`/api/sessions/${activeSessionId}/memories`)
      if (!res.ok) throw new Error('Failed to load memories')
      const data = await res.json()
      setMemoryData(data)
      setMemoryModalOpen(true)
    } catch (err) {
      alert(`Failed to load session memory graphs: ${err.message}`)
    }
  }

  const handleSessionTitleUpdated = (sessionId, newTitle) => {
    setSessions(prev => prev.map(s => (s.id || s.session_id) !== sessionId ? s : { ...s, title: newTitle }))
  }

  const handleLogout = async () => {
    try {
      if (refreshToken) {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ refresh_token: refreshToken }),
        })
      }
    } catch { }

    localStorage.removeItem('graphrag_access_token')
    localStorage.removeItem('graphrag_refresh_token')
    localStorage.removeItem('graphrag_username')
    localStorage.removeItem('graphrag_active_session')
    setToken('')
    setRefreshToken('')
    setUsername('')
    setSessions([])
    selectSession('')
  }

  const activeSessionObj = sessions.find(s => (s.id || s.session_id) === activeSessionId) || { session_id: activeSessionId, id: activeSessionId, title: 'Active Session' }

  // HiTL: handle approval resolved — resume streaming
  const handleHiTLResolved = (resumeResponse) => {
    setHitlPending(null)
    setResumeStream(resumeResponse)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand" onClick={() => setView('workspace')}>
          <div className="brand-icon">⚡</div>
          <div>
            <div className="brand-name">Enterprise RAG</div>
          </div>
          <span className="brand-tag">v2.0</span>
        </div>

        <nav className="nav-tabs">
          <button
            className={`nav-btn ${view === 'workspace' ? 'active' : ''}`}
            onClick={() => setView('workspace')}
          >
            💬 Agent Workspace
          </button>
          <button
            className={`nav-btn ${view === 'history' ? 'active' : ''}`}
            onClick={() => setView('history')}
          >
            🗂 Sessions History
          </button>
          <button
            className={`nav-btn ${view === 'ingest' ? 'active' : ''}`}
            onClick={() => setView('ingest')}
          >
            📥 Ingestion Studio
          </button>
          <button
            className={`nav-btn ${view === 'landing' ? 'active' : ''}`}
            onClick={() => setView('landing')}
          >
            🚀 Overview
          </button>
        </nav>

        <div className="header-actions">
          <div className="status-pill">
            <span className="status-dot" />
            <span>Triton & Weaviate Ready</span>
          </div>

          {token ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <GmailStatusBar token={token} />
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>👤 {username}</span>
              <button className="btn btn-ghost" style={{ padding: '6px 12px', fontSize: 12.5 }} onClick={handleLogout}>
                Sign Out
              </button>
            </div>
          ) : (
            <button className="btn btn-gradient" style={{ padding: '7px 16px', fontSize: 13 }} onClick={() => setView('auth')}>
              Sign In / Register
            </button>
          )}
        </div>
      </header>

      <main style={{ flex: 1 }}>
        {view === 'auth' && (
          <AuthPage
            onAuthSuccess={({ token: acc, username: uname }) => {
              setToken(acc)
              setUsername(localStorage.getItem('graphrag_username') || uname)
              setView('workspace')
              loadSessions()
            }}
          />
        )}

        {view === 'landing' && (
          <LandingPage onExploreWorkspace={() => setView('workspace')} />
        )}

        {view === 'workspace' && (
          <>
            {hitlPending && (
              <HiTLBanner
                sessionId={activeSessionId}
                actionRequests={hitlPending.action_requests}
                reviewConfigs={hitlPending.review_configs}
                token={token}
                onResolved={handleHiTLResolved}
              />
            )}
            <ChatWorkspace
              token={token}
              activeSessionId={activeSessionId}
              activeSession={activeSessionObj}
              sessions={sessions}
              onSessionTitleUpdated={handleSessionTitleUpdated}
              onSessionChange={selectSession}
              onCreateSession={handleCreateSession}
              onEndSession={handleEndSession}
              onViewMemories={handleViewMemories}
              onHiTLRequired={setHitlPending}
              resumeStream={resumeStream}
            />
          </>
        )}

        {view === 'history' && (
          <SessionsPage
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={(sessId) => {
              setActiveSessionId(sessId)
              setView('workspace')
            }}
            onCreateSession={handleCreateSession}
            onDeleteSession={handleDeleteSession}
          />
        )}

        {view === 'ingest' && (
          <IngestionStudio
            token={token}
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSessionChange={setActiveSessionId}
          />
        )}
      </main>

      <MemoriesModal
        isOpen={memoryModalOpen}
        memoryData={memoryData}
        onClose={() => setMemoryModalOpen(false)}
      />
    </div>
  )
}
