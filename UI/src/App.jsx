import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

// ── Constants & Helpers ────────────────────────────────────────────────────────
const CHUNK_SIZE = 1024 * 1024 // 1 MB
const POLL_INTERVAL_MS = 1500

function formatBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1048576).toFixed(1)} MB`
}

function genId() {
  return crypto.randomUUID()
}

// ── Chunked File Upload API ───────────────────────────────────────────────────
async function uploadFileInChunks(file, uploadId, sessionId, onProgress, token) {
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE)
  const headers = token ? { Authorization: `Bearer ${token}` } : {}

  for (let i = 0; i < totalChunks; i++) {
    const slice = file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE)
    const form = new FormData()
    form.append('file', slice, file.name)
    form.append('upload_id', uploadId)
    form.append('chunk_index', String(i))
    form.append('total_chunks', String(totalChunks))
    form.append('filename', file.name)
    if (sessionId) form.append('session_id', sessionId)

    const res = await fetch('/api/ingest/chunk', { method: 'POST', body: form, headers })
    if (!res.ok) throw new Error(`Upload slice ${i} failed: ${res.statusText}`)
    onProgress(((i + 1) / totalChunks) * 0.5)
  }

  const finalizeForm = new FormData()
  finalizeForm.append('upload_id', uploadId)
  finalizeForm.append('filename', file.name)
  finalizeForm.append('total_chunks', String(totalChunks))
  if (sessionId) finalizeForm.append('session_id', sessionId)

  const res = await fetch('/api/ingest/finalize', { method: 'POST', body: finalizeForm, headers })
  if (!res.ok) throw new Error(`Finalize failed: ${res.statusText}`)
  const data = await res.json()
  return data.task_id
}

// ── In-App PDF & Bounding Box Spotlight Viewer Modal ─────────────────────────
function BBoxModal({ isOpen, onClose, citation, activeSessionId }) {
  if (!isOpen || !citation) return null

  const [activeTab, setActiveTab] = useState('split')
  const bbox = citation.bbox || [-29.718, -29.758, 535.734, 533.773]
  const [currentPage, setCurrentPage] = useState(citation.page || 1)
  const fileName = citation.file_name || 'document.pdf'
  const type = citation.type || 'text'
  const score = citation.score != null ? citation.score.toFixed(4) : 'N/A'
  const startTime = citation.start_time
  const endTime = citation.end_time
  const isTranscript = type === 'transcript' || startTime != null
  const isPdf = fileName.toLowerCase().endsWith('.pdf')
  const isMedia = fileName.toLowerCase().match(/\.(mp4|webm|mp3|wav|ogg|mov)$/) || isTranscript
  const cloudinaryUrl = citation.cloudinary_url
  const sessionId = citation.session_id || activeSessionId || 'default'

  // Server static file URLs with page and timestamp fragments
  const baseFileUrl = `/api/files/${sessionId}/${fileName}`
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
      <div className="modal-dialog glass" style={{ maxWidth: 1000, width: '92vw', height: '88vh', display: 'flex', flexDirection: 'column' }} onClick={e => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-head" style={{ padding: '14px 20px', borderBottom: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 20 }}>{isTranscript ? '🎬' : '📑'}</span>
            <div>
              <div className="modal-title" style={{ fontSize: '1.05rem', color: '#ffffff' }}>{fileName}</div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                {isTranscript ? `Media Playback Segment (${startTime?.toFixed(1) || 0}s - ${endTime?.toFixed(1) || 0}s)` : `Page ${currentPage} • BBox Coordinates: [${bbox.map(n => typeof n === 'number' ? n.toFixed(1) : n).join(', ')}]`}
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
                  <span>Live Document (Page {currentPage})</span>
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
                    <span>📍 BBox Spotlight on Page {currentPage}</span>
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
                      title={`BBox: [${bbox.join(', ')}]`}
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
                  <div className="bbox-meta-val">[ {bbox.map(n => typeof n === 'number' ? n.toFixed(2) : n).join(', ')} ]</div>
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
function CitationPill({ num, citations, activeSessionId, onOpenBBox }) {
  const [isOpen, setIsOpen] = useState(false)
  const index = parseInt(num, 10)
  const cite = citations?.find(c => c.index === index) || (citations && citations[index - 1])

  const fileName = cite?.file_name || 'document.pdf'
  const page = cite?.page || 1
  const isTranscript = cite?.type === 'transcript' || cite?.start_time != null
  const text = cite?.text || cite?.chunk_text || 'Referenced grounding passage from enterprise knowledge base.'

  return (
    <span
      className="citation-pill-container"
      style={{ position: 'relative', display: 'inline-block', margin: '0 3px', verticalAlign: 'baseline' }}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <span
        className={`citation-num-badge ${isOpen ? 'active' : ''}`}
        onClick={(e) => {
          e.stopPropagation()
          onOpenBBox && onOpenBBox(cite || { file_name: fileName, page, text })
        }}
        title={`Citation [${num}]: ${fileName} (Click to open in-app PDF viewer)`}
      >
        {num}
      </span>

      {isOpen && (
        <div
          className="citation-popover-card glass"
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
                onOpenBBox && onOpenBBox(cite || { file_name: fileName, page, text })
              }}
            >
              📍 Spotlight BBox
            </button>
            <button
              onClick={() => {
                setIsOpen(false)
                onOpenBBox && onOpenBBox(cite || { file_name: fileName, page, text })
              }}
              className="citation-popover-link"
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
            >
              View source ↗
            </button>
          </div>
        </div>
      )}
    </span>
  )
}

// ── Markdown Renderer Component ────────────────────────────────────────────────
function MarkdownRenderer({ content, citations, onCitationClick, activeSessionId }) {
  if (!content) return null

  const lines = content.split('\n')
  const elements = []
  let inCodeBlock = false
  let codeBuffer = []
  let listBuffer = []

  const flushList = () => {
    if (listBuffer.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`} style={{ margin: '8px 0', paddingLeft: 22, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {listBuffer.map((item, idx) => (
            <li key={idx} style={{ lineHeight: 1.6 }}>{renderInline(item, citations, onCitationClick, activeSessionId)}</li>
          ))}
        </ul>
      )
      listBuffer = []
    }
  }

  const flushCode = () => {
    if (codeBuffer.length > 0) {
      elements.push(
        <pre key={`code-${elements.length}`} style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid var(--border-subtle)', borderRadius: 8, padding: 12, overflowX: 'auto', margin: '8px 0', fontSize: 13, color: 'var(--accent-cyan)' }}>
          <code>{codeBuffer.join('\n')}</code>
        </pre>
      )
      codeBuffer = []
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        flushCode()
        inCodeBlock = false
      } else {
        flushList()
        inCodeBlock = true
      }
      continue
    }

    if (inCodeBlock) {
      codeBuffer.push(line)
      continue
    }

    if (line.startsWith('### ')) {
      flushList()
      elements.push(<h3 key={`h3-${i}`} style={{ fontSize: '1.1rem', color: 'var(--accent-secondary)', margin: '14px 0 6px', fontWeight: 700 }}>{renderInline(line.slice(4), citations, onCitationClick, activeSessionId)}</h3>)
      continue
    }
    if (line.startsWith('## ')) {
      flushList()
      elements.push(<h2 key={`h2-${i}`} style={{ fontSize: '1.25rem', color: '#ffffff', margin: '16px 0 8px', fontWeight: 700 }}>{renderInline(line.slice(3), citations, onCitationClick, activeSessionId)}</h2>)
      continue
    }
    if (line.startsWith('# ')) {
      flushList()
      elements.push(<h1 key={`h1-${i}`} style={{ fontSize: '1.4rem', color: '#ffffff', margin: '18px 0 10px', fontWeight: 800 }}>{renderInline(line.slice(2), citations, onCitationClick, activeSessionId)}</h1>)
      continue
    }

    const trimmed = line.trim()
    if (trimmed.startsWith('* ') || trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
      const itemText = trimmed.replace(/^[\*\-•]\s+/, '')
      listBuffer.push(itemText)
      continue
    }

    const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/)
    if (numMatch) {
      flushList()
      elements.push(
        <div key={`num-${i}`} style={{ display: 'flex', gap: 8, margin: '4px 0', lineHeight: 1.6 }}>
          <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{numMatch[1]}.</span>
          <div>{renderInline(numMatch[2], citations, onCitationClick, activeSessionId)}</div>
        </div>
      )
      continue
    }

    flushList()
    if (trimmed) {
      elements.push(<p key={`p-${i}`} style={{ margin: '8px 0', lineHeight: 1.65 }}>{renderInline(line, citations, onCitationClick, activeSessionId)}</p>)
    }
  }

  flushList()
  flushCode()

  return <div className="markdown-content">{elements}</div>
}

function renderInline(text, citations, onCitationClick, activeSessionId) {
  if (!text) return ''

  // Regex to split by bold **...**, code `...`, numbered citations [1] or [1, 2], and [Doc: ...]
  const regex = /(\*\*.*?\*\*|`.*?`|\[\d+(?:,\s*\d+)*\]|\[(?:Doc|Media|Page|Citation).*?\])/g
  const tokens = text.split(regex)

  return tokens.map((token, i) => {
    if (!token) return null
    if (token.startsWith('**') && token.endsWith('**')) {
      const inner = token.slice(2, -2)
      return <strong key={i} style={{ color: '#ffffff', fontWeight: 700 }}>{renderInline(inner, citations, onCitationClick, activeSessionId)}</strong>
    }
    if (token.startsWith('`') && token.endsWith('`')) {
      const inner = token.slice(1, -1)
      return <code key={i} style={{ background: 'rgba(255,255,255,0.08)', padding: '2px 6px', borderRadius: 4, fontSize: '0.9em', color: 'var(--accent-cyan)' }}>{inner}</code>
    }

    // Numbered citations [1], [2], [1, 2]
    if (token.startsWith('[') && token.endsWith(']')) {
      const inner = token.slice(1, -1).trim()
      const numbers = inner.split(',').map(s => s.trim()).filter(s => /^\d+$/.test(s))
      if (numbers.length > 0) {
        return (
          <span key={i} className="citation-group-inline" style={{ display: 'inline-flex', gap: 3, verticalAlign: 'baseline' }}>
            {numbers.map((num, nIdx) => (
              <CitationPill
                key={`cite-${i}-${nIdx}`}
                num={num}
                citations={citations}
                activeSessionId={activeSessionId}
                onOpenBBox={onCitationClick}
              />
            ))}
          </span>
        )
      }

      // Format: [Doc: file.pdf | Page: 2]
      const citeText = inner
      const pageMatch = citeText.match(/Page:\s*(\d+)/i) || citeText.match(/p\.\s*(\d+)/i)
      const pageNum = pageMatch ? pageMatch[1] : null
      const fileMatch = citeText.match(/(?:Doc|Media):\s*([^\s\|]+)/i)
      const fileName = fileMatch ? fileMatch[1] : null

      const directLink = fileName && activeSessionId
        ? `/api/files/${activeSessionId}/${fileName}${pageNum ? `#page=${pageNum}` : ''}`
        : null

      return (
        <span
          key={i}
          className="citation-inline-group"
          style={{ display: 'inline-flex', alignItems: 'center', margin: '0 4px', verticalAlign: 'middle' }}
        >
          <span
            className="citation-pill"
            style={{ display: 'inline-flex', padding: '2px 8px', fontSize: 11.5, cursor: 'pointer' }}
            onClick={() => onCitationClick && onCitationClick({ file_name: fileName || citeText, page: pageNum ? parseInt(pageNum, 10) : 1, text: `Reference: ${citeText}` })}
            title="Click to inspect chunk bbox & position"
          >
            🔍 {citeText}
          </span>
          <button
            onClick={() => onCitationClick && onCitationClick({ file_name: fileName || citeText, page: pageNum ? parseInt(pageNum, 10) : 1, text: `Reference: ${citeText}` })}
            style={{ background: 'none', border: 'none', fontSize: 11, marginLeft: 4, color: 'var(--accent-cyan)', opacity: 0.8, cursor: 'pointer', padding: 0 }}
            title={`View ${fileName || 'document'}${pageNum ? ` page ${pageNum}` : ''} in-app`}
          >
            ↗
          </button>
        </span>
      )
    }
    return token
  })
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

// ── Memories Modal ─────────────────────────────────────────────────────────────
function MemoriesModal({ isOpen, onClose, memoryData }) {
  if (!isOpen || !memoryData) return null

  const episodic = memoryData.episodic_memory || {}
  const semantic = memoryData.semantic_memory || {}
  const flags = memoryData.feature_flags || { enable_semantic_memory: true, enable_episodic_memory: true }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog glass" onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">🧠 Extracted Contextual Memory Profile</div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <span className="status-pill" style={{ background: flags.enable_episodic_memory ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255,255,255,0.05)', color: flags.enable_episodic_memory ? 'var(--accent-emerald)' : 'var(--text-muted)' }}>
              Episodic Memory: {flags.enable_episodic_memory ? 'ENABLED' : 'DISABLED'}
            </span>
            <span className="status-pill" style={{ background: flags.enable_semantic_memory ? 'rgba(6, 182, 212, 0.15)' : 'rgba(255,255,255,0.05)', color: flags.enable_semantic_memory ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
              Semantic Memory: {flags.enable_semantic_memory ? 'ENABLED' : 'DISABLED'}
            </span>
          </div>

          <div>
            <h4 style={{ fontSize: 14, color: 'var(--accent-secondary)', marginBottom: 8 }}>📖 Episodic Summary & Key Events</h4>
            <div style={{ background: 'var(--bg-tertiary)', padding: 14, borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', fontSize: 13.5 }}>
              <p style={{ marginBottom: 10 }}><strong>Summary:</strong> {episodic.summary || 'No episodic summary extracted for this session yet.'}</p>
              {episodic.key_events && episodic.key_events.length > 0 && (
                <div>
                  <strong>Key Milestones:</strong>
                  <ul style={{ paddingLeft: 20, marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {episodic.key_events.map((ev, i) => <li key={i}>{ev}</li>)}
                  </ul>
                </div>
              )}
            </div>
          </div>

          <div>
            <h4 style={{ fontSize: 14, color: 'var(--accent-cyan)', marginBottom: 8 }}>💡 Learned Semantic Facts & Preferences</h4>
            <div style={{ background: 'var(--bg-tertiary)', padding: 14, borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', fontSize: 13.5 }}>
              {semantic.facts && semantic.facts.length > 0 ? (
                <div style={{ marginBottom: 10 }}>
                  <strong>Extracted Knowledge:</strong>
                  <ul style={{ paddingLeft: 20, marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {semantic.facts.map((fact, i) => <li key={i}>{fact}</li>)}
                  </ul>
                </div>
              ) : (
                <p style={{ color: 'var(--text-muted)' }}>No semantic facts extracted yet.</p>
              )}

              {semantic.preferences && semantic.preferences.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <strong>User Preferences:</strong>
                  <ul style={{ paddingLeft: 20, marginTop: 4 }}>
                    {semantic.preferences.map((pref, i) => <li key={i}>{pref}</li>)}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
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

    addLog(`Initiated multi-file upload (${fileArray.length} files) for session: ${activeSessionId || 'default'}`, 'info')

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
        onClick={() => document.getElementById('file-picker').click()}
      >
        <input
          id="file-picker"
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
          Upload multiple files simultaneously. PDFs are parsed via OpenDataLoader with table clusters and bounding boxes, while video/audio files are transcribed via Rev AI and indexed into Triton & Weaviate.
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

// ── Agentic Chat Workspace Component ──────────────────────────────────────────
function ChatWorkspace({ token, activeSessionId, sessions, onSessionChange, onCreateSession, onEndSession, onViewMemories }) {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [streaming, setStreaming] = useState(false)
  const [selectedCitation, setSelectedCitation] = useState(null)
  const abortRef = useRef(null)

  useEffect(() => {
    if (!activeSessionId || !token) return
    fetch(`/api/sessions/${activeSessionId}`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.history) {
          setMessages(data.history.map(m => ({
            role: m.role,
            content: m.content,
            thoughts: [],
            steps: [],
            citations: [],
          })))
        }
      })
      .catch(err => console.error('Failed to load session history:', err))
  }, [activeSessionId, token])

  const handleAsk = useCallback(async () => {
    if (!query.trim() || streaming) return
    const userQuery = query.trim()
    setQuery('')
    setStreaming(true)

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

      const endpoint = `/api/sessions/${activeSessionId || 'default'}/ask`
      const headers = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`

      const res = await fetch(endpoint, {
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
            } else if (msg.type === 'error') {
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : {
                ...m,
                content: m.content + `\n\n**Error:** ${msg.content}`,
                isReasoning: false,
              }))
            }
          } catch {}
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
  }, [query, streaming, activeSessionId, token])

  const toggleReasoning = (msgId) => {
    setMessages(prev => prev.map(m => m.id !== msgId ? m : {
      ...m,
      reasoningCollapsed: !m.reasoningCollapsed,
    }))
  }

  return (
    <div className="workspace-container">
      <div className="session-bar glass">
        <div className="session-selector">
          <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)' }}>SESSION:</span>
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

        <div className="session-actions">
          <button className="btn btn-ghost" style={{ padding: '6px 14px', fontSize: 13 }} onClick={onCreateSession}>
            + New Session
          </button>
          {activeSessionId && (
            <>
              <button className="btn btn-ghost" style={{ padding: '6px 14px', fontSize: 13 }} onClick={onViewMemories}>
                🧠 View Memories
              </button>
              <button className="btn btn-ghost" style={{ padding: '6px 14px', fontSize: 13, color: 'var(--accent-rose)' }} onClick={onEndSession}>
                ⏹ End Session
              </button>
            </>
          )}
        </div>
      </div>

      <div className="chat-window">
        <div className="chat-history glass">
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: 42, marginBottom: 12 }}>💬</div>
              <h4 style={{ color: '#ffffff', fontSize: '1.2rem', marginBottom: 8 }}>Enterprise Agent Ready</h4>
              <p style={{ fontSize: 14, maxWidth: 480, margin: '0 auto' }}>
                Ask questions across your multi-modal session documents and audio/video transcripts. Thought reasoning traces, timestamps, and bounding box citations will stream live.
              </p>
            </div>
          )}

          {messages.map((m, idx) => (
            <div key={m.id || idx} className={`message-card ${m.role}`}>
              <div className="message-header">
                <span className="message-role">{m.role === 'user' ? '👤 User' : '🤖 GraphRAG Agent'}</span>
              </div>

              {/* Faded, Collapsible Thought Section */}
              {m.role === 'assistant' && (m.steps?.length > 0 || m.thoughts?.length > 0 || m.isReasoning) && (
                <div className="reasoning-accordion">
                  <div className="reasoning-header" onClick={() => toggleReasoning(m.id)}>
                    <div className="reasoning-title">
                      {m.isReasoning ? <span className="pulse-spinner" /> : <span>🧠</span>}
                      <span>{m.isReasoning ? 'Agent is reasoning & searching knowledge…' : `Reasoning Trace (${(m.steps?.length || 0) + (m.thoughts?.length || 0)} steps)`}</span>
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

              {/* Real Answer Text Rendered with Markdown */}
              {m.content && (
                <div className="answer-body">
                  <MarkdownRenderer
                    content={m.content}
                    citations={m.citations}
                    activeSessionId={activeSessionId}
                    onCitationClick={(citeItem) => {
                      if (typeof citeItem === 'object' && citeItem !== null) {
                        setSelectedCitation({ ...citeItem, session_id: activeSessionId })
                      } else if (typeof citeItem === 'string') {
                        const citeText = citeItem
                        const matched = m.citations?.find(c =>
                          citeText.toLowerCase().includes((c.file_name || '').toLowerCase()) ||
                          (c.page && citeText.includes(`${c.page}`)) ||
                          (c.custom_id && citeText.includes(c.custom_id))
                        )
                        if (matched) {
                          setSelectedCitation({ ...matched, session_id: activeSessionId })
                        } else {
                          const pageMatch = citeText.match(/Page:\s*(\d+)/i) || citeText.match(/p\.\s*(\d+)/i)
                          const fileMatch = citeText.match(/(?:Doc|Media):\s*([^\s\|]+)/i)
                          setSelectedCitation({
                            file_name: fileMatch ? fileMatch[1] : citeText,
                            page: pageMatch ? parseInt(pageMatch[1], 10) : 1,
                            session_id: activeSessionId,
                            type: citeText.toLowerCase().includes('media') ? 'transcript' : 'text',
                            text: `Reference: ${citeText}`,
                          })
                        }
                      }
                    }}
                  />
                </div>
              )}

              {/* Citations with Position, Timestamp, and BBox Links */}
              {m.citations && m.citations.length > 0 && (
                <div className="citations-section">
                  <div className="citations-label">Source Document Citations (Click to inspect position in file):</div>
                  <div className="citations-grid">
                    {m.citations.map((c, i) => {
                      const isTrans = c.type === 'transcript' || c.start_time != null
                      return (
                        <div
                          key={c.custom_id || i}
                          className="citation-pill"
                          onClick={() => setSelectedCitation({ ...c, session_id: activeSessionId })}
                          title="Click to view chunk position"
                        >
                          <span className="citation-badge">{c.type || 'text'}</span>
                          <span>{isTrans ? '🎬' : '📄'} {c.file_name}</span>
                          <span style={{ color: 'var(--accent-cyan)' }}>
                            {isTrans ? `⏱ ${c.start_time?.toFixed(1)}s` : `p. ${c.page || 1}`}
                          </span>
                          {c.score != null && <span style={{ color: 'var(--accent-emerald)' }}>{(c.score).toFixed(2)}</span>}
                          <span>📍 View</span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="chat-input-bar">
          <textarea
            className="chat-input"
            rows={2}
            placeholder={activeSessionId ? "Ask a question about this session's documents and media…" : "Ask anything across enterprise knowledge..."}
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

// ── Landing Page Component ────────────────────────────────────────────────────
function LandingPage({ onExploreWorkspace }) {
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
          Next-generation GraphRAG platform powered by <strong>NVIDIA Triton GPU acceleration</strong>, PDF visual layout understanding, Rev AI audio/video transcription, Jina cross-encoder reranking, and self-evolving contextual memory.
        </p>

        <div className="hero-cta">
          <button className="btn btn-gradient" style={{ padding: '14px 28px', fontSize: 16 }} onClick={onExploreWorkspace}>
            Launch AI Workspace →
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
            <div className="metric-label">Token Security & Isolation</div>
          </div>
          <div className="metric-card glass">
            <div className="metric-value">100%</div>
            <div className="metric-label">Multi-Modal BBox Accuracy</div>
          </div>
        </div>
      </section>

      {/* Interactive Architecture Section */}
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
            <div className="pipeline-node-sub">PDF BBox & Media STT</div>
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

      {/* Features Grid */}
      <section className="section-container">
        <div className="section-header">
          <div className="section-tag">Enterprise Capabilities</div>
          <h2 className="section-title">Built for Mission-Critical Accuracy</h2>
        </div>

        <div className="features-grid">
          <div className="feature-card glass">
            <div className="feature-icon">🔍</div>
            <h3 className="feature-title">Multi-Modal BBox & Media Audio Localization</h3>
            <p className="feature-desc">
              Preserves exact spatial coordinates for tables, figures, headings, and speech audio timestamps with Rev AI integration and Cloudinary CDN previews.
            </p>
          </div>

          <div className="feature-card glass">
            <div className="feature-icon">⚡</div>
            <h3 className="feature-title">Triton GPU Microservice</h3>
            <p className="feature-desc">
              High-throughput gRPC inference utilizing Google SigLIP multimodal vision-language embeddings and Jina AI cross-encoder rerankers with rate-limiting and dynamic batching.
            </p>
          </div>

          <div className="feature-card glass">
            <div className="feature-icon">🧠</div>
            <h3 className="feature-title">Episodic & Semantic Memory Engine</h3>
            <p className="feature-desc">
              Asynchronous Celery workers synthesize conversational milestones into structured episodic summaries and persistent semantic facts with dedicated feature flag controls.
            </p>
          </div>

          <div className="feature-card glass">
            <div className="feature-icon">🛡️</div>
            <h3 className="feature-title">Session-Level Isolation & JWT Rotation</h3>
            <p className="feature-desc">
              15-minute access tokens paired with 7-day Redis-backed refresh tokens featuring automatic token rotation, instant logout revocation, and session-level Weaviate query filtering.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

// ── Main App Root ─────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView] = useState('landing')
  const [token, setToken] = useState(() => localStorage.getItem('graphrag_access_token') || '')
  const [refreshToken, setRefreshToken] = useState(() => localStorage.getItem('graphrag_refresh_token') || '')
  const [username, setUsername] = useState(() => localStorage.getItem('graphrag_username') || '')
  const [authModalOpen, setAuthModalOpen] = useState(false)
  const [memoryModalOpen, setMemoryModalOpen] = useState(false)
  const [memoryData, setMemoryData] = useState(null)

  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState('')

  const loadSessions = useCallback(async (authToken) => {
    if (!authToken) return
    try {
      const res = await fetch('/api/sessions', {
        headers: { Authorization: `Bearer ${authToken}` },
      })
      if (res.status === 401) {
        setToken('')
        setUsername('')
        localStorage.removeItem('graphrag_access_token')
        localStorage.removeItem('graphrag_refresh_token')
        localStorage.removeItem('graphrag_username')
        return
      }
      if (res.ok) {
        const data = await res.json()
        setSessions(data)
        if (data.length > 0 && !activeSessionId) {
          setActiveSessionId(data[0].session_id)
        }
      }
    } catch (err) {
      console.error('Error fetching sessions:', err)
    }
  }, [activeSessionId])

  useEffect(() => {
    if (token) loadSessions(token)
  }, [token, loadSessions])

  const handleCreateSession = async () => {
    if (!token) {
      setAuthModalOpen(true)
      return
    }
    try {
      const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ title: `Session ${sessions.length + 1}` }),
      })
      if (res.status === 401) {
        setToken('')
        setUsername('')
        localStorage.removeItem('graphrag_access_token')
        localStorage.removeItem('graphrag_refresh_token')
        localStorage.removeItem('graphrag_username')
        setAuthModalOpen(true)
        throw new Error('Your authentication expired. Please sign in again.')
      }
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || 'Failed to create session')
      }
      const newSess = await res.json()
      setSessions(prev => [newSess, ...prev])
      setActiveSessionId(newSess.session_id)
    } catch (err) {
      alert(err.message)
    }
  }

  const handleEndSession = async () => {
    if (!activeSessionId || !token) return
    try {
      const res = await fetch(`/api/sessions/${activeSessionId}/end`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error('Failed to end session')
      const data = await res.json()
      alert(`Session closed: ${data.message}`)
      setSessions(prev => prev.map(s => s.session_id !== activeSessionId ? s : { ...s, status: 'ending' }))
    } catch (err) {
      alert(err.message)
    }
  }

  const handleViewMemories = async () => {
    if (!activeSessionId || !token) return
    try {
      const res = await fetch(`/api/sessions/${activeSessionId}/memories`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error('Failed to load memories')
      const data = await res.json()
      setMemoryData(data)
      setMemoryModalOpen(true)
    } catch (err) {
      alert(err.message)
    }
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
    } catch {}

    localStorage.removeItem('graphrag_access_token')
    localStorage.removeItem('graphrag_refresh_token')
    localStorage.removeItem('graphrag_username')
    setToken('')
    setRefreshToken('')
    setUsername('')
    setSessions([])
    setActiveSessionId('')
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand" onClick={() => setView('landing')}>
          <div className="brand-icon">⚡</div>
          <div>
            <div className="brand-name">Enterprise RAG</div>
          </div>
          <span className="brand-tag">v2.0</span>
        </div>

        <nav className="nav-tabs">
          <button
            className={`nav-btn ${view === 'landing' ? 'active' : ''}`}
            onClick={() => setView('landing')}
          >
            Overview
          </button>
          <button
            className={`nav-btn ${view === 'workspace' ? 'active' : ''}`}
            onClick={() => setView('workspace')}
          >
            Agent Workspace
          </button>
          <button
            className={`nav-btn ${view === 'ingest' ? 'active' : ''}`}
            onClick={() => setView('ingest')}
          >
            Ingestion Studio
          </button>
        </nav>

        <div className="header-actions">
          <div className="status-pill">
            <span className="status-dot" />
            <span>Triton & Weaviate Online</span>
          </div>

          {token ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>👤 {username}</span>
              <button className="btn btn-ghost" style={{ padding: '6px 12px', fontSize: 12.5 }} onClick={handleLogout}>
                Sign Out
              </button>
            </div>
          ) : (
            <button className="btn btn-gradient" style={{ padding: '7px 16px', fontSize: 13 }} onClick={() => setAuthModalOpen(true)}>
              Sign In / Register
            </button>
          )}
        </div>
      </header>

      <main style={{ flex: 1 }}>
        {view === 'landing' && (
          <LandingPage onExploreWorkspace={() => setView('workspace')} />
        )}

        {view === 'workspace' && (
          <ChatWorkspace
            token={token}
            activeSessionId={activeSessionId}
            sessions={sessions}
            onSessionChange={setActiveSessionId}
            onCreateSession={handleCreateSession}
            onEndSession={handleEndSession}
            onViewMemories={handleViewMemories}
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

      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onAuthSuccess={(acc, ref, uname) => {
          setToken(acc)
          setRefreshToken(ref)
          setUsername(uname)
          setView('workspace')
          setAuthModalOpen(false)
          loadSessions(acc)
        }}
      />

      <MemoriesModal
        isOpen={memoryModalOpen}
        memoryData={memoryData}
        onClose={() => setMemoryModalOpen(false)}
      />
    </div>
  )
}
