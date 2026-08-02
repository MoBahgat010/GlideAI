import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

// ── constants ──────────────────────────────────────────────────────────────
const CHUNK_SIZE = 1024 * 1024   // 1 MB slices
const POLL_INTERVAL_MS = 1500

// ── helpers ────────────────────────────────────────────────────────────────
function formatBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1048576).toFixed(1)} MB`
}

function genId() {
  return crypto.randomUUID()
}

// ── Upload logic ───────────────────────────────────────────────────────────
async function uploadFileInChunks(file, uploadId, onProgress) {
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE)

  for (let i = 0; i < totalChunks; i++) {
    const slice = file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE)
    const form = new FormData()
    form.append('file', slice, file.name)
    form.append('upload_id', uploadId)
    form.append('chunk_index', String(i))
    form.append('total_chunks', String(totalChunks))
    form.append('filename', file.name)

    const res = await fetch('/api/ingest/chunk', { method: 'POST', body: form })
    if (!res.ok) throw new Error(`Chunk ${i} failed: ${res.statusText}`)

    onProgress((i + 1) / totalChunks * 0.5)   // first half = upload
  }

  // Finalise
  const form = new FormData()
  form.append('upload_id', uploadId)
  form.append('filename', file.name)
  form.append('total_chunks', String(totalChunks))

  const res = await fetch('/api/ingest/finalize', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Finalize failed: ${res.statusText}`)
  const data = await res.json()
  return data.task_id
}

// ── ProgressBar ────────────────────────────────────────────────────────────
function ProgressBar({ pct }) {
  return (
    <div className="progress-wrap">
      <div className="progress-bar-bg">
        <div className="progress-bar-fill" style={{ width: `${Math.round(pct * 100)}%` }} />
      </div>
    </div>
  )
}

// ── FileCard ───────────────────────────────────────────────────────────────
function FileCard({ item }) {
  const statusClass = item.status === 'done' ? 'done'
    : item.status === 'error' ? 'error'
    : item.status === 'running' ? 'running' : 'pending'

  const statusText = item.status === 'done' ? '✓ Done'
    : item.status === 'error' ? '✗ Failed'
    : item.status === 'running' ? item.stage || 'Processing…'
    : 'Queued'

  return (
    <div className="file-card glass">
      <span style={{ fontSize: 22 }}>📄</span>
      <div className="file-info">
        <div className="file-name">{item.name}</div>
        <div className="file-size">{formatBytes(item.size)}</div>
        {item.status === 'running' && <ProgressBar pct={item.pct || 0} />}
        {item.message && (
          <div className="progress-label">{item.message}</div>
        )}
      </div>
      <span className={`file-status ${statusClass}`}>{statusText}</span>
    </div>
  )
}

// ── StageLog ───────────────────────────────────────────────────────────────
function StageLog({ lines }) {
  const ref = useRef(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [lines])
  if (!lines.length) return null
  return (
    <div className="stage-log" ref={ref}>
      {lines.map((l, i) => (
        <div key={i} className={`log-line ${l.cls}`}>{l.text}</div>
      ))}
    </div>
  )
}

// ── SourceCard ─────────────────────────────────────────────────────────────
function SourceCard({ result, index }) {
  const type = result.type || 'text'
  const typeClass = `${type}-type`
  return (
    <div className={`source-card ${typeClass}`}>
      <div className="source-meta">
        <span className="source-badge">{type}</span>
        <span className="source-file">{result.file_name}</span>
        {result.rerank_score != null && (
          <span className="source-file">score: {result.rerank_score?.toFixed(3)}</span>
        )}
      </div>
      {result.chunk_text && (
        <div className="source-text">{result.chunk_text.slice(0, 400)}{result.chunk_text.length > 400 ? '…' : ''}</div>
      )}
      {type === 'image' && result.image_path && (
        <img
          className="source-image"
          src={`/uploads/${encodeURIComponent(result.image_path.split('/').pop())}`}
          alt={`Source image ${index + 1}`}
          loading="lazy"
        />
      )}
      {result.linked_image?.image_path && (
        <img
          className="source-image"
          src={`/uploads/${encodeURIComponent(result.linked_image.image_path.split('/').pop())}`}
          alt="Linked image"
          loading="lazy"
        />
      )}
    </div>
  )
}

// ── IngestTab ──────────────────────────────────────────────────────────────
function IngestTab() {
  const [files, setFiles] = useState([])
  const [dragOver, setDragOver] = useState(false)
  const [logLines, setLogLines] = useState([])
  const pollersRef = useRef({})

  const addLog = useCallback((text, cls = '') => {
    setLogLines(prev => [...prev, { text, cls }])
  }, [])

  const startPolling = useCallback((uploadId, taskId) => {
    const iv = setInterval(async () => {
      try {
        const res = await fetch(`/api/ingest/status/${taskId}`)
        const data = await res.json()

        setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : {
          ...f,
          status: data.state === 'SUCCESS' ? 'done'
            : data.state === 'FAILURE' ? 'error'
            : 'running',
          stage: data.stage,
          message: data.message,
          pct: 0.5 + (data.pct || 0) * 0.5,
        }))

        const stateClass = data.state === 'SUCCESS' ? 'DONE'
          : data.state === 'FAILURE' ? 'FAILED' : 'active'
        addLog(`[${data.stage}] ${data.message}`, stateClass)

        if (data.state === 'SUCCESS' || data.state === 'FAILURE') {
          clearInterval(iv)
          delete pollersRef.current[uploadId]
        }
      } catch (e) {
        addLog(`Poll error: ${e.message}`, 'FAILED')
      }
    }, POLL_INTERVAL_MS)
    pollersRef.current[uploadId] = iv
  }, [addLog])

  const processFile = useCallback(async (file) => {
    const uploadId = genId()
    const entry = {
      uploadId,
      name: file.name,
      size: file.size,
      status: 'running',
      stage: 'UPLOADING',
      message: 'Uploading…',
      pct: 0,
    }
    setFiles(prev => [...prev, entry])
    addLog(`⬆ Uploading ${file.name} (${formatBytes(file.size)})`, 'active')

    try {
      const taskId = await uploadFileInChunks(file, uploadId, (pct) => {
        setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : { ...f, pct, message: `Uploading… ${Math.round(pct * 100)}%` }))
      })
      addLog(`✓ Upload complete — ingestion task dispatched`, 'DONE')
      addLog(`  Task ID: ${taskId}`)
      setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : { ...f, stage: 'QUEUED', message: 'Ingestion queued…', pct: 0.5 }))
      startPolling(uploadId, taskId)
    } catch (err) {
      addLog(`✗ ${file.name}: ${err.message}`, 'FAILED')
      setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : { ...f, status: 'error', message: err.message }))
    }
  }, [addLog, startPolling])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    Array.from(e.dataTransfer.files).forEach(processFile)
  }, [processFile])

  const handleChange = useCallback((e) => {
    Array.from(e.target.files || []).forEach(processFile)
    e.target.value = ''
  }, [processFile])

  return (
    <>
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input
          id="file-input"
          type="file"
          multiple
          accept=".pdf,.mp4,.mp3,.wav,.mov"
          onChange={handleChange}
        />
        <div className="upload-icon">📁</div>
        <h3>Drop files here or click to browse</h3>
        <p>PDF, MP4, MP3, WAV, MOV — files are uploaded in 1 MB chunks</p>
      </div>

      {files.length > 0 && (
        <div className="file-list">
          {files.map(f => <FileCard key={f.uploadId} item={f} />)}
        </div>
      )}

      <StageLog lines={logLines} />
    </>
  )
}

// ── AskTab ─────────────────────────────────────────────────────────────────
function AskTab() {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState('')
  const abortRef = useRef(null)

  const handleAsk = useCallback(async () => {
    if (!query.trim() || streaming) return
    setAnswer('')
    setSources([])
    setError('')
    setStreaming(true)

    try {
      const ctrl = new AbortController()
      abortRef.current = ctrl

      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
        signal: ctrl.signal,
      })

      if (!res.ok) throw new Error(`Server error: ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })

        const parts = buf.split('\n\n')
        buf = parts.pop()   // keep incomplete tail

        for (const part of parts) {
          if (!part.startsWith('data:')) continue
          try {
            const msg = JSON.parse(part.slice(5).trim())
            if (msg.type === 'token') {
              setAnswer(prev => prev + msg.content)
            } else if (msg.type === 'done') {
              setSources(msg.results || [])
            } else if (msg.type === 'error') {
              setError(msg.content)
            }
          } catch {}
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message)
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }, [query, streaming])

  const handleKey = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleAsk()
  }

  return (
    <>
      <div className="ask-form">
        <div className="ask-input-wrap">
          <label htmlFor="ask-query">Your question</label>
          <textarea
            id="ask-query"
            className="ask-input"
            rows={3}
            placeholder="Ask anything about your indexed documents…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKey}
            disabled={streaming}
          />
        </div>
        <button
          id="ask-submit"
          className="btn btn-primary"
          onClick={handleAsk}
          disabled={streaming || !query.trim()}
          style={{ marginBottom: 0 }}
        >
          {streaming ? '⏹ Stop' : '▶ Ask'}
        </button>
      </div>

      <div className={`answer-box glass ${!answer && !streaming ? 'empty' : ''}`}>
        {answer || (!streaming && <span>Your answer will appear here…</span>)}
        {streaming && <span className="answer-cursor" />}
      </div>

      {error && (
        <div className="glass" style={{ padding: 14, color: 'var(--error)', fontSize: 13 }}>
          ⚠ {error}
        </div>
      )}

      {sources.length > 0 && (
        <div>
          <div className="sources-header">Sources ({sources.length})</div>
          <div className="source-cards">
            {sources.map((r, i) => <SourceCard key={r.id || i} result={r} index={i} />)}
          </div>
        </div>
      )}
    </>
  )
}

// ── App ────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState('ingest')

  return (
    <div className="app">
      <header className="header glass" style={{ borderRadius: 0 }}>
        <div className="logo">
          <span className="logo-dot" />
          GraphRAG
        </div>
        <div className="tabs" role="tablist">
          <button
            id="tab-ingest"
            role="tab"
            className={`tab ${tab === 'ingest' ? 'active' : ''}`}
            onClick={() => setTab('ingest')}
          >
            ⬆ Ingest
          </button>
          <button
            id="tab-ask"
            role="tab"
            className={`tab ${tab === 'ask' ? 'active' : ''}`}
            onClick={() => setTab('ask')}
          >
            💬 Ask
          </button>
        </div>
      </header>

      <main className="content" role="tabpanel">
        {tab === 'ingest' ? <IngestTab /> : <AskTab />}
      </main>
    </div>
  )
}
