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
async function uploadFileInChunks(file, uploadId, onProgress, token) {
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

    const res = await fetch('/api/ingest/chunk', { method: 'POST', body: form, headers })
    if (!res.ok) throw new Error(`Chunk ${i} failed: ${res.statusText}`)

    onProgress((i + 1) / totalChunks * 0.5)   // first half = upload
  }

  // Finalise
  const form = new FormData()
  form.append('upload_id', uploadId)
  form.append('filename', file.name)
  form.append('total_chunks', String(totalChunks))

  const res = await fetch('/api/ingest/finalize', { method: 'POST', body: form, headers })
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

// ── Auth Modal Component ───────────────────────────────────────────────────
function AuthModal({ isOpen, onClose, onSuccess }) {
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
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail || 'Registration failed')
        // Auto login
        setMode('login')
      }

      // Login
      const form = new URLSearchParams()
      form.append('username', username)
      form.append('password', password)

      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: form,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Login failed')

      localStorage.setItem('graphrag_token', data.access_token)
      localStorage.setItem('graphrag_username', data.username)
      onSuccess(data.access_token, data.username)
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-overlay">
      <div className="modal-card glass">
        <div className="modal-header">
          <span className="modal-title">{mode === 'login' ? '🔐 Sign In' : '✨ Register Account'}</span>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="input-group">
            <label>Username</label>
            <input
              className="form-input"
              type="text"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>

          {mode === 'register' && (
            <div className="input-group">
              <label>Email Address</label>
              <input
                className="form-input"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
          )}

          <div className="input-group">
            <label>Password</label>
            <input
              className="form-input"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          {error && <div style={{ color: 'var(--error)', fontSize: 13 }}>⚠ {error}</div>}

          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? 'Processing…' : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <div style={{ fontSize: 13, textAlign: 'center', color: 'var(--text-muted)' }}>
          {mode === 'login' ? (
            <span>Don't have an account? <a href="#" style={{ color: 'var(--accent2)' }} onClick={() => setMode('register')}>Register</a></span>
          ) : (
            <span>Already have an account? <a href="#" style={{ color: 'var(--accent2)' }} onClick={() => setMode('login')}>Sign In</a></span>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Memories Modal Component ───────────────────────────────────────────────
function MemoriesModal({ isOpen, onClose, memoryData }) {
  if (!isOpen || !memoryData) return null

  const episodic = memoryData.episodic_memory || {}
  const semantic = memoryData.semantic_memory || {}

  return (
    <div className="modal-overlay">
      <div className="modal-card glass" style={{ maxWidth: 640 }}>
        <div className="modal-header">
          <span className="modal-title">🧠 Extracted Memory Profile</span>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxHeight: 420, overflowY: 'auto' }}>
          <div>
            <h4 style={{ fontSize: 14, color: 'var(--accent2)', marginBottom: 6 }}>📖 Episodic Memory</h4>
            <div className="memory-box">
              <p><strong>Summary:</strong> {episodic.summary || 'No episodic summary extracted yet.'}</p>
              {episodic.key_events && episodic.key_events.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <strong>Key Events:</strong>
                  <ul style={{ paddingLeft: 18, marginTop: 4 }}>
                    {episodic.key_events.map((ev, i) => <li key={i}>{ev}</li>)}
                  </ul>
                </div>
              )}
            </div>
          </div>

          <div>
            <h4 style={{ fontSize: 14, color: '#22d3ee', marginBottom: 6 }}>💡 Semantic Memory</h4>
            <div className="memory-box">
              {semantic.facts && semantic.facts.length > 0 ? (
                <div>
                  <strong>Learned Facts:</strong>
                  <ul style={{ paddingLeft: 18, marginTop: 4 }}>
                    {semantic.facts.map((fact, i) => <li key={i}>{fact}</li>)}
                  </ul>
                </div>
              ) : (
                <p>No semantic facts extracted yet.</p>
              )}

              {semantic.preferences && semantic.preferences.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <strong>User Preferences:</strong>
                  <ul style={{ paddingLeft: 18, marginTop: 4 }}>
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

// ── IngestTab ──────────────────────────────────────────────────────────────
function IngestTab({ token }) {
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
      }, token)
      addLog(`✓ Upload complete — ingestion task dispatched`, 'DONE')
      addLog(`  Task ID: ${taskId}`)
      setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : { ...f, stage: 'QUEUED', message: 'Ingestion queued…', pct: 0.5 }))
      startPolling(uploadId, taskId)
    } catch (err) {
      addLog(`✗ ${file.name}: ${err.message}`, 'FAILED')
      setFiles(prev => prev.map(f => f.uploadId !== uploadId ? f : { ...f, status: 'error', message: err.message }))
    }
  }, [addLog, startPolling, token])

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
function AskTab({ activeSessionId, token, history, setHistory }) {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState('')
  const abortRef = useRef(null)

  const handleAsk = useCallback(async () => {
    if (!query.trim() || streaming) return
    const userMsg = query.trim()
    setQuery('')
    setAnswer('')
    setSources([])
    setError('')
    setStreaming(true)

    // Optimistically add user message to transcript
    setHistory(prev => [...prev, { role: 'user', content: userMsg }])

    try {
      const ctrl = new AbortController()
      abortRef.current = ctrl

      const endpoint = activeSessionId ? `/api/sessions/${activeSessionId}/ask` : '/api/ask'
      const headers = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`

      const res = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({ query: userMsg }),
        signal: ctrl.signal,
      })

      if (!res.ok) throw new Error(`Server error: ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      let fullText = ''

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
              fullText += msg.content
              setAnswer(fullText)
            } else if (msg.type === 'done') {
              setSources(msg.results || [])
              // Append assistant message to local history
              if (fullText) {
                setHistory(prev => [...prev, { role: 'assistant', content: fullText }])
              }
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
  }, [query, streaming, activeSessionId, token, setHistory])

  const handleKey = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleAsk()
  }

  return (
    <>
      {history.length > 0 && (
        <div className="chat-history glass" style={{ padding: 16 }}>
          {history.map((msg, i) => (
            <div key={i} className={`chat-bubble ${msg.role}`}>
              <div className="chat-role">{msg.role}</div>
              <div>{msg.content}</div>
            </div>
          ))}
        </div>
      )}

      <div className="ask-form">
        <div className="ask-input-wrap">
          <label htmlFor="ask-query">Your question</label>
          <textarea
            id="ask-query"
            className="ask-input"
            rows={3}
            placeholder={activeSessionId ? "Ask a question in this session…" : "Ask anything about your indexed documents…"}
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
        {answer || (!streaming && <span>Your answer will stream here…</span>)}
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
  const [token, setToken] = useState(() => localStorage.getItem('graphrag_token') || '')
  const [username, setUsername] = useState(() => localStorage.getItem('graphrag_username') || '')
  const [authModalOpen, setAuthModalOpen] = useState(false)
  
  // Sessions state
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState('')
  const [history, setHistory] = useState([])
  const [memoryModalOpen, setMemoryModalOpen] = useState(false)
  const [memoryData, setMemoryData] = useState(null)

  // Load sessions on auth
  useEffect(() => {
    if (!token) return
    fetch('/api/sessions', {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(res => res.ok ? res.json() : [])
      .then(data => {
        setSessions(data)
        if (data.length > 0 && !activeSessionId) {
          setActiveSessionId(data[0].session_id)
        }
      })
      .catch(err => console.error('Failed to load sessions:', err))
  }, [token])

  // Load active session history
  useEffect(() => {
    if (!activeSessionId || !token) {
      setHistory([])
      return
    }
    fetch(`/api/sessions/${activeSessionId}`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.history) {
          setHistory(data.history)
        }
      })
      .catch(err => console.error('Failed to load session history:', err))
  }, [activeSessionId, token])

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
      if (!res.ok) throw new Error('Failed to create session')
      const newSess = await res.json()
      setSessions(prev => [newSess, ...prev])
      setActiveSessionId(newSess.session_id)
      setHistory([])
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
      alert(`Session ended! Celery task ${data.task_id} dispatched for memory extraction.`)

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

  const handleLogout = () => {
    localStorage.removeItem('graphrag_token')
    localStorage.removeItem('graphrag_username')
    setToken('')
    setUsername('')
    setSessions([])
    setActiveSessionId('')
    setHistory([])
  }

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

        <div className="header-right">
          {token ? (
            <div className="auth-badge">
              <span>👤 <span className="username-tag">{username}</span></span>
              <button className="btn btn-ghost" style={{ padding: '6px 12px', fontSize: 12 }} onClick={handleLogout}>Logout</button>
            </div>
          ) : (
            <button className="btn btn-primary" style={{ padding: '6px 14px', fontSize: 13 }} onClick={() => setAuthModalOpen(true)}>
              Sign In / Register
            </button>
          )}
        </div>
      </header>

      {/* Session toolbar if authenticated */}
      {token && (
        <div style={{ maxWidth: 1050, width: '100%', margin: '16px auto 0', padding: '0 28px' }}>
          <div className="session-bar glass">
            <div className="session-selector">
              <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)' }}>SESSION:</span>
              <select
                className="session-select"
                value={activeSessionId}
                onChange={(e) => setActiveSessionId(e.target.value)}
              >
                {sessions.map(s => (
                  <option key={s.session_id} value={s.session_id}>
                    {s.title} ({s.status})
                  </option>
                ))}
              </select>
            </div>

            <div className="session-actions">
              <button className="btn btn-ghost" style={{ padding: '6px 12px', fontSize: 12 }} onClick={handleCreateSession}>
                + New Session
              </button>
              {activeSessionId && (
                <>
                  <button className="btn btn-ghost" style={{ padding: '6px 12px', fontSize: 12 }} onClick={handleViewMemories}>
                    🧠 View Memories
                  </button>
                  <button className="btn btn-ghost" style={{ padding: '6px 12px', fontSize: 12, color: 'var(--error)' }} onClick={handleEndSession}>
                    ⏹ End Session
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      <main className="content" role="tabpanel">
        {tab === 'ingest' ? (
          <IngestTab token={token} />
        ) : (
          <AskTab
            activeSessionId={activeSessionId}
            token={token}
            history={history}
            setHistory={setHistory}
          />
        )}
      </main>

      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onSuccess={(tok, uname) => {
          setToken(tok)
          setUsername(uname)
        }}
      />

      <MemoriesModal
        isOpen={memoryModalOpen}
        onClose={() => setMemoryModalOpen(false)}
        memoryData={memoryData}
      />
    </div>
  )
}
