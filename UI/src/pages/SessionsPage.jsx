import React from 'react'

export default function SessionsPage({ sessions, activeSessionId, onSelectSession, onCreateSession, onDeleteSession }) {
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
            const sid = s.session_id || s.id
            const isActive = sid === activeSessionId
            const createdStr = s.created_at ? new Date(s.created_at).toLocaleString() : 'Recent'
            const fileCount = s.files?.length || 0

            return (
              <div
                key={sid}
                className={`session-card glass ${isActive ? 'active' : ''}`}
                onClick={() => onSelectSession(sid)}
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
                      onSelectSession(sid)
                    }}
                  >
                    Continue Chat →
                  </button>
                  <button
                    className="file-action-btn delete"
                    style={{ padding: '6px 10px' }}
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteSession && onDeleteSession(sid)
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
