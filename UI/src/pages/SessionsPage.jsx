import React from 'react'

export default function SessionsPage({ sessions, activeSessionId, onSelectSession, onCreateSession, onDeleteSession }) {
  return (
    <div className="sessions-history-container">
      <div className="sessions-history-header">
        <div>
          <h2 className="sessions-history-title">🗂 Conversation Sessions</h2>
          <p className="sessions-history-desc">
            Select a session to start or continue your conversation with isolated documents and memory.
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
            const createdStr = s.created_at ? new Date(s.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'Recent'

            return (
              <div
                key={sid}
                className={`session-card glass clickable-session-card ${isActive ? 'active' : ''}`}
                onClick={() => onSelectSession(sid)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    onSelectSession(sid)
                  }
                }}
              >
                <div className="session-card-header">
                  <div className="session-card-title-wrap">
                    <span className="session-card-icon">💬</span>
                    <div className="session-card-title" title={s.title || 'Conversation Session'}>
                      {s.title || 'Conversation Session'}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="session-delete-btn"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteSession && onDeleteSession(sid)
                    }}
                    title="Delete Session"
                    aria-label="Delete Session"
                  >
                    🗑️
                  </button>
                </div>

                <div className="session-card-footer">
                  <span className="session-card-date">🕒 {createdStr}</span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
