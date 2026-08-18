import React from 'react'

export default function MemoriesModal({ isOpen, onClose, memoryData }) {
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
            <h4 style={{ fontSize: 14, color: 'var(--accent-secondary)', marginBottom: 8 }}>📖 Episodic Summary &amp; Key Events</h4>
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
            <h4 style={{ fontSize: 14, color: 'var(--accent-cyan)', marginBottom: 8 }}>💡 Learned Semantic Facts &amp; Preferences</h4>
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
