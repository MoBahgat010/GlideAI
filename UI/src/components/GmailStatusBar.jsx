import React, { useState, useEffect, useCallback, useRef } from 'react'

/**
 * GmailStatusBar — shows Gmail connection status in the header / sidebar.
 * Clicking "Connect Gmail" redirects to /api/gmail/connect with the active JWT token.
 */
export default function GmailStatusBar({ token }) {
  const [status, setStatus] = useState(null)
  const loadingRef = useRef(false)

  const activeToken = token || localStorage.getItem('graphrag_access_token') || ''

  const checkStatus = useCallback(async () => {
    if (!activeToken || loadingRef.current) return
    loadingRef.current = true
    try {
      const res = await fetch('/api/gmail/status', {
        headers: { Authorization: `Bearer ${activeToken}` },
      })
      if (res.ok) {
        const data = await res.json()
        setStatus(data)
      }
    } catch {
      // silently fail
    } finally {
      loadingRef.current = false
    }
  }, [activeToken])

  // Check status once on mount or when activeToken changes
  useEffect(() => {
    checkStatus()
  }, [checkStatus])

  const handleConnect = () => {
    const t = token || localStorage.getItem('graphrag_access_token') || ''
    if (!t) {
      alert('Please sign in first before connecting your Gmail account.')
      return
    }
    window.location.href = `/api/gmail/connect?token=${encodeURIComponent(t)}`
  }

  if (!activeToken) return null

  return (
    <div className="gmail-status-bar">
      {status?.connected ? (
        <div className="service-badge connected" title={status.email || 'Gmail connected'}>
          <span className="service-badge-dot">●</span>
          <span>Gmail connected</span>
          {status.email && <span className="service-badge-email">{status.email}</span>}
        </div>
      ) : (
        <button
          id="btn-connect-gmail"
          className="connect-service-btn"
          onClick={handleConnect}
          title="Connect your Gmail account to enable email tools"
        >
          <span>✉</span> Connect Gmail
        </button>
      )}
    </div>
  )
}
