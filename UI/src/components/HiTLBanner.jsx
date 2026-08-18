import React, { useState } from 'react'

/**
 * HiTLBanner — shown in the chat area when the agent requests user approval (e.g. for sending emails).
 * Provides a user-friendly email dispatch preview with inline editing, approve, and reject capabilities.
 */
export default function HiTLBanner({ sessionId, actionRequests, reviewConfigs, token, onResolved }) {
  const [editedEmail, setEditedEmail] = useState({})
  const [isEditingEmail, setIsEditingEmail] = useState({})
  const [editedArgs, setEditedArgs] = useState({})
  const [editingIdx, setEditingIdx] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  if (!actionRequests || actionRequests.length === 0) return null

  const configs = reviewConfigs || []

  const getAllowedDecisions = (idx) => {
    const cfg = configs[idx]
    if (!cfg) return ['approve', 'edit', 'reject', 'respond']
    return cfg.allowed_decisions || ['approve', 'edit', 'reject', 'respond']
  }

  const buildDecisions = (type, idx, message, customArgs = null) => {
    return actionRequests.map((req, i) => {
      if (i !== idx) return { type: 'approve' }
      const base = { type }
      if (message) base.message = message

      if (type === 'edit') {
        if (customArgs) {
          base.edited_action = { name: req.name, args: customArgs }
        } else {
          const argsStr = editedArgs[i] || JSON.stringify(req.arguments || {}, null, 2)
          try {
            base.edited_action = { name: req.name, args: JSON.parse(argsStr) }
          } catch {
            base.edited_action = { name: req.name, args: req.arguments || {} }
          }
        }
      }
      return base
    })
  }

  const submit = async (decisions) => {
    setSubmitting(true)
    setError('')
    try {
      const res = await fetch(`/api/sessions/${sessionId}/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ decisions }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Decision submission failed')
      }
      onResolved(res)
    } catch (err) {
      setError(err.message)
      setSubmitting(false)
    }
  }

  return (
    <div className="hitl-banner" role="alert" aria-label="Action approval required">
      <div className="hitl-banner-header">
        <span className="hitl-banner-icon">🛡️</span>
        <span className="hitl-banner-title">Action Requires Your Confirmation</span>
      </div>

      {actionRequests.map((req, idx) => {
        const isEmailAction = req.name === 'send_email' || req.name === 'gmail_send_email'
        const allowed = getAllowedDecisions(idx)
        const args = req.arguments || {}

        // Friendly Email UI
        if (isEmailAction) {
          const emailTo = (editedEmail[idx]?.to !== undefined ? editedEmail[idx].to : args.to) || ''
          const emailSubj = (editedEmail[idx]?.subject !== undefined ? editedEmail[idx].subject : args.subject) || ''
          const emailBody = (editedEmail[idx]?.body !== undefined ? editedEmail[idx].body : args.body) || ''
          const isEditing = isEditingEmail[idx]

          const handleSend = () => {
            if (isEditing || editedEmail[idx]) {
              const finalArgs = {
                to: emailTo,
                subject: emailSubj,
                body: emailBody,
              }
              submit(buildDecisions('edit', idx, null, finalArgs))
            } else {
              submit(buildDecisions('approve', idx))
            }
          }

          return (
            <div key={idx} className="hitl-email-card">
              <div className="hitl-email-card-header">
                <span className="hitl-email-icon">✉️</span>
                <span className="hitl-email-card-title">Send Email Confirmation</span>
                <span className="hitl-email-badge">Gmail API</span>
              </div>

              <div className="hitl-email-fields">
                <div className="hitl-email-row">
                  <label className="hitl-email-label">To:</label>
                  {isEditing ? (
                    <input
                      type="email"
                      className="hitl-email-input"
                      value={emailTo}
                      onChange={(e) => setEditedEmail((prev) => ({
                        ...prev,
                        [idx]: { ...prev[idx], to: e.target.value, subject: emailSubj, body: emailBody }
                      }))}
                      placeholder="recipient@example.com"
                    />
                  ) : (
                    <div className="hitl-email-chip">{emailTo || 'No recipient specified'}</div>
                  )}
                </div>

                <div className="hitl-email-row">
                  <label className="hitl-email-label">Subject:</label>
                  {isEditing ? (
                    <input
                      type="text"
                      className="hitl-email-input"
                      value={emailSubj}
                      onChange={(e) => setEditedEmail((prev) => ({
                        ...prev,
                        [idx]: { ...prev[idx], subject: e.target.value, to: emailTo, body: emailBody }
                      }))}
                      placeholder="Email subject"
                    />
                  ) : (
                    <div className="hitl-email-value-subject">{emailSubj || '(No Subject)'}</div>
                  )}
                </div>

                <div className="hitl-email-body-section">
                  <label className="hitl-email-label">Message:</label>
                  {isEditing ? (
                    <textarea
                      className="hitl-email-textarea"
                      rows={5}
                      value={emailBody}
                      onChange={(e) => setEditedEmail((prev) => ({
                        ...prev,
                        [idx]: { ...prev[idx], body: e.target.value, to: emailTo, subject: emailSubj }
                      }))}
                      placeholder="Email content..."
                    />
                  ) : (
                    <div className="hitl-email-body-preview">{emailBody || '(Empty message)'}</div>
                  )}
                </div>
              </div>

              <div className="hitl-action-btns">
                <button
                  id={`btn-hitl-send-email-${idx}`}
                  className="hitl-btn hitl-btn-send-email"
                  disabled={submitting || !emailTo}
                  onClick={handleSend}
                >
                  🚀 Approve & Send Email
                </button>

                <button
                  id={`btn-hitl-toggle-edit-${idx}`}
                  className="hitl-btn hitl-btn-edit"
                  disabled={submitting}
                  onClick={() => setIsEditingEmail((prev) => ({ ...prev, [idx]: !prev[idx] }))}
                >
                  {isEditing ? '💾 Done Editing' : '✏ Edit Draft'}
                </button>

                <button
                  id={`btn-hitl-reject-email-${idx}`}
                  className="hitl-btn hitl-btn-reject"
                  disabled={submitting}
                  onClick={() => submit(buildDecisions('reject', idx, 'User cancelled sending the email.'))}
                >
                  ✗ Cancel
                </button>
              </div>
            </div>
          )
        }

        // Generic Action UI
        const argsJson = JSON.stringify(args, null, 2)
        const isEditing = editingIdx === idx

        return (
          <div key={idx} className="hitl-action-card">
            <div className="hitl-action-name">
              🔧 <strong>{req.name}</strong>
            </div>

            {req.description && (
              <div className="hitl-action-desc">{req.description}</div>
            )}

            <div className="hitl-action-args">
              {isEditing ? (
                <textarea
                  className="hitl-args-editor"
                  rows={Math.min(10, argsJson.split('\n').length + 2)}
                  defaultValue={editedArgs[idx] || argsJson}
                  onChange={(e) => setEditedArgs((prev) => ({ ...prev, [idx]: e.target.value }))}
                />
              ) : (
                <pre className="hitl-args-pre">{argsJson}</pre>
              )}
            </div>

            <div className="hitl-action-btns">
              {allowed.includes('approve') && (
                <button
                  id={`btn-hitl-approve-${idx}`}
                  className="hitl-btn hitl-btn-approve"
                  disabled={submitting}
                  onClick={() => submit(buildDecisions('approve', idx))}
                >
                  ✓ Approve
                </button>
              )}

              {allowed.includes('edit') && (
                isEditing ? (
                  <button
                    id={`btn-hitl-edit-submit-${idx}`}
                    className="hitl-btn hitl-btn-edit"
                    disabled={submitting}
                    onClick={() => {
                      setEditingIdx(null)
                      submit(buildDecisions('edit', idx))
                    }}
                  >
                    ✏ Submit Edit
                  </button>
                ) : (
                  <button
                    id={`btn-hitl-edit-${idx}`}
                    className="hitl-btn hitl-btn-edit"
                    disabled={submitting}
                    onClick={() => setEditingIdx(idx)}
                  >
                    ✏ Edit
                  </button>
                )
              )}

              {allowed.includes('reject') && (
                <button
                  id={`btn-hitl-reject-${idx}`}
                  className="hitl-btn hitl-btn-reject"
                  disabled={submitting}
                  onClick={() => submit(buildDecisions('reject', idx, 'User rejected this action.'))}
                >
                  ✗ Reject
                </button>
              )}
            </div>
          </div>
        )
      })}

      {error && <div className="hitl-error">{error}</div>}
      {submitting && <div className="hitl-loading">Processing your request…</div>}
    </div>
  )
}
