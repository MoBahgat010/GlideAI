import React, { useState, useEffect } from 'react'

/**
 * AuthPage — full-page sign in / register with Google OAuth support.
 * On mount, reads the URL hash for OAuth callback params (#access_token=...&refresh_token=...&username=...)
 * and auto-logs the user in if found.
 */
export default function AuthPage({ onAuthSuccess, initialTab = 'login' }) {
  const [tab, setTab] = useState(initialTab || 'login')   // 'login' | 'register'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (initialTab) setTab(initialTab)
  }, [initialTab])

  // Handle Google OAuth callback: tokens arrive in the URL hash
  useEffect(() => {
    const hash = window.location.hash
    if (!hash) return
    const params = new URLSearchParams(hash.slice(1))
    const accessToken = params.get('access_token')
    const refreshToken = params.get('refresh_token')
    const user = params.get('username')
    if (accessToken && user) {
      localStorage.setItem('graphrag_access_token', accessToken)
      if (refreshToken) localStorage.setItem('graphrag_refresh_token', refreshToken)
      localStorage.setItem('graphrag_username', user)
      window.history.replaceState(null, '', window.location.pathname)
      onAuthSuccess({ token: accessToken, username: user })
    }
  }, [onAuthSuccess])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const endpoint = tab === 'login' ? '/api/auth/login' : '/api/auth/register'
      const body = tab === 'login'
        ? { username, password }
        : { username, password, email }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Authentication failed')

      localStorage.setItem('graphrag_access_token', data.access_token)
      if (data.refresh_token) localStorage.setItem('graphrag_refresh_token', data.refresh_token)
      if (data.username || username) localStorage.setItem('graphrag_username', data.username || username)
      onAuthSuccess({ token: data.access_token, username: data.username || username })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleLogin = () => {
    window.location.href = '/api/auth/google/login'
  }

  return (
    <div className="auth-page">
      <div className="auth-card glass">
        {/* Logo / Brand */}
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>🤖</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>Enterprise RAG</div>
          <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>Sign in to your workspace</div>
        </div>

        {/* Tabs */}
        <div className="auth-tabs">
          <button
            id="auth-tab-login"
            className={`auth-tab-btn ${tab === 'login' ? 'active' : ''}`}
            onClick={() => { setTab('login'); setError('') }}
          >
            Sign In
          </button>
          <button
            id="auth-tab-register"
            className={`auth-tab-btn ${tab === 'register' ? 'active' : ''}`}
            onClick={() => { setTab('register'); setError('') }}
          >
            Register
          </button>
        </div>

        {/* Google Sign-In */}
        <button id="btn-google-signin" className="google-btn" onClick={handleGoogleLogin} type="button">
          <svg width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
            <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
            <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
            <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
            <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
          </svg>
          Continue with Google
        </button>

        <div className="oauth-divider"><span>or</span></div>

        {/* Credentials Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="auth-field">
            <label htmlFor="auth-username">Username</label>
            <input
              id="auth-username"
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
              autoComplete="username"
            />
          </div>

          {tab === 'register' && (
            <div className="auth-field">
              <label htmlFor="auth-email">Email</label>
              <input
                id="auth-email"
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
            </div>
          )}

          <div className="auth-field">
            <label htmlFor="auth-password">Password</label>
            <input
              id="auth-password"
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
            />
          </div>

          {error && <div className="auth-error">{error}</div>}

          <button
            id="btn-auth-submit"
            type="submit"
            className="btn btn-gradient"
            disabled={loading}
            style={{ width: '100%', padding: '12px', fontSize: 15, marginTop: 4 }}
          >
            {loading ? 'Please wait…' : tab === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  )
}
