import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';

const LockIcon = () => (
  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
  </svg>
);

const MailIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
    <polyline points="22,6 12,13 2,6"></polyline>
  </svg>
);

const KeyIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"></path>
  </svg>
);

const AlertIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="8" x2="12" y2="12"></line>
    <line x1="12" y1="16" x2="12.01" y2="16"></line>
  </svg>
);

export default function Login({ onLoggedIn }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [needsVerification, setNeedsVerification] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    setNeedsVerification(false);
    setLoading(true);

    try {
      const form = new FormData();
      form.append('username', email);
      form.append('password', password);
      const res = await api.post('/auth/login', form);
      onLoggedIn(res.data.access_token);
    } catch (e) {
      const errorDetail = e.response?.data?.detail || 'Login failed. Please check your credentials.';
      
      // Check if it's an email verification error
      if (e.response?.status === 403 || errorDetail.toLowerCase().includes('verify your email')) {
        setNeedsVerification(true);
        setError(errorDetail);
      } else {
        setError(errorDetail);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResendVerification = () => {
    navigate('/verify-email', { state: { email } });
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      {/* Background Glow Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[var(--gradient-start)] rounded-full filter blur-[128px] opacity-20"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[var(--gradient-mid)] rounded-full filter blur-[128px] opacity-15"></div>
      </div>

      <div className="relative max-w-md w-full">
        {/* Main Card */}
        <div className="glass-strong rounded-2xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="p-8 text-center border-b border-[var(--border-subtle)]">
            <div className="w-20 h-20 mx-auto rounded-2xl bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-mid)] flex items-center justify-center mb-6 shadow-lg glow">
              <LockIcon />
            </div>
            <h1 className="text-2xl font-bold gradient-text mb-2">
              Welcome Back
            </h1>
            <p className="text-[var(--text-muted)]">
              Sign in to continue to HippoSync
            </p>
          </div>

          {/* Form */}
          <form onSubmit={submit} className="p-8 space-y-6">
            {/* Email Input */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--text-secondary)]">
                Email Address
              </label>
              <div className="relative">
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)]">
                  <MailIcon />
                </div>
                <input
                  type="email"
                  className="w-full pl-12 pr-4 py-3.5 rounded-xl input-field text-sm"
                  placeholder="your@email.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                  disabled={loading}
                />
              </div>
            </div>

            {/* Password Input */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--text-secondary)]">
                Password
              </label>
              <div className="relative">
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)]">
                  <KeyIcon />
                </div>
                <input
                  type="password"
                  className="w-full pl-12 pr-4 py-3.5 rounded-xl input-field text-sm"
                  placeholder="Enter your password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  disabled={loading}
                />
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-4 rounded-xl bg-red-900/20 border border-red-700/50">
                <p className="text-sm text-red-400 flex items-start gap-2">
                  <AlertIcon className="flex-shrink-0 mt-0.5" />
                  <span>{error}</span>
                </p>
                
                {/* Resend Verification Button */}
                {needsVerification && email && (
                  <button
                    type="button"
                    onClick={handleResendVerification}
                    className="mt-3 text-sm text-[var(--gradient-mid)] hover:text-[var(--gradient-end)] font-medium underline"
                  >
                    → Resend verification email
                  </button>
                )}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full btn-primary py-4 rounded-xl font-semibold text-base flex items-center justify-center gap-2"
              disabled={loading}
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" strokeOpacity="0.25"></circle>
                    <path d="M12 2a10 10 0 0 1 10 10" strokeOpacity="0.75"></path>
                  </svg>
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </button>

            {/* Sign Up Link */}
            <div className="text-center text-sm text-[var(--text-muted)]">
              Don't have an account?{' '}
              <button
                type="button"
                onClick={() => navigate('/signup')}
                className="text-[var(--gradient-mid)] hover:text-[var(--gradient-end)] font-medium transition-colors"
                disabled={loading}
              >
                Create Account
              </button>
            </div>
          </form>
        </div>

        {/* Powered by MemMachine Badge */}
        <div className="mt-6 flex items-center justify-center gap-2">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-[var(--gradient-mid)]">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M12 1v4"></path>
            <path d="M12 19v4"></path>
            <path d="M4.22 4.22l2.83 2.83"></path>
            <path d="M16.95 16.95l2.83 2.83"></path>
            <path d="M1 12h4"></path>
            <path d="M19 12h4"></path>
            <path d="M4.22 19.78l2.83-2.83"></path>
            <path d="M16.95 7.05l2.83-2.83"></path>
          </svg>
          <span className="text-xs text-[var(--text-muted)]">
            Powered by <span className="text-[var(--gradient-mid)] font-medium">MemMachine</span>
          </span>
        </div>
      </div>
    </div>
  );
}