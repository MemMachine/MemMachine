import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { api } from '../api';

const MailIcon = () => (
  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
    <polyline points="22,6 12,13 2,6"></polyline>
  </svg>
);

const CheckIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"></polyline>
  </svg>
);

const AlertIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="8" x2="12" y2="12"></line>
    <line x1="12" y1="16" x2="12.01" y2="16"></line>
  </svg>
);

const VerifyEmailPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const email = location.state?.email || '';
  
  const [isResending, setIsResending] = useState(false);
  const [resendMessage, setResendMessage] = useState('');
  const [resendError, setResendError] = useState('');

  const handleResendEmail = async () => {
    if (!email) {
      setResendError('Email not found. Please sign up again.');
      return;
    }

    setIsResending(true);
    setResendMessage('');
    setResendError('');

    try {
      await api.post('/auth/resend-verification', { email });
      setResendMessage('✅ Verification email sent! Check your inbox.');
      setResendError('');
    } catch (error) {
      console.error('Resend failed:', error);
      setResendError('Failed to resend email. Please try again.');
      setResendMessage('');
    } finally {
      setIsResending(false);
    }
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
              <MailIcon />
            </div>
            <h1 className="text-2xl font-bold gradient-text mb-2">
              Check Your Email
            </h1>
            <p className="text-[var(--text-muted)]">
              We've sent a verification link to:
            </p>
          </div>

          {/* Content */}
          <div className="p-8 space-y-6">
            {/* Email Display */}
            <div className="p-4 rounded-xl bg-gradient-to-r from-[var(--gradient-start)]/10 to-[var(--gradient-mid)]/5 border border-[var(--border-subtle)]">
              <p className="text-center text-[var(--gradient-mid)] font-semibold break-all">
                {email || 'your email address'}
              </p>
            </div>

            {/* Instructions */}
            <div className="p-4 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)]">
              <h2 className="font-semibold text-[var(--text-primary)] mb-3 flex items-center gap-2">
                <span className="text-[var(--gradient-mid)]">📱</span>
                Next Steps:
              </h2>
              <ol className="space-y-2 text-sm text-[var(--text-secondary)]">
                <li className="flex items-start gap-2">
                  <span className="font-bold text-[var(--gradient-mid)]">1.</span>
                  <span>Check your inbox (and spam folder)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-bold text-[var(--gradient-mid)]">2.</span>
                  <span>Click the verification link in the email</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-bold text-[var(--gradient-mid)]">3.</span>
                  <span>You can open the link on any device (phone, tablet, computer)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-bold text-[var(--gradient-mid)]">4.</span>
                  <span>Come back here and login!</span>
                </li>
              </ol>
            </div>

            {/* Important Note */}
            <div className="p-4 rounded-xl bg-amber-900/20 border border-amber-700/50">
              <p className="text-sm text-amber-400 flex items-start gap-2">
                <AlertIcon className="flex-shrink-0 mt-0.5" />
                <span>
                  <strong>Link expires in 24 hours.</strong> If expired, use the resend button below.
                </span>
              </p>
            </div>

            {/* Success/Error Messages */}
            {resendMessage && (
              <div className="p-4 rounded-xl bg-green-900/20 border border-green-700/50">
                <p className="text-sm text-green-400 flex items-center gap-2">
                  <CheckIcon />
                  {resendMessage}
                </p>
              </div>
            )}

            {resendError && (
              <div className="p-4 rounded-xl bg-red-900/20 border border-red-700/50">
                <p className="text-sm text-red-400">{resendError}</p>
              </div>
            )}

            {/* Resend Section */}
            <div className="space-y-3">
              <p className="text-center text-sm text-[var(--text-muted)]">
                Didn't receive the email?
              </p>

              <button
                onClick={handleResendEmail}
                disabled={isResending}
                className="w-full btn-primary py-4 rounded-xl font-semibold text-base"
              >
                {isResending ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" strokeOpacity="0.25"></circle>
                      <path d="M12 2a10 10 0 0 1 10 10" strokeOpacity="0.75"></path>
                    </svg>
                    Sending...
                  </span>
                ) : (
                  '🔄 Resend Verification Email'
                )}
              </button>

              {/* Back to Login */}
              <button
                onClick={() => navigate('/login')}
                className="w-full py-3 px-4 bg-transparent border-2 border-[var(--gradient-mid)] text-[var(--gradient-mid)] rounded-xl font-semibold hover:bg-[var(--gradient-mid)]/10 transition-all"
              >
                ← Back to Login
              </button>
            </div>
          </div>
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
};

export default VerifyEmailPage;