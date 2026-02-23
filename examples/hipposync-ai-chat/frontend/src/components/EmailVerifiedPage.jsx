import React, { useEffect, useState, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { api } from '../api';

const CheckIcon = () => (
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"></polyline>
  </svg>
);

const XIcon = () => (
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const InfoIcon = () => (
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="16" x2="12" y2="12"></line>
    <line x1="12" y1="8" x2="12.01" y2="8"></line>
  </svg>
);

const EmailVerifiedPage = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get('token');
  const verificationAttempted = useRef(false);

  const [status, setStatus] = useState('verifying'); // verifying, success, error, already_verified
  const [message, setMessage] = useState('');
  const [email, setEmail] = useState('');
  const [countdown, setCountdown] = useState(3);

  useEffect(() => {
    const verifyEmail = async () => {
      // Prevent duplicate calls
      if (verificationAttempted.current) {
        return;
      }
      verificationAttempted.current = true;

      if (!token) {
        setStatus('error');
        setMessage('Invalid verification link. Token is missing.');
        return;
      }

      try {
        // Keep verifying state visible for at least 300ms
        const [response] = await Promise.all([
          api.get(`/auth/verify-email?token=${token}`),
          new Promise(resolve => setTimeout(resolve, 300))
        ]);

        if (response.data.status === 'success') {
          setStatus('success');
          setMessage(response.data.message);
          setEmail(response.data.email);
          
          // Auto-redirect after 3 seconds
          const timer = setInterval(() => {
            setCountdown((prev) => {
              if (prev <= 1) {
                clearInterval(timer);
                navigate('/login');
                return 0;
              }
              return prev - 1;
            });
          }, 1000);

          return () => clearInterval(timer);
        } else if (response.data.status === 'already_verified') {
          setStatus('already_verified');
          setMessage(response.data.message);
        } else {
          setStatus('error');
          setMessage('Unexpected response from server.');
        }
      } catch (error) {
        console.error('Verification failed:', error);
        setStatus('error');
        
        if (error.response?.data?.detail) {
          setMessage(error.response.data.detail);
        } else {
          setMessage('Verification failed. The link may be invalid or expired.');
        }
      }
    };

    verifyEmail();
  }, [token]); // Removed navigate from dependencies

  // Verifying State
  if (status === 'verifying') {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[var(--gradient-start)] rounded-full filter blur-[128px] opacity-20"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[var(--gradient-mid)] rounded-full filter blur-[128px] opacity-15"></div>
        </div>

        <div className="relative max-w-md w-full">
          <div className="glass-strong rounded-2xl shadow-2xl p-8 text-center">
            <div className="w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-mid)] flex items-center justify-center mb-6 animate-pulse glow">
              <svg className="w-10 h-10 text-white animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold gradient-text mb-2">Verifying Your Email...</h1>
            <p className="text-[var(--text-muted)]">Please wait a moment</p>
          </div>
        </div>
      </div>
    );
  }

  // Success State
  if (status === 'success') {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[var(--gradient-start)] rounded-full filter blur-[128px] opacity-20"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[var(--gradient-mid)] rounded-full filter blur-[128px] opacity-15"></div>
        </div>

        <div className="relative max-w-md w-full">
          <div className="glass-strong rounded-2xl shadow-2xl overflow-hidden">
            <div className="p-8 text-center border-b border-[var(--border-subtle)]">
              <div className="w-20 h-20 mx-auto rounded-full bg-green-500 flex items-center justify-center mb-6 animate-bounce">
                <CheckIcon />
              </div>
              <h1 className="text-2xl font-bold gradient-text mb-2">
                Email Verified! 🎉
              </h1>
              <p className="text-[var(--text-muted)]">
                {message}
              </p>
            </div>

            <div className="p-8 space-y-6">
              {/* Email Display */}
              {email && (
                <div className="p-4 rounded-xl bg-green-900/20 border border-green-700/50">
                  <p className="text-center text-green-400 font-semibold break-all">
                    {email}
                  </p>
                </div>
              )}

              {/* Success Message */}
              <div className="p-4 rounded-xl bg-gradient-to-r from-green-900/20 to-emerald-900/20 border border-green-700/50">
                <div className="flex items-start gap-3">
                  <svg className="w-6 h-6 text-green-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <div>
                    <p className="font-semibold text-green-400 mb-1">Your account is now active!</p>
                    <p className="text-sm text-green-500">You can now login and start using all features.</p>
                  </div>
                </div>
              </div>

              {/* Auto-redirect countdown */}
              <div className="p-4 rounded-xl bg-gradient-to-r from-[var(--gradient-start)]/10 to-[var(--gradient-mid)]/5 border border-[var(--border-subtle)]">
                <p className="text-sm text-[var(--text-secondary)] text-center">
                  Redirecting to login in <span className="font-bold text-xl text-[var(--gradient-mid)]">{countdown}</span> seconds...
                </p>
              </div>

              {/* Login Button */}
              <button
                onClick={() => navigate('/login')}
                className="w-full btn-primary py-4 rounded-xl font-semibold text-base"
              >
                Go to Login Now →
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Already Verified State
  if (status === 'already_verified') {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[var(--gradient-start)] rounded-full filter blur-[128px] opacity-20"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[var(--gradient-mid)] rounded-full filter blur-[128px] opacity-15"></div>
        </div>

        <div className="relative max-w-md w-full">
          <div className="glass-strong rounded-2xl shadow-2xl overflow-hidden">
            <div className="p-8 text-center border-b border-[var(--border-subtle)]">
              <div className="w-20 h-20 mx-auto rounded-full bg-blue-500 flex items-center justify-center mb-6">
                <InfoIcon />
              </div>
              <h1 className="text-2xl font-bold gradient-text mb-2">
                Already Verified
              </h1>
              <p className="text-[var(--text-muted)]">
                {message}
              </p>
            </div>

            <div className="p-8 space-y-6">
              {/* Info Box */}
              <div className="p-4 rounded-xl bg-blue-900/20 border border-blue-700/50">
                <p className="text-sm text-blue-400 text-center">
                  Your email is already verified. You can login to your account.
                </p>
              </div>

              {/* Login Button */}
              <button
                onClick={() => navigate('/login')}
                className="w-full btn-primary py-4 rounded-xl font-semibold text-base"
              >
                Go to Login →
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Error State
  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[var(--gradient-start)] rounded-full filter blur-[128px] opacity-20"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[var(--gradient-mid)] rounded-full filter blur-[128px] opacity-15"></div>
      </div>

      <div className="relative max-w-md w-full">
        <div className="glass-strong rounded-2xl shadow-2xl overflow-hidden">
          <div className="p-8 text-center border-b border-[var(--border-subtle)]">
            <div className="w-20 h-20 mx-auto rounded-full bg-red-500 flex items-center justify-center mb-6">
              <XIcon />
            </div>
            <h1 className="text-2xl font-bold gradient-text mb-2">
              Verification Failed
            </h1>
          </div>

          <div className="p-8 space-y-6">
            {/* Error Message */}
            <div className="p-4 rounded-xl bg-red-900/20 border border-red-700/50">
              <p className="text-sm text-red-400 text-center">
                {message}
              </p>
            </div>

            {/* Help Box */}
            <div className="p-4 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)]">
              <p className="text-sm text-[var(--text-primary)] mb-3 font-semibold">Possible reasons:</p>
              <ul className="space-y-2 text-sm text-[var(--text-secondary)]">
                <li className="flex items-start gap-2">
                  <span className="text-red-500">•</span>
                  <span>Link has expired (24 hours)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500">•</span>
                  <span>Link has already been used</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500">•</span>
                  <span>Invalid or corrupted link</span>
                </li>
              </ul>
            </div>

            {/* Action Buttons */}
            <div className="space-y-3">
              <button
                onClick={() => navigate('/verify-email')}
                className="w-full btn-primary py-4 rounded-xl font-semibold text-base"
              >
                Request New Verification Email
              </button>

              <button
                onClick={() => navigate('/signup')}
                className="w-full py-3 px-4 bg-transparent border-2 border-[var(--gradient-mid)] text-[var(--gradient-mid)] rounded-xl font-semibold hover:bg-[var(--gradient-mid)]/10 transition-all"
              >
                Sign Up Again
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmailVerifiedPage;