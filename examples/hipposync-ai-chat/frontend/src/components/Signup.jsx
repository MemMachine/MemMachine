import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';

// ============================================================================
// ICONS
// ============================================================================

const UserPlusIcon = () => (
  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
    <circle cx="8.5" cy="7" r="4"></circle>
    <line x1="20" y1="8" x2="20" y2="14"></line>
    <line x1="23" y1="11" x2="17" y2="11"></line>
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

const UserIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
    <circle cx="12" cy="7" r="4"></circle>
  </svg>
);

const BriefcaseIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
    <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>
  </svg>
);

const MapPinIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
    <circle cx="12" cy="10" r="3"></circle>
  </svg>
);

const CheckIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"></polyline>
  </svg>
);

const XIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const AlertIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="8" x2="12" y2="12"></line>
    <line x1="12" y1="16" x2="12.01" y2="16"></line>
  </svg>
);

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

const DISPOSABLE_DOMAINS = new Set([
  'tempmail.com', 'throwaway.email', 'guerrillamail.com', 'mailinator.com',
  '10minutemail.com', 'trashmail.com', 'fakeinbox.com', 'temp-mail.org',
  'yopmail.com', 'maildrop.cc', 'getnada.com', 'sharklasers.com'
]);

const COMMON_PASSWORDS = new Set([
  'password', '123456', '12345678', 'qwerty', 'abc123', 'letmein',
  'password1', '123456789', '12345', 'password123', 'admin', 'welcome'
]);

/**
 * Validate email format and domain
 */
const validateEmail = (email) => {
  const errors = [];
  
  if (!email) {
    return { isValid: false, errors: ['Email is required'] };
  }
  
  // Basic format check
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    errors.push('Invalid email format');
    return { isValid: false, errors };
  }
  
  // Check domain
  const domain = email.split('@')[1]?.toLowerCase();
  if (!domain) {
    errors.push('Invalid email domain');
    return { isValid: false, errors };
  }
  
  // Block disposable emails
  if (DISPOSABLE_DOMAINS.has(domain)) {
    errors.push('Disposable email addresses are not allowed. Please use a legitimate email.');
  }
  
  // Block test domains
  if (domain.includes('test') || domain.includes('example') || domain.includes('temp')) {
    errors.push('Test or temporary email domains are not allowed');
  }
  
  // Domain should have at least one dot
  if (!domain.includes('.')) {
    errors.push('Email domain is invalid');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * Validate password strength with comprehensive checks
 */
const validatePassword = (password) => {
  const checks = {
    length: { passed: false, label: 'At least 8 characters', error: '' },
    uppercase: { passed: false, label: 'One uppercase letter', error: '' },
    lowercase: { passed: false, label: 'One lowercase letter', error: '' },
    number: { passed: false, label: 'One number', error: '' },
    special: { passed: false, label: 'One special character (!@#$...)', error: '' },
    common: { passed: true, label: 'Not a common password', error: '' },
    sequential: { passed: true, label: 'No sequential characters', error: '' },
    repeated: { passed: true, label: 'No repeated characters', error: '' }
  };
  
  if (!password) {
    return { isValid: false, checks, errors: ['Password is required'] };
  }
  
  // Length check
  checks.length.passed = password.length >= 8;
  if (!checks.length.passed) {
    checks.length.error = 'Password must be at least 8 characters';
  }
  
  // Complexity checks
  checks.uppercase.passed = /[A-Z]/.test(password);
  if (!checks.uppercase.passed) {
    checks.uppercase.error = 'Add at least one uppercase letter (A-Z)';
  }
  
  checks.lowercase.passed = /[a-z]/.test(password);
  if (!checks.lowercase.passed) {
    checks.lowercase.error = 'Add at least one lowercase letter (a-z)';
  }
  
  checks.number.passed = /\d/.test(password);
  if (!checks.number.passed) {
    checks.number.error = 'Add at least one number (0-9)';
  }
  
  checks.special.passed = /[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\\/;`~]/.test(password);
  if (!checks.special.passed) {
    checks.special.error = 'Add at least one special character';
  }
  
  // Common password check
  checks.common.passed = !COMMON_PASSWORDS.has(password.toLowerCase());
  if (!checks.common.passed) {
    checks.common.error = 'This password is too common';
  }
  
  // Sequential characters check
  const sequences = ['123', '234', '345', '456', '567', '678', '789', 'abc', 'bcd', 'cde'];
  checks.sequential.passed = !sequences.some(seq => password.toLowerCase().includes(seq));
  if (!checks.sequential.passed) {
    checks.sequential.error = 'Avoid sequential characters like 123 or abc';
  }
  
  // Repeated characters check (e.g., "aaa", "111")
  checks.repeated.passed = !/(.)\1{2,}/.test(password);
  if (!checks.repeated.passed) {
    checks.repeated.error = 'Avoid repeating the same character 3+ times';
  }
  
  const errors = Object.values(checks)
    .filter(check => !check.passed && check.error)
    .map(check => check.error);
  
  return {
    isValid: errors.length === 0,
    checks,
    errors
  };
};

// ============================================================================
// SIGNUP COMPONENT
// ============================================================================

export default function Signup({ onSignedUp }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [occupation, setOccupation] = useState('');
  const [allowLocation, setAllowLocation] = useState(true);
  const [location, setLocation] = useState(null);
  const [locationError, setLocationError] = useState('');
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  
  // Validation states
  const [emailValidation, setEmailValidation] = useState({ isValid: null, errors: [] });
  const [passwordValidation, setPasswordValidation] = useState({ isValid: null, checks: {}, errors: [] });
  const [showPasswordRequirements, setShowPasswordRequirements] = useState(false);
  const [touched, setTouched] = useState({ email: false, password: false });

  // Real-time email validation
  useEffect(() => {
    if (touched.email && email) {
      const validation = validateEmail(email);
      setEmailValidation(validation);
    }
  }, [email, touched.email]);

  // Real-time password validation
  useEffect(() => {
    if (touched.password && password) {
      const validation = validatePassword(password);
      setPasswordValidation(validation);
    }
  }, [password, touched.password]);

  // Request location permission
  const requestLocation = () => {
    if (!navigator.geolocation) {
      setLocationError('Geolocation not supported by your browser');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        
        try {
          const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`,
            { headers: { 'User-Agent': 'HippoSync/1.0' } }
          );
          const data = await response.json();
          
          const addr = data.address || {};
          const city = addr.city || addr.town || addr.village || 'Unknown';
          const state = addr.state || '';
          const country = addr.country || '';
          const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
          
          setLocation({
            city, state, country,
            latitude: String(latitude),
            longitude: String(longitude),
            timezone,
            formatted: `${city}${state ? ', ' + state : ''}${country ? ', ' + country : ''}`,
          });
          setLocationError('');
        } catch (err) {
          console.error('Geocoding error:', err);
          setLocationError('Could not determine location details');
        }
      },
      (error) => {
        setLocationError('Location access denied. You can set it later in settings.');
      }
    );
  };

  useEffect(() => {
    if (allowLocation && !location && !locationError) {
      requestLocation();
    }
  }, [allowLocation]);

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    
    // Mark all fields as touched
    setTouched({ email: true, password: true });
    
    // Final validation
    const emailVal = validateEmail(email);
    const passwordVal = validatePassword(password);
    
    setEmailValidation(emailVal);
    setPasswordValidation(passwordVal);
    
    if (!emailVal.isValid || !passwordVal.isValid) {
      setError('Please fix the validation errors before submitting');
      return;
    }
    
    setLoading(true);

    try {
      const payload = { email, password };
      
      if (name) payload.name = name;
      if (occupation) payload.occupation = occupation;
      
      if (location) {
        payload.location_city = location.city;
        payload.location_state = location.state;
        payload.location_country = location.country;
        payload.location_latitude = location.latitude;
        payload.location_longitude = location.longitude;
        payload.location_timezone = location.timezone;
        payload.location_formatted = location.formatted;
      }

      await api.post('/auth/signup', payload);
      setSuccess(true);
      
      // Clear any old tokens before verification flow
      localStorage.removeItem('token');
      sessionStorage.removeItem('token');
      
      // ✅ UPDATED: Redirect to verify email page
      setTimeout(() => {
        navigate('/verify-email', { state: { email } });
      }, 1500);
    } catch (e) {
      const errorMsg = e.response?.data?.detail || 'Signup failed. Please try again.';
      setError(errorMsg);
    } finally {
      setLoading(false);
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
        <div className="glass-strong rounded-2xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="p-8 text-center border-b border-[var(--border-subtle)]">
            <div className="w-20 h-20 mx-auto rounded-2xl bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-mid)] flex items-center justify-center mb-6 shadow-lg glow">
              <UserPlusIcon />
            </div>
            <h1 className="text-2xl font-bold gradient-text mb-2">Create Account</h1>
            <p className="text-[var(--text-muted)]">Get started with HippoSync</p>
          </div>

          {/* Form */}
          <form onSubmit={submit} className="p-8 space-y-5">
            {/* Name Input (Optional) */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--text-secondary)]">
                Name <span className="text-[var(--text-muted)] font-normal">(Optional)</span>
              </label>
              <div className="relative">
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)]">
                  <UserIcon />
                </div>
                <input
                  type="text"
                  className="w-full pl-12 pr-4 py-3 rounded-xl input-field text-sm"
                  placeholder="Your name"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  disabled={loading}
                  maxLength={100}
                />
              </div>
            </div>

            {/* Email Input with Validation */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--text-secondary)]">
                Email Address *
              </label>
              <div className="relative">
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)]">
                  <MailIcon />
                </div>
                <input
                  type="email"
                  className={`w-full pl-12 pr-12 py-3 rounded-xl input-field text-sm ${
                    touched.email && !emailValidation.isValid ? 'border-red-500' : ''
                  } ${touched.email && emailValidation.isValid ? 'border-green-500' : ''}`}
                  placeholder="your@email.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  onBlur={() => setTouched(prev => ({ ...prev, email: true }))}
                  required
                  disabled={loading}
                />
                {touched.email && emailValidation.isValid !== null && (
                  <div className="absolute right-4 top-1/2 -translate-y-1/2">
                    {emailValidation.isValid ? (
                      <div className="text-green-500"><CheckIcon /></div>
                    ) : (
                      <div className="text-red-500"><XIcon /></div>
                    )}
                  </div>
                )}
              </div>
              {touched.email && !emailValidation.isValid && emailValidation.errors.length > 0 && (
                <div className="space-y-1">
                  {emailValidation.errors.map((err, i) => (
                    <p key={i} className="text-xs text-red-400 flex items-start gap-2">
                      <AlertIcon className="flex-shrink-0 mt-0.5" />
                      <span>{err}</span>
                    </p>
                  ))}
                </div>
              )}
            </div>

            {/* Password Input with Validation */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--text-secondary)]">
                Password *
              </label>
              <div className="relative">
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)]">
                  <KeyIcon />
                </div>
                <input
                  type="password"
                  className={`w-full pl-12 pr-12 py-3 rounded-xl input-field text-sm ${
                    touched.password && !passwordValidation.isValid ? 'border-red-500' : ''
                  } ${touched.password && passwordValidation.isValid ? 'border-green-500' : ''}`}
                  placeholder="Create a strong password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  onFocus={() => setShowPasswordRequirements(true)}
                  onBlur={() => {
                    setTouched(prev => ({ ...prev, password: true }));
                    setShowPasswordRequirements(false);
                  }}
                  required
                  disabled={loading}
                  maxLength={128}
                />
                {touched.password && passwordValidation.isValid !== null && (
                  <div className="absolute right-4 top-1/2 -translate-y-1/2">
                    {passwordValidation.isValid ? (
                      <div className="text-green-500"><CheckIcon /></div>
                    ) : (
                      <div className="text-red-500"><XIcon /></div>
                    )}
                  </div>
                )}
              </div>
              
              {/* Password Requirements */}
              {(showPasswordRequirements || (touched.password && !passwordValidation.isValid)) && (
                <div className="p-4 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)] space-y-2">
                  <p className="text-xs font-medium text-[var(--text-secondary)] mb-2">Password Requirements:</p>
                  {Object.entries(passwordValidation.checks).map(([key, check]) => (
                    <div key={key} className="flex items-center gap-2 text-xs">
                      <div className={check.passed ? 'text-green-500' : 'text-[var(--text-muted)]'}>
                        {check.passed ? <CheckIcon /> : <XIcon />}
                      </div>
                      <span className={check.passed ? 'text-green-500' : 'text-[var(--text-secondary)]'}>
                        {check.label}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Occupation Input (Optional) */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--text-secondary)]">
                Occupation <span className="text-[var(--text-muted)] font-normal">(Optional)</span>
              </label>
              <div className="relative">
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)]">
                  <BriefcaseIcon />
                </div>
                <input
                  type="text"
                  className="w-full pl-12 pr-4 py-3 rounded-xl input-field text-sm"
                  placeholder="e.g., ML Engineer, Student"
                  value={occupation}
                  onChange={e => setOccupation(e.target.value)}
                  disabled={loading}
                  maxLength={100}
                />
              </div>
            </div>

            {/* Location Permission */}
            <div className="p-4 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)]">
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={allowLocation}
                  onChange={(e) => setAllowLocation(e.target.checked)}
                  className="mt-1 w-4 h-4 accent-[var(--gradient-mid)]"
                  disabled={loading}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2 text-sm font-medium text-[var(--text-primary)]">
                    <MapPinIcon />
                    Allow location access
                  </div>
                  <p className="text-xs text-[var(--text-muted)] mt-1">
                    Used for weather, time, and local recommendations
                  </p>
                </div>
              </label>

              {allowLocation && location && (
                <div className="mt-3 pt-3 border-t border-[var(--border-subtle)]">
                  <div className="flex items-start gap-2 text-sm">
                    <div className="text-[var(--accent-success)] mt-0.5">
                      <CheckIcon />
                    </div>
                    <div className="flex-1">
                      <p className="text-[var(--accent-success)] font-medium">Location detected</p>
                      <p className="text-[var(--text-muted)] text-xs mt-1">{location.formatted}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-4 rounded-xl bg-red-900/20 border border-red-700/50">
                <p className="text-sm text-red-400">{error}</p>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div className="p-4 rounded-xl bg-green-900/20 border border-green-700/50">
                <p className="text-sm text-green-400 flex items-center gap-2">
                  <CheckIcon />
                  Account created successfully! Redirecting...
                </p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full btn-primary py-4 rounded-xl font-semibold text-base"
              disabled={loading || success}
            >
              {loading ? 'Creating Account...' : success ? 'Success! ✓' : 'Create Account'}
            </button>

            {/* Sign In Link */}
            <div className="text-center text-sm text-[var(--text-muted)]">
              Already have an account?{' '}
              <button
                type="button"
                onClick={() => navigate('/login')}
                className="text-[var(--gradient-mid)] hover:text-[var(--gradient-end)] font-medium transition-colors"
                disabled={loading}
              >
                Sign in
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}