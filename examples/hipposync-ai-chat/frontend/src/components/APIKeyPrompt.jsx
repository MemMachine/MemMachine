import React from 'react';

const KeyIcon = () => (
  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"></path>
  </svg>
);

const SettingsIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3"></circle>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
  </svg>
);

const InfoIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="16" x2="12" y2="12"></line>
    <line x1="12" y1="8" x2="12.01" y2="8"></line>
  </svg>
);

const ExternalLinkIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
    <polyline points="15 3 21 3 21 9"></polyline>
    <line x1="10" y1="14" x2="21" y2="3"></line>
  </svg>
);

const ShieldIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
  </svg>
);

/**
 * APIKeyPrompt - Shows when user tries to chat without API keys
 * Updated with purple gradient theme
 */
export default function APIKeyPrompt({ onOpenSettings }) {
  const providers = [
    { name: 'OpenAI', models: 'GPT-4o, GPT-4o-mini', color: 'var(--accent-success)' },
    { name: 'Anthropic', models: 'Claude Sonnet, Opus', color: 'var(--gradient-mid)' },
    { name: 'Google', models: 'Gemini Pro, Flash', color: 'var(--accent-info)' },
    { name: 'Tavily', models: 'Web Search (Optional)', color: 'var(--accent-warning)' },
  ];

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
              <KeyIcon />
            </div>
            <h1 className="text-2xl font-bold gradient-text mb-2">
              API Keys Required
            </h1>
            <p className="text-[var(--text-muted)]">
              To start chatting, you'll need to add at least one API key
            </p>
          </div>

          {/* Content */}
          <div className="p-6 space-y-5">
            {/* Why API Keys Info Box */}
            <div className="p-4 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)]">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-[var(--gradient-mid)]/20 flex items-center justify-center text-[var(--gradient-mid)] flex-shrink-0">
                  <InfoIcon />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-1">
                    Why do I need API keys?
                  </h3>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    HippoSync uses your own API keys to connect to AI models. This means you only pay for what you use and have full control over your data.
                  </p>
                </div>
              </div>
            </div>

            {/* Supported Providers */}
            <div>
              <h3 className="text-sm font-semibold text-[var(--text-secondary)] mb-3">
                Supported Providers
              </h3>
              <div className="space-y-2">
                {providers.map((provider) => (
                  <div 
                    key={provider.name}
                    className="flex items-center gap-3 p-3 rounded-xl bg-[var(--surface-dark)] border border-[var(--border-subtle)]"
                  >
                    <div 
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: provider.color }}
                    ></div>
                    <div className="flex-1 min-w-0">
                      <span className="text-sm font-medium text-[var(--text-primary)]">
                        {provider.name}
                      </span>
                      <span className="text-xs text-[var(--text-muted)] ml-2">
                        {provider.models}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Get Keys Links */}
            <div className="p-4 rounded-xl bg-gradient-to-r from-[var(--gradient-start)]/10 to-[var(--gradient-mid)]/5 border border-[var(--border-subtle)]">
              <p className="text-sm text-[var(--text-secondary)] mb-2">
                <strong className="text-[var(--gradient-mid)]">Don't have API keys yet?</strong>
              </p>
              <div className="flex flex-wrap gap-3">
                <a 
                  href="https://platform.openai.com/api-keys" 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="inline-flex items-center gap-1 text-xs text-[var(--text-muted)] hover:text-[var(--gradient-mid)] transition-colors"
                >
                  OpenAI <ExternalLinkIcon />
                </a>
                <a 
                  href="https://console.anthropic.com/settings/keys" 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="inline-flex items-center gap-1 text-xs text-[var(--text-muted)] hover:text-[var(--gradient-mid)] transition-colors"
                >
                  Anthropic <ExternalLinkIcon />
                </a>
                <a 
                  href="https://aistudio.google.com/app/apikey" 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="inline-flex items-center gap-1 text-xs text-[var(--text-muted)] hover:text-[var(--gradient-mid)] transition-colors"
                >
                  Google <ExternalLinkIcon />
                </a>
              </div>
            </div>
          </div>

          {/* Footer / Action */}
          <div className="p-6 border-t border-[var(--border-subtle)]">
            <button
              onClick={onOpenSettings}
              className="w-full btn-primary py-4 rounded-xl font-semibold flex items-center justify-center gap-2 text-lg"
            >
              <SettingsIcon />
              Add API Keys
            </button>
            <div className="flex items-center justify-center gap-2 mt-4 text-xs text-[var(--text-muted)]">
              <ShieldIcon />
              Your keys are encrypted and stored securely
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
}