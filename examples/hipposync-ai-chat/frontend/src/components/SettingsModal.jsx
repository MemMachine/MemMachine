import React, { useEffect, useState } from 'react';
import { api } from '../api';

// SVG Icons
const XIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const UserIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
    <circle cx="12" cy="7" r="4"></circle>
  </svg>
);

const KeyIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"></path>
  </svg>
);

const CpuIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
    <rect x="9" y="9" width="6" height="6"></rect>
    <line x1="9" y1="1" x2="9" y2="4"></line>
    <line x1="15" y1="1" x2="15" y2="4"></line>
    <line x1="9" y1="20" x2="9" y2="23"></line>
    <line x1="15" y1="20" x2="15" y2="23"></line>
    <line x1="20" y1="9" x2="23" y2="9"></line>
    <line x1="20" y1="14" x2="23" y2="14"></line>
    <line x1="1" y1="9" x2="4" y2="9"></line>
    <line x1="1" y1="14" x2="4" y2="14"></line>
  </svg>
);

const EyeIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
    <circle cx="12" cy="12" r="3"></circle>
  </svg>
);

const EyeOffIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
    <line x1="1" y1="1" x2="23" y2="23"></line>
  </svg>
);

const ExternalLinkIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
    <polyline points="15 3 21 3 21 9"></polyline>
    <line x1="10" y1="14" x2="21" y2="3"></line>
  </svg>
);

const CheckIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"></polyline>
  </svg>
);

const MapPinIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
    <circle cx="12" cy="10" r="3"></circle>
  </svg>
);

export default function SettingsModal({ open, onClose, settings, onSave, user }) {
  const [activeTab, setActiveTab] = useState('profile');

  // Model settings state
  const [models, setModels] = useState({ openai: [], anthropic: [], google: [] });
  const [selectedProvider, setSelectedProvider] = useState('openai');
  const [model, setModel] = useState(settings.model || 'gpt-4o-mini');
  const [temperature, setTemperature] = useState(settings.temperature ?? 1.0);
  const [systemPrompt, setSystemPrompt] = useState(settings.system_prompt || '');
  const [loading, setLoading] = useState(false);

  // Profile settings state
  const [name, setName] = useState('');
  const [occupation, setOccupation] = useState('');
  const [preferences, setPreferences] = useState('');
  const [location, setLocation] = useState(null);

  // API Keys state
  const [apiKeys, setApiKeys] = useState({
    openai: '',
    anthropic: '',
    google: '',
    tavily: '',
  });
  const [showKeys, setShowKeys] = useState({
    openai: false,
    anthropic: false,
    google: false,
    tavily: false,
  });
  const [saveStatus, setSaveStatus] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open) {
      loadModels();
      loadUserSettings();
      detectProviderFromModel();
    }
  }, [open]);

  const loadUserSettings = async () => {
    try {
      const res = await api.get('/api/user/settings');
      setName(res.data.name || '');
      setOccupation(res.data.occupation || '');
      setPreferences(res.data.preferences || '');
      setLocation(res.data.location);
    } catch (err) {
      console.error('Failed to load user settings:', err);
    }
  };

  const loadModels = async () => {
    setLoading(true);
    try {
      const res = await api.get('/models/available/fast');
      setModels(res.data);
    } catch (err) {
      console.error('Failed to load models:', err);
      setModels({ openai: [], anthropic: [], google: [] });
    } finally {
      setLoading(false);
    }
  };

  const detectProviderFromModel = () => {
    const m = (settings.model || '').toLowerCase();
    if (m.includes('gpt') || m.includes('o1')) {
      setSelectedProvider('openai');
    } else if (m.includes('claude')) {
      setSelectedProvider('anthropic');
    } else if (m.includes('gemini')) {
      setSelectedProvider('google');
    } else {
      setSelectedProvider('openai');
    }
  };

  const handleProviderChange = (provider) => {
    setSelectedProvider(provider);
    const providerModels = models[provider] || [];
    if (providerModels.length > 0) {
      setModel(providerModels[0].id);
    }
  };

  const getCurrentProviderModels = () => {
    return models[selectedProvider] || [];
  };

  const handleModelSave = () => {
    onSave({
      model,
      provider: selectedProvider,
      temperature,
      system_prompt: systemPrompt,
    });
    onClose();
  };

  const handleProfileSave = async () => {
    setSaving(true);
    setSaveStatus(null);

    try {
      await api.post('/api/user/settings', {
        name: name || null,
        occupation: occupation || null,
        preferences: preferences || null,
      });

      setSaveStatus({ type: 'success', message: 'Profile saved successfully!' });
      setTimeout(() => setSaveStatus(null), 3000);
    } catch (err) {
      setSaveStatus({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to save profile',
      });
    } finally {
      setSaving(false);
    }
  };

  const handleApiKeysSave = async () => {
    setSaving(true);
    setSaveStatus(null);

    const hasKey = Object.values(apiKeys).some((key) => key.trim().length > 0);
    if (!hasKey) {
      setSaveStatus({ type: 'error', message: 'Please provide at least one API key' });
      setSaving(false);
      return;
    }

    try {
      const payload = {};
      if (apiKeys.openai.trim()) payload.openai_api_key = apiKeys.openai.trim();
      if (apiKeys.anthropic.trim()) payload.anthropic_api_key = apiKeys.anthropic.trim();
      if (apiKeys.google.trim()) payload.google_api_key = apiKeys.google.trim();
      if (apiKeys.tavily.trim()) payload.tavily_api_key = apiKeys.tavily.trim();

      const res = await api.post('/api/user/api-keys', payload);

      setSaveStatus({
        type: 'success',
        message: `Saved ${res.data.saved.length} API key(s) successfully!`,
      });

      setApiKeys({ openai: '', anthropic: '', google: '', tavily: '' });

      setTimeout(() => setSaveStatus(null), 3000);
    } catch (err) {
      setSaveStatus({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to save API keys',
      });
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  const tabs = [
    { id: 'profile', label: 'Profile', icon: UserIcon },
    { id: 'apikeys', label: 'API Keys', icon: KeyIcon },
    { id: 'model', label: 'Model', icon: CpuIcon },
  ];

  return (
    <div className="fixed inset-0 modal-overlay flex items-center justify-center z-50 p-4">
      <div className="glass-strong rounded-2xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-[var(--border-subtle)]">
          <h2 className="text-xl font-bold gradient-text">Settings</h2>
          <button onClick={onClose} className="icon-btn">
            <XIcon />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-[var(--border-subtle)]">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-4 font-medium transition-all ${
                  activeTab === tab.id
                    ? 'text-[var(--gradient-mid)] border-b-2 border-[var(--gradient-mid)] bg-[var(--surface-light)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--surface-dark)]'
                }`}
              >
                <Icon />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium mb-2 text-[var(--text-secondary)]">
                  Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  className="w-full p-3 rounded-xl input-field"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-[var(--text-secondary)]">
                  Occupation
                </label>
                <input
                  type="text"
                  value={occupation}
                  onChange={(e) => setOccupation(e.target.value)}
                  placeholder="e.g., ML Engineer, Student"
                  className="w-full p-3 rounded-xl input-field"
                />
              </div>

              {location && (
                <div>
                  <label className="block text-sm font-medium mb-2 text-[var(--text-secondary)]">
                    Location
                  </label>
                  <div className="p-4 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)]">
                    <div className="flex items-center gap-2 text-[var(--text-primary)]">
                      <MapPinIcon />
                      {location.formatted}
                    </div>
                    {location.timezone && (
                      <div className="text-xs text-[var(--text-muted)] mt-1.5 ml-5">
                        Timezone: {location.timezone}
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium mb-2 text-[var(--text-secondary)]">
                  Interests & Preferences
                </label>
                <textarea
                  value={preferences}
                  onChange={(e) => setPreferences(e.target.value)}
                  placeholder="Tell us about your interests, preferences, or anything you'd like AI to remember..."
                  rows={4}
                  className="w-full p-3 rounded-xl input-field resize-none"
                />
              </div>

              {saveStatus && (
                <div
                  className={`flex items-center gap-2 p-4 rounded-xl text-sm ${
                    saveStatus.type === 'success'
                      ? 'bg-[var(--accent-success)]/10 text-[var(--accent-success)] border border-[var(--accent-success)]/30'
                      : 'bg-[var(--accent-danger)]/10 text-[var(--accent-danger)] border border-[var(--accent-danger)]/30'
                  }`}
                >
                  {saveStatus.type === 'success' && <CheckIcon />}
                  {saveStatus.message}
                </div>
              )}

              <button
                onClick={handleProfileSave}
                disabled={saving}
                className="w-full py-3 btn-primary rounded-xl font-medium disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save Profile'}
              </button>
            </div>
          )}

          {/* API Keys Tab */}
          {activeTab === 'apikeys' && (
            <div className="space-y-5">
              <div className="p-4 rounded-xl bg-gradient-to-r from-[var(--gradient-start)]/10 to-[var(--gradient-mid)]/5 border border-[var(--border-subtle)]">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-[var(--gradient-mid)]/20 flex items-center justify-center">
                    <KeyIcon />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-[var(--text-primary)]">
                      Your API keys are encrypted
                    </p>
                    <p className="text-xs text-[var(--text-muted)] mt-1">
                      Keys are stored securely and never shared. You only pay for what you use directly
                      with each provider.
                    </p>
                  </div>
                </div>
              </div>

              {/* OpenAI */}
              <div>
                <label className="flex items-center gap-2 text-sm font-medium mb-2 text-[var(--text-secondary)]">
                  <div className="w-2 h-2 rounded-full bg-[var(--accent-success)]"></div>
                  OpenAI API Key
                </label>
                <div className="flex gap-2">
                  <input
                    type={showKeys.openai ? 'text' : 'password'}
                    value={apiKeys.openai}
                    onChange={(e) => setApiKeys({ ...apiKeys, openai: e.target.value })}
                    placeholder="sk-..."
                    className="flex-1 p-3 rounded-xl input-field"
                  />
                  <button
                    type="button"
                    onClick={() => setShowKeys({ ...showKeys, openai: !showKeys.openai })}
                    className="icon-btn"
                  >
                    {showKeys.openai ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                </div>
                <div className="flex justify-between items-center mt-1.5">
                  <p className="text-xs text-[var(--text-muted)]">Format: sk-...</p>
                  <a
                    href="https://platform.openai.com/api-keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-xs text-[var(--gradient-mid)] hover:text-[var(--gradient-end)]"
                  >
                    Get key <ExternalLinkIcon />
                  </a>
                </div>
              </div>

              {/* Anthropic */}
              <div>
                <label className="flex items-center gap-2 text-sm font-medium mb-2 text-[var(--text-secondary)]">
                  <div className="w-2 h-2 rounded-full bg-[var(--gradient-mid)]"></div>
                  Anthropic API Key
                </label>
                <div className="flex gap-2">
                  <input
                    type={showKeys.anthropic ? 'text' : 'password'}
                    value={apiKeys.anthropic}
                    onChange={(e) => setApiKeys({ ...apiKeys, anthropic: e.target.value })}
                    placeholder="sk-ant-..."
                    className="flex-1 p-3 rounded-xl input-field"
                  />
                  <button
                    type="button"
                    onClick={() =>
                      setShowKeys({ ...showKeys, anthropic: !showKeys.anthropic })
                    }
                    className="icon-btn"
                  >
                    {showKeys.anthropic ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                </div>
                <div className="flex justify-between items-center mt-1.5">
                  <p className="text-xs text-[var(--text-muted)]">Format: sk-ant-...</p>
                  <a
                    href="https://console.anthropic.com/settings/keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-xs text-[var(--gradient-mid)] hover:text-[var(--gradient-end)]"
                  >
                    Get key <ExternalLinkIcon />
                  </a>
                </div>
              </div>

              {/* Google */}
              <div>
                <label className="flex items-center gap-2 text-sm font-medium mb-2 text-[var(--text-secondary)]">
                  <div className="w-2 h-2 rounded-full bg-[var(--accent-info)]"></div>
                  Google API Key
                </label>
                <div className="flex gap-2">
                  <input
                    type={showKeys.google ? 'text' : 'password'}
                    value={apiKeys.google}
                    onChange={(e) => setApiKeys({ ...apiKeys, google: e.target.value })}
                    placeholder="AIza..."
                    className="flex-1 p-3 rounded-xl input-field"
                  />
                  <button
                    type="button"
                    onClick={() => setShowKeys({ ...showKeys, google: !showKeys.google })}
                    className="icon-btn"
                  >
                    {showKeys.google ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                </div>
                <div className="flex justify-between items-center mt-1.5">
                  <p className="text-xs text-[var(--text-muted)]">Format: AIza...</p>
                  <a
                    href="https://aistudio.google.com/app/apikey"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-xs text-[var(--gradient-mid)] hover:text-[var(--gradient-end)]"
                  >
                    Get key <ExternalLinkIcon />
                  </a>
                </div>
              </div>

              {/* Tavily */}
              <div>
                <label className="flex items-center gap-2 text-sm font-medium mb-2 text-[var(--text-secondary)]">
                  <div className="w-2 h-2 rounded-full bg-[var(--accent-warning)]"></div>
                  Tavily API Key{' '}
                  <span className="text-[var(--text-muted)] font-normal">
                    (Optional - Web Search)
                  </span>
                </label>
                <div className="flex gap-2">
                  <input
                    type={showKeys.tavily ? 'text' : 'password'}
                    value={apiKeys.tavily}
                    onChange={(e) => setApiKeys({ ...apiKeys, tavily: e.target.value })}
                    placeholder="tvly-..."
                    className="flex-1 p-3 rounded-xl input-field"
                  />
                  <button
                    type="button"
                    onClick={() => setShowKeys({ ...showKeys, tavily: !showKeys.tavily })}
                    className="icon-btn"
                  >
                    {showKeys.tavily ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                </div>
                <div className="flex justify-between items-center mt-1.5">
                  <p className="text-xs text-[var(--text-muted)]">Format: tvly-...</p>
                  <a
                    href="https://tavily.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-xs text-[var(--gradient-mid)] hover:text-[var(--gradient-end)]"
                  >
                    Get key <ExternalLinkIcon />
                  </a>
                </div>
              </div>

              {saveStatus && (
                <div
                  className={`flex items-center gap-2 p-4 rounded-xl text-sm ${
                    saveStatus.type === 'success'
                      ? 'bg-[var(--accent-success)]/10 text-[var(--accent-success)] border border-[var(--accent-success)]/30'
                      : 'bg-[var(--accent-danger)]/10 text-[var(--accent-danger)] border border-[var(--accent-danger)]/30'
                  }`}
                >
                  {saveStatus.type === 'success' && <CheckIcon />}
                  {saveStatus.message}
                </div>
              )}

              <button
                onClick={handleApiKeysSave}
                disabled={saving}
                className="w-full py-3 btn-primary rounded-xl font-medium disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save API Keys'}
              </button>
            </div>
          )}

          {/* Model Tab */}
          {activeTab === 'model' && (
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium mb-3 text-[var(--text-secondary)]">
                  Provider
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { id: 'openai', name: 'OpenAI', color: 'var(--accent-success)' },
                    { id: 'anthropic', name: 'Anthropic', color: 'var(--gradient-mid)' },
                    { id: 'google', name: 'Google', color: 'var(--accent-info)' },
                  ].map((p) => (
                    <button
                      key={p.id}
                      onClick={() => handleProviderChange(p.id)}
                      className={`p-4 rounded-xl border transition-all ${
                        selectedProvider === p.id
                          ? 'border-[var(--gradient-mid)] bg-[var(--surface-hover)]'
                          : 'border-[var(--border-subtle)] bg-[var(--surface-dark)] hover:border-[var(--border-medium)]'
                      }`}
                    >
                      <div className="flex items-center justify-center gap-2">
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: p.color }}
                        ></div>
                        <span
                          className={
                            selectedProvider === p.id
                              ? 'text-[var(--text-primary)]'
                              : 'text-[var(--text-secondary)]'
                          }
                        >
                          {p.name}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-[var(--text-secondary)]">
                  Model
                </label>
                {loading ? (
                  <div className="p-4 bg-[var(--surface-dark)] rounded-xl border border-[var(--border-subtle)] text-[var(--text-muted)]">
                    Loading models...
                  </div>
                ) : (
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full p-3 rounded-xl select-dropdown"
                  >
                    {getCurrentProviderModels().length === 0 ? (
                      <option>No models available</option>
                    ) : (
                      getCurrentProviderModels().map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))
                    )}
                  </select>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium mb-3 text-[var(--text-secondary)]">
                  Temperature:{' '}
                  <span className="text-[var(--gradient-mid)]">
                    {temperature.toFixed(2)}
                  </span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-2 rounded-full appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, var(--gradient-start) 0%, var(--gradient-mid) ${
                      (temperature / 2) * 100
                    }%, var(--bg-secondary) ${(temperature / 2) * 100}%, var(--bg-secondary) 100%)`,
                  }}
                />
                <div className="flex justify-between text-xs text-[var(--text-muted)] mt-2">
                  <span>Precise</span>
                  <span>Balanced</span>
                  <span>Creative</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-[var(--text-secondary)]">
                  System Prompt{' '}
                  <span className="text-[var(--text-muted)] font-normal">
                    (Optional)
                  </span>
                </label>
                <textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="Custom instructions for the AI..."
                  rows={4}
                  className="w-full p-3 rounded-xl input-field resize-none"
                />
              </div>

              <button
                onClick={handleModelSave}
                className="w-full py-3 btn-primary rounded-xl font-medium"
              >
                Save Model Settings
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}