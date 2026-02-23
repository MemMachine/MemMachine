import React, { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import SettingsModal from './SettingsModal';
import Sidebar from './Sidebar';
import APIKeyPrompt from './APIKeyPrompt';
import ProjectModal from './ProjectModal';

// SVG Icons
const EditIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
  </svg>
);

const TrashIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"></polyline>
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
    <line x1="10" y1="11" x2="10" y2="17"></line>
    <line x1="14" y1="11" x2="14" y2="17"></line>
  </svg>
);

const SettingsIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3"></circle>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
  </svg>
);

const ChevronDownIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9"></polyline>
  </svg>
);

const CpuIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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

export default function Chat({ user }) {
  const [messages, setMessages] = useState([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [config, setConfig] = useState(() => {
    const saved = localStorage.getItem('chat-settings');
    return saved
      ? JSON.parse(saved)
      : { model: 'gpt-4o-mini', temperature: 1.0, system_prompt: '' };
  });
  const [thread, setThread] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [hasApiKeys, setHasApiKeys] = useState(false);
  const [checkingKeys, setCheckingKeys] = useState(true);
  const [models, setModels] = useState({ openai: [], anthropic: [], google: [] });
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);

  // Project modal state
  const [projectModalOpen, setProjectModalOpen] = useState(false);
  const [projectModalMode, setProjectModalMode] = useState('create'); // 'create' or 'invite'
  const [selectedProject, setSelectedProject] = useState(null);

  const sessionId = useMemo(() => `sess-${user.id}`, [user.id]);

  useEffect(() => {
    localStorage.setItem('chat-settings', JSON.stringify(config));
  }, [config]);

  // Load available models
  useEffect(() => {
    const loadModels = async () => {
      try {
        const res = await api.get('/models/available/fast');
        setModels(res.data);
      } catch (err) {
        console.error('Failed to load models:', err);
      }
    };
    loadModels();
  }, []);

  // Check if user has any API keys configured
  useEffect(() => {
    const checkKeys = async () => {
      try {
        const res = await api.get('/api/user/api-keys');
        const hasKeys =
          res.data.has_openai || res.data.has_anthropic || res.data.has_google;
        setHasApiKeys(hasKeys);
      } catch (err) {
        console.error('Key check failed:', err);
      } finally {
        setCheckingKeys(false);
      }
    };
    checkKeys();
  }, []);

  useEffect(() => {
    const initializeThread = async () => {
      setLoading(true);
      try {
        const savedThreadId = localStorage.getItem('last-thread-id');
        
        if (savedThreadId) {
          try {
            const res = await api.get('/threads');
            const threads = res.data;
            const lastThread = threads.find(t => t.id === parseInt(savedThreadId));
            
            if (lastThread) {
              await selectThread(lastThread);
              setLoading(false);
              return;
            }
          } catch (err) {
            console.warn('Could not restore last thread:', err);
          }
        }

        const res = await api.get('/threads');
        if (res.data && res.data.length > 0) {
          await selectThread(res.data[0]);
        } else {
          await newPersonal();
        }
      } catch (err) {
        console.error('Failed to initialize thread:', err);
        await newPersonal();
      } finally {
        setLoading(false);
      }
    };

    initializeThread();
  }, [user.id]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (modelDropdownOpen && !e.target.closest('.model-dropdown-container')) {
        setModelDropdownOpen(false);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [modelDropdownOpen]);

  const loadMessages = async (t) => {
    if (!t) return;
    try {
      const res = await api.get(`/threads/${t.id}/messages`);
      const ms = res.data.map((m) => {
        if (m.type === 'file') {
          return { 
            role: 'user', 
            type: 'file', 
            filename: m.filename,
            userEmail: m.user_email || user.email  // ✅ ADDED: Include user email
          };
        }
        return {
          role: m.sender === 'user' ? 'user' : 'assistant',
          content: m.content,
          model_used: m.model_used,
          userEmail: m.user_email || (m.sender === 'user' ? user.email : null)  // ✅ ADDED: Include user email
        };
      });
      setMessages(ms);
    } catch (e) {
      console.error('Failed to load messages:', e);
    }
  };

  const selectThread = async (t) => {
    setThread(t);
    localStorage.setItem('last-thread-id', String(t.id));
    await loadMessages(t);
    
    if (t.active_model && t.active_model !== config.model) {
      setConfig(prev => ({ ...prev, model: t.active_model }));
    }
  };

  const newPersonal = async (projectId = null) => {
    try {
      const payload = { title: 'New chat' };
      if (projectId) payload.project_id = projectId;
      
      const res = await api.post('/threads', payload);
      const newThread = res.data;
      
      setThread(newThread);
      localStorage.setItem('last-thread-id', String(newThread.id));
      setMessages([]);
      
      window.dispatchEvent(new Event('refresh-threads'));
    } catch (error) {
      console.error('Failed to create new chat:', error);
      alert(`Failed to create new chat: ${error.response?.data?.detail || error.message}`);
    }
  };

  // Project handler functions
  const handleNewProject = () => {
    setSelectedProject(null);
    setProjectModalMode('create');
    setProjectModalOpen(true);
  };

  const handleInvite = (project) => {
    setSelectedProject(project);
    setProjectModalMode('invite');
    setProjectModalOpen(true);
  };

  const handleProjectModalSave = async (data) => {
    try {
      if (projectModalMode === 'create') {
        const res = await api.post('/projects', data);
        console.log('Project created:', res.data);
        setProjectModalOpen(false);
        window.dispatchEvent(new Event('refresh-threads'));
      } else if (projectModalMode === 'invite') {
        const res = await api.post(`/projects/${selectedProject.id}/members`, data);
        console.log('Member invited:', res.data);
        if (res.data.status === 'already_member') {
          alert('User is already a member of this project');
        }
        setProjectModalOpen(false);
      }
    } catch (error) {
      console.error('Project operation failed:', error);
      throw error;
    }
  };

  const renameChat = async () => {
    if (!thread) return alert('No chat selected');
    const newTitle = prompt('Rename chat:', thread.title);
    if (!newTitle || newTitle === thread.title) return;
    try {
      await api.put(`/threads/${thread.id}`, { title: newTitle });
      const updated = { ...thread, title: newTitle };
      setThread(updated);
      window.dispatchEvent(new Event('refresh-threads'));
    } catch {
      alert('Rename failed');
    }
  };

  const deleteChat = async () => {
    if (!thread) return alert('No chat selected');
    if (!window.confirm(`Delete chat "${thread.title}"?`)) return;
    try {
      await api.delete(`/threads/${thread.id}`);
      
      const res = await api.get('/threads');
      if (res.data && res.data.length > 0) {
        await selectThread(res.data[0]);
      } else {
        await newPersonal();
      }
      
      window.dispatchEvent(new Event('refresh-threads'));
    } catch {
      alert('Delete failed');
    }
  };

  const handleModelSelect = (modelId) => {
    setConfig(prev => ({ ...prev, model: modelId }));
    setModelDropdownOpen(false);
    
    if (thread) {
      const updatedThread = { ...thread, active_model: modelId };
      setThread(updatedThread);
    }
  };

  const getAllModels = () => {
    const allModels = [];
    if (models.openai?.length) {
      allModels.push({ provider: 'OpenAI', models: models.openai });
    }
    if (models.anthropic?.length) {
      allModels.push({ provider: 'Anthropic', models: models.anthropic });
    }
    if (models.google?.length) {
      allModels.push({ provider: 'Google', models: models.google });
    }
    return allModels;
  };

  const getCurrentModelName = () => {
    const allModels = [...(models.openai || []), ...(models.anthropic || []), ...(models.google || [])];
    const current = allModels.find(m => m.id === config.model);
    return current?.name || config.model;
  };

  const send = async ({ text, file }) => {
    if (isStreaming) return;
    
    let t = thread;

    if (!t) {
      const res = await api.post('/threads', { title: 'New chat' });
      t = res.data;
      setThread(t);
      localStorage.setItem('last-thread-id', String(t.id));
    }

    // ✅ UPDATED: Add userEmail to user messages
    if (text)
      setMessages((m) => [...m, { role: 'user', content: text, model_used: config.model, userEmail: user.email }]);
    if (file)
      setMessages((m) => [
        ...m,
        { role: 'user', type: 'file', filename: file.name, model_used: config.model, userEmail: user.email },
      ]);

    const assistantMessageIndex = messages.length + (text ? 1 : 0) + (file ? 1 : 0);
    setMessages((m) => [
      ...m,
      { role: 'assistant', content: '', model_used: config.model, streaming: true },
    ]);
    setIsStreaming(true);

    const form = new FormData();
    form.append('thread_id', String(t.id));
    form.append('message', text || '');
    form.append('model', config.model);
    form.append('temperature', String(config.temperature));
    form.append('system_prompt', config.system_prompt || '');
    if (file) form.append('file', file);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${api.defaults.baseURL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: form,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              break;
            }
            try {
              const parsed = JSON.parse(data);
              if (parsed.content) {
                accumulatedContent += parsed.content;
                setMessages((prevMessages) => {
                  const newMessages = [...prevMessages];
                  newMessages[assistantMessageIndex] = {
                    role: 'assistant',
                    content: accumulatedContent,
                    model_used: config.model,
                    streaming: true,
                  };
                  return newMessages;
                });
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }

      setMessages((prevMessages) => {
        const newMessages = [...prevMessages];
        newMessages[assistantMessageIndex] = {
          role: 'assistant',
          content: accumulatedContent,
          model_used: config.model,
          streaming: false,
        };
        return newMessages;
      });
      
      if (t.active_model !== config.model) {
        const updatedThread = { ...t, active_model: config.model };
        setThread(updatedThread);
      }

      if (t.title === 'New chat' && messages.length === 0 && text) {
        autoRenameThread(t.id, text);
      }
      
      window.dispatchEvent(new Event('refresh-threads'));
    } catch (e) {
      console.error('Streaming error:', e);
      setMessages((prevMessages) => {
        const newMessages = [...prevMessages];
        newMessages[assistantMessageIndex] = {
          role: 'assistant',
          content: '⚠️ Error processing your message.',
          model_used: config.model,
          streaming: false,
        };
        return newMessages;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const autoRenameThread = async (threadId, firstMessage) => {
    try {
      let title = firstMessage.trim();
      
      const firstSentence = title.match(/^[^.!?]+[.!?]/);
      if (firstSentence) {
        title = firstSentence[0];
      }
      
      if (title.length > 50) {
        title = title.substring(0, 47) + '...';
      }
      
      await api.put(`/threads/${threadId}`, { title });
      
      setThread(prev => ({ ...prev, title }));
      
      window.dispatchEvent(new Event('refresh-threads'));
    } catch (err) {
      console.error('Failed to auto-rename thread:', err);
    }
  };

  const handleModelChange = (newConfig) => {
    setConfig(newConfig);
    
    if (thread && newConfig.model !== thread.active_model) {
      const updatedThread = { ...thread, active_model: newConfig.model };
      setThread(updatedThread);
    }
  };

  if (checkingKeys) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-mid)] flex items-center justify-center mx-auto mb-4 animate-pulse">
            <CpuIcon />
          </div>
          <div className="text-lg text-[var(--text-secondary)]">Loading...</div>
        </div>
      </div>
    );
  }

  if (!hasApiKeys) {
    return (
      <>
        <APIKeyPrompt onOpenSettings={() => setSettingsOpen(true)} />
        <SettingsModal
          open={settingsOpen}
          onClose={() => {
            setSettingsOpen(false);
            const recheckKeys = async () => {
              try {
                const res = await api.get('/api/user/api-keys');
                const hasKeys =
                  res.data.has_openai || res.data.has_anthropic || res.data.has_google;
                setHasApiKeys(hasKeys);
              } catch (err) {
                console.error('Key check failed:', err);
              }
            };
            recheckKeys();
          }}
          settings={config}
          onSave={handleModelChange}
        />
      </>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-mid)] flex items-center justify-center mx-auto mb-4 animate-pulse">
            <CpuIcon />
          </div>
          <div className="text-lg text-[var(--text-secondary)]">Loading your chat...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <Sidebar
        user={user}
        selected={thread}
        onSelectThread={selectThread}
        onNewPersonal={newPersonal}
        onNewProject={handleNewProject}  
        onInvite={handleInvite}          
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-screen bg-[var(--bg-primary)]">
        {/* Header */}
        <header className="border-b border-[var(--border-subtle)] p-4 flex items-center justify-between glass-strong">
          <div className="flex flex-col gap-1">
            <h1 className="font-semibold text-lg text-[var(--text-primary)]">
              {thread ? thread.title : 'HippoSync'}
            </h1>
            
            {/* Model Dropdown */}
            {thread && (
              <div className="model-dropdown-container relative">
                <button
                  onClick={() => setModelDropdownOpen(!modelDropdownOpen)}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--surface-light)] border border-[var(--border-subtle)] hover:border-[var(--border-medium)] transition-all text-sm"
                >
                  <CpuIcon />
                  <span className="text-[var(--text-secondary)]">{getCurrentModelName()}</span>
                  <ChevronDownIcon />
                </button>
                
                {/* Dropdown Menu */}
                {modelDropdownOpen && (
                  <div className="absolute top-full left-0 mt-2 w-64 rounded-xl glass-strong shadow-xl z-50 py-2 max-h-80 overflow-y-auto">
                    {getAllModels().map((group) => (
                      <div key={group.provider}>
                        <div className="px-3 py-2 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                          {group.provider}
                        </div>
                        {group.models.map((m) => (
                          <button
                            key={m.id}
                            onClick={() => handleModelSelect(m.id)}
                            className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                              config.model === m.id
                                ? 'bg-[var(--surface-hover)] text-[var(--gradient-mid)]'
                                : 'hover:bg-[var(--surface-light)] text-[var(--text-secondary)]'
                            }`}
                          >
                            <div className="font-medium">{m.name}</div>
                            {m.context_window && (
                              <div className="text-xs text-[var(--text-muted)]">
                                {(m.context_window / 1000).toFixed(0)}K context
                              </div>
                            )}
                          </button>
                        ))}
                      </div>
                    ))}
                    
                    {getAllModels().length === 0 && (
                      <div className="px-3 py-4 text-sm text-[var(--text-muted)] text-center">
                        No models available
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
          
          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <button
              className="icon-btn"
              onClick={renameChat}
              disabled={!thread}
              title="Rename Chat"
            >
              <EditIcon />
            </button>
            <button
              className="icon-btn danger"
              onClick={deleteChat}
              disabled={!thread}
              title="Delete Chat"
            >
              <TrashIcon />
            </button>
            <button
              className="btn-primary px-4 py-2 rounded-xl text-sm font-medium flex items-center gap-2"
              onClick={() => setSettingsOpen(true)}
            >
              <SettingsIcon />
              Settings
            </button>
          </div>
        </header>

        {/* Messages Container */}
        <div className="flex-1 overflow-hidden">
          {/* ✅ UPDATED: Pass currentUserEmail to MessageList */}
          <MessageList messages={messages} currentUserEmail={user.email} />
        </div>

        {/* Input Area */}
        <div className="border-t border-[var(--border-subtle)] glass">
          <MessageInput onSend={send} disabled={isStreaming} />
        </div>

        {/* Settings Modal */}
        <SettingsModal
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          settings={config}
          onSave={handleModelChange}
        />

        {/* Project Modal */}
        <ProjectModal
          open={projectModalOpen}
          onClose={() => setProjectModalOpen(false)}
          project={selectedProject}
          onSave={handleProjectModalSave}
          mode={projectModalMode}
        />
      </div>
    </div>
  );
} 