import React, { useRef, useState } from 'react';

const PaperclipIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
  </svg>
);

const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

const XIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const FileIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
  </svg>
);

const LoaderIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
    <line x1="12" y1="2" x2="12" y2="6"></line>
    <line x1="12" y1="18" x2="12" y2="22"></line>
    <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
    <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
    <line x1="2" y1="12" x2="6" y2="12"></line>
    <line x1="18" y1="12" x2="22" y2="12"></line>
    <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
    <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
  </svg>
);

export default function MessageInput({ onSend, disabled }) {
  const [value, setValue] = useState('');
  const [attachedFile, setAttachedFile] = useState(null);
  const fileRef = useRef(null);
  const inputRef = useRef(null);

  const send = () => {
    if (disabled) return;
    const text = value.trim();
    if (!text && !attachedFile) return;
    onSend({ text, file: attachedFile });
    setValue('');
    setAttachedFile(null);
  };

  const onFile = (e) => {
    const file = e.target.files?.[0];
    if (file) setAttachedFile(file);
    e.target.value = '';
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="p-4">
      {/* Attached File Preview */}
      {attachedFile && (
        <div className="mb-3 flex items-center gap-3 px-4 py-3 bg-[var(--surface-light)] border border-[var(--border-subtle)] rounded-xl">
          <div className="w-10 h-10 rounded-lg bg-[var(--bg-secondary)] flex items-center justify-center text-[var(--gradient-mid)]">
            <FileIcon />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-[var(--text-primary)] truncate">
              {attachedFile.name}
            </div>
            <div className="text-xs text-[var(--text-muted)]">
              {formatFileSize(attachedFile.size)}
            </div>
          </div>
          <button
            onClick={() => setAttachedFile(null)}
            className="icon-btn danger !w-8 !h-8"
            disabled={disabled}
            title="Remove file"
          >
            <XIcon />
          </button>
        </div>
      )}

      {/* Input Container */}
      <div className="flex items-center gap-3">
        {/* Attach Button */}
        <button 
          className={`icon-btn ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          onClick={() => !disabled && fileRef.current?.click()}
          title="Attach file"
          disabled={disabled}
        >
          <PaperclipIcon />
        </button>
        <input 
          ref={fileRef} 
          type="file" 
          className="hidden" 
          onChange={onFile} 
          disabled={disabled} 
        />
        
        {/* Text Input */}
        <div className="flex-1 relative">
          <input
            ref={inputRef}
            className={`w-full px-4 py-3 rounded-xl input-field text-sm ${
              disabled ? 'opacity-60 cursor-not-allowed' : ''
            }`}
            placeholder={disabled ? "AI is responding..." : "Type your message..."}
            value={value}
            onChange={e => setValue(e.target.value)}
            onKeyDown={e => { 
              if (e.key === 'Enter' && !e.shiftKey && !disabled) { 
                e.preventDefault(); 
                send(); 
              }
            }}
            disabled={disabled}
          />
        </div>

        {/* Send Button */}
        <button 
          className={`btn-primary px-5 py-3 rounded-xl flex items-center gap-2 font-medium ${
            disabled ? 'opacity-70 cursor-not-allowed' : ''
          }`}
          onClick={send}
          disabled={disabled}
        >
          {disabled ? (
            <>
              <LoaderIcon />
              <span className="hidden sm:inline">Sending</span>
            </>
          ) : (
            <>
              <SendIcon />
              <span className="hidden sm:inline">Send</span>
            </>
          )}
        </button>
      </div>

      {/* Keyboard Hint */}
      <div className="mt-2 text-center">
        <span className="text-xs text-[var(--text-muted)]">
          Press <kbd className="px-1.5 py-0.5 rounded bg-[var(--surface-light)] border border-[var(--border-subtle)] text-[var(--text-secondary)]">Enter</kbd> to send
        </span>
      </div>
    </div>
  );
}