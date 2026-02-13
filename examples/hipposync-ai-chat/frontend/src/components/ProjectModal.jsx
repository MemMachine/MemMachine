import React, { useState } from 'react';

const ProjectModal = ({ open, onClose, project, onSave, mode = 'create' }) => {
  const [name, setName] = useState(project?.name || '');
  const [description, setDescription] = useState(project?.description || '');
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('member');
  const [inviting, setInviting] = useState(false);

  if (!open) return null;

  const handleSave = () => {
    if (!name.trim()) {
      alert('Project name is required');
      return;
    }
    onSave({ name: name.trim(), description: description.trim() });
  };

  const handleInvite = async () => {
    if (!email.trim()) {
      alert('Email is required');
      return;
    }
    setInviting(true);
    try {
      await onSave({ email: email.trim(), role });
      setEmail('');
      setRole('member');
      alert('Member invited successfully!');
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to invite member');
    } finally {
      setInviting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-[var(--bg-primary)] rounded-2xl shadow-2xl w-full max-w-md p-6 m-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold gradient-text">
            {mode === 'create' ? 'Create New Project' : mode === 'invite' ? 'Invite Member' : 'Edit Project'}
          </h2>
          <button
            onClick={onClose}
            className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        {mode === 'invite' ? (
          /* Invite Member Form */
          <>
            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
                  Email Address
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="user@example.com"
                  className="w-full px-4 py-2 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--gradient-mid)]"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
                  Role
                </label>
                <select
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  className="w-full px-4 py-2 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)] text-[var(--text-primary)] focus:outline-none focus:border-[var(--gradient-mid)]"
                >
                  <option value="member">Member</option>
                  <option value="admin">Admin</option>
                  <option value="viewer">Viewer</option>
                </select>
                <p className="text-xs text-[var(--text-muted)] mt-1">
                  {role === 'admin' && 'Can manage project and members'}
                  {role === 'member' && 'Can create and view chats'}
                  {role === 'viewer' && 'Can only view chats'}
                </p>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 rounded-xl btn-secondary font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleInvite}
                disabled={inviting}
                className="flex-1 px-4 py-2 rounded-xl btn-primary font-medium disabled:opacity-50"
              >
                {inviting ? 'Inviting...' : 'Invite'}
              </button>
            </div>
          </>
        ) : (
          /* Create/Edit Project Form */
          <>
            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
                  Project Name *
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., Website Redesign, ML Research"
                  className="w-full px-4 py-2 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--gradient-mid)]"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
                  Description (Optional)
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Brief description of the project..."
                  rows={3}
                  className="w-full px-4 py-2 rounded-xl bg-[var(--surface-light)] border border-[var(--border-subtle)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--gradient-mid)] resize-none"
                />
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 rounded-xl btn-secondary font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="flex-1 px-4 py-2 rounded-xl btn-primary font-medium"
              >
                {mode === 'create' ? 'Create Project' : 'Save Changes'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ProjectModal;