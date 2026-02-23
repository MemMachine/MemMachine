import axios from 'axios';

  // Base URL of your FastAPI backend
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://44.199.198.123:8001';

// Create axios instance
export const api = axios.create({
  baseURL: API_BASE,
});

// --- AUTH MANAGEMENT HELPERS ---

// Attach or remove JWT token
export function setAuth(token) {
  if (token) {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    localStorage.setItem('token', token);
  } else {
    delete api.defaults.headers.common['Authorization'];
    localStorage.removeItem('token');
  }
}

// Retrieve token from localStorage (for reload persistence)
export function getToken() {
  return localStorage.getItem('token');
}

// Try to automatically attach token on page load
const existingToken = getToken();
if (existingToken) {
  api.defaults.headers.common['Authorization'] = `Bearer ${existingToken}`;
}

// --- BASIC API SHORTCUTS ---

// AUTH
export const login = async (email, password) => {
  const form = new FormData();
  form.append('username', email);
  form.append('password', password);
  const res = await api.post('/auth/login', form);
  return res.data;
};

export const signup = async (email, password) => {
  const res = await api.post('/auth/signup', { email, password });
  return res.data;
};

export const getCurrentUser = async () => {
  const res = await api.get('/auth/me');
  return res.data;
};

// PROJECTS
export const getProjects = async () => {
  const res = await api.get('/projects');
  return res.data;
};

export const createProject = async (name) => {
  const res = await api.post('/projects', { name });
  return res.data;
};

export const inviteToProject = async (projectId, email) => {
  const res = await api.post(`/projects/${projectId}/members`, { email });
  return res.data;
};

// THREADS (CHATS)
export const getThreads = async (projectId = null) => {
  const params = projectId ? { project_id: projectId } : {};
  const res = await api.get('/threads', { params });
  return res.data;
};

export const createThread = async (title = 'New chat', projectId = null) => {
  const body = { title };
  if (projectId) body.project_id = projectId;
  const res = await api.post('/threads', body);
  return res.data;
};

export const renameThread = async (threadId, newTitle) => {
  const res = await api.put(`/threads/${threadId}`, { title: newTitle });
  return res.data;
};

export const deleteThread = async (threadId) => {
  const res = await api.delete(`/threads/${threadId}`);
  return res.data;
};

export const getMessages = async (threadId) => {
  const res = await api.get(`/threads/${threadId}/messages`);
  return res.data;
};

// CHAT (Send messages or files)
export const sendChat = async (threadId, message, file, config = {}) => {
  const form = new FormData();
  form.append('thread_id', threadId);
  form.append('message', message || '');
  if (file) form.append('file', file);
  if (config.model) form.append('model', config.model);
  if (config.temperature) form.append('temperature', String(config.temperature));
  if (config.system_prompt) form.append('system_prompt', config.system_prompt);

  const res = await api.post('/chat', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export default api;
