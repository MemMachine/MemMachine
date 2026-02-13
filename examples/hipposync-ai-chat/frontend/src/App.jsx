import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { api, setAuth, getToken } from './api';
import Chat from './components/Chat';
import Login from './components/Login';
import Signup from './components/Signup';
import VerifyEmailPage from './components/VerifyEmailPage';
import EmailVerifiedPage from './components/EmailVerifiedPage';

export default function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = getToken();
    if (token) {
      setAuth(token);
      api.get('/auth/me')
        .then(res => {
          setUser(res.data);
        })
        .catch(() => {
          setAuth(null);
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, []);

  const onLoggedIn = (token) => {
    setAuth(token);
    api.get('/auth/me').then(res => {
      setUser(res.data);
    });
  };

  const onLogout = () => {
    setAuth(null);
    setUser(null);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-500 via-purple-600 to-indigo-700">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        {/* Public Routes */}
        <Route 
          path="/login" 
          element={
            user ? <Navigate to="/chat" /> : <Login onLoggedIn={onLoggedIn} />
          } 
        />
        <Route 
          path="/signup" 
          element={
            user ? <Navigate to="/chat" /> : <Signup />
          } 
        />
        <Route 
          path="/verify-email" 
          element={<VerifyEmailPage />} 
        />
        <Route 
          path="/email-verified" 
          element={<EmailVerifiedPage />} 
        />

        {/* Protected Routes */}
        <Route 
          path="/chat" 
          element={
            user ? <Chat user={user} onLogout={onLogout} /> : <Navigate to="/login" />
          } 
        />

        {/* Default Route */}
        <Route 
          path="/" 
          element={<Navigate to={user ? "/chat" : "/login"} />} 
        />

        {/* 404 Route */}
        <Route 
          path="*" 
          element={<Navigate to="/" />} 
        />
      </Routes>
    </Router>
  );
}