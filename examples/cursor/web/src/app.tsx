import { Routes, Route } from 'react-router-dom'
import {toast} from  'sonner'
import { LoginPage, SignupPage, HomePage } from './pages'
import './app.css'
import { UserProvider } from './contexts/UserContext'
import useCatchError from '@/hooks/useCatchError';
import { useTranslation } from 'react-i18next'
import MainLayout from './components/main-layout'
import NotFoundPage from './pages/not-found'

function App() {
  const {t} = useTranslation()
  useCatchError({
    toast: {
      error: (msg: string) => {
        toast.error(msg)
      },
    },
    onAuthError: (ev: any) => {
      const error = ev.reason;
      if (error.redirectUnauthorized !== false) {
        // resetCurrentUser();
        // location.href = error.redirect || LOGIN_URL;
      }
    },
    t: t,
  });

  return (
    <UserProvider>
      <MainLayout>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/" element={<HomePage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </MainLayout>
    </UserProvider>
  )
}

export default App
