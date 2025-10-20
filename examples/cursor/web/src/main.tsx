import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import './index.css'
import './i18n' 
import App from './app.tsx'
import { NotificationProvider } from './contexts/NotificationContext.tsx'
import { Toaster } from '@/components/ui/sonner'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <NotificationProvider>
        <Toaster richColors position='top-center' />
        <App />
      </NotificationProvider>
    </HashRouter>
  </StrictMode>,
)
