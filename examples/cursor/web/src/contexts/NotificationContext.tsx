import { toast, ExternalToast } from 'sonner';
import React, { createContext, useContext, useMemo } from 'react';

type NotificationContextType = {
  success: (message: string, data?: ExternalToast) => void;
  info: (message: string, data?: ExternalToast) => void;
  warning: (message: string, data?: ExternalToast) => void;
  error: (message: string, data?: ExternalToast) => void;
};

const NotificationContext = createContext<NotificationContextType>({
  success: () => {},
  info: () => {},
  warning: () => {},
  error: () => {},
});

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const contextValue: NotificationContextType = useMemo(() => ({
    success: (message: string, data?: ExternalToast) => {
      toast.success(message, data)
    },
    info: (message: string, data?: ExternalToast) => {
      toast.info(message, data)
    },
    warning: (message: string, data?: ExternalToast) => {
      toast.warning(message, data)
    },
    error: (message: string, data?: ExternalToast) => {
      toast.error(message, data)
    },
  }), []);

  return (
    <NotificationContext.Provider value={contextValue}>
      {children}
    </NotificationContext.Provider>
  );
};

export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};
