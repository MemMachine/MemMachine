/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, ReactNode, useEffect, useState, useCallback } from 'react';
import { userService } from '@/services/user';
import { IUser, ILoginReq, IRegisterReq } from '@/models/user';
import authToken from '@/base/authToken';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';

interface UserContextType {
  user: IUser | null;
  token: string | null;
  expiresAt: string | null;
  loading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
  clear: () => void;
  logout: () => Promise<void>;
  login: (credentials: ILoginReq) => Promise<boolean>;
  register: (userData: IRegisterReq) => Promise<boolean>;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

interface UserProviderProps {
  children: ReactNode;
}

export function UserProvider({ children }: UserProviderProps) {
  const [user, setUser] = useState<IUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [expiresAt, setExpiresAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const navigate = useNavigate()

  const fetchUserInfo = useCallback(async () => {
    const token = authToken.get();
    
    if (!token) {
      setUser(null);
      setToken(null);
      setExpiresAt(null);
      setError(null);
      navigate('/login');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const response = await userService.getCurrentUser();
      
      if (response.success && response.data) {
        setUser(response.data.user);
        setToken(response.data.token);
        setExpiresAt(response.data.expires_at);
      } else {
        setError(new Error(response.message || 'Failed to fetch user info'));
        setUser(null);
        setToken(null);
        setExpiresAt(null);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch user info';
      setError(new Error(errorMessage));
      setUser(null);
      
      // If there's an authentication error, clear the token
      if (err instanceof Error && err.message.includes('No access permission')) {
        authToken.remove();
        navigate('/login');
      }
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  const clear = () => {
    setUser(null);
    setToken(null);
    setExpiresAt(null);
    setError(null);
    setLoading(false);
  };


  const logout = useCallback(async () => {
    await userService.logout();
    authToken.remove();
    clear();
    navigate('/login');
  }, [clear, navigate]);

  const login = useCallback(async (credentials: ILoginReq): Promise<boolean> => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await userService.login(credentials);
      
      if (response.success && response.data) {
        setUser(response.data.user);
        setToken(response.data.token);
        setExpiresAt(response.data.expires_at);
        authToken.set(response.data.token);
        toast.success('Login successful!');
        navigate('/');
        return true;
      } else {
        const errorMessage = response.message || 'Login failed';
        setError(new Error(errorMessage));
        toast.error(errorMessage);
        return false;
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Login failed. Please try again.';
      setError(new Error(errorMessage));
      toast.error(errorMessage);
      return false;
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  const register = useCallback(async (userData: IRegisterReq): Promise<boolean> => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await userService.register(userData);
      
      if (response.success && response.data) {
        setUser(response.data.user);
        setToken(response.data.token);
        setExpiresAt(response.data.expires_at);
        authToken.set(response.data.token);
        toast.success('Registration successful!');
        navigate('/');
        return true;
      } else {
        const errorMessage = response.message || 'Registration failed';
        setError(new Error(errorMessage));
        toast.error(errorMessage);
        return false;
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Registration failed. Please try again.';
      setError(new Error(errorMessage));
      toast.error(errorMessage);
      return false;
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  // Fetch user info on mount if token exists
  useEffect(() => {
    fetchUserInfo();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refresh = async () => {
    await fetchUserInfo();
  };

  const userInfo: UserContextType = {
    user,
    token,
    expiresAt,
    loading,
    error,
    refresh,
    clear,
    logout,
    login,
    register,
  };

  return (
    <UserContext.Provider value={userInfo}>
      {children}
    </UserContext.Provider>
  );
}

export function useUserContext(): UserContextType {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUserContext must be used within a UserProvider');
  }
  return context;
}
