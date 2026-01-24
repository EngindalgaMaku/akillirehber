"use client";

import { createContext, useContext, useState, useEffect, ReactNode, useCallback, useRef } from "react";
import { api, User, LoginRequest, RegisterRequest, setOnUnauthorizedCallback } from "./api";
import { useRouter } from "next/navigation";

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  login: (data: LoginRequest) => Promise<void>;
  register: (data: RegisterRequest) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const TOKEN_KEY = "akilli_rehber_token";
const REFRESH_TOKEN_KEY = "akilli_rehber_refresh_token";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  const isInitialized = useRef(false);

  // Handle unauthorized (401) responses - auto logout
  const handleUnauthorized = useCallback(() => {
    setUser(null);
    router.push("/login");
  }, [router]);

  // Set up the unauthorized callback
  useEffect(() => {
    setOnUnauthorizedCallback(handleUnauthorized);
    return () => {
      setOnUnauthorizedCallback(null);
    };
  }, [handleUnauthorized]);

  // Initialize auth on mount
  useEffect(() => {
    // Prevent double initialization in React Strict Mode
    if (isInitialized.current) return;
    isInitialized.current = true;

    const initAuth = async () => {
      const token = localStorage.getItem(TOKEN_KEY);
      const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
      
      if (token) {
        api.setToken(token);
      }
      if (refreshToken) {
        api.setRefreshToken(refreshToken);
      }
      
      if (token || refreshToken) {
        try {
          // This will automatically refresh the token if needed
          const userData = await api.getMe();
          setUser(userData);
        } catch {
          // Token invalid and refresh failed, clear everything
          localStorage.removeItem(TOKEN_KEY);
          localStorage.removeItem(REFRESH_TOKEN_KEY);
          api.setToken(null);
          api.setRefreshToken(null);
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = useCallback(async (data: LoginRequest) => {
    const response = await api.login(data);
    const { access_token, refresh_token } = response;
    
    // Save tokens
    localStorage.setItem(TOKEN_KEY, access_token);
    localStorage.setItem(REFRESH_TOKEN_KEY, refresh_token);
    api.setToken(access_token);
    api.setRefreshToken(refresh_token);
    
    // Get user data
    const userData = await api.getMe();
    setUser(userData);
  }, []);

  const register = useCallback(async (data: RegisterRequest) => {
    // Register user
    await api.register(data);
    
    // Auto login after register
    await login({ email: data.email, password: data.password });
  }, [login]);

  const logout = useCallback(async () => {
    // Call API logout to revoke refresh token
    await api.logout();
    setUser(null);
    router.push("/login");
  }, [router]);

  const refreshUser = useCallback(async () => {
    try {
      const userData = await api.getMe();
      setUser(userData);
    } catch (error) {
      console.error("Failed to refresh user:", error);
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        login,
        register,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
