import React from 'react';
import Navbar from '@/components/navbar';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {

  return (
    <div className="min-h-screen flex flex-col">
      <header className="shrink-0">
        <Navbar />
      </header>
      <main className="grow-1 calc(100vh - 64px)">
        {children}
      </main>
    </div>
  );
};

export default MainLayout;
