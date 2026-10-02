import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar, type PageId } from './components/Sidebar';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { DashboardPage } from './pages/DashboardPage';
import { ImageDetectPage } from './pages/ImageDetectPage';
import { LiveCameraPage } from './pages/LiveCameraPage';
import { ModelsPage } from './pages/ModelsPage';
import { RTSPCameraPage } from './pages/RTSPCameraPage';
import { TrainingPage } from './pages/TrainingPage';
import { VideoDetectPage } from './pages/VideoDetectPage';
import { api } from './services/api';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('dashboard');
  const [activeModel, setActiveModel] = useState<string>('yolov8n.pt');
  const [hardwareDevice, setHardwareDevice] = useState<string>('CPU');

  useEffect(() => {
    // Initial health check
    api.getHealth()
      .then((data) => {
        setHardwareDevice(data.compute_device || 'CPU');
        setActiveModel(data.default_model || 'yolov8n.pt');
      })
      .catch((err) => console.warn('Could not contact API health endpoint:', err));

    // Fetch active model from registry
    api.listModels()
      .then((models) => {
        const active = models.find((m) => m.is_active);
        if (active) setActiveModel(active.name);
      })
      .catch(console.error);
  }, []);

  const renderActivePage = () => {
    switch (activePage) {
      case 'dashboard':
        return <DashboardPage />;
      case 'image':
        return <ImageDetectPage />;
      case 'video':
        return <VideoDetectPage />;
      case 'live':
        return <LiveCameraPage />;
      case 'rtsp':
        return <RTSPCameraPage />;
      case 'models':
        return <ModelsPage />;
      case 'training':
        return <TrainingPage />;
      case 'analytics':
        return <AnalyticsPage />;
      default:
        return <DashboardPage />;
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col antialiased selection:bg-emerald-500 selection:text-slate-950">
      <Navbar activeModel={activeModel} hardwareDevice={hardwareDevice} />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar activePage={activePage} onSelectPage={setActivePage} />

        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <div className="max-w-7xl mx-auto">{renderActivePage()}</div>
        </main>
      </div>
    </div>
  );
};

export default App;
