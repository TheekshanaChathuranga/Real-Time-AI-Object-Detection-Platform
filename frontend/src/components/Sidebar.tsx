import React from 'react';
import {
  BarChart3,
  Camera,
  Cpu,
  Flame,
  ImageIcon,
  LayoutDashboard,
  Radio,
  Video
} from 'lucide-react';

export type PageId =
  | 'dashboard'
  | 'image'
  | 'video'
  | 'live'
  | 'rtsp'
  | 'models'
  | 'training'
  | 'analytics';

interface SidebarProps {
  activePage: PageId;
  onSelectPage: (page: PageId) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activePage, onSelectPage }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'image', label: 'Image Detection', icon: ImageIcon },
    { id: 'video', label: 'Video Detection', icon: Video },
    { id: 'live', label: 'Live Camera', icon: Camera, badge: 'Real-Time' },
    { id: 'rtsp', label: 'RTSP Camera', icon: Radio },
    { id: 'models', label: 'Model Registry', icon: Cpu },
    { id: 'training', label: 'Custom Training', icon: Flame },
    { id: 'analytics', label: 'Analytics & Logs', icon: BarChart3 },
  ];

  return (
    <aside className="w-64 border-r border-slate-800 bg-slate-950/60 backdrop-blur-md flex flex-col justify-between p-4 shrink-0">
      <div className="space-y-1">
        <p className="px-3 text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
          Perception Systems
        </p>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectPage(item.id as PageId)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                isActive
                  ? 'bg-gradient-to-r from-emerald-500/20 to-emerald-500/5 text-emerald-400 border border-emerald-500/30 shadow-md shadow-emerald-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span className="px-1.5 py-0.5 text-[9px] font-bold uppercase bg-emerald-500/20 text-emerald-300 rounded border border-emerald-500/30">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* System Info Card */}
      <div className="p-3.5 rounded-xl bg-slate-900/70 border border-slate-800 text-xs">
        <div className="flex items-center justify-between text-slate-400 mb-1">
          <span className="font-semibold text-slate-300">Vision Core</span>
          <span className="text-[10px] text-emerald-400 font-mono">v1.0.0</span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          High-throughput perception engine for real-time AI object detection.
        </p>
      </div>
    </aside>
  );
};
