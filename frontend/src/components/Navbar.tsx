import React from 'react';
import { Activity, Cpu, Sparkles } from 'lucide-react';

interface NavbarProps {
  activeModel: string;
  hardwareDevice: string;
}

export const Navbar: React.FC<NavbarProps> = ({ activeModel, hardwareDevice }) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-30">
      {/* Brand & Title */}
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-cyan-500 p-0.5 shadow-lg shadow-emerald-500/20">
          <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-emerald-400 animate-pulse" />
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
              VISION CORE
            </span>
            <span className="px-2 py-0.5 text-[10px] uppercase font-bold tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full">
              YOLO v8/v11
            </span>
          </div>
          <p className="text-xs text-slate-400">Real-Time AI Object Detection Platform</p>
        </div>
      </div>

      {/* Hardware & Active Model Status Pills */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="text-slate-400">Compute:</span>
          <span className="font-semibold text-cyan-300 uppercase">{hardwareDevice || 'CPU'}</span>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
          <Activity className="w-4 h-4 text-emerald-400" />
          <span className="text-slate-400">Active Model:</span>
          <span className="font-semibold text-emerald-300 font-mono">{activeModel || 'yolov8n.pt'}</span>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-ping"></span>
          <span>Engine Online</span>
        </div>
      </div>
    </header>
  );
};
