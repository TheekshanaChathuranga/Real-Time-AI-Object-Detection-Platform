import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  color?: 'emerald' | 'cyan' | 'indigo' | 'amber' | 'rose';
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  color = 'emerald',
}) => {
  const colorMap = {
    emerald: {
      border: 'border-emerald-500/20',
      bgIcon: 'bg-emerald-500/10 text-emerald-400',
      glow: 'shadow-emerald-500/5',
    },
    cyan: {
      border: 'border-cyan-500/20',
      bgIcon: 'bg-cyan-500/10 text-cyan-400',
      glow: 'shadow-cyan-500/5',
    },
    indigo: {
      border: 'border-indigo-500/20',
      bgIcon: 'bg-indigo-500/10 text-indigo-400',
      glow: 'shadow-indigo-500/5',
    },
    amber: {
      border: 'border-amber-500/20',
      bgIcon: 'bg-amber-500/10 text-amber-400',
      glow: 'shadow-amber-500/5',
    },
    rose: {
      border: 'border-rose-500/20',
      bgIcon: 'bg-rose-500/10 text-rose-400',
      glow: 'shadow-rose-500/5',
    },
  };

  const scheme = colorMap[color] || colorMap.emerald;

  return (
    <div
      className={`glass-panel p-5 rounded-2xl border ${scheme.border} shadow-lg ${scheme.glow} relative overflow-hidden group`}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">{title}</p>
          <h3 className="text-2xl font-bold text-white mt-1 tracking-tight">{value}</h3>
          {subtitle && <p className="text-xs text-slate-400 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-xl ${scheme.bgIcon} shrink-0`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  );
};
