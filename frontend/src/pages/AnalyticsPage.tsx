import React, { useEffect, useState } from 'react';
import {
  BarChart3,
  PieChart,
  RefreshCw,
  Sparkles
} from 'lucide-react';
import { api } from '../services/api';
import type { AnalyticsBreakdown } from '../types';

export const AnalyticsPage: React.FC = () => {
  const [data, setData] = useState<AnalyticsBreakdown | null>(null);

  const fetchAnalytics = async () => {
    try {
      const res = await api.getAnalyticsBreakdown();
      setData(res);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-emerald-400" />
            Detection Analytics & Observability
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Aggregated object frequencies, inference performance metrics, and server hardware utilization.
          </p>
        </div>

        <button
          onClick={fetchAnalytics}
          className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Top Telemetry Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <p className="text-xs uppercase font-medium text-slate-400">Total Detections</p>
          <h3 className="text-2xl font-bold text-emerald-400 mt-1">
            {data?.total_detected_objects.toLocaleString() || '0'}
          </h3>
          <p className="text-[11px] text-slate-500 mt-1">Across all inference pipelines</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <p className="text-xs uppercase font-medium text-slate-400">Average FPS</p>
          <h3 className="text-2xl font-bold text-indigo-400 mt-1 font-mono">
            {data?.average_fps || '0.0'}
          </h3>
          <p className="text-[11px] text-slate-500 mt-1">Live frame throughput</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <p className="text-xs uppercase font-medium text-slate-400">CPU Utilization</p>
          <h3 className="text-2xl font-bold text-cyan-400 mt-1 font-mono">
            {data?.system_hardware?.cpu_percent || 0}%
          </h3>
          <p className="text-[11px] text-slate-500 mt-1">Host processor load</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <p className="text-xs uppercase font-medium text-slate-400">Memory Utilization</p>
          <h3 className="text-2xl font-bold text-amber-400 mt-1 font-mono">
            {data?.system_hardware?.memory_percent || 0}%
          </h3>
          <p className="text-[11px] text-slate-500 mt-1">RAM allocation</p>
        </div>
      </div>

      {/* Class Frequency Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              Most Frequent Classes
            </h2>
            <span className="text-xs text-slate-400">Top Classes</span>
          </div>

          {data && Object.keys(data.class_distribution).length > 0 ? (
            <div className="space-y-3">
              {Object.entries(data.class_distribution).map(([className, count]) => {
                const maxVal = Math.max(...Object.values(data.class_distribution), 1);
                const pct = (count / maxVal) * 100;

                return (
                  <div key={className} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="capitalize text-slate-200 font-medium">{className}</span>
                      <span className="font-mono text-emerald-400 font-semibold">{count}</span>
                    </div>
                    <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="py-12 text-center text-slate-500 text-xs">
              No detection class data recorded yet.
            </div>
          )}
        </div>

        {/* Pipeline Distribution */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <PieChart className="w-4 h-4 text-cyan-400" />
              Pipeline Distribution
            </h2>
            <span className="text-xs text-slate-400">By Stream Type</span>
          </div>

          {data && (
            <div className="grid grid-cols-2 gap-3 pt-2">
              {Object.entries(data.session_type_distribution).map(([type, count]) => (
                <div key={type} className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                  <p className="text-xs uppercase font-medium text-slate-400 capitalize">{type}</p>
                  <p className="text-xl font-bold text-white font-mono mt-1">{count}</p>
                  <p className="text-[10px] text-slate-500 mt-0.5">Sessions logged</p>
                </div>
              ))}
            </div>
          )}

          <div className="mt-4 p-4 rounded-xl bg-slate-900/50 border border-slate-800/80 text-xs text-slate-400">
            <div className="flex items-center justify-between">
              <span>Primary Compute:</span>
              <span className="font-mono text-cyan-400 uppercase font-semibold">
                {data?.system_hardware?.compute_device || 'CPU'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
