import React, { useEffect, useState } from 'react';
import {
  Activity,
  Camera,
  CheckCircle2,
  Clock,
  Cpu,
  Layers,
  Sparkles,
  Zap
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { api } from '../services/api';
import type { DashboardStats } from '../types';

export const DashboardPage: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchStats = async () => {
    try {
      const data = await api.getDashboardStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to load dashboard metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center h-full min-h-[400px]">
        <div className="flex flex-col items-center gap-3 text-slate-400">
          <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
          <span className="text-sm">Connecting to AI Perception Engine...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Hero Welcome & Platform Status */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-emerald-950/40 via-slate-900/60 to-slate-950 border border-emerald-500/20 shadow-xl shadow-emerald-500/5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            Perception Overview
            <span className="px-2.5 py-0.5 text-xs font-semibold bg-emerald-500/20 text-emerald-300 rounded-full border border-emerald-500/30">
              Active
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Real-time multi-stream YOLO perception engine running high-throughput object detection,
            video analytics, webcam tracking, and custom model registries.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="px-4 py-2 rounded-xl bg-slate-900/80 border border-slate-800 text-right">
            <p className="text-[11px] text-slate-400 uppercase font-bold tracking-wider">Compute Target</p>
            <p className="text-sm font-bold text-cyan-400 uppercase font-mono">
              {stats?.hardware_device || 'CPU'}
            </p>
          </div>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Inferences"
          value={stats?.total_sessions.toLocaleString() || '0'}
          subtitle="Image, Video, & Streams"
          icon={Layers}
          color="emerald"
        />
        <StatCard
          title="Detected Objects"
          value={stats?.total_detected_objects.toLocaleString() || '0'}
          subtitle="Bounding boxes parsed"
          icon={Sparkles}
          color="cyan"
        />
        <StatCard
          title="Average Latency"
          value={`${stats?.average_latency_ms || 0} ms`}
          subtitle="Inference pipeline time"
          icon={Clock}
          color="amber"
        />
        <StatCard
          title="Throughput (FPS)"
          value={`${stats?.average_fps || 0} FPS`}
          subtitle="Real-time frame rate"
          icon={Zap}
          color="indigo"
        />
      </div>

      {/* Second Row: Active Camera Streams & Model Registry Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Camera className="w-4 h-4 text-emerald-400" />
                Live Camera Feeds
              </h2>
              <span className="px-2 py-0.5 text-xs font-mono bg-emerald-500/10 text-emerald-400 rounded-md border border-emerald-500/20">
                {stats?.active_cameras || 0} Connected
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Thread-isolated RTSP capture loops continuously process IP camera streams with automatic
              reconnection and low-latency MJPEG preview.
            </p>
          </div>
          <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
            <span>Resilient Worker Isolation</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-cyan-400" />
                Model Registry
              </h2>
              <span className="px-2 py-0.5 text-xs font-mono bg-cyan-500/10 text-cyan-400 rounded-md border border-cyan-500/20">
                {stats?.available_models || 0} Models
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Pretrained YOLO weights and registered custom models with mAP50, precision, recall
              evaluation benchmarks and one-click hot swapping.
            </p>
          </div>
          <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
            <span>Dynamic Model Loading</span>
            <CheckCircle2 className="w-4 h-4 text-cyan-400" />
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Activity className="w-4 h-4 text-indigo-400" />
                Pipeline Telemetry
              </h2>
              <span className="px-2 py-0.5 text-xs font-mono bg-indigo-500/10 text-indigo-400 rounded-md border border-indigo-500/20">
                Healthy
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Separation of preprocessing, forward inference pass, and non-max suppression postprocessing
              metrics for complete transparency.
            </p>
          </div>
          <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
            <span>Decoupled ML Engine</span>
            <CheckCircle2 className="w-4 h-4 text-indigo-400" />
          </div>
        </div>
      </div>

      {/* Recent Inference Sessions Table */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
        <div className="p-5 border-b border-slate-800/80 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-white">Recent Detection Sessions</h2>
            <p className="text-xs text-slate-400 mt-0.5">Audit log of latest inference activities</p>
          </div>
          <button
            onClick={fetchStats}
            className="text-xs px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition"
          >
            Refresh
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="py-3 px-5">Session ID</th>
                <th className="py-3 px-5">Type</th>
                <th className="py-3 px-5">Model</th>
                <th className="py-3 px-5">Objects</th>
                <th className="py-3 px-5">FPS</th>
                <th className="py-3 px-5">Latency</th>
                <th className="py-3 px-5">Timestamp</th>
                <th className="py-3 px-5">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {stats?.recent_sessions && stats.recent_sessions.length > 0 ? (
                stats.recent_sessions.map((sess) => (
                  <tr key={sess.session_id} className="hover:bg-slate-900/40 transition">
                    <td className="py-3 px-5 font-mono text-slate-300">
                      {sess.session_id.slice(0, 8)}...
                    </td>
                    <td className="py-3 px-5">
                      <span className="capitalize px-2 py-0.5 rounded text-[11px] font-medium bg-slate-800 text-slate-300">
                        {sess.session_type}
                      </span>
                    </td>
                    <td className="py-3 px-5 font-mono text-emerald-400">{sess.model_name}</td>
                    <td className="py-3 px-5 font-semibold text-white">{sess.object_count}</td>
                    <td className="py-3 px-5 text-slate-300">{sess.fps.toFixed(1)}</td>
                    <td className="py-3 px-5 text-slate-300">{sess.latency_ms.toFixed(1)} ms</td>
                    <td className="py-3 px-5 text-slate-400">
                      {new Date(sess.start_time).toLocaleTimeString()}
                    </td>
                    <td className="py-3 px-5">
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        {sess.status}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-slate-500">
                    No inference sessions recorded yet. Run an image or video detection to begin!
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
