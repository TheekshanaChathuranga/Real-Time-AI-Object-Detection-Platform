import React, { useEffect, useState } from 'react';
import {
  Camera,
  Play,
  Plus,
  Radio,
  RefreshCw,
  Square,
  Trash2,
  Tv
} from 'lucide-react';
import { api } from '../services/api';
import type { CameraRecord, ModelRecord } from '../types';

export const RTSPCameraPage: React.FC = () => {
  const [cameras, setCameras] = useState<CameraRecord[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<CameraRecord | null>(null);
  const [showAddModal, setShowAddModal] = useState<boolean>(false);

  // Form states
  const [name, setName] = useState<string>('Warehouse North Entrance');
  const [rtspUrl, setRtspUrl] = useState<string>('rtsp://127.0.0.1:8554/live');
  const [resolution, setResolution] = useState<string>('1280x720');
  const [targetFps, setTargetFps] = useState<number>(30);
  const [modelName, setModelName] = useState<string>('yolov8n.pt');
  const [error, setError] = useState<string | null>(null);

  const fetchCameras = async () => {
    try {
      const data = await api.listCameras();
      setCameras(data);
      if (data.length > 0 && !selectedCamera) {
        setSelectedCamera(data[0]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchCameras();
    api.listModels().then(setModels).catch(console.error);
    const interval = setInterval(fetchCameras, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleCreateCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const newCam = await api.createCamera({
        name,
        rtsp_url: rtspUrl,
        resolution,
        target_fps: targetFps,
        model_name: modelName,
      });
      setShowAddModal(false);
      fetchCameras();
      setSelectedCamera(newCam);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to register camera.');
    }
  };

  const handleStartStream = async (camId: string) => {
    try {
      await api.startCamera(camId);
      fetchCameras();
    } catch (err) {
      console.error(err);
    }
  };

  const handleStopStream = async (camId: string) => {
    try {
      await api.stopCamera(camId);
      fetchCameras();
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeleteCamera = async (camId: string) => {
    try {
      await api.deleteCamera(camId);
      if (selectedCamera?.camera_id === camId) {
        setSelectedCamera(null);
      }
      fetchCameras();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Radio className="w-6 h-6 text-emerald-400" />
            RTSP & IP Camera Surveillance
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Thread-isolated continuous capture loops for CCTV streams with auto-reconnect logic.
          </p>
        </div>

        <button
          onClick={() => setShowAddModal(true)}
          className="px-4 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold text-xs flex items-center gap-2 shadow-lg shadow-emerald-500/20 transition"
        >
          <Plus className="w-4 h-4" />
          Add RTSP Camera
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Cameras List */}
        <div className="space-y-3">
          <div className="flex items-center justify-between px-1">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Registered Feeds ({cameras.length})
            </span>
            <button onClick={fetchCameras} className="text-slate-400 hover:text-white transition">
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>

          {cameras.length > 0 ? (
            cameras.map((cam) => {
              const isSelected = selectedCamera?.camera_id === cam.camera_id;
              const isRunning = cam.status === 'RUNNING';

              return (
                <div
                  key={cam.camera_id}
                  onClick={() => setSelectedCamera(cam)}
                  className={`p-4 rounded-2xl border transition cursor-pointer ${
                    isSelected
                      ? 'bg-slate-900 border-emerald-500/40 shadow-lg shadow-emerald-500/5'
                      : 'glass-panel border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h2 className="text-sm font-semibold text-white">{cam.name}</h2>
                      <p className="text-xs text-slate-400 font-mono truncate max-w-[200px] mt-0.5">
                        {cam.rtsp_url}
                      </p>
                    </div>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${
                        isRunning
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                          : 'bg-slate-800 text-slate-400 border-slate-700'
                      }`}
                    >
                      {cam.status}
                    </span>
                  </div>

                  <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
                    <span className="text-slate-400 font-mono">
                      {cam.resolution} @ {cam.target_fps}fps
                    </span>

                    <div className="flex items-center gap-2">
                      {!isRunning ? (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartStream(cam.camera_id);
                          }}
                          className="px-2.5 py-1 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 font-medium flex items-center gap-1 border border-emerald-500/30 transition"
                        >
                          <Play className="w-3 h-3 fill-current" />
                          Start
                        </button>
                      ) : (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStopStream(cam.camera_id);
                          }}
                          className="px-2.5 py-1 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 font-medium flex items-center gap-1 border border-rose-500/30 transition"
                        >
                          <Square className="w-3 h-3 fill-current" />
                          Stop
                        </button>
                      )}

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteCamera(cam.camera_id);
                        }}
                        className="p-1 rounded-lg hover:bg-slate-800 text-slate-500 hover:text-rose-400 transition"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="glass-panel p-8 rounded-2xl border border-slate-800 text-center text-slate-500 text-xs">
              No RTSP cameras registered yet. Click "Add RTSP Camera" to configure an IP feed.
            </div>
          )}
        </div>

        {/* Live Preview Viewport */}
        <div className="lg:col-span-2 space-y-5">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 min-h-[500px] flex flex-col justify-between">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
              <div>
                <h2 className="text-sm font-semibold text-white">
                  {selectedCamera ? selectedCamera.name : 'Select a Camera'}
                </h2>
                {selectedCamera && (
                  <p className="text-xs text-slate-400 font-mono mt-0.5">{selectedCamera.rtsp_url}</p>
                )}
              </div>

              {selectedCamera && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Model:</span>
                  <span className="text-xs font-mono text-emerald-400">{selectedCamera.model_name}</span>
                </div>
              )}
            </div>

            {/* Video Canvas / MJPEG Stream */}
            <div className="flex-1 bg-slate-950 rounded-xl overflow-hidden relative flex items-center justify-center min-h-[380px]">
              {selectedCamera && selectedCamera.status === 'RUNNING' ? (
                <img
                  src={`/api/v1/cameras/${selectedCamera.camera_id}/preview`}
                  alt="Live MJPEG Camera Stream"
                  className="max-h-[460px] w-full object-contain"
                />
              ) : selectedCamera ? (
                <div className="flex flex-col items-center gap-3 text-slate-500">
                  <Tv className="w-12 h-12 stroke-[1.5]" />
                  <p className="text-sm">Camera stream is currently {selectedCamera.status.toLowerCase()}</p>
                  <button
                    onClick={() => handleStartStream(selectedCamera.camera_id)}
                    className="px-4 py-2 rounded-xl bg-emerald-500 text-slate-950 text-xs font-semibold flex items-center gap-1.5"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    Start Stream
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 text-slate-500 text-sm">
                  <Radio className="w-12 h-12 stroke-[1.5]" />
                  Select an RTSP camera to view live telemetry
                </div>
              )}
            </div>

            {/* Telemetry Footer */}
            {selectedCamera && (
              <div className="mt-3 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
                <span className="flex items-center gap-1">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      selectedCamera.status === 'RUNNING' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'
                    }`}
                  ></span>
                  Status: <b className="text-white font-mono">{selectedCamera.status}</b>
                </span>
                <span>
                  Config: <b className="text-white font-mono">{selectedCamera.resolution}</b>
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Add Camera Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-panel bg-slate-950 p-6 rounded-2xl border border-slate-800 w-full max-w-md space-y-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Camera className="w-5 h-5 text-emerald-400" />
              Configure RTSP Camera
            </h2>

            <form onSubmit={handleCreateCamera} className="space-y-3 text-xs">
              <div>
                <label className="text-slate-400 font-medium block mb-1">Camera Name</label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div>
                <label className="text-slate-400 font-medium block mb-1">RTSP Stream URL</label>
                <input
                  type="text"
                  required
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  placeholder="rtsp://192.168.1.100:554/stream"
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-sm font-mono focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-slate-400 font-medium block mb-1">Resolution</label>
                  <input
                    type="text"
                    value={resolution}
                    onChange={(e) => setResolution(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="text-slate-400 font-medium block mb-1">Target FPS</label>
                  <input
                    type="number"
                    min="5"
                    max="60"
                    value={targetFps}
                    onChange={(e) => setTargetFps(parseInt(e.target.value) || 30)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div>
                <label className="text-slate-400 font-medium block mb-1">Inference Model</label>
                <select
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                >
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>

              {error && (
                <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
                  {error}
                </div>
              )}

              <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold transition"
                >
                  Save Camera
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
