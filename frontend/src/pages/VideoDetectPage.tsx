import React, { useEffect, useState } from 'react';
import {
  Download,
  Film,
  Play,
  Sliders,
  Video
} from 'lucide-react';
import { api } from '../services/api';
import type { ModelRecord, VideoDetectionResponse } from '../types';

export const VideoDetectPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('yolov8n.pt');
  const [confThreshold, setConfThreshold] = useState<number>(0.50);
  const [classFilter, setClassFilter] = useState<string>('vehicles');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<VideoDetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listModels().then((data) => {
      setModels(data);
      const active = data.find((m) => m.is_active);
      if (active) setSelectedModel(active.name);
    }).catch(console.error);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleStartProcessing = async () => {
    if (!selectedFile) {
      setError('Please select a video file.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_name', selectedModel);
    formData.append('conf_threshold', confThreshold.toString());

    if (classFilter === 'vehicles') {
      formData.append('classes', 'car,truck,bus,motorcycle,bicycle');
    } else if (classFilter === 'vehicles_pedestrians') {
      formData.append('classes', 'car,truck,bus,motorcycle,bicycle,person');
    }

    try {
      const res = await api.detectVideo(formData);
      setResult(res);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.error || err.message || 'Video processing failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <Video className="w-6 h-6 text-emerald-400" />
          Video Stream Detection
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Memory-efficient frame-by-frame streaming inference with annotated video export.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Settings & Upload */}
        <div className="space-y-5">
          <div
            className="glass-panel p-6 rounded-2xl border-2 border-dashed border-slate-700 hover:border-emerald-500/50 transition flex flex-col items-center justify-center text-center cursor-pointer group"
            onClick={() => document.getElementById('video-upload-input')?.click()}
          >
            <input
              id="video-upload-input"
              type="file"
              accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="p-4 rounded-2xl bg-emerald-500/10 text-emerald-400 group-hover:scale-110 transition">
              <Film className="w-6 h-6" />
            </div>
            <p className="text-sm font-semibold text-white mt-3">Select Video File</p>
            <p className="text-xs text-slate-400 mt-1">MP4, AVI, MOV (Frame-by-frame streaming)</p>
            {selectedFile && (
              <span className="mt-3 px-3 py-1 bg-emerald-500/10 text-emerald-400 text-xs font-mono rounded-lg border border-emerald-500/30">
                {selectedFile.name} ({(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)
              </span>
            )}
          </div>

          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-emerald-400" />
              Processing Config
            </h2>

            <div>
              <label className="text-xs font-medium text-slate-400 block mb-1.5">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
              >
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Confidence Threshold</span>
                <span className="text-emerald-400 font-mono font-semibold">
                  {(confThreshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                className="w-full accent-emerald-500 cursor-pointer"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-slate-400 block mb-1.5">Target Detection Classes</label>
              <select
                value={classFilter}
                onChange={(e) => setClassFilter(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
              >
                <option value="vehicles">🚗 Vehicles Only (Road/Highway Traffic)</option>
                <option value="vehicles_pedestrians">🚶 Vehicles + Pedestrians</option>
                <option value="all">🌐 All Classes (COCO 80)</option>
              </select>
              <p className="text-[11px] text-slate-400 mt-1">
                Filters out non-road objects (e.g. overhead structures mistagged as train).
              </p>
            </div>

            <button
              onClick={handleStartProcessing}
              disabled={loading || !selectedFile}
              className={`w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition ${
                loading || !selectedFile
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-emerald-500 hover:bg-emerald-600 text-slate-950 shadow-lg shadow-emerald-500/20'
              }`}
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></div>
                  Streaming Frames...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  Process Video
                </>
              )}
            </button>

            {error && (
              <div className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Video Player & Results */}
        <div className="lg:col-span-2 space-y-5">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 min-h-[460px] flex flex-col justify-between">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 mb-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Annotated Video Stream
              </span>
              {result?.processed_video_url && (
                <a
                  href={result.processed_video_url}
                  download={`detected_${result.session_id.slice(0, 8)}.mp4`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 text-xs font-medium border border-emerald-500/30 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download Processed Video
                </a>
              )}
            </div>

            <div className="flex-1 flex items-center justify-center bg-slate-950/80 rounded-xl overflow-hidden p-2 min-h-[340px]">
              {result?.processed_video_url ? (
                <video
                  src={result.processed_video_url}
                  controls
                  className="max-h-[440px] w-full object-contain rounded-lg"
                />
              ) : loading ? (
                <div className="flex flex-col items-center gap-3 text-slate-400">
                  <div className="w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                  <span className="text-sm">Iterating video frames with YOLO detector...</span>
                  <p className="text-xs text-slate-500">Writing annotated MP4 video output to disk</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 text-slate-500">
                  <Video className="w-12 h-12 stroke-[1.5]" />
                  <span className="text-sm">Upload a video to stream detections</span>
                </div>
              )}
            </div>

            {result && (
              <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-3 bg-slate-900 rounded-xl border border-slate-800">
                  <p className="text-slate-400 text-[11px]">Total Frames</p>
                  <p className="text-base font-bold text-white font-mono mt-0.5">{result.total_frames}</p>
                </div>
                <div className="p-3 bg-slate-900 rounded-xl border border-slate-800">
                  <p className="text-slate-400 text-[11px]">Average FPS</p>
                  <p className="text-base font-bold text-indigo-400 font-mono mt-0.5">{result.average_fps}</p>
                </div>
                <div className="p-3 bg-slate-900 rounded-xl border border-slate-800">
                  <p className="text-slate-400 text-[11px]">Average Latency</p>
                  <p className="text-base font-bold text-amber-400 font-mono mt-0.5">{result.average_latency_ms} ms</p>
                </div>
                <div className="p-3 bg-slate-900 rounded-xl border border-slate-800">
                  <p className="text-slate-400 text-[11px]">Detections</p>
                  <p className="text-base font-bold text-emerald-400 font-mono mt-0.5">{result.total_detections}</p>
                </div>
              </div>
            )}
          </div>

          {/* Class Breakdown */}
          {result && Object.keys(result.class_distribution).length > 0 && (
            <div className="glass-panel p-5 rounded-2xl border border-slate-800">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-3">
                Detected Class Distribution
              </h2>
              <div className="flex flex-wrap gap-2">
                {Object.entries(result.class_distribution).map(([cls, count]) => (
                  <span
                    key={cls}
                    className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs flex items-center gap-2"
                  >
                    <span className="capitalize text-slate-300 font-medium">{cls}</span>
                    <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-mono font-bold text-[11px]">
                      {count}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
