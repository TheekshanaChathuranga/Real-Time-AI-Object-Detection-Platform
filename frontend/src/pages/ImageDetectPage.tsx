import React, { useEffect, useState } from 'react';
import {
  Clock,
  Download,
  FileCheck,
  ImageIcon,
  Play,
  Sliders,
  Sparkles,
  Upload,
  Zap
} from 'lucide-react';
import { api } from '../services/api';
import type { ImageDetectionResponse, ModelRecord } from '../types';

export const ImageDetectPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('yolov8n.pt');
  const [confThreshold, setConfThreshold] = useState<number>(0.50);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [classFilter, setClassFilter] = useState<string>('vehicles');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ImageDetectionResponse | null>(null);
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
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleRunDetection = async () => {
    if (!selectedFile) {
      setError('Please select an image file first.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_name', selectedModel);
    formData.append('conf_threshold', confThreshold.toString());
    formData.append('iou_threshold', iouThreshold.toString());

    if (classFilter === 'vehicles') {
      formData.append('classes', 'car,truck,bus,motorcycle,bicycle');
    } else if (classFilter === 'vehicles_pedestrians') {
      formData.append('classes', 'car,truck,bus,motorcycle,bicycle,person');
    }

    try {
      const res = await api.detectImage(formData);
      setResult(res);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.error || err.message || 'Failed to perform image detection.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result?.annotated_image_base64) return;
    const a = document.createElement('a');
    a.href = result.annotated_image_base64;
    a.download = `detection_${result.session_id.slice(0, 8)}.jpg`;
    a.click();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <ImageIcon className="w-6 h-6 text-emerald-400" />
          Image Object Detection
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Perform high-precision bounding box inference with configurable confidence and IoU thresholds.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Column */}
        <div className="space-y-5">
          {/* Upload Area */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="glass-panel p-6 rounded-2xl border-2 border-dashed border-slate-700 hover:border-emerald-500/50 transition flex flex-col items-center justify-center text-center cursor-pointer group"
            onClick={() => document.getElementById('image-upload-input')?.click()}
          >
            <input
              id="image-upload-input"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="p-4 rounded-2xl bg-emerald-500/10 text-emerald-400 group-hover:scale-110 transition">
              <Upload className="w-6 h-6" />
            </div>
            <p className="text-sm font-semibold text-white mt-3">Click or Drag Image Here</p>
            <p className="text-xs text-slate-400 mt-1">Supports JPEG, PNG, WEBP (up to 100MB)</p>
            {selectedFile && (
              <span className="mt-3 px-3 py-1 bg-emerald-500/10 text-emerald-400 text-xs font-mono rounded-lg border border-emerald-500/30 flex items-center gap-1.5">
                <FileCheck className="w-3.5 h-3.5" />
                {selectedFile.name}
              </span>
            )}
          </div>

          {/* Model & Hyperparameters Config */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-emerald-400" />
              Inference Parameters
            </h2>

            {/* Model Selection */}
            <div>
              <label className="text-xs font-medium text-slate-400 block mb-1.5">Detection Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
              >
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name} {m.is_active ? '(Active)' : ''}
                  </option>
                ))}
              </select>
            </div>

            {/* Confidence Threshold */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Confidence Threshold</span>
                <span className="text-emerald-400 font-mono font-semibold">
                  {(confThreshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                className="w-full accent-emerald-500 cursor-pointer"
              />
            </div>

            {/* IoU Threshold */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">IoU (NMS) Threshold</span>
                <span className="text-cyan-400 font-mono font-semibold">
                  {(iouThreshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={iouThreshold}
                onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
            </div>

            {/* Target Detection Classes */}
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
                Suppresses false alarms like overhead bridges or structures detected as train.
              </p>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunDetection}
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
                  Inferencing...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  Run Object Detection
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

        {/* Display Canvas & Output Column */}
        <div className="lg:col-span-2 space-y-5">
          {/* Main Visualizer Window */}
          <div className="glass-panel p-4 rounded-2xl border border-slate-800 min-h-[460px] flex flex-col justify-between">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 mb-3">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                <Sparkles className="w-4 h-4 text-emerald-400" />
                Detection Canvas
              </div>
              {result && (
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-emerald-400 text-xs font-medium border border-emerald-500/30 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download Result
                </button>
              )}
            </div>

            {/* Image Preview / Annotated Result */}
            <div className="flex-1 flex items-center justify-center bg-slate-950/80 rounded-xl overflow-hidden relative min-h-[360px] p-2">
              {result?.annotated_image_base64 ? (
                <img
                  src={result.annotated_image_base64}
                  alt="Annotated detection output"
                  className="max-h-[500px] w-auto object-contain rounded-lg shadow-2xl"
                />
              ) : previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Uploaded preview"
                  className="max-h-[500px] w-auto object-contain rounded-lg opacity-80"
                />
              ) : (
                <div className="flex flex-col items-center gap-2 text-slate-500">
                  <ImageIcon className="w-12 h-12 stroke-[1.5]" />
                  <span className="text-sm">Upload an image to visualize detections</span>
                </div>
              )}
            </div>

            {/* Performance Bar */}
            {result && (
              <div className="mt-3 pt-3 border-t border-slate-800 flex flex-wrap items-center justify-between text-xs gap-3">
                <div className="flex items-center gap-4">
                  <span className="text-slate-400 flex items-center gap-1">
                    <Zap className="w-3.5 h-3.5 text-indigo-400" />
                    FPS: <b className="text-white font-mono">{result.metrics.fps}</b>
                  </span>
                  <span className="text-slate-400 flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-amber-400" />
                    Latency: <b className="text-white font-mono">{result.metrics.total_latency_ms} ms</b>
                    <span className="text-[10px] text-slate-500 font-mono">
                      (Inf: {result.metrics.inference_time_ms}ms)
                    </span>
                  </span>
                </div>
                <div className="text-slate-400">
                  Resolution:{' '}
                  <span className="text-white font-mono">
                    {result.image_width} × {result.image_height}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Detections Summary Table / Badges */}
          {result && (
            <div className="glass-panel p-5 rounded-2xl border border-slate-800">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-3 flex items-center justify-between">
                <span>Detected Classes ({result.object_count})</span>
                <span className="text-xs font-mono text-emerald-400">Session: {result.session_id.slice(0, 8)}</span>
              </h2>

              {result.detections.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                  {result.detections.map((det, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between"
                    >
                      <div>
                        <p className="text-xs font-semibold text-white capitalize">{det.class_name}</p>
                        <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                          [{det.bbox.x1.toFixed(0)}, {det.bbox.y1.toFixed(0)}, {det.bbox.x2.toFixed(0)}, {det.bbox.y2.toFixed(0)}]
                        </p>
                      </div>
                      <span className="px-2 py-0.5 rounded text-xs font-bold font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        {(det.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500 text-center py-4">
                  No objects detected above {(confThreshold * 100).toFixed(0)}% confidence.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
