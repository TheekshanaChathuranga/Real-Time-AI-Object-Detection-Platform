import React, { useEffect, useRef, useState } from 'react';
import {
  Camera,
  CameraOff,
  Clock,
  Sliders,
  Zap
} from 'lucide-react';
import { api } from '../services/api';
import type { Detection, ModelRecord } from '../types';

export const LiveCameraPage: React.FC = () => {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('yolov8n.pt');
  const [confThreshold, setConfThreshold] = useState<number>(0.50);
  const [iouThreshold] = useState<number>(0.45);
  const [classFilter, setClassFilter] = useState<string>('vehicles');

  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [objectCount, setObjectCount] = useState<number>(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const loopRef = useRef<number | null>(null);

  useEffect(() => {
    api.listModels().then((data) => {
      setModels(data);
      const active = data.find((m) => m.is_active);
      if (active) setSelectedModel(active.name);
    }).catch(console.error);

    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      streamRef.current = stream;
      setIsStreaming(true);

      // Connect WebSocket
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/live`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('Live webcam WebSocket connected.');
        startFramePipeline(ws);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.detections) {
            setDetections(data.detections);
            setObjectCount(data.object_count || data.detections.length);
            setFps(data.fps || 0);
            setLatencyMs(data.inference_latency_ms || 0);
            drawBoxes(data.detections);
          }
        } catch (e) {
          console.error(e);
        }
      };

      ws.onerror = (e) => {
        console.warn('WebSocket error, falling back to REST capture:', e);
      };

      wsRef.current = ws;
    } catch (err: any) {
      console.error('Camera access error:', err);
      setError(err.message || 'Unable to access browser webcam. Please grant permission.');
    }
  };

  const stopCamera = () => {
    if (loopRef.current) {
      cancelAnimationFrame(loopRef.current);
      loopRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setIsStreaming(false);
    setDetections([]);
    setObjectCount(0);
    setFps(0);
    setLatencyMs(0);
  };

  const startFramePipeline = (ws: WebSocket) => {
    const hiddenCanvas = document.createElement('canvas');
    hiddenCanvas.width = 640;
    hiddenCanvas.height = 480;
    const hiddenCtx = hiddenCanvas.getContext('2d');

    let isBusy = false;

    const sendFrame = () => {
      if (!streamRef.current || !videoRef.current) return;

      if (
        ws.readyState === WebSocket.OPEN &&
        !isBusy &&
        videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA
      ) {
        hiddenCtx?.drawImage(videoRef.current, 0, 0, 640, 480);
        const frameBase64 = hiddenCanvas.toDataURL('image/jpeg', 0.6);

        isBusy = true;
        const classesParam =
          classFilter === 'vehicles'
            ? 'car,truck,bus,motorcycle,bicycle'
            : classFilter === 'vehicles_pedestrians'
            ? 'car,truck,bus,motorcycle,bicycle,person'
            : undefined;

        ws.send(
          JSON.stringify({
            frame: frameBase64,
            model: selectedModel,
            conf: confThreshold,
            iou: iouThreshold,
            classes: classesParam,
          })
        );
        isBusy = false;
      }

      loopRef.current = requestAnimationFrame(sendFrame);
    };

    loopRef.current = requestAnimationFrame(sendFrame);
  };

  const drawBoxes = (currentDetections: Detection[]) => {
    const canvas = canvasRef.current;
    if (!canvas || !videoRef.current) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 480;

    currentDetections.forEach((det) => {
      const x1 = det.bbox.x1 * scaleX;
      const y1 = det.bbox.y1 * scaleY;
      const width = (det.bbox.x2 - det.bbox.x1) * scaleX;
      const height = (det.bbox.y2 - det.bbox.y1) * scaleY;

      // Draw outer bounding box
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, width, height);

      // Label badge
      const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = 'bold 12px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = '#22c55e';
      ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 8, 22);

      ctx.fillStyle = '#0b0f19';
      ctx.fillText(label, x1 + 4, Math.max(15, y1 - 6));
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <Camera className="w-6 h-6 text-emerald-400" />
          Live Webcam Detection
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Real-time low-latency WebSocket perception pipeline directly in your web browser.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Column */}
        <div className="space-y-5">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-emerald-400" />
              Stream Controls
            </h2>

            {/* Model Selection */}
            <div>
              <label className="text-xs font-medium text-slate-400 block mb-1.5">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isStreaming}
                className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500 disabled:opacity-50"
              >
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Confidence Slider */}
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

            {/* Target Classes */}
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
            </div>

            {/* Start / Stop Camera Button */}
            {!isStreaming ? (
              <button
                onClick={startCamera}
                className="w-full py-3 rounded-xl font-semibold text-sm bg-emerald-500 hover:bg-emerald-600 text-slate-950 flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20 transition"
              >
                <Camera className="w-4 h-4" />
                Start Live Camera
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="w-full py-3 rounded-xl font-semibold text-sm bg-rose-500 hover:bg-rose-600 text-white flex items-center justify-center gap-2 shadow-lg shadow-rose-500/20 transition"
              >
                <CameraOff className="w-4 h-4" />
                Stop Camera Stream
              </button>
            )}

            {error && (
              <div className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl">
                {error}
              </div>
            )}
          </div>

          {/* Real-time Detections List */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-3 flex items-center justify-between">
              <span>Active Detections</span>
              <span className="text-xs font-mono text-emerald-400">{objectCount} items</span>
            </h2>

            {detections.length > 0 ? (
              <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
                {detections.map((det, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between text-xs"
                  >
                    <span className="capitalize font-medium text-slate-200">{det.class_name}</span>
                    <span className="px-2 py-0.5 rounded font-mono font-bold text-[11px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      {(det.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-500 py-3 text-center">
                {isStreaming ? 'Scanning video frames...' : 'Camera stream idle.'}
              </p>
            )}
          </div>
        </div>

        {/* Live Canvas View */}
        <div className="lg:col-span-2 space-y-5">
          <div className="glass-panel p-4 rounded-2xl border border-slate-800 min-h-[500px] flex flex-col justify-between">
            {/* Top HUD */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
              <div className="flex items-center gap-2">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    isStreaming ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'
                  }`}
                ></span>
                <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
                  {isStreaming ? 'Live Viewport' : 'Viewport Inactive'}
                </span>
              </div>

              {isStreaming && (
                <div className="flex items-center gap-4 text-xs font-mono">
                  <span className="flex items-center gap-1.5 text-indigo-400 bg-indigo-500/10 px-2.5 py-1 rounded-lg border border-indigo-500/20">
                    <Zap className="w-3.5 h-3.5" />
                    {fps} FPS
                  </span>
                  <span className="flex items-center gap-1.5 text-amber-400 bg-amber-500/10 px-2.5 py-1 rounded-lg border border-amber-500/20">
                    <Clock className="w-3.5 h-3.5" />
                    {latencyMs} ms
                  </span>
                </div>
              )}
            </div>

            {/* Video + Canvas Stack */}
            <div className="relative w-full aspect-video bg-slate-950 rounded-xl overflow-hidden flex items-center justify-center">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`w-full h-full object-cover ${!isStreaming ? 'hidden' : 'block'}`}
              />
              <canvas
                ref={canvasRef}
                width={640}
                height={480}
                className="absolute inset-0 w-full h-full pointer-events-none z-10"
              />

              {!isStreaming && (
                <div className="flex flex-col items-center gap-3 text-slate-500">
                  <Camera className="w-14 h-14 stroke-[1.5]" />
                  <p className="text-sm">Click "Start Live Camera" to initialize inference</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
