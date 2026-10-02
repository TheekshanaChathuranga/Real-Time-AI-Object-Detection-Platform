import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Flame,
  Play,
  RefreshCw,
  Zap
} from 'lucide-react';
import { api } from '../services/api';
import type { DatasetValidationReport, ModelRecord, TrainingJob } from '../types';

export const TrainingPage: React.FC = () => {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);

  // Form states
  const [datasetYaml, setDatasetYaml] = useState<string>('data/datasets/coco8/data.yaml');
  const [selectedModel, setSelectedModel] = useState<string>('yolov8n.pt');
  const [epochs, setEpochs] = useState<number>(10);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [imageSize, setImageSize] = useState<number>(640);
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [device, setDevice] = useState<string>('auto');

  // Validation report state
  const [validationReport, setValidationReport] = useState<DatasetValidationReport | null>(null);
  const [validating, setValidating] = useState<boolean>(false);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = async () => {
    try {
      const data = await api.listTrainingJobs();
      setJobs(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchJobs();
    api.listModels().then(setModels).catch(console.error);
    const interval = setInterval(fetchJobs, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleValidateDataset = async () => {
    if (!datasetYaml.trim()) {
      setError('Please provide a data.yaml path.');
      return;
    }
    setValidating(true);
    setError(null);
    try {
      const report = await api.validateDataset(datasetYaml);
      setValidationReport(report);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Validation failed.');
    } finally {
      setValidating(false);
    }
  };

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await api.startTrainingJob({
        model_name: selectedModel,
        dataset_yaml: datasetYaml,
        epochs,
        batch_size: batchSize,
        image_size: imageSize,
        learning_rate: learningRate,
        device,
      });
      fetchJobs();
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to dispatch training job.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Flame className="w-6 h-6 text-amber-500" />
            Custom YOLO Model Training
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            End-to-end dataset validation, background training workers, and automated model registry evaluation.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Training Dispatch Form */}
        <div className="space-y-5">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Zap className="w-4 h-4 text-emerald-400" />
              Configure Training Run
            </h2>

            <form onSubmit={handleStartTraining} className="space-y-3.5 text-xs">
              <div>
                <label className="text-slate-400 font-medium block mb-1">Dataset Config (data.yaml)</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    required
                    value={datasetYaml}
                    onChange={(e) => setDatasetYaml(e.target.value)}
                    placeholder="path/to/data.yaml"
                    className="flex-1 bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-xs font-mono focus:outline-none focus:border-emerald-500"
                  />
                  <button
                    type="button"
                    onClick={handleValidateDataset}
                    disabled={validating}
                    className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl font-medium shrink-0 transition"
                  >
                    {validating ? 'Testing...' : 'Validate'}
                  </button>
                </div>
              </div>

              <div>
                <label className="text-slate-400 font-medium block mb-1">Base Pretrained Weights</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
                >
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-slate-400 font-medium block mb-1">Epochs</label>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="text-slate-400 font-medium block mb-1">Batch Size</label>
                  <input
                    type="number"
                    min="1"
                    max="128"
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value) || 16)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-slate-400 font-medium block mb-1">Image Size (px)</label>
                  <input
                    type="number"
                    min="320"
                    max="1280"
                    step="32"
                    value={imageSize}
                    onChange={(e) => setImageSize(parseInt(e.target.value) || 640)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="text-slate-400 font-medium block mb-1">Learning Rate</label>
                  <input
                    type="number"
                    step="0.001"
                    min="0.0001"
                    max="0.1"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div>
                <label className="text-slate-400 font-medium block mb-1">Compute Target</label>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
                >
                  <option value="auto">Auto (CUDA if available, else CPU)</option>
                  <option value="cpu">CPU Only</option>
                  <option value="cuda:0">NVIDIA GPU (cuda:0)</option>
                </select>
              </div>

              {error && (
                <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={submitting}
                className="w-full py-3 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-amber-500/20 transition"
              >
                <Play className="w-4 h-4 fill-current" />
                Launch Background Training Job
              </button>
            </form>
          </div>

          {/* Dataset Validation Card */}
          {validationReport && (
            <div
              className={`p-4 rounded-2xl border text-xs space-y-2 ${
                validationReport.is_valid
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                  : 'bg-rose-500/10 border-rose-500/30 text-rose-300'
              }`}
            >
              <div className="flex items-center gap-2 font-semibold">
                {validationReport.is_valid ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    Dataset Validated Successfully
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4 text-rose-400" />
                    Validation Issues Found
                  </>
                )}
              </div>
              <div className="space-y-1 text-slate-300 text-[11px]">
                <p>Train Images: {validationReport.train_images_count}</p>
                <p>Val Images: {validationReport.val_images_count}</p>
                <p>Total Classes: {validationReport.num_classes} ({validationReport.class_names.join(', ')})</p>
                <p>Annotations: {validationReport.total_annotations}</p>
              </div>
              {validationReport.errors.length > 0 && (
                <div className="text-[10px] text-rose-400 mt-2 space-y-0.5">
                  {validationReport.errors.map((err, i) => (
                    <p key={i}>• {err}</p>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Training Jobs Table & Telemetry */}
        <div className="lg:col-span-2 space-y-5">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 mb-4">
              <div>
                <h2 className="text-sm font-semibold text-white">Active & Past Training Jobs</h2>
                <p className="text-xs text-slate-400 mt-0.5">Non-blocking background training execution</p>
              </div>
              <button onClick={fetchJobs} className="text-slate-400 hover:text-white transition">
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>

            {jobs.length > 0 ? (
              <div className="space-y-4">
                {jobs.map((job) => {
                  const total = job.total_epochs || job.epochs || 1;
                  const pct = total > 0 ? (job.current_epoch / total) * 100 : 0;

                  return (
                    <div
                      key={job.job_id}
                      className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3"
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-white text-xs font-mono">{job.job_id}</span>
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${
                                job.status === 'COMPLETED'
                                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                                  : job.status === 'RUNNING'
                                  ? 'bg-amber-500/10 text-amber-400 border-amber-500/20 animate-pulse'
                                  : 'bg-slate-800 text-slate-400 border-slate-700'
                              }`}
                            >
                              {job.status}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 mt-0.5">
                            Model: <span className="font-mono text-slate-300">{job.model_name}</span> | Dataset:{' '}
                            <span className="font-mono text-slate-300">{job.dataset_path}</span>
                          </p>
                        </div>

                        <div className="text-right text-xs">
                          <span className="font-mono text-slate-300">
                            Epoch {job.current_epoch} / {total}
                          </span>
                        </div>
                      </div>

                      {/* Progress bar */}
                      <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${
                            job.status === 'COMPLETED' ? 'bg-emerald-500' : 'bg-amber-500'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>

                      {/* Metrics row */}
                      <div className="grid grid-cols-4 gap-2 pt-2 border-t border-slate-800/80 text-[11px]">
                        <div>
                          <span className="text-slate-500 block">mAP50</span>
                          <span className="font-mono font-semibold text-emerald-400">
                            {(job.map50 * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500 block">mAP50-95</span>
                          <span className="font-mono text-cyan-400">
                            {(job.map50_95 * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500 block">Precision</span>
                          <span className="font-mono text-slate-300">
                            {(job.precision * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500 block">Recall</span>
                          <span className="font-mono text-slate-300">
                            {(job.recall * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {job.best_model_path && (
                        <p className="text-[10px] text-slate-400 font-mono truncate">
                          Best Weights: {job.best_model_path}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="py-12 text-center text-slate-500 text-xs">
                No custom training jobs submitted yet. Fill out the configuration to launch a run.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
