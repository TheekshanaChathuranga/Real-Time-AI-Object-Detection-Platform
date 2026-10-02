import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  Cpu,
  Upload,
  Zap
} from 'lucide-react';
import { api } from '../services/api';
import type { ModelRecord } from '../types';

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [showUploadModal, setShowUploadModal] = useState<boolean>(false);

  // Upload modal states
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [version, setVersion] = useState<string>('v1.0');
  const [datasetName, setDatasetName] = useState<string>('custom-dataset');
  const [notes, setNotes] = useState<string>('Fine-tuned custom YOLO weights');
  const [uploading, setUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = async () => {
    try {
      const data = await api.listModels();
      setModels(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleActivate = async (modelName: string) => {
    try {
      await api.activateModel(modelName);
      fetchModels();
    } catch (e) {
      console.error(e);
    }
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) {
      setError('Please select a .pt weights file.');
      return;
    }
    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('version', version);
    formData.append('dataset_name', datasetName);
    formData.append('notes', notes);

    try {
      await api.uploadModel(formData);
      setShowUploadModal(false);
      setUploadFile(null);
      fetchModels();
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Cpu className="w-6 h-6 text-emerald-400" />
            Model Registry & Version Management
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Seamlessly switch between baseline pretrained YOLO weights and fine-tuned custom models.
          </p>
        </div>

        <button
          onClick={() => setShowUploadModal(true)}
          className="px-4 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold text-xs flex items-center gap-2 shadow-lg shadow-emerald-500/20 transition"
        >
          <Upload className="w-4 h-4" />
          Upload Custom Weights
        </button>
      </div>

      {/* Models Table */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/70 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="py-3 px-5">Model</th>
                <th className="py-3 px-5">Type</th>
                <th className="py-3 px-5">Version</th>
                <th className="py-3 px-5">Dataset</th>
                <th className="py-3 px-5">mAP50</th>
                <th className="py-3 px-5">mAP50-95</th>
                <th className="py-3 px-5">Precision</th>
                <th className="py-3 px-5">Recall</th>
                <th className="py-3 px-5">State</th>
                <th className="py-3 px-5 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {models.map((m) => {
                const latestVersion = m.versions && m.versions.length > 0 ? m.versions[m.versions.length - 1] : null;

                return (
                  <tr key={m.id} className="hover:bg-slate-900/40 transition">
                    <td className="py-4 px-5">
                      <div className="flex items-center gap-2.5">
                        <div className="h-8 w-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center shrink-0">
                          <Zap className="w-4 h-4 text-emerald-400" />
                        </div>
                        <div>
                          <p className="font-semibold text-white font-mono">{m.name}</p>
                          <p className="text-[11px] text-slate-400 truncate max-w-[220px]">
                            {m.description || 'YOLO Object Detector'}
                          </p>
                        </div>
                      </div>
                    </td>
                    <td className="py-4 px-5">
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[11px]">
                        {m.model_type}
                      </span>
                    </td>
                    <td className="py-4 px-5 font-mono text-slate-300">
                      {latestVersion ? latestVersion.version : 'v1.0'}
                    </td>
                    <td className="py-4 px-5 text-slate-400">
                      {latestVersion ? latestVersion.dataset_name : 'COCO'}
                    </td>
                    <td className="py-4 px-5 font-mono font-semibold text-emerald-400">
                      {latestVersion ? (latestVersion.map50 * 100).toFixed(1) + '%' : '-'}
                    </td>
                    <td className="py-4 px-5 font-mono text-cyan-400">
                      {latestVersion ? (latestVersion.map50_95 * 100).toFixed(1) + '%' : '-'}
                    </td>
                    <td className="py-4 px-5 font-mono text-slate-300">
                      {latestVersion ? (latestVersion.precision * 100).toFixed(1) + '%' : '-'}
                    </td>
                    <td className="py-4 px-5 font-mono text-slate-300">
                      {latestVersion ? (latestVersion.recall * 100).toFixed(1) + '%' : '-'}
                    </td>
                    <td className="py-4 px-5">
                      {m.is_active ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                          <CheckCircle2 className="w-3 h-3" />
                          Active
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400">
                          Available
                        </span>
                      )}
                    </td>
                    <td className="py-4 px-5 text-right">
                      {!m.is_active ? (
                        <button
                          onClick={() => handleActivate(m.name)}
                          className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-emerald-400 border border-emerald-500/30 text-xs font-semibold transition"
                        >
                          Set Active
                        </button>
                      ) : (
                        <span className="text-xs text-slate-500 italic">Current Engine</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Upload Model Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-panel bg-slate-950 p-6 rounded-2xl border border-slate-800 w-full max-w-md space-y-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Upload className="w-5 h-5 text-emerald-400" />
              Upload Custom Model Weights
            </h2>

            <form onSubmit={handleUploadSubmit} className="space-y-3 text-xs">
              <div>
                <label className="text-slate-400 font-medium block mb-1">Weights File (.pt, .onnx)</label>
                <input
                  type="file"
                  required
                  accept=".pt,.onnx"
                  onChange={(e) => setUploadFile(e.target.files ? e.target.files[0] : null)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-slate-400 font-medium block mb-1">Version Tag</label>
                  <input
                    type="text"
                    required
                    value={version}
                    onChange={(e) => setVersion(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="text-slate-400 font-medium block mb-1">Dataset Name</label>
                  <input
                    type="text"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div>
                <label className="text-slate-400 font-medium block mb-1">Notes</label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={2}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                />
              </div>

              {error && (
                <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
                  {error}
                </div>
              )}

              <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowUploadModal(false)}
                  className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={uploading}
                  className="px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold transition"
                >
                  {uploading ? 'Uploading...' : 'Register Model'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
