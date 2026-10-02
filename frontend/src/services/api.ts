import axios from 'axios';
import type {
  AnalyticsBreakdown,
  CameraRecord,
  CameraTelemetry,
  DashboardStats,
  DatasetValidationReport,
  ImageDetectionResponse,
  ModelRecord,
  TrainingJob,
  VideoDetectionResponse
} from '../types';

const client = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Health
  getHealth: async () => {
    const res = await client.get('/health');
    return res.data;
  },

  // Detection
  detectImage: async (formData: FormData): Promise<ImageDetectionResponse> => {
    const res = await client.post('/detection/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  detectVideo: async (formData: FormData): Promise<VideoDetectionResponse> => {
    const res = await client.post('/detection/video', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  getSession: async (sessionId: string) => {
    const res = await client.get(`/detection/${sessionId}`);
    return res.data;
  },

  processLiveFrame: async (payload: {
    frame_base64: string;
    model_name?: string;
    conf_threshold?: number;
    iou_threshold?: number;
  }) => {
    const res = await client.post('/live-frame', payload);
    return res.data;
  },

  // Models
  listModels: async (): Promise<ModelRecord[]> => {
    const res = await client.get('/models');
    return res.data;
  },

  activateModel: async (modelName: string): Promise<ModelRecord> => {
    const res = await client.post('/models/activate', { model_name: modelName });
    return res.data;
  },

  uploadModel: async (formData: FormData): Promise<ModelRecord> => {
    const res = await client.post('/models/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Cameras
  listCameras: async (): Promise<CameraRecord[]> => {
    const res = await client.get('/cameras');
    return res.data;
  },

  createCamera: async (data: {
    name: string;
    rtsp_url: string;
    resolution?: string;
    target_fps?: number;
    model_name?: string;
  }): Promise<CameraRecord> => {
    const res = await client.post('/cameras', data);
    return res.data;
  },

  startCamera: async (cameraId: string) => {
    const res = await client.post(`/cameras/${cameraId}/start`);
    return res.data;
  },

  stopCamera: async (cameraId: string) => {
    const res = await client.post(`/cameras/${cameraId}/stop`);
    return res.data;
  },

  getCameraStatus: async (cameraId: string): Promise<CameraTelemetry> => {
    const res = await client.get(`/cameras/${cameraId}/status`);
    return res.data;
  },

  deleteCamera: async (cameraId: string) => {
    const res = await client.delete(`/cameras/${cameraId}`);
    return res.data;
  },

  // Training
  listTrainingJobs: async (): Promise<TrainingJob[]> => {
    const res = await client.get('/training');
    return res.data;
  },

  startTrainingJob: async (data: {
    model_name: string;
    dataset_yaml: string;
    epochs: number;
    batch_size: number;
    image_size: number;
    learning_rate: number;
    device: string;
  }): Promise<TrainingJob> => {
    const res = await client.post('/training', data);
    return res.data;
  },

  getTrainingJob: async (jobId: string): Promise<TrainingJob> => {
    const res = await client.get(`/training/${jobId}`);
    return res.data;
  },

  validateDataset: async (yamlPath: string): Promise<DatasetValidationReport> => {
    const res = await client.post('/training/validate-dataset', { yaml_path: yamlPath });
    return res.data;
  },

  // Analytics
  getDashboardStats: async (): Promise<DashboardStats> => {
    const res = await client.get('/analytics/dashboard');
    return res.data;
  },

  getAnalyticsBreakdown: async (): Promise<AnalyticsBreakdown> => {
    const res = await client.get('/analytics/breakdown');
    return res.data;
  },
};
