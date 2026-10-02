export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BBox;
}

export interface PerformanceMetrics {
  preprocessing_time_ms: number;
  inference_time_ms: number;
  postprocessing_time_ms: number;
  total_latency_ms: number;
  fps: number;
  number_of_detections: number;
  number_of_frames: number;
}

export interface ImageDetectionResponse {
  session_id: string;
  model: string;
  image_width: number;
  image_height: number;
  object_count: number;
  detections: Detection[];
  metrics: PerformanceMetrics;
  annotated_image_url?: string;
  annotated_image_base64?: string;
}

export interface VideoDetectionResponse {
  session_id: string;
  model: string;
  total_frames: number;
  total_detections: number;
  average_fps: number;
  average_latency_ms: number;
  class_distribution: Record<string, number>;
  processed_video_url?: string;
  status: string;
}

export interface ModelVersion {
  id: number;
  version: string;
  file_path: string;
  dataset_name: string;
  training_date: string;
  map50: number;
  map50_95: number;
  precision: number;
  recall: number;
  f1_score: number;
  status: string;
  notes?: string;
}

export interface ModelRecord {
  id: number;
  name: string;
  description: string;
  model_type: string;
  is_active: boolean;
  created_at: string;
  versions: ModelVersion[];
}

export interface CameraRecord {
  id: number;
  camera_id: string;
  name: string;
  rtsp_url: string;
  resolution: string;
  target_fps: number;
  model_name: string;
  status: string;
  is_active: boolean;
  created_at: string;
}

export interface CameraTelemetry {
  camera_id: string;
  name: string;
  rtsp_url: string;
  status: string;
  fps: number;
  resolution: string;
  total_frames_received: number;
  reconnect_count: number;
  uptime_seconds: number;
  error_message?: string;
}

export interface TrainingJob {
  id: number;
  job_id: string;
  model_name: string;
  dataset_path: string;
  epochs: number;
  total_epochs?: number;
  batch_size: number;
  image_size: number;
  learning_rate: number;
  device: string;
  status: string;
  current_epoch: number;
  train_loss: number;
  val_loss: number;
  precision: number;
  recall: number;
  map50: number;
  map50_95: number;
  best_model_path?: string;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

export interface RecentSession {
  session_id: string;
  session_type: string;
  model_name: string;
  start_time: string;
  duration_ms: number;
  object_count: number;
  fps: number;
  latency_ms: number;
  status: string;
}

export interface DashboardStats {
  total_sessions: number;
  total_detected_objects: number;
  active_cameras: number;
  available_models: number;
  average_fps: number;
  average_latency_ms: number;
  hardware_device: string;
  recent_sessions: RecentSession[];
}

export interface AnalyticsBreakdown {
  total_sessions: number;
  total_detected_objects: number;
  average_fps: number;
  average_latency_ms: number;
  class_distribution: Record<string, number>;
  session_type_distribution: Record<string, number>;
  system_hardware: {
    cpu_percent: number;
    memory_percent: number;
    compute_device: string;
  };
}

export interface DatasetValidationReport {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  train_images_count: number;
  val_images_count: number;
  num_classes: number;
  class_names: string[];
  total_annotations: number;
  class_distribution: Record<string, number>;
}
