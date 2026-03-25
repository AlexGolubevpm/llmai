export type JobType = "REWRITE" | "TRANSLATE" | "POSTPROCESS" | "AI_PROCESS";
export type JobStatus = "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";

export interface JobConfig {
  model?: string;
  systemPrompt?: string;
  userPrompt?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  minP?: number;
  topK?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  repetitionPenalty?: number;
  multiplier?: number;
  titleCol?: string;
  chunkSize?: number;
  maxWorkers?: number;
  sourceLanguage?: string;
  targetLanguage?: string;
  harmfulPatterns?: string[];
  applyStopWords?: boolean;
}

export interface Job {
  id: string;
  type: JobType;
  status: JobStatus;
  config: JobConfig;
  inputFileUrl: string;
  outputFileUrl: string | null;
  totalRows: number;
  processedRows: number;
  failedRows: number;
  currentPass: number;
  totalPasses: number;
  errorLog: ErrorLogEntry[] | null;
  startedAt: string | null;
  completedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ErrorLogEntry {
  row: number;
  error: string;
  retries: number;
}

export interface JobProgress {
  jobId: string;
  status: JobStatus;
  processedRows: number;
  totalRows: number;
  failedRows: number;
  currentPass: number;
  totalPasses: number;
  eta?: number; // seconds remaining
  speed?: number; // rows per second
}

export interface Preset {
  id: string;
  name: string;
  systemPrompt: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  minP: number;
  topK: number;
  presencePenalty: number;
  frequencyPenalty: number;
  repetitionPenalty: number;
  isDefault: boolean;
}

export interface StopWord {
  id: string;
  word: string;
  replacement: string | null;
  isActive: boolean;
}

export interface AllowedTag {
  id: string;
  name: string;
  category: string;
}

export interface AllowedCategory {
  id: string;
  name: string;
  slug: string;
}

export interface ChatCompletionParams {
  model: string;
  systemPrompt: string;
  userPrompt: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  minP: number;
  topK: number;
  presencePenalty: number;
  frequencyPenalty: number;
  repetitionPenalty: number;
}

export interface WDTaggerResult {
  tags: string[];
  rating: { label: string; confidence: number }[];
  characters: string[];
  generalTags: { tag: string; confidence: number }[];
  caption: string;
}

export interface NovitaModel {
  id: string;
  object: string;
}
