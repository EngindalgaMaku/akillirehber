const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Token keys for localStorage
const TOKEN_KEY = "akilli_rehber_token";
const REFRESH_TOKEN_KEY = "akilli_rehber_refresh_token";

// Event for auth state changes (logout on 401)
type AuthEventCallback = () => void;
let onUnauthorizedCallback: AuthEventCallback | null = null;

// Flag to prevent multiple refresh attempts
let isRefreshing = false;
let refreshPromise: Promise<string | null> | null = null;

export function setOnUnauthorizedCallback(callback: AuthEventCallback | null) {
  onUnauthorizedCallback = callback;
}

export interface ApiError {
  detail: string;
}

export interface User {
  id: number;
  email: string;
  full_name: string;
  role: "admin" | "teacher" | "student";
  created_at: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  full_name: string;
  role: "teacher" | "student";
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface Course {
  id: number;
  name: string;
  description: string | null;
  teacher_id: number;
  is_active: boolean;
  created_at: string;
}

export interface Document {
  id: number;
  filename: string;
  original_filename: string;
  file_type: string;
  file_size: number;
  char_count: number | null;
  course_id: number;
  created_at: string;
  processed: boolean;
  chunk_count: number;
  embedding_status: "pending" | "processing" | "completed" | "error";
  embedding_model: string | null;
  embedded_at: string | null;
  vector_count: number;
}

export interface Chunk {
  content: string;
  start_index: number;
  end_index: number;
  metadata: Record<string, unknown>;
}

export interface ChunkStats {
  total_chunks: number;
  avg_chunk_size: number;
  min_chunk_size: number;
  max_chunk_size: number;
  total_characters: number;
}

export interface ChunkQualityMetrics {
  chunk_index: number;
  semantic_coherence: number;
  sentence_count: number;
  topic_consistency: number;
  has_questions: boolean;
  has_qa_pairs: boolean;
}

export interface QualityReport {
  total_chunks: number;
  avg_coherence: number;
  min_coherence: number;
  max_coherence: number;
  chunks_below_threshold: number;
  inter_chunk_similarities: number[];
  merge_recommendations: [number, number][];
  split_recommendations: number[];
  overall_quality_score: number;
  recommendations: string[];
}

export interface ChunkResponse {
  chunks: Chunk[];
  stats: ChunkStats;
  strategy: string;
  // Semantic chunking specific fields
  quality_metrics?: ChunkQualityMetrics[];
  quality_report?: QualityReport;
  detected_language?: string;
  adaptive_threshold_used?: number;
  processing_time_ms?: number;
  fallback_used?: string;
  warning_message?: string;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface CourseListResponse {
  courses: Course[];
  total: number;
}

export interface CourseSettings {
  id: number;
  course_id: number;
  default_chunk_strategy: string;
  default_chunk_size: number;
  default_overlap: number;
  default_embedding_model: string;
  search_alpha: number;
  search_top_k: number;
  min_relevance_score: number;
  llm_provider: string;
  llm_model: string;
  llm_temperature: number;
  llm_max_tokens: number;
  system_prompt: string | null;
  enable_reranker: boolean;
  reranker_provider: string | null;
  reranker_model: string | null;
  reranker_top_k: number;
  created_at: string;
  updated_at: string;
}

export interface CourseSettingsUpdate {
  default_chunk_strategy?: string;
  default_chunk_size?: number;
  default_overlap?: number;
  default_embedding_model?: string;
  search_alpha?: number;
  search_top_k?: number;
  min_relevance_score?: number;
  llm_provider?: string;
  llm_model?: string;
  llm_temperature?: number;
  llm_max_tokens?: number;
  system_prompt?: string;
  enable_reranker?: boolean;
  reranker_provider?: string;
  reranker_model?: string;
  reranker_top_k?: number;
}

export interface EmbedResponse {
  document_id: number;
  status: string;
  vector_count: number;
  model: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChunkReference {
  document_id: number;
  document_name: string;
  chunk_index: number;
  content_preview: string;
  full_content: string;
  score: number;
}

export interface ChatResponse {
  message: string;
  sources: ChunkReference[];
}

export interface ChatHistoryMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  sources?: ChunkReference[] | null;
  response_time_ms?: number | null;
}

export interface ChatHistoryResponse {
  messages: ChatHistoryMessage[];
  has_more: boolean;
}

export interface ChatHistoryClearResponse {
  success: boolean;
  deleted_count: number;
}

class ApiClient {
  private token: string | null = null;
  private refreshToken: string | null = null;

  setToken(token: string | null) {
    this.token = token;
  }

  setRefreshToken(token: string | null) {
    this.refreshToken = token;
  }

  getRefreshToken(): string | null {
    return this.refreshToken;
  }

  private async tryRefreshToken(): Promise<string | null> {
    // If already refreshing, wait for the existing promise
    if (isRefreshing && refreshPromise) {
      return refreshPromise;
    }

    const refreshToken = this.refreshToken || localStorage.getItem(REFRESH_TOKEN_KEY);
    if (!refreshToken) {
      return null;
    }

    isRefreshing = true;
    refreshPromise = (async () => {
      try {
        const response = await fetch(`${API_URL}/api/auth/refresh`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!response.ok) {
          // Refresh token is invalid or expired
          return null;
        }

        const data: TokenResponse = await response.json();
        
        // Update tokens
        this.token = data.access_token;
        localStorage.setItem(TOKEN_KEY, data.access_token);
        
        return data.access_token;
      } catch {
        return null;
      } finally {
        isRefreshing = false;
        refreshPromise = null;
      }
    })();

    return refreshPromise;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    isRetry: boolean = false
  ): Promise<T> {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
      ...options.headers,
    };

    if (this.token) {
      (headers as Record<string, string>)["Authorization"] = `Bearer ${this.getToken()}${this.token}`;
    }

    const response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      // Handle 401 Unauthorized - token expired or invalid
      if (response.status === 401 && !isRetry) {
        // Try to refresh the token
        const newToken = await this.tryRefreshToken();
        
        if (newToken) {
          // Retry the request with the new token
          return this.request<T>(endpoint, options, true);
        }
        
        // Refresh failed - clear tokens and logout
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(REFRESH_TOKEN_KEY);
        this.token = null;
        this.refreshToken = null;
        
        // Trigger logout callback if set
        if (onUnauthorizedCallback) {
          onUnauthorizedCallback();
        }
        
        throw new Error("Oturum süresi doldu. Lütfen tekrar giriş yapın.");
      }
      
      const error: ApiError = await response.json().catch(() => ({
        detail: "Bir hata oluştu",
      }));
      throw new Error(error.detail);
    }

    // Handle 204 No Content responses
    if (response.status === 204) {
      return undefined as T;
    }

    // Check if response has content
    const contentLength = response.headers.get("content-length");
    if (contentLength === "0") {
      return undefined as T;
    }

    return response.json();
  }

  // Auth endpoints
  async login(data: LoginRequest): Promise<TokenResponse> {
    const formData = new URLSearchParams();
    formData.append("username", data.email);
    formData.append("password", data.password);

    const headers: HeadersInit = {
      "Content-Type": "application/x-www-form-urlencoded",
    };

    const response = await fetch(`${API_URL}/api/auth/login`, {
      method: "POST",
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: "Giriş başarısız",
      }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  async register(data: RegisterRequest): Promise<User> {
    return this.request<User>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getMe(): Promise<User> {
    return this.request<User>("/api/auth/me");
  }

  async updateProfile(data: { full_name?: string; email?: string }): Promise<User> {
    return this.request<User>("/api/auth/me", {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async changePassword(data: { current_password: string; new_password: string }): Promise<{ message: string }> {
    return this.request<{ message: string }>("/api/auth/change-password", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async logout(): Promise<void> {
    const refreshToken = this.refreshToken || localStorage.getItem(REFRESH_TOKEN_KEY);
    if (refreshToken) {
      try {
        await fetch(`${API_URL}/api/auth/logout`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
      } catch {
        // Ignore errors during logout
      }
    }
    
    // Clear tokens regardless of API call result
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    this.token = null;
    this.refreshToken = null;
  }

  // Course endpoints
  async getCourses(): Promise<Course[]> {
    const response = await this.request<CourseListResponse>("/api/courses");
    return response.courses;
  }

  async getCourse(id: number): Promise<Course> {
    return this.request<Course>(`/api/courses/${id}`);
  }

  async createCourse(data: { name: string; description?: string }): Promise<Course> {
    return this.request<Course>("/api/courses", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateCourse(id: number, data: { name?: string; description?: string; is_active?: boolean }): Promise<Course> {
    return this.request<Course>(`/api/courses/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async deleteCourse(id: number): Promise<void> {
    await this.request(`/api/courses/${id}`, { method: "DELETE" });
  }

  // Course Settings endpoints
  async getCourseSettings(courseId: number): Promise<CourseSettings> {
    return this.request<CourseSettings>(`/api/courses/${courseId}/settings`);
  }

  async updateCourseSettings(courseId: number, data: CourseSettingsUpdate): Promise<CourseSettings> {
    return this.request<CourseSettings>(`/api/courses/${courseId}/settings`, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async getLLMProviders(): Promise<Record<string, string[]>> {
    return this.request<Record<string, string[]>>("/api/llm-providers");
  }

  // Custom LLM Models endpoints
  async getCustomLLMModels(provider?: string): Promise<CustomLLMModelListResponse> {
    const params = provider ? `?provider=${provider}` : "";
    return this.request<CustomLLMModelListResponse>(`/api/llm-models${params}`);
  }

  async getModelsByProvider(provider: string): Promise<LLMModelsResponse> {
    return this.request<LLMModelsResponse>(`/api/llm-models/by-provider/${provider}`);
  }

  async createCustomLLMModel(data: CustomLLMModelCreate): Promise<CustomLLMModel> {
    return this.request<CustomLLMModel>("/api/llm-models", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async deleteCustomLLMModel(modelId: number): Promise<void> {
    await this.request(`/api/llm-models/${modelId}`, { method: "DELETE" });
  }

  async getLLMProvidersList(): Promise<{ providers: string[] }> {
    return this.request<{ providers: string[] }>("/api/llm-models/providers");
  }

  // Document endpoints
  async getCourseDocuments(courseId: number): Promise<Document[]> {
    const response = await this.request<DocumentListResponse>(`/api/courses/${courseId}/documents`);
    return response.documents;
  }

  async uploadDocument(courseId: number, file: File): Promise<Document> {
    const formData = new FormData();
    formData.append("file", file);

    const headers: HeadersInit = {};
    if (this.token) {
      headers["Authorization"] = `Bearer ${this.getToken()}${this.token}`;
    }

    const response = await fetch(`${API_URL}/api/courses/${courseId}/documents`, {
      method: "POST",
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: "Dosya yüklenirken hata oluştu",
      }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  async deleteDocument(id: number): Promise<void> {
    await this.request(`/api/documents/${id}`, { method: "DELETE" });
  }

  async getDocumentChunks(documentId: number): Promise<{ chunks: Chunk[]; total: number }> {
    return this.request(`/api/documents/${documentId}/chunks`);
  }

  async deleteDocumentChunks(documentId: number): Promise<void> {
    await this.request(`/api/documents/${documentId}/chunks`, { method: "DELETE" });
  }

  async processDocument(
    id: number,
    options: { 
      strategy: string; 
      chunk_size?: number; 
      overlap?: number;
      similarity_threshold?: number;
      embedding_model?: string;
      min_chunk_size?: number;
      max_chunk_size?: number;
    }
  ): Promise<{ chunks: Chunk[]; total: number; document_id: number }> {
    return this.request(`/api/documents/${id}/process`, {
      method: "POST",
      body: JSON.stringify(options),
    });
  }

  // Embedding endpoints
  async embedDocument(documentId: number, model: string = "openai/text-embedding-3-small"): Promise<EmbedResponse> {
    return this.request<EmbedResponse>(`/api/documents/${documentId}/embed`, {
      method: "POST",
      body: JSON.stringify({ model }),
    });
  }

  async deleteDocumentVectors(documentId: number): Promise<void> {
    await this.request(`/api/documents/${documentId}/vectors`, { method: "DELETE" });
  }

  async getDocumentVectorCount(documentId: number): Promise<{ document_id: number; vector_count: number }> {
    return this.request(`/api/documents/${documentId}/vectors/count`);
  }

  // Chat endpoint
  async chat(
    courseId: number,
    message: string,
    history: ChatMessage[] = []
  ): Promise<ChatResponse> {
    return this.request<ChatResponse>(`/api/courses/${courseId}/chat`, {
      method: "POST",
      body: JSON.stringify({
        message,
        history,
        search_type: "hybrid",
      }),
    });
  }

  async getChatHistory(
    courseId: number,
    params?: { limit?: number; before_id?: number }
  ): Promise<ChatHistoryResponse> {
    const queryParams = new URLSearchParams();
    if (params?.limit) queryParams.append("limit", params.limit.toString());
    if (params?.before_id) queryParams.append("before_id", params.before_id.toString());

    const url = `/api/courses/${courseId}/chat/history${
      queryParams.toString() ? `?${queryParams.toString()}` : ""
    }`;
    return this.request<ChatHistoryResponse>(url);
  }

  async clearChatHistory(courseId: number): Promise<ChatHistoryClearResponse> {
    return this.request<ChatHistoryClearResponse>(
      `/api/courses/${courseId}/chat/history`,
      { method: "DELETE" }
    );
  }

  // Health check
  async health(): Promise<{ status: string }> {
    return this.request<{ status: string }>("/health");
  }

  // Dashboard stats
  async getDashboardStats(): Promise<{ course_count: number; document_count: number; chunk_count: number }> {
    return this.request("/api/courses/dashboard/stats");
  }

  // Legacy chunking endpoint
  async chunk(data: { 
    text: string; 
    strategy: string; 
    chunk_size?: number; 
    overlap?: number; 
    similarity_threshold?: number;
    include_quality_metrics?: boolean;
    enable_qa_detection?: boolean;
    enable_adaptive_threshold?: boolean;
    enable_cache?: boolean;
    min_chunk_size?: number;
    max_chunk_size?: number;
    buffer_size?: number;
  }): Promise<ChunkResponse> {
    return this.request<ChunkResponse>("/api/chunk", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }
  // ==================== RAGAS Endpoints ====================

  // Test Sets
  async getTestSets(courseId: number): Promise<TestSet[]> {
    return this.request<TestSet[]>(`/api/ragas/test-sets?course_id=${courseId}`);
  }

  async getTestSet(testSetId: number): Promise<TestSetDetail> {
    return this.request<TestSetDetail>(`/api/ragas/test-sets/${testSetId}`);
  }

  async createTestSet(data: { course_id: number; name: string; description?: string }): Promise<TestSet> {
    return this.request<TestSet>("/api/ragas/test-sets", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateTestSet(testSetId: number, data: { name?: string; description?: string }): Promise<TestSet> {
    return this.request<TestSet>(`/api/ragas/test-sets/${testSetId}`, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async deleteTestSet(testSetId: number): Promise<void> {
    await this.request(`/api/ragas/test-sets/${testSetId}`, { method: "DELETE" });
  }

  async duplicateTestSet(testSetId: number): Promise<TestSet> {
    return this.request<TestSet>(`/api/ragas/test-sets/${testSetId}/duplicate`, {
      method: "POST",
    });
  }

  async importQuestions(testSetId: number, questions: { question: string; ground_truth: string; alternative_ground_truths?: string[]; expected_contexts?: string[] }[]): Promise<TestSetDetail> {
    return this.request<TestSetDetail>(`/api/ragas/test-sets/${testSetId}/import`, {
      method: "POST",
      body: JSON.stringify({ questions }),
    });
  }

  async exportTestSet(testSetId: number): Promise<{ name: string; description?: string; questions: { question: string; ground_truth: string }[] }> {
    return this.request(`/api/ragas/test-sets/${testSetId}/export`);
  }

  // Questions
  async addQuestion(testSetId: number, data: { question: string; ground_truth: string; alternative_ground_truths?: string[]; expected_contexts?: string[] }): Promise<TestQuestion> {
    return this.request<TestQuestion>(`/api/ragas/test-sets/${testSetId}/questions`, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateQuestion(questionId: number, data: { question?: string; ground_truth?: string; alternative_ground_truths?: string[]; expected_contexts?: string[] }): Promise<TestQuestion> {
    return this.request<TestQuestion>(`/api/ragas/questions/${questionId}`, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async deleteQuestion(questionId: number): Promise<void> {
    await this.request(`/api/ragas/questions/${questionId}`, { method: "DELETE" });
  }

  // Evaluation Runs
  async startEvaluation(data: { test_set_id: number; course_id: number; name?: string; config?: EvaluationConfig; evaluation_model?: string; question_ids?: number[] }): Promise<EvaluationRun> {
    return this.request<EvaluationRun>("/api/ragas/evaluate", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getEvaluationRuns(courseId: number, testSetId?: number): Promise<EvaluationRun[]> {
    let url = `/api/ragas/runs?course_id=${courseId}`;
    if (testSetId !== undefined) {
      url += `&test_set_id=${testSetId}`;
    }
    return this.request<EvaluationRun[]>(url);
  }

  async getEvaluationRun(runId: number): Promise<EvaluationRunDetail> {
    return this.request<EvaluationRunDetail>(`/api/ragas/runs/${runId}`);
  }

  async getRunStatus(runId: number): Promise<{ id: number; status: string; total_questions: number; processed_questions: number; error_message?: string }> {
    return this.request(`/api/ragas/runs/${runId}/status`);
  }

  async deleteEvaluationRun(runId: number): Promise<void> {
    await this.request(`/api/ragas/runs/${runId}`, { method: "DELETE" });
  }

  async fixRunSummary(runId: number): Promise<{
    message: string;
    run_id: number;
    status: string;
    total_questions: number;
    successful_questions: number;
    failed_questions: number;
  }> {
    return this.request(`/api/ragas/runs/${runId}/fix-summary`, {
      method: "POST",
    });
  }

  async compareRuns(runIds: number[]): Promise<{ runs: EvaluationRun[]; summaries: RunSummary[] }> {
    return this.request("/api/ragas/compare", {
      method: "POST",
      body: JSON.stringify({ run_ids: runIds }),
    });
  }

  // RAGAS Settings
  async getRagasSettings(): Promise<RagasSettings> {
    return this.request<RagasSettings>("/api/ragas/settings");
  }

  async updateRagasSettings(data: { provider?: string; model?: string }): Promise<RagasSettings> {
    const params = new URLSearchParams();
    if (data.provider !== undefined) params.append("provider", data.provider);
    if (data.model !== undefined) params.append("model", data.model);
    return this.request<RagasSettings>(`/api/ragas/settings?${params.toString()}`, {
      method: "POST",
    });
  }

  async getRagasProviders(): Promise<RagasProvidersResponse> {
    return this.request<RagasProvidersResponse>("/api/ragas/providers");
  }

  async quickTest(data: QuickTestRequest): Promise<QuickTestResponse> {
    return this.request<QuickTestResponse>("/api/ragas/quick-test", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // ==================== Admin User Management Endpoints ====================

  async getAdminUsers(params?: {
    page?: number;
    limit?: number;
    role?: string;
    search?: string;
    sort_by?: string;
    sort_order?: "asc" | "desc";
  }): Promise<AdminUserListResponse> {
    const queryParams = new URLSearchParams();
    if (params?.page) queryParams.append("page", params.page.toString());
    if (params?.limit) queryParams.append("limit", params.limit.toString());
    if (params?.role) queryParams.append("role", params.role);
    if (params?.search) queryParams.append("search", params.search);
    if (params?.sort_by) queryParams.append("sort_by", params.sort_by);
    if (params?.sort_order) queryParams.append("sort_order", params.sort_order);
    
    const url = `/api/admin/users${queryParams.toString() ? `?${queryParams.toString()}` : ""}`;
    return this.request<AdminUserListResponse>(url);
  }

  async updateAdminUser(userId: number, data: AdminUserUpdate): Promise<AdminUser> {
    return this.request<AdminUser>(`/api/admin/users/${userId}`, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async deleteAdminUser(userId: number): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/api/admin/users/${userId}`, {
      method: "DELETE",
    });
  }

  async deactivateUser(userId: number): Promise<AdminUser> {
    return this.request<AdminUser>(`/api/admin/users/${userId}/deactivate`, {
      method: "PATCH",
    });
  }

  async activateUser(userId: number): Promise<AdminUser> {
    return this.request<AdminUser>(`/api/admin/users/${userId}/activate`, {
      method: "PATCH",
    });
  }

  async getAdminStatistics(): Promise<AdminStatistics> {
    return this.request<AdminStatistics>("/api/admin/statistics");
  }

  async resetUserPassword(userId: number): Promise<{
    success: boolean;
    temporary_password: string;
    expires_at: string;
  }> {
    return this.request<{
      success: boolean;
      temporary_password: string;
      expires_at: string;
    }>(`/api/admin/users/${userId}/reset-password`, {
      method: "POST",
    });
  }

  // Quick Test Results
  async saveQuickTestResult(data: QuickTestResultCreate): Promise<QuickTestResult> {
    return this.request<QuickTestResult>("/api/ragas/quick-test-results", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getQuickTestResults(
    courseId: number,
    groupName?: string,
    skip: number = 0,
    limit: number = 10
  ): Promise<QuickTestResultListResponse> {
    const params = new URLSearchParams({
      course_id: courseId.toString(),
      skip: skip.toString(),
      limit: limit.toString()
    });
    if (groupName) params.append("group_name", groupName);
    return this.request<QuickTestResultListResponse>(`/api/ragas/quick-test-results?${params.toString()}`);
  }

  async getQuickTestResult(resultId: number): Promise<QuickTestResult> {
    return this.request<QuickTestResult>(`/api/ragas/quick-test-results/${resultId}`);
  }

  async deleteQuickTestResult(resultId: number): Promise<void> {
    await this.request(`/api/ragas/quick-test-results/${resultId}`, { method: "DELETE" });
  }

  // ==================== Semantic Similarity Endpoints ====================

  async semanticSimilarityQuickTest(data: SemanticSimilarityQuickTestRequest): Promise<SemanticSimilarityQuickTestResponse> {
    return this.request<SemanticSimilarityQuickTestResponse>("/api/semantic-similarity/quick-test", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async semanticSimilarityBatchTest(data: SemanticSimilarityBatchTestRequest): Promise<SemanticSimilarityBatchTestResponse> {
    return this.request<SemanticSimilarityBatchTestResponse>("/api/semantic-similarity/batch-test", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async saveSemanticSimilarityResult(data: Omit<SemanticSimilarityResult, 'id' | 'created_by' | 'created_at'>): Promise<SemanticSimilarityResult> {
    return this.request<SemanticSimilarityResult>("/api/semantic-similarity/results", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getSemanticSimilarityResults(
    courseId: number,
    groupName?: string,
    skip: number = 0,
    limit: number = 10
  ): Promise<SemanticSimilarityResultListResponse> {
    const params = new URLSearchParams({
      course_id: courseId.toString(),
      skip: skip.toString(),
      limit: limit.toString()
    });
    if (groupName) params.append("group_name", groupName);
    return this.request<SemanticSimilarityResultListResponse>(`/api/semantic-similarity/results?${params.toString()}`);
  }

  async getSemanticSimilarityResult(resultId: number): Promise<SemanticSimilarityResult> {
    return this.request<SemanticSimilarityResult>(`/api/semantic-similarity/results/${resultId}`);
  }

  async deleteSemanticSimilarityResult(resultId: number): Promise<void> {
    await this.request(`/api/semantic-similarity/results/${resultId}`, { method: "DELETE" });
  }

  // ==================== System Settings Endpoints ====================

  async getSystemSettings(): Promise<SystemSettings> {
    return this.request<SystemSettings>("/api/system/settings");
  }

  async updateSystemSettings(data: SystemSettingsUpdate): Promise<SystemSettings> {
    return this.request<SystemSettings>("/api/system/settings", {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async getPublicSettings(): Promise<PublicSettings> {
    const response = await fetch(`${API_URL}/api/system/public-settings`);
    if (!response.ok) {
      throw new Error("Failed to get public settings");
    }
    return response.json();
  }

  async verifyRegistrationKey(role: string, key: string): Promise<{ valid: boolean }> {
    const response = await fetch(`${API_URL}/api/system/verify-key`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, key }),
    });
    if (!response.ok) {
      throw new Error("Failed to verify key");
    }
    return response.json();
  }
}

export const api = new ApiClient();

// ==================== System Settings Types ====================

export interface SystemSettings {
  id: number;
  teacher_registration_key: string | null;
  student_registration_key: string | null;
  hcaptcha_site_key: string | null;
  captcha_enabled: boolean;
}

export interface SystemSettingsUpdate {
  teacher_registration_key?: string;
  student_registration_key?: string;
  hcaptcha_site_key?: string;
  hcaptcha_secret_key?: string;
  captcha_enabled?: boolean;
}

export interface PublicSettings {
  captcha_enabled: boolean;
  hcaptcha_site_key: string | null;
  registration_key_required: boolean;
}


// ==================== RAGAS Types ====================

export interface TestQuestion {
  id: number;
  test_set_id: number;
  question: string;
  ground_truth: string;
  alternative_ground_truths?: string[];
  expected_contexts?: string[];
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface TestSet {
  id: number;
  course_id: number;
  name: string;
  description?: string;
  created_by: number;
  created_at: string;
  updated_at: string;
  question_count: number;
}

export interface TestSetDetail extends TestSet {
  questions: TestQuestion[];
}

export interface EvaluationConfig {
  search_type?: string;
  search_alpha?: number;
  top_k?: number;
  llm_model?: string;
  llm_temperature?: number;
}

export interface EvaluationRun {
  id: number;
  test_set_id: number;
  test_set_name?: string;
  course_id: number;
  name?: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  config?: EvaluationConfig;
  total_questions: number;
  processed_questions: number;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  created_at: string;
  // Average metrics from summary
  avg_faithfulness?: number;
  avg_answer_relevancy?: number;
  avg_context_precision?: number;
  avg_context_recall?: number;
  avg_answer_correctness?: number;
}

export interface EvaluationResult {
  id: number;
  run_id: number;
  question_id: number;
  question_text: string;
  ground_truth_text: string;
  generated_answer?: string;
  retrieved_contexts?: string[];
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
  answer_correctness?: number;
  latency_ms?: number;
  llm_provider?: string;
  llm_model?: string;
  embedding_model?: string;
  evaluation_model?: string;
  search_alpha?: number;
  search_top_k?: number;
  error_message?: string;
  created_at: string;
}

export interface RunSummary {
  id: number;
  run_id: number;
  avg_faithfulness?: number;
  avg_answer_relevancy?: number;
  avg_context_precision?: number;
  avg_context_recall?: number;
  avg_answer_correctness?: number;
  avg_latency_ms?: number;
  total_questions: number;
  successful_questions: number;
  failed_questions: number;
  created_at: string;
}

export interface EvaluationRunDetail extends EvaluationRun {
  results: EvaluationResult[];
  summary?: RunSummary;
}

// ==================== RAGAS Settings Types ====================

export interface RagasSettings {
  provider: string | null;
  model: string | null;
  current_provider: string | null;
  current_model: string | null;
  is_free: boolean;
}

export interface RagasProvider {
  name: string;
  available: boolean;
  is_free: boolean;
  default_model: string;
  models: string[];
  priority: number;
}

export interface RagasProvidersResponse {
  providers: RagasProvider[];
  current: {
    provider: string;
    model: string;
    is_free: boolean;
  } | null;
}

// ==================== Quick Test Types ====================

export interface RetrievedContext {
  text: string;
  score: number;
}

export interface QuickTestRequest {
  course_id: number;
  question: string;
  ground_truth: string;
  alternative_ground_truths?: string[];
  system_prompt?: string;
  llm_provider?: string;
  llm_model?: string;
}

export interface QuickTestResponse {
  question: string;
  ground_truth: string;
  generated_answer: string;
  retrieved_contexts: RetrievedContext[];
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
  answer_correctness?: number;
  latency_ms: number;
  system_prompt_used: string;
  llm_provider_used: string;
  llm_model_used: string;
  // Reranker metadata
  reranker_used?: boolean;
  reranker_provider?: string;
  reranker_model?: string;
}

// ==================== Quick Test Result Types ====================

export interface QuickTestResultCreate {
  course_id: number;
  group_name?: string;
  question: string;
  ground_truth: string;
  alternative_ground_truths?: string[];
  system_prompt?: string;
  llm_provider: string;
  llm_model: string;
  generated_answer: string;
  retrieved_contexts?: RetrievedContext[];
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
  answer_correctness?: number;
  latency_ms: number;
}

export interface QuickTestResult {
  id: number;
  course_id: number;
  group_name?: string;
  question: string;
  ground_truth: string;
  alternative_ground_truths?: string[];
  system_prompt?: string;
  llm_provider: string;
  llm_model: string;
  generated_answer: string;
  retrieved_contexts?: RetrievedContext[];
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
  answer_correctness?: number;
  latency_ms: number;
  created_by: number;
  created_at: string;
  // Reranker metadata
  reranker_used?: boolean;
  reranker_provider?: string;
  reranker_model?: string;
}

export interface QuickTestResultListResponse {
  results: QuickTestResult[];
  total: number;
  groups: string[];
}

// ==================== Custom LLM Model Types ====================

export interface CustomLLMModel {
  id: number;
  provider: string;
  model_id: string;
  display_name: string;
  is_active: boolean;
  created_by: number;
  created_at: string;
}

export interface CustomLLMModelCreate {
  provider: string;
  model_id: string;
  display_name: string;
}

export interface CustomLLMModelListResponse {
  models: CustomLLMModel[];
  total: number;
}

export interface LLMModelsResponse {
  default_models: string[];
  custom_models: CustomLLMModel[];
}

// ==================== Semantic Similarity Types ====================

export interface SemanticSimilarityQuickTestRequest {
  course_id: number;
  question: string;
  ground_truth: string;
  alternative_ground_truths?: string[];
  generated_answer?: string;
  llm_provider?: string;
  llm_model?: string;
}

export interface SemanticSimilarityQuickTestResponse {
  question: string;
  ground_truth: string;
  generated_answer: string;
  similarity_score: number;
  best_match_ground_truth: string;
  all_scores: Array<{ ground_truth: string; score: number }>;
  latency_ms: number;
  embedding_model_used: string;
  llm_model_used?: string;
  retrieved_contexts?: string[];
  system_prompt_used?: string;
  // ROUGE metrics
  rouge1?: number;
  rouge2?: number;
  rougel?: number;
  // BERTScore metrics
  bertscore_precision?: number;
  bertscore_recall?: number;
  bertscore_f1?: number;
}

export interface SemanticSimilarityBatchTestRequest {
  course_id: number;
  test_cases: Array<{
    question: string;
    ground_truth: string;
    alternative_ground_truths?: string[];
    generated_answer?: string;
  }>;
  llm_provider?: string;
  llm_model?: string;
}

export interface SemanticSimilarityBatchTestResponse {
  results: Array<{
    question: string;
    ground_truth: string;
    generated_answer: string;
    similarity_score: number;
    best_match_ground_truth: string;
    latency_ms: number;
    retrieved_contexts?: string[];
    system_prompt_used?: string;
    rouge1?: number;
    rouge2?: number;
    rougel?: number;
    bertscore_precision?: number;
    bertscore_recall?: number;
    bertscore_f1?: number;
  }>;
  aggregate: {
    avg_similarity: number;
    min_similarity: number;
    max_similarity: number;
    total_latency_ms: number;
    test_count: number;
    avg_rouge1?: number;
    avg_rouge2?: number;
    avg_rougel?: number;
    avg_bertscore_precision?: number;
    avg_bertscore_recall?: number;
    avg_bertscore_f1?: number;
  };
  embedding_model_used: string;
  llm_model_used?: string;
}

export interface SemanticSimilarityResult {
  id: number;
  course_id: number;
  group_name?: string;
  question: string;
  ground_truth: string;
  alternative_ground_truths?: string[];
  generated_answer: string;
  similarity_score: number;
  best_match_ground_truth: string;
  all_scores?: Array<{ ground_truth: string; score: number }>;
  retrieved_contexts?: string[];
  system_prompt_used?: string;
  latency_ms: number;
  embedding_model_used: string;
  llm_model_used?: string;
  // ROUGE metrics
  rouge1?: number;
  rouge2?: number;
  rougel?: number;
  // BERTScore metrics
  bertscore_precision?: number;
  bertscore_recall?: number;
  bertscore_f1?: number;
  // Retrieval metrics
  hit_at_1?: number;
  mrr?: number;
  created_by: number;
  created_at: string;
}

export interface SemanticSimilarityResultListResponse {
  results: SemanticSimilarityResult[];
  total: number;
  groups: string[];
  aggregate?: {
    avg_similarity: number;
    avg_rouge1?: number;
    avg_rouge2?: number;
    avg_rougel?: number;
    avg_bertscore_precision?: number;
    avg_bertscore_recall?: number;
    avg_bertscore_f1?: number;
    test_count: number;
  };
}

// ==================== Admin User Management Types ====================

export interface AdminUser {
  id: number;
  full_name: string;
  email: string;
  role: "admin" | "teacher" | "student";
  is_active: boolean;
  created_at: string;
  last_login: string | null;
}

export interface AdminUserListResponse {
  users: AdminUser[];
  total: number;
  page: number;
  total_pages: number;
}

export interface AdminUserUpdate {
  full_name?: string;
  email?: string;
  role?: "admin" | "teacher" | "student";
  is_active?: boolean;
}

export interface AdminStatistics {
  total_users: number;
  active_teachers: number;
  active_students: number;
  inactive_users: number;
  new_users_this_month: number;
}

// ==================== Chunking Progress Types ====================

export interface ChunkingProgressEvent {
  event_type: 'progress' | 'complete' | 'error';
  stage: string;
  progress: number;
  message: string;
  details?: Record<string, unknown>;
  result?: ChunkResponse;
}

export type ChunkingProgressCallback = (event: ChunkingProgressEvent) => void;

/**
 * Stream chunking progress via Server-Sent Events
 * @param data Chunking request parameters
 * @param onProgress Callback for progress updates
 * @returns Promise that resolves with the final result
 */
export async function streamChunking(
  data: {
    text: string;
    strategy: string;
    chunk_size?: number;
    overlap?: number;
    similarity_threshold?: number;
    include_quality_metrics?: boolean;
    enable_qa_detection?: boolean;
    enable_adaptive_threshold?: boolean;
    enable_cache?: boolean;
    min_chunk_size?: number;
    max_chunk_size?: number;
    buffer_size?: number;
  },
  onProgress: ChunkingProgressCallback
): Promise<ChunkResponse> {
  return new Promise((resolve, reject) => {
    const controller = new AbortController();
    
    fetch(`${API_URL}/api/chunk/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const error = await response.json().catch(() => ({ detail: 'Streaming failed' }));
          throw new Error(error.detail);
        }
        
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }
        
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          
          // Process complete SSE events
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const eventData: ChunkingProgressEvent = JSON.parse(line.slice(6));
                onProgress(eventData);
                
                if (eventData.event_type === 'complete' && eventData.result) {
                  resolve(eventData.result);
                  return;
                }
                
                if (eventData.event_type === 'error') {
                  reject(new Error(eventData.message));
                  return;
                }
              } catch (e) {
                console.error('Failed to parse SSE event:', e);
              }
            }
          }
        }
        
        // If we get here without a complete event, something went wrong
        reject(new Error('Stream ended without completion'));

     / /   U s e r   d o c u m e n t   e n d p o i n t s 
     a s y n c   g e t U s e r D o c u m e n t s ( ) :   P r o m i s e < D o c u m e n t [ ] >   { 
         c o n s t   r e s p o n s e   =   a w a i t   t h i s . r e q u e s t < D o c u m e n t L i s t R e s p o n s e > ( \ / a p i / u s e r s / d o c u m e n t s \ ) ; 
         r e t u r n   r e s p o n s e . d o c u m e n t s ; 
     } 
 
     a s y n c   u p l o a d U s e r D o c u m e n t ( f i l e :   F i l e ,   c o u r s e I d ? :   n u m b e r ) :   P r o m i s e < D o c u m e n t >   { 
         c o n s t   f o r m D a t a   =   n e w   F o r m D a t a ( ) ; 
         f o r m D a t a . a p p e n d ( ' f i l e ' ,   f i l e ) ; 
         i f   ( c o u r s e I d )   { 
             f o r m D a t a . a p p e n d ( ' c o u r s e _ i d ' ,   c o u r s e I d . t o S t r i n g ( ) ) ; 
         } 
 
         c o n s t   r e s p o n s e   =   a w a i t   f e t c h ( \ \ / a p i / u s e r s / d o c u m e n t s \ ,   { 
             m e t h o d :   ' P O S T ' , 
             h e a d e r s :   { 
                 ' A u t h o r i z a t i o n ' :   \ B e a r e r   \ \ , 
             } , 
             b o d y :   f o r m D a t a , 
         } ) ; 
 
         i f   ( ! r e s p o n s e . o k )   { 
             c o n s t   e r r o r   =   a w a i t   r e s p o n s e . t e x t ( ) ; 
             t h r o w   n e w   E r r o r ( e r r o r   | |   ' U p l o a d   f a i l e d ' ) ; 
         } 
 
         r e t u r n   r e s p o n s e . j s o n ( ) ; 
     } 
 }  
 