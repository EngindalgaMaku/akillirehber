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
  message?: string;
}

export interface GenerateTestSetQuestionsResponse {
  success: boolean;
  test_set_id: number;
  generated_count: number;
  saved_count: number;
  persona_used: string;
  llm_used?: string;
}

export interface GenerateFromCourseResponse {
  success: boolean;
  test_set_id: number;
  generated_count: number;
  saved_count: number;
  statistics?: Record<string, unknown>;
  message?: string;
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
  course_id: number | null;
  created_at: string;
  processed: boolean;
  chunk_count: number;
  embedding_status: "pending" | "processing" | "completed" | "error";
  embedding_model: string | null;
  embedded_at: string | null;
  vector_count: number;
}

export interface Chunk {
  id?: number;
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
  active_prompt_template_id?: number | null;
  system_prompt_remembering?: string | null;
  system_prompt_understanding_applying?: string | null;
  system_prompt_analyzing_evaluating?: string | null;
  enable_reranker: boolean;
  reranker_provider: string | null;
  reranker_model: string | null;
  reranker_top_k: number;
  enable_direct_llm: boolean;
  enable_pii_filter: boolean;
  vector_store: string; // "weaviate" or "chromadb"
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
  active_prompt_template_id?: number | null;
  system_prompt_remembering?: string;
  system_prompt_understanding_applying?: string;
  system_prompt_analyzing_evaluating?: string;
  enable_reranker?: boolean;
  reranker_provider?: string;
  reranker_model?: string;
  reranker_top_k?: number;
  enable_direct_llm?: boolean;
  enable_pii_filter?: boolean;
  vector_store?: string; // "weaviate" or "chromadb"
}

export interface CoursePromptTemplate {
  id: number;
  course_id: number;
  name: string;
  content: string;
  created_at: string;
  updated_at: string;
}

export interface CoursePromptTemplateListResponse {
  templates: CoursePromptTemplate[];
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

  private getToken(): string {
    return this.token || localStorage.getItem(TOKEN_KEY) || "";
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
    const isFormDataBody = typeof FormData !== "undefined" && options.body instanceof FormData;

    const headers: HeadersInit = {
      ...options.headers,
    };

    if (!isFormDataBody && (headers as Record<string, string>)["Content-Type"] === undefined) {
      (headers as Record<string, string>)["Content-Type"] = "application/json";
    }

    const token = this.getToken();
    if (token) {
      (headers as Record<string, string>)["Authorization"] = `Bearer ${token}`;
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
        
        throw new Error("Oturum sÃ¼resi doldu. LÃ¼tfen tekrar giriÅŸ yapÄ±n.");
      }
      
      const error: ApiError = await response.json().catch(() => ({
        detail: "Bir hata oluÅŸtu",
      }));
      
      // Handle detail as string or object
      let errorMessage = "Bir hata oluÅŸtu";
      if (typeof error.detail === "string") {
        errorMessage = error.detail;
      } else if (error.detail && typeof error.detail === "object") {
        // If detail is an object, try to extract message
        errorMessage = JSON.stringify(error.detail);
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      throw new Error(errorMessage);
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
        detail: "GiriÅŸ baÅŸarÄ±sÄ±z",
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

  async getCoursePromptTemplates(courseId: number): Promise<CoursePromptTemplateListResponse> {
    return this.request<CoursePromptTemplateListResponse>(
      `/api/courses/${courseId}/prompt-templates`
    );
  }

  async createCoursePromptTemplate(
    courseId: number,
    data: { name: string; content: string }
  ): Promise<CoursePromptTemplate> {
    return this.request<CoursePromptTemplate>(
      `/api/courses/${courseId}/prompt-templates`,
      {
        method: "POST",
        body: JSON.stringify(data),
      }
    );
  }

  async updateCoursePromptTemplate(
    courseId: number,
    templateId: number,
    data: { name?: string; content?: string }
  ): Promise<CoursePromptTemplate> {
    return this.request<CoursePromptTemplate>(
      `/api/courses/${courseId}/prompt-templates/${templateId}`,
      {
        method: "PUT",
        body: JSON.stringify(data),
      }
    );
  }

  async deleteCoursePromptTemplate(
    courseId: number,
    templateId: number
  ): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      `/api/courses/${courseId}/prompt-templates/${templateId}`,
      { method: "DELETE" }
    );
  }

  async activateCoursePromptTemplate(
    courseId: number,
    templateId: number | null
  ): Promise<{ success: boolean; active_prompt_template_id: number | null }> {
    const params = new URLSearchParams();
    if (templateId !== null) params.append("template_id", templateId.toString());
    const qs = params.toString();
    return this.request<{ success: boolean; active_prompt_template_id: number | null }>(
      `/api/courses/${courseId}/prompt-templates/activate${qs ? `?${qs}` : ""}`,
      { method: "POST" }
    );
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
      headers["Authorization"] = `Bearer ${this.getToken()}`;
    }

    const response = await fetch(`${API_URL}/api/courses/${courseId}/documents`, {
      method: "POST",
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: "Dosya yÃ¼klenirken hata oluÅŸtu",
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

  async deleteChunk(documentId: number, chunkId: number): Promise<void> {
    await this.request(`/api/documents/${documentId}/chunks/${chunkId}`, { method: "DELETE" });
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

  async deleteCourseCollection(courseId: number): Promise<void> {
    await this.request(`/api/courses/${courseId}/collection`, { method: "DELETE" });
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

  async generateFromCourse(data: {
    test_set_id: number;
    total_questions?: number;
    remembering_ratio?: number;
    understanding_applying_ratio?: number;
    analyzing_evaluating_ratio?: number;
  }): Promise<GenerateFromCourseResponse> {
    const formData = new FormData();
    formData.append("test_set_id", data.test_set_id.toString());
    if (data.total_questions !== undefined) formData.append("total_questions", data.total_questions.toString());
    if (data.remembering_ratio !== undefined) formData.append("remembering_ratio", data.remembering_ratio.toString());
    if (data.understanding_applying_ratio !== undefined) {
      formData.append("understanding_applying_ratio", data.understanding_applying_ratio.toString());
    }
    if (data.analyzing_evaluating_ratio !== undefined) {
      formData.append("analyzing_evaluating_ratio", data.analyzing_evaluating_ratio.toString());
    }

    return this.request<GenerateFromCourseResponse>("/api/test-generation/generate-from-course", {
      method: "POST",
      body: formData,
    });
  }

  async *generateWithQualityFilter(data: {
    test_set_id: number;
    target_questions: number;
    min_rouge1_score?: number;
    remembering_ratio?: number;
    understanding_applying_ratio?: number;
    analyzing_evaluating_ratio?: number;
  }): AsyncGenerator<{
    event: string;
    message?: string;
    question?: string;
    bloom_level?: string;
    rouge1?: number;
    accepted?: number;
    rejected?: number;
    target?: number;
    reason?: string;
    error?: string;
  }> {
    const formData = new FormData();
    formData.append("test_set_id", data.test_set_id.toString());
    formData.append("target_questions", data.target_questions.toString());
    if (data.min_rouge1_score !== undefined) {
      formData.append("min_rouge1_score", data.min_rouge1_score.toString());
    }
    if (data.remembering_ratio !== undefined) {
      formData.append("remembering_ratio", data.remembering_ratio.toString());
    }
    if (data.understanding_applying_ratio !== undefined) {
      formData.append("understanding_applying_ratio", data.understanding_applying_ratio.toString());
    }
    if (data.analyzing_evaluating_ratio !== undefined) {
      formData.append("analyzing_evaluating_ratio", data.analyzing_evaluating_ratio.toString());
    }

    const token = this.getToken();
    const response = await fetch(`${API_URL}/api/test-generation/generate-with-quality-filter`, {
      method: "POST",
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error("No response body");
    }

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data.trim()) {
              try {
                const parsed = JSON.parse(data);
                yield parsed;
              } catch (e) {
                console.error("Failed to parse SSE data:", e);
              }
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
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

  async createTestSet(data: { course_id: number; name: string; description?: string }): Promise<TestSet> {
    return this.request<TestSet>("/api/ragas/test-sets", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getTestSet(testSetId: number): Promise<TestSetDetail> {
    return this.request<TestSetDetail>(`/api/ragas/test-sets/${testSetId}`);
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

  async mergeTestSets(targetTestSetId: number, sourceTestSetId: number): Promise<TestSetDetail> {
    return this.request<TestSetDetail>(`/api/ragas/test-sets/${targetTestSetId}/merge/${sourceTestSetId}`, {
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

  async findDuplicateQuestions(testSetId: number, similarityThreshold: number = 0.85): Promise<FindDuplicatesResponse> {
    return this.request<FindDuplicatesResponse>("/api/ragas/test-sets/find-duplicates", {
      method: "POST",
      body: JSON.stringify({
        test_set_id: testSetId,
        similarity_threshold: similarityThreshold,
      }),
    });
  }

  async deleteMultipleQuestions(testSetId: number, questionIds: number[]): Promise<{ deleted_count: number; question_ids: number[] }> {
    return this.request(`/api/ragas/test-sets/${testSetId}/delete-questions`, {
      method: "POST",
      body: JSON.stringify({
        question_ids: questionIds,
      }),
    });
  }

  async generateTestSetQuestions(
    testSetId: number,
    data: { num_questions?: number; persona?: string }
  ): Promise<GenerateTestSetQuestionsResponse> {
    const params = new URLSearchParams();
    if (data.num_questions !== undefined) params.append("num_questions", data.num_questions.toString());
    if (data.persona !== undefined) params.append("persona", data.persona);
    const qs = params.toString();

    return this.request<GenerateTestSetQuestionsResponse>(
      `/api/ragas/test-sets/${testSetId}/generate-questions${qs ? `?${qs}` : ""}`,
      {
        method: "POST",
      }
    );
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
  async startEvaluation(data: { test_set_id: number; course_id: number; name?: string; config?: EvaluationConfig; evaluation_provider?: string; evaluation_model?: string; question_ids?: number[] }): Promise<EvaluationRun> {
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

  async getRunStatus(runId: number): Promise<{ id: number; status: string; total_questions: number; processed_questions: number; error_message?: string; wandb_run_url?: string; wandb_run_id?: string }> {
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

  async wandbExportRagasRun(data: {
    course_id: number;
    run_id: number;
  }): Promise<{ success: boolean; run_name: string; run_url?: string; exported_count: number }> {
    return this.request<{ success: boolean; run_name: string; run_url?: string; exported_count: number }>(
      "/api/ragas/wandb-export",
      {
        method: "POST",
        body: JSON.stringify(data),
      }
    );
  }

  async getRagasWandbRuns(
    courseId: number,
    page: number = 1,
    limit: number = 10,
    search?: string,
    stateFilter?: string,
    tagFilter?: string
  ): Promise<{
    runs: Array<{
      id: string;
      name: string;
      state: string;
      created_at: string | null;
      config: Record<string, unknown>;
      missing_fields: string[];
    }>;
    pagination: {
      currentPage: number;
      totalPages: number;
      totalItems: number;
      itemsPerPage: number;
    };
  }> {
    const params = new URLSearchParams({
      course_id: courseId.toString(),
      page: page.toString(),
      limit: limit.toString(),
    });
    if (search) params.append("search", search);
    if (stateFilter && stateFilter !== "all") params.append("state", stateFilter);
    if (tagFilter) params.append("tag", tagFilter);

    return this.request(
      `/api/ragas/wandb-runs?${params.toString()}`
    );
  }

  async updateRagasWandbRun(data: {
    run_id: string;
    course_id: number;
    evaluation_run_id: number;
    tags?: string[];
  }): Promise<{ success: boolean; updated_fields?: string[]; message?: string; run_name?: string }> {
    return this.request<{ success: boolean; updated_fields?: string[]; message?: string; run_name?: string }>(
      "/api/ragas/wandb-runs/update",
      {
        method: "POST",
        body: JSON.stringify(data),
      }
    );
  }

  async quickTest(data: QuickTestRequest): Promise<QuickTestResponse> {
    return this.request<QuickTestResponse>("/api/ragas/quick-test", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async ragasBatchTest(data: {
    course_id: number;
    test_cases: Array<{
      question: string;
      ground_truth: string;
      alternative_ground_truths?: string[];
    }>;
    llm_provider?: string;
    llm_model?: string;
  }): Promise<{
    results: Array<{
      question: string;
      ground_truth: string;
      generated_answer: string;
      faithfulness?: number;
      answer_relevancy?: number;
      context_precision?: number;
      context_recall?: number;
      answer_correctness?: number;
      latency_ms: number;
    }>;
    aggregate: {
      avg_faithfulness?: number;
      avg_answer_relevancy?: number;
      avg_context_precision?: number;
      avg_context_recall?: number;
      avg_answer_correctness?: number;
      test_count: number;
    };
  }> {
    // RAGAS'ta batch test yok, her soruyu tek tek Ã§alÄ±ÅŸtÄ±rmamÄ±z gerekiyor
    const results: Array<{
      question: string;
      ground_truth: string;
      generated_answer: string;
      faithfulness?: number;
      answer_relevancy?: number;
      context_precision?: number;
      context_recall?: number;
      answer_correctness?: number;
      latency_ms: number;
    }> = [];

    for (const testCase of data.test_cases) {
      try {
        const result = await this.quickTest({
          course_id: data.course_id,
          question: testCase.question,
          ground_truth: testCase.ground_truth,
          alternative_ground_truths: testCase.alternative_ground_truths,
          llm_provider: data.llm_provider,
          llm_model: data.llm_model,
        });

        results.push({
          question: result.question,
          ground_truth: result.ground_truth,
          generated_answer: result.generated_answer,
          faithfulness: result.faithfulness,
          answer_relevancy: result.answer_relevancy,
          context_precision: result.context_precision,
          context_recall: result.context_recall,
          answer_correctness: result.answer_correctness,
          latency_ms: result.latency_ms,
        });
      } catch (error) {
        // Hata durumunda da sonucu ekle
        results.push({
          question: testCase.question,
          ground_truth: testCase.ground_truth,
          generated_answer: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
          latency_ms: 0,
        });
      }
    }

    // Aggregate hesapla
    const validResults = results.filter(r => r.faithfulness !== undefined);
    const aggregate = {
      avg_faithfulness: validResults.length > 0 
        ? validResults.reduce((sum, r) => sum + (r.faithfulness || 0), 0) / validResults.length 
        : undefined,
      avg_answer_relevancy: validResults.length > 0
        ? validResults.reduce((sum, r) => sum + (r.answer_relevancy || 0), 0) / validResults.length
        : undefined,
      avg_context_precision: validResults.length > 0
        ? validResults.reduce((sum, r) => sum + (r.context_precision || 0), 0) / validResults.length
        : undefined,
      avg_context_recall: validResults.length > 0
        ? validResults.reduce((sum, r) => sum + (r.context_recall || 0), 0) / validResults.length
        : undefined,
      avg_answer_correctness: validResults.length > 0
        ? validResults.reduce((sum, r) => sum + (r.answer_correctness || 0), 0) / validResults.length
        : undefined,
      test_count: results.length,
    };

    return { results, aggregate };
  }

  // ==================== Admin User Management Endpoints ====================

  async createAdminUser(data: AdminUserCreate): Promise<AdminUser> {
    const res = await this.request<{ success: boolean; user: AdminUser; message: string }>("/api/admin/users", {
      method: "POST",
      body: JSON.stringify(data),
    });
    return res.user;
  }

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

  // ==================== Giskard Endpoints ====================
  
  async giskardQuickTest(data: {
    course_id: number;
    question: string;
    question_type: "relevant" | "irrelevant";
    expected_answer: string;
    system_prompt?: string;
    llm_provider?: string;
    llm_model?: string;
  }): Promise<GiskardQuickTestResponse> {
    return this.request<GiskardQuickTestResponse>("/api/giskard/quick-test", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getGiskardQuickTestResults(
    courseId: number,
    groupName?: string,
    skip: number = 0,
    limit: number = 10
  ): Promise<GiskardQuickTestResultListResponse> {
    const params = new URLSearchParams({
      course_id: courseId.toString(),
      skip: skip.toString(),
      limit: limit.toString()
    });
    if (groupName) params.append("group_name", groupName);
    return this.request<GiskardQuickTestResultListResponse>(
      `/api/giskard/courses/${courseId}/quick-test-results?${params.toString()}`,
    );
  }

  async saveGiskardQuickTestResult(data: {
    course_id: number;
    group_name?: string;
    question: string;
    question_type: "relevant" | "irrelevant";
    expected_answer: string;
    generated_answer: string;
    score?: number;
    correct_refusal?: boolean;
    hallucinated?: boolean;
    provided_answer?: boolean;
    language?: string;
    quality_score?: number;
    system_prompt?: string;
    llm_provider?: string;
    llm_model?: string;
    embedding_model?: string;
    latency_ms?: number;
    error_message?: string;
  }): Promise<GiskardQuickTestSavedResult> {
    return this.request<GiskardQuickTestSavedResult>("/api/giskard/quick-test/save", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async deleteGiskardQuickTestResult(resultId: number): Promise<void> {
    await this.request(`/api/giskard/quick-test-results/${resultId}`, { method: "DELETE" });
  }

  // Giskard Test Sets
  async getGiskardTestSets(courseId: number): Promise<GiskardTestSet[]> {
    return this.request<GiskardTestSet[]>(`/api/giskard/courses/${courseId}/test-sets`);
  }

  async getGiskardTestSet(testSetId: number): Promise<GiskardTestSet> {
    return this.request<GiskardTestSet>(`/api/giskard/test-sets/${testSetId}`);
  }

  async createGiskardTestSet(data: { course_id: number; name: string; description?: string }): Promise<GiskardTestSet> {
    return this.request<GiskardTestSet>("/api/giskard/test-sets", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async deleteGiskardTestSet(testSetId: number): Promise<void> {
    await this.request(`/api/giskard/test-sets/${testSetId}`, { method: "DELETE" });
  }

  async generateGiskardRagetTestset(data: {
    course_id: number;
    num_questions: number;
    language?: "tr" | "en";
    agent_description?: string;
  }): Promise<{
    num_questions: number;
    samples: Array<Record<string, unknown>>;
  }> {
    return this.request(`/api/giskard/raget/generate-testset`, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Giskard Questions
  async addGiskardQuestion(
    testSetId: number,
    data: {
      question: string;
      question_type: "relevant" | "irrelevant";
      expected_answer: string;
      question_metadata?: Record<string, unknown>;
    },
  ): Promise<GiskardQuestion> {
    return this.request<GiskardQuestion>(`/api/giskard/test-sets/${testSetId}/questions`, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getGiskardQuestions(testSetId: number): Promise<GiskardQuestion[]> {
    return this.request<GiskardQuestion[]>(`/api/giskard/test-sets/${testSetId}/questions`);
  }

  async deleteGiskardQuestion(questionId: number): Promise<void> {
    await this.request(`/api/giskard/questions/${questionId}`, { method: "DELETE" });
  }

  // Giskard Evaluation Runs
  async startGiskardEvaluation(data: {
    test_set_id: number;
    course_id: number;
    name: string;
    total_questions: number;
    config?: Record<string, unknown>;
  }): Promise<GiskardEvaluationRun> {
    return this.request<GiskardEvaluationRun>(
      `/api/giskard/test-sets/${data.test_set_id}/runs`,
      {
      method: "POST",
        body: JSON.stringify(data),
      },
    );
  }

  async getGiskardEvaluationRuns(courseId: number): Promise<GiskardEvaluationRun[]> {
    return this.request<GiskardEvaluationRun[]>(`/api/giskard/courses/${courseId}/runs`);
  }

  async getGiskardEvaluationRun(runId: number): Promise<GiskardEvaluationRun> {
    return this.request<GiskardEvaluationRun>(`/api/giskard/runs/${runId}`);
  }

  async deleteGiskardEvaluationRun(runId: number): Promise<void> {
    await this.request(`/api/giskard/runs/${runId}`, { method: "DELETE" });
  }

  async getGiskardRunResults(runId: number): Promise<GiskardResult[]> {
    return this.request<GiskardResult[]>(`/api/giskard/runs/${runId}/results`);
  }

  async getGiskardRunSummary(runId: number): Promise<GiskardSummary> {
    return this.request<GiskardSummary>(`/api/giskard/runs/${runId}/summary`);
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

  async getExistingQuickTestQuestions(
    courseId: number,
    groupName: string
  ): Promise<{ questions: string[] }> {
    const params = new URLSearchParams({
      course_id: courseId.toString(),
      group_name: groupName,
    });
    return this.request<{ questions: string[] }>(
      `/api/ragas/quick-test-results/existing-questions?${params.toString()}`
    );
  }

  async getQuickTestResult(resultId: number): Promise<QuickTestResult> {
    return this.request<QuickTestResult>(`/api/ragas/quick-test-results/${resultId}`);
  }

  async deleteQuickTestResult(resultId: number): Promise<void> {
    await this.request(`/api/ragas/quick-test-results/${resultId}`, { method: "DELETE" });
  }

  async exportQuickTestResultsToWandB(courseId: number, groupName: string): Promise<{
    success: boolean;
    run_name: string;
    run_id: string;
    run_url: string;
    exported_count: number;
    aggregate_metrics: {
      avg_faithfulness: number | null;
      avg_answer_relevancy: number | null;
      avg_context_precision: number | null;
      avg_context_recall: number | null;
      avg_answer_correctness: number | null;
      avg_latency_ms: number | null;
    };
  }> {
    return this.request(`/api/ragas/quick-test-results/wandb-export`, {
      method: "POST",
      body: JSON.stringify({
        course_id: courseId,
        group_name: groupName,
      }),
    });
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

  async saveSemanticSimilarityResultsBatch(data: {
    course_id: number;
    group_name?: string;
    results: Array<{
      question: string;
      ground_truth: string;
      generated_answer: string;
      similarity_score: number;
      best_match_ground_truth: string;
      rouge1?: number | null;
      rouge2?: number | null;
      rougel?: number | null;
      bertscore_precision?: number | null;
      bertscore_recall?: number | null;
      bertscore_f1?: number | null;
      original_bertscore_precision?: number | null;
      original_bertscore_recall?: number | null;
      original_bertscore_f1?: number | null;
      latency_ms?: number;
      embedding_model_used?: string;
      llm_model_used?: string;
      retrieved_contexts?: string[];
      system_prompt_used?: string;
      search_top_k?: number;
      search_alpha?: number;
      reranker_used?: boolean;
      reranker_provider?: string;
      reranker_model?: string;
    }>;
  }): Promise<{ saved_count: number; failed_count: number; group_name?: string }> {
    return this.request("/api/semantic-similarity/results/batch", {
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

  async renameSemanticSimilarityGroup(courseId: number, oldGroupName: string, newGroupName: string): Promise<{ success: boolean; message: string; updated_count: number }> {
    return this.request<{ success: boolean; message: string; updated_count: number }>(
      `/api/semantic-similarity/groups/rename?course_id=${courseId}&old_group_name=${encodeURIComponent(oldGroupName)}&new_group_name=${encodeURIComponent(newGroupName)}`,
      { method: "PUT" }
    );
  }

  async deleteSemanticSimilarityGroup(courseId: number, groupName: string): Promise<{ success: boolean; message: string; deleted_count: number }> {
    return this.request<{ success: boolean; message: string; deleted_count: number }>(
      `/api/semantic-similarity/groups/${encodeURIComponent(groupName)}?course_id=${courseId}`,
      { method: "DELETE" }
    );
  }

  // ==================== RAGAS Group Management ====================

  async renameRagasGroup(
    courseId: number, 
    oldGroupName: string, 
    newGroupName: string
  ): Promise<{ success: boolean; message: string; updated_count: number }> {
    return this.request<{ success: boolean; message: string; updated_count: number }>(
      `/api/ragas/groups/rename?course_id=${courseId}&old_group_name=${encodeURIComponent(oldGroupName)}&new_group_name=${encodeURIComponent(newGroupName)}`,
      { method: "PUT" }
    );
  }

  async deleteRagasGroup(
    courseId: number, 
    groupName: string
  ): Promise<{ success: boolean; message: string; deleted_count: number }> {
    return this.request<{ success: boolean; message: string; deleted_count: number }>(
      `/api/ragas/groups/${encodeURIComponent(groupName)}?course_id=${courseId}`,
      { method: "DELETE" }
    );
  }

  async wandbExportSemanticSimilarityGroup(data: {
    course_id: number;
    group_name: string;
  }): Promise<{ success: boolean; run_name: string; run_url?: string; exported_count: number }> {
    return this.request<{ success: boolean; run_name: string; run_url?: string; exported_count: number }>(
      "/api/semantic-similarity/wandb-export",
      {
        method: "POST",
        body: JSON.stringify(data),
      }
    );
  }

  async getWandbRuns(courseId: number, page: number = 1, limit: number = 10, search?: string, stateFilter?: string, tagFilter?: string): Promise<{
    runs: Array<{
      id: string;
      name: string;
      state: string;
      created_at: string | null;
      config: Record<string, unknown>;
      missing_fields: string[];
    }>;
    pagination: {
      currentPage: number;
      totalPages: number;
      totalItems: number;
      itemsPerPage: number;
    };
  }> {
    const params = new URLSearchParams({
      course_id: courseId.toString(),
      page: page.toString(),
      limit: limit.toString(),
    });
    
    if (search) params.append('search', search);
    if (stateFilter && stateFilter !== 'all') params.append('state', stateFilter);
    if (tagFilter) params.append('tag', tagFilter);

    return this.request<{
      runs: Array<{
        id: string;
        name: string;
        state: string;
        created_at: string | null;
        config: Record<string, unknown>;
        missing_fields: string[];
      }>;
      pagination: {
        currentPage: number;
        totalPages: number;
        totalItems: number;
        itemsPerPage: number;
      };
    }>(
      `/api/semantic-similarity/wandb-runs?${params.toString()}`
    );
  }

  async updateWandbRun(data: {
    run_id: string;
    group_name: string;
    course_id: number;
    tags?: string[];
  }): Promise<{ success: boolean; updated_fields?: string[]; message?: string; run_name?: string }> {
    return this.request<{ success: boolean; updated_fields?: string[]; message?: string; run_name?: string }>(
      "/api/semantic-similarity/wandb-runs/update",
      {
        method: "POST",
        body: JSON.stringify(data),
      }
    );
  }

  // ==================== Batch Test Session Endpoints ====================

  async createBatchTestSession(data: {
    course_id: number;
    test_cases: Array<{
      question: string;
      ground_truth: string;
      alternative_ground_truths?: string[];
      generated_answer?: string;
    }>;
    embedding_provider?: string;
    embedding_model?: string;
    llm_provider?: string;
    llm_model?: string;
    reranker_used?: boolean;
    reranker_provider?: string;
    reranker_model?: string;
  }): Promise<BatchTestSession> {
    return this.request<BatchTestSession>("/api/semantic-similarity/batch-test-sessions", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getBatchTestSessions(courseId: number, skip: number = 0, limit: number = 20): Promise<BatchTestSessionListResponse> {
    return this.request<BatchTestSessionListResponse>(
      `/api/semantic-similarity/batch-test-sessions?course_id=${courseId}&skip=${skip}&limit=${limit}`
    );
  }

  async getBatchTestSession(sessionId: number): Promise<BatchTestSession> {
    return this.request<BatchTestSession>(`/api/semantic-similarity/batch-test-sessions/${sessionId}`);
  }

  async resumeBatchTestSession(sessionId: number): Promise<void> {
    // This is a streaming endpoint, handled separately
    void sessionId;
    return Promise.resolve();
  }

  async cancelBatchTestSession(sessionId: number): Promise<void> {
    return this.request(`/api/semantic-similarity/batch-test-sessions/${sessionId}`, {
      method: "DELETE",
    });
  }

  async deleteBatchTestSession(sessionId: number): Promise<void> {
    return this.request(`/api/semantic-similarity/batch-test-sessions/${sessionId}/delete`, {
      method: "DELETE",
    });
  }

  // ==================== Test Dataset Endpoints ====================

  async saveTestDataset(data: {
    course_id: number;
    name: string;
    description?: string;
    test_cases: Array<{
      question: string;
      ground_truth: string;
      alternative_ground_truths?: string[];
      generated_answer?: string;
    }>;
  }): Promise<{
    id: number;
    name: string;
    description?: string;
    total_test_cases: number;
    created_at: string;
  }> {
    return this.request("/api/semantic-similarity/test-datasets", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getTestDatasets(courseId: number): Promise<{
    datasets: Array<{
      id: number;
      name: string;
      description?: string;
      total_test_cases: number;
      created_at: string;
      updated_at: string;
    }>;
  }> {
    return this.request(`/api/semantic-similarity/test-datasets?course_id=${courseId}`);
  }

  async getTestDataset(datasetId: number): Promise<{
    id: number;
    name: string;
    description?: string;
    test_cases: Array<{
      question: string;
      ground_truth: string;
      alternative_ground_truths?: string[];
      generated_answer?: string;
    }>;
    total_test_cases: number;
    created_at: string;
    updated_at: string;
  }> {
    return this.request(`/api/semantic-similarity/test-datasets/${datasetId}`);
  }

  async deleteTestDataset(datasetId: number): Promise<{ message: string }> {
    return this.request(`/api/semantic-similarity/test-datasets/${datasetId}`, {
      method: "DELETE",
    });
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

  // User document endpoints
  async getUserDocuments(): Promise<Document[]> {
    const response = await this.request<DocumentListResponse>("/api/users/documents");
    return response.documents;
  }

  async uploadUserDocument(file: File, courseId?: number): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);
    if (courseId) {
      formData.append('course_id', courseId.toString());
    }

    const response = await fetch(`${API_URL}/api/users/documents`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.getToken()}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(error || 'Upload failed');
    }

    return response.json();
  }

  // ==================== Backup Endpoints ====================

  async listBackups(): Promise<BackupListResponse> {
    return this.request<BackupListResponse>("/api/admin/backup/list");
  }

  async createPostgresBackup(): Promise<BackupCreateResponse> {
    return this.request<BackupCreateResponse>("/api/admin/backup/create/postgres", {
      method: "POST",
    });
  }

  async createWeaviateBackup(): Promise<BackupCreateResponse> {
    return this.request<BackupCreateResponse>("/api/admin/backup/create/weaviate", {
      method: "POST",
    });
  }

  async createFullBackup(): Promise<BackupCreateResponse> {
    return this.request<BackupCreateResponse>("/api/admin/backup/create/full", {
      method: "POST",
    });
  }

  async downloadBackup(filename: string): Promise<Blob> {
    const token = this.getToken();
    const response = await fetch(`${API_URL}/api/admin/backup/download/${filename}`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error("Yedek indirilemedi");
    }

    return response.blob();
  }

  async restorePostgresBackup(file: File): Promise<BackupRestoreResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const token = this.getToken();
    const response = await fetch(`${API_URL}/api/admin/backup/restore/postgres`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Restore baÅŸarÄ±sÄ±z" }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  async restoreWeaviateBackup(file: File): Promise<BackupRestoreResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const token = this.getToken();
    const response = await fetch(`${API_URL}/api/admin/backup/restore/weaviate`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Restore baÅŸarÄ±sÄ±z" }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  async deleteBackup(filename: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(
      `/api/admin/backup/delete/${filename}`,
      { method: "DELETE" }
    );
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

// ==================== Backup Types ====================

export interface BackupInfo {
  filename: string;
  size: number;
  created_at: string;
  type: string;
}

export interface BackupListResponse {
  backups: BackupInfo[];
  total: number;
}

export interface BackupCreateResponse {
  success: boolean;
  message: string;
  filename: string;
  size: number;
  created_at: string;
}

export interface BackupRestoreResponse {
  success: boolean;
  message: string;
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
  question_metadata?: {
    bloom_level?: string;
    generated_by?: string;
    generated_at?: string;
    chunk_id?: string;
    llm_provider?: string;
    llm_model?: string;
    [key: string]: unknown;
  };
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

export interface DuplicateGroup {
  similarity_score: number;
  questions: Array<{
    id: number;
    question: string;
    ground_truth: string;
  }>;
}

export interface FindDuplicatesResponse {
  test_set_id: number;
  test_set_name: string;
  total_questions: number;
  duplicate_groups: DuplicateGroup[];
  total_duplicates: number;
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
  wandb_run_url?: string;
  wandb_run_id?: string;
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

// ==================== Giskard Types ====================

export interface GiskardTestSet {
  id: number;
  course_id: number;
  name: string;
  description?: string;
  created_by: number;
  created_at: string;
  updated_at: string;
  question_count: number;
}

export interface GiskardQuestion {
  id: number;
  test_set_id: number;
  question: string;
  question_type: "relevant" | "irrelevant";
  expected_answer: string;
  question_metadata?: Record<string, unknown>;
  created_at: string;
}

export interface GiskardEvaluationRun {
  id: number;
  test_set_id: number;
  test_set_name?: string;
  course_id: number;
  name?: string;
  status: "pending" | "running" | "completed" | "failed";
  total_questions: number;
  processed_questions: number;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  created_at: string;
}

export interface GiskardResult {
  id: number;
  run_id: number;
  question_id: number;
  question_text: string;
  expected_answer: string;
  question_type: "relevant" | "irrelevant";
  generated_answer?: string;
  retrieved_contexts?: string[];
  score?: number;
  hallucinated?: boolean;
  correct_refusal?: boolean;
  language?: string;
  quality_score?: number;
  latency_ms?: number;
  error_message?: string;
  created_at: string;
}

export interface GiskardSummary {
  id: number;
  run_id: number;
  relevant_count: number;
  relevant_avg_score?: number;
  relevant_success_rate?: number;
  irrelevant_count: number;
  irrelevant_avg_score?: number;
  irrelevant_success_rate?: number;
  hallucination_rate?: number;
  correct_refusal_rate?: number;
  language_consistency?: number;
  turkish_response_rate?: number;
  overall_score?: number;
  total_questions: number;
  successful_questions: number;
  failed_questions: number;
  avg_latency_ms?: number;
  created_at: string;
}

export interface GiskardQuickTestSavedResult {
  id: number;
  course_id: number;
  group_name?: string;
  question: string;
  expected_answer: string;
  system_prompt?: string;
  llm_provider?: string;
  llm_model?: string;
  generated_answer: string;
  retrieved_contexts?: string[];
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
  answer_correctness?: number;
  latency_ms?: number;
  created_by: number;
  created_at: string;
}

export interface GiskardQuickTestResultListResponse {
  results: GiskardQuickTestSavedResult[];
  total: number;
  groups: string[];
}

export interface GiskardQuickTestResponse {
  question: string;
  expected_answer: string;
  generated_answer: string;
  question_type: "relevant" | "irrelevant";
  score?: number;
  hallucinated?: boolean;
  correct_refusal?: boolean;
  provided_answer?: boolean;
  language?: string;
  quality_score?: number;
  system_prompt_used?: string;
  llm_provider_used?: string;
  llm_model_used?: string;
  embedding_model_used?: string;
  latency_ms?: number;
  retrieved_contexts?: string[];
  error_message?: string;
}

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
  ragas_embedding_model?: string;
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
  evaluation_model_used?: string;
  embedding_model_used?: string;
  search_top_k_used?: number;
  search_alpha_used?: number;
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
  evaluation_model?: string;
  embedding_model?: string;
  search_top_k?: number;
  search_alpha?: number;
  reranker_used?: boolean;
  reranker_provider?: string;
  reranker_model?: string;
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
  evaluation_model?: string;
  embedding_model?: string;
  search_top_k?: number;
  search_alpha?: number;
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

export interface RagasGroupInfo {
  name: string;
  created_at: string | null;
  test_count?: number | null;
  llm_provider?: string | null;
  llm_model?: string | null;
  evaluation_model?: string | null;
  embedding_model?: string | null;
  search_top_k?: number | null;
  search_alpha?: number | null;
  reranker_used?: boolean | null;
  reranker_provider?: string | null;
  reranker_model?: string | null;
  avg_faithfulness?: number | null;
  avg_answer_relevancy?: number | null;
  avg_context_precision?: number | null;
  avg_context_recall?: number | null;
  avg_answer_correctness?: number | null;
}

export interface QuickTestResultListResponse {
  results: QuickTestResult[];
  total: number;
  groups: RagasGroupInfo[];
  aggregate?: {
    avg_faithfulness?: number;
    avg_answer_relevancy?: number;
    avg_context_precision?: number;
    avg_context_recall?: number;
    avg_answer_correctness?: number;
    test_count?: number;
    test_parameters?: {
      llm_model?: string;
      llm_provider?: string;
      embedding_model?: string;
      evaluation_model?: string;
      search_alpha?: number;
      search_top_k?: number;
      reranker_used?: boolean;
      reranker_provider?: string | null;
      reranker_model?: string | null;
    };
  };
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
  use_direct_llm?: boolean;
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

  original_bertscore_precision?: number;
  original_bertscore_recall?: number;
  original_bertscore_f1?: number;
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
  use_direct_llm?: boolean;
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

    original_bertscore_precision?: number;
    original_bertscore_recall?: number;
    original_bertscore_f1?: number;
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

    avg_original_bertscore_precision?: number;
    avg_original_bertscore_recall?: number;
    avg_original_bertscore_f1?: number;
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
  bloom_level?: string;
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

  original_bertscore_precision?: number;
  original_bertscore_recall?: number;
  original_bertscore_f1?: number;
  // Retrieval metrics
  hit_at_1?: number;
  mrr?: number;
  // Search and reranker parameters
  search_top_k?: number;
  search_alpha?: number;
  reranker_used?: boolean;
  reranker_provider?: string;
  reranker_model?: string;
  created_by: number;
  created_at: string;
}

export interface SemanticSimilarityGroupInfo {
  name: string;
  created_at: string | null;
  test_count: number;
  avg_rouge1?: number;
  avg_rouge2?: number;
  avg_rougel?: number;
  avg_bertscore_precision?: number;
  avg_bertscore_recall?: number;
  avg_bertscore_f1?: number;
  avg_original_bertscore_precision?: number;
  avg_original_bertscore_recall?: number;
  avg_original_bertscore_f1?: number;
  avg_latency_ms?: number;
  llm_model?: string;
  embedding_model?: string;
  search_top_k?: number;
  search_alpha?: number;
  reranker_used?: boolean;
  reranker_provider?: string;
  reranker_model?: string;
}

export interface SemanticSimilarityResultListResponse {
  results: SemanticSimilarityResult[];
  total: number;
  groups: SemanticSimilarityGroupInfo[];
  aggregate?: {
    avg_similarity: number;
    avg_rouge1?: number;
    avg_rouge2?: number;
    avg_rougel?: number;
    avg_bertscore_precision?: number;
    avg_bertscore_recall?: number;
    avg_bertscore_f1?: number;

    avg_original_bertscore_precision?: number;
    avg_original_bertscore_recall?: number;
    avg_original_bertscore_f1?: number;
    test_count: number;
  };
}

// ==================== Batch Test Session Types ====================

export interface BatchTestSession {
  id: number;
  course_id: number;
  user_id: number;
  group_name: string;
  test_cases: string; // JSON string of test cases
  total_tests: number;
  completed_tests: number;
  failed_tests: number;
  current_index: number;
  status: "in_progress" | "completed" | "cancelled" | "failed";
  llm_provider?: string;
  llm_model?: string;
  embedding_model_used?: string;
  reranker_used?: boolean;
  reranker_provider?: string;
  reranker_model?: string;
  started_at: string;
  completed_at?: string | null;
  updated_at: string;
}

export interface BatchTestSessionListResponse {
  sessions: BatchTestSession[];
  total: number;
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

export interface AdminUserCreate {
  full_name: string;
  email: string;
  password: string;
  role: "teacher" | "student";
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
      })
      .catch((error) => {
        if (error.name === 'AbortError') {
          reject(new Error('Request cancelled'));
        } else {
          reject(error);
        }
      });
  });
}


// ==================== Backup Types ====================

export interface BackupInfo {
  filename: string;
  size: number;
  created_at: string;
  type: string;
}

export interface BackupListResponse {
  backups: BackupInfo[];
  total: number;
}

export interface BackupCreateResponse {
  success: boolean;
  message: string;
  filename: string;
  size: number;
  created_at: string;
}

export interface BackupRestoreResponse {
  success: boolean;
  message: string;
}
