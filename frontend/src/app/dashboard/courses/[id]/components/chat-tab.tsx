"use client";

import { useState, useRef, useEffect } from "react";
import { api, ChatMessage, ChunkReference } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { 
  Send, 
  Loader2, 
  Trash2, 
  FileText, 
  ChevronDown, 
  ChevronUp,
  MessageSquare,
  Bot,
  User,
  Clock,
  X
} from "lucide-react";

interface ChatTabProps {
  courseId: number;
}

interface Message extends ChatMessage {
  id?: number;
  sources?: ChunkReference[];
  timestamp?: string;
  responseTime?: number; // in milliseconds
}

const MESSAGES_PER_PAGE = 20;

export function ChatTab({ courseId }: ChatTabProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [expandedSources, setExpandedSources] = useState<number | null>(null);
  const [hasMoreMessages, setHasMoreMessages] = useState(false);
  const [oldestMessageId, setOldestMessageId] = useState<number | null>(null);
  const [selectedSource, setSelectedSource] = useState<ChunkReference | null>(null);
  const [isDirectLlmMode, setIsDirectLlmMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const dedupeMessagesByIdKeepLast = (items: Message[]) => {
    const seen = new Set<number>();
    const result: Message[] = [];
    for (let i = items.length - 1; i >= 0; i--) {
      const msg = items[i];
      if (msg.id != null) {
        if (seen.has(msg.id)) continue;
        seen.add(msg.id);
      }
      result.push(msg);
    }
    result.reverse();
    return result;
  };

  // Load chat history from server (DB)
  useEffect(() => {
    let cancelled = false;

    const loadHistory = async () => {
      try {
        const response = await api.getChatHistory(courseId, { limit: MESSAGES_PER_PAGE });
        if (cancelled) return;

        const serverMessages: Message[] = response.messages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          sources: m.sources ?? undefined,
          timestamp: m.created_at,
          responseTime: m.response_time_ms ?? undefined,
        }));

        const uniqueServerMessages = dedupeMessagesByIdKeepLast(serverMessages);
        setMessages(uniqueServerMessages);
        setHasMoreMessages(response.has_more);
        setOldestMessageId(
          uniqueServerMessages.length > 0
            ? uniqueServerMessages[0].id ?? null
            : null
        );
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Sohbet geçmişi yüklenemedi");
      }
    };

    loadHistory();

    return () => {
      cancelled = true;
    };
  }, [courseId]);

  // Load course settings to check Direct LLM mode
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const settings = await api.getCourseSettings(courseId);
        setIsDirectLlmMode(settings.enable_direct_llm || false);
      } catch {
        // Settings not available, keep default
      }
    };
    loadSettings();
  }, [courseId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (messages.length > 0) {
      scrollToBottom();
    }
  }, [messages.length]);

  // Count conversations (question-answer pairs)
  const conversationCount = Math.floor(messages.filter(m => m.role === "assistant").length);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    
    const newUserMessage: Message = { 
      role: "user", 
      content: userMessage,
      timestamp: new Date().toISOString()
    };
    
    const updatedMessages = [...messages, newUserMessage];
    setMessages(updatedMessages);
    setIsLoading(true);

    const startTime = Date.now();

    try {
      const history = messages.slice(-10).map((m) => ({ role: m.role, content: m.content }));
      const response = await api.chat(courseId, userMessage, history);
      
      const responseTime = Date.now() - startTime;
      
      const assistantMessage: Message = {
        role: "assistant",
        content: response.message,
        sources: response.sources,
        timestamp: new Date().toISOString(),
        responseTime
      };
      
      const finalMessages = [...updatedMessages, assistantMessage];
      setMessages(finalMessages);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Sohbet hatası");
      setMessages(messages);
      setInput(userMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = async () => {
    if (messages.length === 0) return;
    if (!confirm("Sohbet geçmişini temizlemek istediğinizden emin misiniz?")) return;

    try {
      await api.clearChatHistory(courseId);
      setMessages([]);
      setExpandedSources(null);
      setHasMoreMessages(false);
      setOldestMessageId(null);
      toast.success("Sohbet geçmişi temizlendi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Sohbet geçmişi temizlenemedi");
    }
  };

  const loadMoreMessages = async () => {
    if (!hasMoreMessages || isLoadingMore) return;

    try {
      setIsLoadingMore(true);

      const container = messagesContainerRef.current;
      const prevScrollHeight = container?.scrollHeight ?? 0;

      const response = await api.getChatHistory(courseId, {
        limit: MESSAGES_PER_PAGE,
        before_id: oldestMessageId ?? undefined,
      });

      const olderMessages: Message[] = response.messages.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        sources: m.sources ?? undefined,
        timestamp: m.created_at,
        responseTime: m.response_time_ms ?? undefined,
      }));

      setMessages((prev) => {
        const merged = [...olderMessages, ...prev];
        return dedupeMessagesByIdKeepLast(merged);
      });
      setHasMoreMessages(response.has_more);
      setOldestMessageId(olderMessages.length > 0 ? olderMessages[0].id ?? null : oldestMessageId);

      requestAnimationFrame(() => {
        const nextScrollHeight = container?.scrollHeight ?? 0;
        if (container) {
          container.scrollTop += nextScrollHeight - prevScrollHeight;
        }
      });
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Mesajlar yüklenemedi");
    } finally {
      setIsLoadingMore(false);
    }
  };

  const visibleMessages = messages;

  const formatTime = (timestamp?: string) => {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    return date.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" });
  };

  const formatResponseTime = (ms?: number) => {
    if (!ms) return "";
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <div className="bg-white rounded-lg border border-slate-200 flex flex-col h-[calc(100vh-280px)] lg:h-[calc(100vh-240px)] min-h-[400px]">
      {/* Header */}
      <div className="flex items-center justify-between px-3 sm:px-4 py-2.5 sm:py-3 border-b border-slate-200 bg-gradient-to-r from-indigo-50 to-purple-50">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center shrink-0">
            <MessageSquare className="w-4 h-4 text-indigo-600" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="font-medium text-slate-900 text-sm sm:text-base">Ders Asistanı</h3>
              {isDirectLlmMode && (
                <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 border border-amber-200">
                  Direct LLM
                </span>
              )}
            </div>
            <p className="text-xs text-slate-500">{conversationCount} sohbet</p>
          </div>
        </div>
        {messages.length > 0 && (
          <Button
            size="sm"
            variant="ghost"
            onClick={handleClear}
            className="text-slate-400 hover:text-red-600 hover:bg-red-50 shrink-0"
          >
            <Trash2 className="w-4 h-4 sm:mr-1" />
            <span className="hidden sm:inline">Temizle</span>
          </Button>
        )}
      </div>

      {/* Messages */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-3 sm:space-y-4 bg-slate-50/50"
      >
        {/* Load More Button */}
        {hasMoreMessages && (
          <div className="flex justify-center">
            <Button
              size="sm"
              variant="outline"
              onClick={loadMoreMessages}
              className="text-xs"
            >
              <ChevronUp className="w-3 h-3 mr-1" />
              Daha eski mesajları göster
            </Button>
          </div>
        )}

        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Bot className="w-8 h-8 text-indigo-600" />
              </div>
              <p className="text-slate-700 font-medium">Ders materyalleri hakkında soru sorun</p>
              <p className="text-sm text-slate-400 mt-1">
                Örnek: &quot;Bu dersin ana konuları nelerdir?&quot;
              </p>
            </div>
          </div>
        ) : (
          visibleMessages.map((message, idx) => {
            const globalIdx = idx;
            return (
              <div
                key={
                  message.id != null
                    ? `msg-${message.id}`
                    : `tmp-${message.timestamp ?? globalIdx}-${message.role}`
                }
                className={`flex gap-2 sm:gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.role === "assistant" && (
                  <div className="w-7 h-7 sm:w-8 sm:h-8 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0">
                    <Bot className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-indigo-600" />
                  </div>
                )}
                
                <div
                  className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-3 sm:px-4 py-2.5 sm:py-3 ${
                    message.role === "user"
                      ? "bg-indigo-600 text-white rounded-br-md"
                      : "bg-white text-slate-900 border border-slate-200 rounded-bl-md shadow-sm"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                  
                  {/* Timestamp and Response Time */}
                  <div className={`flex items-center gap-2 text-xs mt-1 ${
                    message.role === "user" ? "text-indigo-200" : "text-slate-400"
                  }`}>
                    <span>{formatTime(message.timestamp)}</span>
                    {message.responseTime && (
                      <>
                        <span>•</span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {formatResponseTime(message.responseTime)}
                        </span>
                      </>
                    )}
                  </div>
                  
                  {/* Sources */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-100">
                      <button
                        className="flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800"
                        onClick={() => setExpandedSources(expandedSources === globalIdx ? null : globalIdx)}
                      >
                        <FileText className="w-3 h-3" />
                        {message.sources.length} kaynak
                        {expandedSources === globalIdx ? (
                          <ChevronUp className="w-3 h-3" />
                        ) : (
                          <ChevronDown className="w-3 h-3" />
                        )}
                      </button>
                      
                      {expandedSources === globalIdx && (
                        <div className="mt-2 space-y-2">
                          {message.sources.map((source, sIdx) => (
                            <button
                              key={sIdx}
                              onClick={() => setSelectedSource(source)}
                              className="w-full text-left bg-slate-50 rounded-lg p-2 text-xs border border-slate-100 hover:border-indigo-300 hover:bg-indigo-50 transition-colors"
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-medium text-slate-700">
                                  {source.document_name}
                                </span>
                                <span className="text-indigo-600 font-medium">
                                  {(source.score * 100).toFixed(0)}%
                                </span>
                              </div>
                              <p className="text-slate-600 line-clamp-2">{source.content_preview}</p>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                {message.role === "user" && (
                  <div className="w-7 h-7 sm:w-8 sm:h-8 bg-indigo-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <User className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-white" />
                  </div>
                )}
              </div>
            );
          })
        )}
        
        {isLoading && (
          <div className="flex gap-2 sm:gap-3 justify-start">
            <div className="w-7 h-7 sm:w-8 sm:h-8 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0">
              <Bot className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-indigo-600" />
            </div>
            <div className="bg-white rounded-2xl rounded-bl-md px-4 py-3 border border-slate-200 shadow-sm">
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin text-indigo-600" />
                <span className="text-sm text-slate-500">Düşünüyor...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-3 sm:p-4 border-t border-slate-200 bg-white">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Mesajınızı yazın..."
            disabled={isLoading}
            className="flex-1 rounded-full px-4 text-sm sm:text-base h-10 sm:h-10"
          />
          <Button 
            onClick={handleSend} 
            disabled={!input.trim() || isLoading}
            className="rounded-full w-10 h-10 p-0"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Source Modal */}
      {selectedSource && (
        <div className="fixed inset-0 bg-black/50 flex items-end sm:items-center justify-center z-50 p-0 sm:p-4">
          <div className="bg-white rounded-t-xl sm:rounded-xl w-full sm:max-w-2xl max-h-[90vh] sm:max-h-[85vh] overflow-hidden shadow-xl flex flex-col">
            <div className="flex items-center justify-between px-4 sm:px-6 py-3 sm:py-4 border-b border-slate-200 bg-gradient-to-r from-indigo-50 to-purple-50 shrink-0">
              <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                <div className="w-8 h-8 sm:w-10 sm:h-10 bg-indigo-100 rounded-lg flex items-center justify-center shrink-0">
                  <FileText className="w-4 h-4 sm:w-5 sm:h-5 text-indigo-600" />
                </div>
                <div className="min-w-0">
                  <h3 className="font-medium text-slate-900 text-sm sm:text-base truncate">{selectedSource.document_name}</h3>
                  <p className="text-xs text-slate-500">
                    Chunk #{selectedSource.chunk_index + 1} • Benzerlik: {(selectedSource.score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <button
                onClick={() => setSelectedSource(null)}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors shrink-0"
              >
                <X className="w-5 h-5 text-slate-500" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 sm:p-6">
              <div className="prose prose-sm max-w-none">
                <p className="text-slate-700 whitespace-pre-wrap leading-relaxed text-sm">
                  {selectedSource.full_content || selectedSource.content_preview}
                </p>
              </div>
            </div>
            <div className="px-4 sm:px-6 py-3 sm:py-4 border-t border-slate-200 bg-slate-50 flex justify-end shrink-0">
              <Button variant="outline" onClick={() => setSelectedSource(null)}>
                Kapat
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
