"use client";

import { useState, useRef, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, Course } from "@/lib/api";
import {
  StoredMessage,
  createMessage,
  formatTimestamp,
  formatResponseTime,
} from "@/lib/chat-history";
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
  X,
  ArrowLeft
} from "lucide-react";

const MESSAGES_PER_PAGE = 20;

interface Message extends StoredMessage {
  id?: number;
}

export default function StudentChatPage() {
  const params = useParams();
  const router = useRouter();
  const courseId = Number(params.courseId);
  
  const [course, setCourse] = useState<Course | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [isCourseLoading, setIsCourseLoading] = useState(true);
  const [expandedSources, setExpandedSources] = useState<number | null>(null);
  const [hasMoreMessages, setHasMoreMessages] = useState(false);
  const [oldestMessageId, setOldestMessageId] = useState<number | null>(null);
  const [selectedSource, setSelectedSource] = useState<import("@/lib/api").ChunkReference | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Load course info
  useEffect(() => {
    const loadCourse = async () => {
      try {
        const courseData = await api.getCourse(courseId);
        setCourse(courseData);
      } catch {
        toast.error("Ders bulunamadı");
        router.push("/student");
      } finally {
        setIsCourseLoading(false);
      }
    };
    loadCourse();
  }, [courseId, router]);

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
          timestamp: m.created_at,
          sources: m.sources ?? undefined,
          responseTime: m.response_time_ms ?? undefined,
        }));

        setMessages(serverMessages);
        setHasMoreMessages(response.has_more);
        setOldestMessageId(serverMessages.length > 0 ? serverMessages[0].id ?? null : null);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Sohbet geçmişi yüklenemedi");
      }
    };

    loadHistory();

    return () => {
      cancelled = true;
    };
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
    
    // Create user message using chat-history service
    const newUserMessage = createMessage("user", userMessage);
    
    const updatedMessages: Message[] = [...messages, { ...newUserMessage, id: undefined }];
    setMessages(updatedMessages);
    setIsLoading(true);

    const startTime = Date.now();

    try {
      const history = messages.slice(-10).map((m) => ({ role: m.role, content: m.content }));
      const response = await api.chat(courseId, userMessage, history);
      
      const responseTime = Date.now() - startTime;
      
      // Create assistant message using chat-history service
      const assistantMessage = createMessage(
        "assistant",
        response.message,
        response.sources,
        responseTime
      );
      
      const finalMessages: Message[] = [
        ...updatedMessages,
        { ...assistantMessage, id: undefined },
      ];
      setMessages(finalMessages);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Mesaj gönderilemedi. Lütfen tekrar deneyin.");
      // Restore previous state - remove the user message we just added
      setMessages(prevMessages => prevMessages.slice(0, -1));
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
        timestamp: m.created_at,
        sources: m.sources ?? undefined,
        responseTime: m.response_time_ms ?? undefined,
      }));

      setMessages((prev) => [...olderMessages, ...prev]);
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

  const handleBack = () => {
    router.push("/student");
  };

  const visibleMessages = messages;

  if (isCourseLoading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-120px)]">
        <Loader2 className="w-8 h-8 text-slate-400 animate-spin" />
      </div>
    );
  }

  if (!course) {
    return null;
  }

  return (
    <div className="h-[calc(100vh-120px)] flex flex-col">
      {/* Header with back button and course name */}
      <div className="flex items-center gap-4 mb-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleBack}
          className="text-slate-600 hover:text-slate-900"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          Geri
        </Button>
        <div className="flex-1">
          <h1 className="text-xl font-semibold text-slate-900">{course.name}</h1>
          {course.description && (
            <p className="text-sm text-slate-500">{course.description}</p>
          )}
        </div>
      </div>

      {/* Chat Container */}
      <div className="bg-white rounded-lg border border-slate-200 flex flex-col flex-1 min-h-0">
        {/* Chat Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 bg-gradient-to-r from-indigo-50 to-purple-50 shrink-0">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center">
              <MessageSquare className="w-4 h-4 text-indigo-600" />
            </div>
            <div>
              <h3 className="font-medium text-slate-900">Ders Asistanı</h3>
              <p className="text-xs text-slate-500">{conversationCount} sohbet</p>
            </div>
          </div>
          {messages.length > 0 && (
            <Button
              size="sm"
              variant="ghost"
              onClick={handleClear}
              className="text-slate-400 hover:text-red-600 hover:bg-red-50"
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Temizle
            </Button>
          )}
        </div>

        {/* Messages */}
        <div 
          ref={messagesContainerRef}
          className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50"
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
                  key={message.id ?? globalIdx}
                  className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {message.role === "assistant" && (
                    <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center shrink-0">
                      <Bot className="w-4 h-4 text-indigo-600" />
                    </div>
                  )}
                  
                  <div
                    className={`max-w-[75%] rounded-2xl px-4 py-3 ${
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
                      <span>{formatTimestamp(message.timestamp)}</span>
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
                                key={`source-${globalIdx}-${sIdx}`}
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
                    <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center shrink-0">
                      <User className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
              );
            })
          )}
          
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center shrink-0">
                <Bot className="w-4 h-4 text-indigo-600" />
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
        <div className="p-4 border-t border-slate-200 bg-white shrink-0">
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Mesajınızı yazın..."
              disabled={isLoading}
              className="flex-1 rounded-full px-4"
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
      </div>

      {/* Source Modal */}
      {selectedSource && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-2xl w-full max-h-[85vh] overflow-hidden shadow-xl flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-indigo-50 to-purple-50 shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                  <FileText className="w-5 h-5 text-indigo-600" />
                </div>
                <div>
                  <h3 className="font-medium text-slate-900">{selectedSource.document_name}</h3>
                  <p className="text-xs text-slate-500">
                    Chunk #{selectedSource.chunk_index + 1} • Benzerlik: {(selectedSource.score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <button
                onClick={() => setSelectedSource(null)}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-500" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6">
              <div className="prose prose-sm max-w-none">
                <p className="text-slate-700 whitespace-pre-wrap leading-relaxed">
                  {selectedSource.full_content || selectedSource.content_preview}
                </p>
              </div>
            </div>
            <div className="px-6 py-4 border-t border-slate-200 bg-slate-50 flex justify-end shrink-0">
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
