"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import {
  Play,
  BarChart3,
  Clock,
  CheckCircle,
  AlertCircle,
  Download,
  RefreshCw,
  Settings,
  Zap,
  Target,
  Trophy,
  Activity,
  ExternalLink,
  TrendingUp,
  Sparkles
} from "lucide-react";

interface BenchmarkTask {
  name: string;
  description?: string;
  category: string;
}

interface BenchmarkResult {
  model_name: string;
  summary: {
    overall_average: number;
    total_tasks: number;
    task_scores: Record<string, any>;
  };
  total_time_seconds: number;
  wb_url?: string;
}

interface ComparisonResult {
  model_results: Record<string, any>;
  comparison_table: Array<Record<string, any>>;
  best_model: string;
  wb_url?: string;
}

interface RerankerInfo {
  provider: string;
  model: string;
  name: string;
  languages: string[];
  description: string;
}

interface RerankerTestResult {
  id: string;
  content: string;
  relevance_score: number;
  rerank_index: number;
}

export default function MTEBBenchmarkPage() {
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedTasks, setSelectedTasks] = useState<string[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [availableTasks, setAvailableTasks] = useState<BenchmarkTask[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult | null>(null);
  const [comparisonResults, setComparisonResults] = useState<ComparisonResult | null>(null);
  const [history, setHistory] = useState<BenchmarkResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("single");

  // Reranker test states
  const [availableRerankers, setAvailableRerankers] = useState<RerankerInfo[]>([]);
  const [selectedRerankerProvider, setSelectedRerankerProvider] = useState("");
  const [selectedRerankerModel, setSelectedRerankerModel] = useState("");
  const [rerankerQuery, setRerankerQuery] = useState("");
  const [rerankerDocuments, setRerankerDocuments] = useState<string[]>([]);
  const [rerankerTestResults, setRerankerTestResults] = useState<RerankerTestResult[] | null>(null);
  const [rerankerLatency, setRerankerLatency] = useState<number | null>(null);
  const [isRerankerLoading, setIsRerankerLoading] = useState(false);

  // Available models
  const models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "ollama/bge-m3",
    "ollama/nomic-embed-text",
    "alibaba/text-embedding-v4",
    "cohere/embed-multilingual-v3.0",
    "jina/jina-embeddings-v3",
    "voyage/voyage-4-large",
    "voyage/voyage-3-large",
    "voyage/voyage-3-lite",
    "voyage/voyage-2"
  ];

  // Available categories
  const categories = [
    { value: "english", label: "English Tasks" },
    { value: "multilingual", label: "Multilingual Tasks" },
    { value: "retrieval", label: "Retrieval Tasks" },
    { value: "clustering", label: "Clustering Tasks" },
    { value: "reranking", label: "Reranking Tasks" }
  ];

  const fetchAvailableTasks = async () => {
    try {
      const response = await fetch("/api/benchmark/tasks");
      const data = await response.json();
      setAvailableTasks(data.tasks || []);
    } catch (error) {
      console.error("Failed to fetch tasks:", error);
      // Fallback to default MTEB tasks if API fails
      setAvailableTasks([
        // English Retrieval Tasks
        { name: "ArguAna", category: "english", description: "Argument retrieval" },
        { name: "ClimateFEVER", category: "english", description: "Climate fact verification" },
        { name: "CQADupstackRetrieval", category: "english", description: "Community QA retrieval" },
        { name: "DBPedia", category: "english", description: "DBPedia entity retrieval" },
        { name: "FEVER", category: "english", description: "Fact verification" },
        { name: "FiQA2018", category: "english", description: "Financial QA" },
        { name: "HotpotQA", category: "english", description: "Multi-hop QA" },
        { name: "MSMARCO", category: "english", description: "MS MARCO passage retrieval" },
        { name: "NFCorpus", category: "english", description: "Medical information retrieval" },
        { name: "NQ", category: "english", description: "Natural Questions" },
        { name: "QuoraRetrieval", category: "english", description: "Quora duplicate questions" },
        { name: "SCIDOCS", category: "english", description: "Scientific document retrieval" },
        { name: "SciFact", category: "english", description: "Scientific fact verification" },
        { name: "Touche2020", category: "english", description: "Argument retrieval" },
        { name: "TRECCOVID", category: "english", description: "COVID-19 retrieval" },
        
        // Multilingual Tasks
        { name: "MultilingualSentiment", category: "multilingual", description: "Multilingual sentiment" },
        { name: "XPQARetrieval", category: "multilingual", description: "Cross-lingual QA" },
        
        // Clustering Tasks
        { name: "ArxivClusteringP2P", category: "clustering", description: "ArXiv paper clustering" },
        { name: "ArxivClusteringS2S", category: "clustering", description: "ArXiv clustering S2S" },
        { name: "BiorxivClusteringP2P", category: "clustering", description: "BioRxiv clustering" },
        { name: "BiorxivClusteringS2S", category: "clustering", description: "BioRxiv clustering S2S" },
        { name: "MedrxivClusteringP2P", category: "clustering", description: "MedRxiv clustering" },
        { name: "MedrxivClusteringS2S", category: "clustering", description: "MedRxiv clustering S2S" },
        { name: "RedditClustering", category: "clustering", description: "Reddit post clustering" },
        { name: "StackExchangeClustering", category: "clustering", description: "StackExchange clustering" },
        { name: "TwentyNewsgroupsClustering", category: "clustering", description: "20 Newsgroups clustering" },
        
        // Reranking Tasks
        { name: "AskUbuntuDupQuestions", category: "reranking", description: "Ubuntu duplicate questions" },
        { name: "MindSmallReranking", category: "reranking", description: "News reranking" },
        { name: "SciDocsRR", category: "reranking", description: "Scientific doc reranking" },
        { name: "StackOverflowDupQuestions", category: "reranking", description: "StackOverflow duplicates" },
      ]);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch("/api/benchmark/models");
      const data = await response.json();
      setAvailableModels(data.models || []);
    } catch (error) {
      console.error("Failed to fetch models:", error);
      setAvailableModels([]);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await fetch("/api/benchmark/history");
      const data = await response.json();
      setHistory(data.history || []);
    } catch (error) {
      console.error("Failed to fetch history:", error);
      setHistory([]);
    }
  };

  const fetchAvailableRerankers = async () => {
    try {
      console.log("Fetching available rerankers...");
      const response = await fetch("/api/benchmark/rerankers");
      console.log("Response status:", response.status);
      console.log("Response ok:", response.ok);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Rerankers response:", data);
      console.log("Setting available rerankers:", data.rerankers || []);
      setAvailableRerankers(data.rerankers || []);
      console.log("Available rerankers set successfully");
    } catch (error) {
      console.error("Failed to fetch rerankers:", error);
      if (error instanceof Error) {
        console.error("Error details:", error.message);
      }
      setAvailableRerankers([]);
    }
  };

  const runMTEBRerankerTest = async (taskName: string) => {
    if (!selectedRerankerProvider) {
      alert("Please select a reranker provider first");
      return;
    }

    try {
      console.log(`Running MTEB reranker test: ${taskName}`);
      
      const response = await fetch("/api/benchmark/mteb-reranker-test", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          task_name: taskName,
          provider: selectedRerankerProvider,
          model: selectedRerankerModel || undefined,
        }),
      });

      const result = await response.json();
      console.log("MTEB reranker test result:", result);

      if (result.success) {
        alert(`MTEB reranker test completed successfully!\n\nTask: ${result.task_name}\nProvider: ${result.provider}\nModel: ${result.model}\nMain Score: ${result.results?.main_score}\n\nResults logged to W&B.`);
      } else {
        alert(`MTEB reranker test failed: ${result.message}`);
      }
    } catch (error) {
      console.error("MTEB reranker test failed:", error);
      alert("MTEB reranker test failed. Please check console for details.");
    }
  };

  const testReranker = async () => {
    if (!selectedRerankerProvider || !rerankerQuery || rerankerDocuments.length === 0) {
      return;
    }

    setIsRerankerLoading(true);
    try {
      const response = await fetch("/api/benchmark/reranker-test", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: rerankerQuery,
          documents: rerankerDocuments,
          provider: selectedRerankerProvider,
          model: selectedRerankerModel || undefined,
          top_k: 5,
        }),
      });

      const result = await response.json();
      setRerankerTestResults(result.results || []);
      setRerankerLatency(result.latency_seconds || 0);
    } catch (error) {
      console.error("Reranker test failed:", error);
    } finally {
      setIsRerankerLoading(false);
    }
  };

  useEffect(() => {
    fetchAvailableTasks();
    fetchAvailableModels();
    fetchHistory();
    fetchAvailableRerankers();
  }, []);

  const runBenchmark = async () => {
    if (!selectedModel || selectedTasks.length === 0) {
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch("/api/benchmark/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_name: selectedModel,
          tasks: selectedTasks,
        }),
      });

      const result = await response.json();
      // Extract data from BenchmarkResponse wrapper
      setBenchmarkResults(result.data || result);
      await fetchHistory(); // Refresh history
    } catch (error) {
      console.error("Benchmark failed:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const runComparison = async () => {
    if (selectedModels.length < 2 || selectedTasks.length === 0) {
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch("/api/benchmark/compare", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_names: selectedModels,
          tasks: selectedTasks,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("Comparison API response:", result);
      
      // Extract data from BenchmarkResponse wrapper
      const comparisonData = result.data || result;
      console.log("Comparison data:", comparisonData);
      
      // Check if we got valid data
      if (!comparisonData.comparison_table || comparisonData.comparison_table.length === 0) {
        alert("Benchmark comparison returned no data. This could be because:\n\n" +
              "1. MTEB library is not installed on the backend\n" +
              "2. Models require API keys that aren't configured\n" +
              "3. Benchmarks take hours to run and may have timed out\n\n" +
              "Check the browser console and backend logs for details.");
      }
      
      setComparisonResults(comparisonData);
      await fetchHistory(); // Refresh history
    } catch (error) {
      console.error("Comparison failed:", error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      alert(`Comparison failed: ${errorMessage}\n\nCheck the browser console for details.`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCategoryChange = (category: string, checked: boolean) => {
    console.log("handleCategoryChange called:", { category, checked, availableTasksLength: availableTasks.length });
    
    if (checked) {
      setSelectedCategories([...selectedCategories, category]);
      // Add all tasks from this category
      const categoryTasks = availableTasks
        .filter(task => task.category === category)
        .map(task => task.name);
      console.log("Adding category tasks:", { category, categoryTasks });
      setSelectedTasks([...new Set([...selectedTasks, ...categoryTasks])]);
    } else {
      setSelectedCategories(selectedCategories.filter(c => c !== category));
      // Remove all tasks from this category
      const categoryTasks = availableTasks
        .filter(task => task.category === category)
        .map(task => task.name);
      console.log("Removing category tasks:", { category, categoryTasks });
      setSelectedTasks(selectedTasks.filter(task => !categoryTasks.includes(task)));
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">MTEB Benchmark</h1>
          <p className="text-muted-foreground">
            Massive Text Embedding Benchmark - Test embedding models on 100+ tasks
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            <Activity className="w-3 h-3 mr-1" />
            MTEB Integration
          </Badge>
          {benchmarkResults?.wb_url && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.open(benchmarkResults.wb_url, '_blank')}
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              View in W&B
            </Button>
          )}
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="single">Single Benchmark</TabsTrigger>
          <TabsTrigger value="comparison">Model Comparison</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="tasks">Available Tasks</TabsTrigger>
          <TabsTrigger value="reranker">Reranker Test</TabsTrigger>
        </TabsList>

        <TabsContent value="single" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="w-5 h-5" />
                  Configuration
                </CardTitle>
                <CardDescription>
                  Select model and tasks for benchmarking
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Model</label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select embedding model" />
                    </SelectTrigger>
                    <SelectContent>
                      {models.map((model) => (
                        <SelectItem key={model} value={model}>
                          {model}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Categories</label>
                  <div className="space-y-2">
                    {categories.map((category) => (
                      <div key={category.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={category.value}
                          checked={selectedCategories.includes(category.value)}
                          onCheckedChange={(checked) => 
                            handleCategoryChange(category.value, checked as boolean)
                          }
                        />
                        <label
                          htmlFor={category.value}
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                        >
                          {category.label}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Selected Tasks ({selectedTasks.length})
                  </label>
                  <ScrollArea className="h-32 w-full border rounded-md p-2">
                    <div className="space-y-1">
                      {selectedTasks.map((task) => (
                        <div key={task} className="text-xs bg-muted px-2 py-1 rounded">
                          {task}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>

                <Button 
                  onClick={runBenchmark} 
                  disabled={!selectedModel || selectedTasks.length === 0 || isLoading}
                  className="w-full"
                >
                  {isLoading ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Running Benchmark...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Run Benchmark
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {benchmarkResults && (
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Trophy className="w-5 h-5" />
                    Benchmark Results
                  </CardTitle>
                  <CardDescription>
                    Results for {benchmarkResults.model_name}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {benchmarkResults.summary.overall_average.toFixed(3)}
                      </div>
                      <div className="text-sm text-muted-foreground">Overall Average</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {benchmarkResults.summary.total_tasks}
                      </div>
                      <div className="text-sm text-muted-foreground">Tasks Completed</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {formatTime(benchmarkResults.total_time_seconds)}
                      </div>
                      <div className="text-sm text-muted-foreground">Total Time</div>
                    </div>
                  </div>

                  {benchmarkResults.wb_url && (
                    <Alert className="mb-6">
                      <TrendingUp className="h-4 w-4" />
                      <AlertDescription>
                        Results logged to Weights & Biases. 
                        <Button 
                          variant="link" 
                          className="p-0 h-auto ml-2"
                          onClick={() => window.open(benchmarkResults.wb_url, '_blank')}
                        >
                          View detailed analysis →
                        </Button>
                      </AlertDescription>
                    </Alert>
                  )}

                  <div className="space-y-4">
                    <h4 className="font-semibold">Task Scores</h4>
                    <ScrollArea className="h-64">
                      <div className="space-y-2">
                        {Object.entries(benchmarkResults.summary.task_scores).map(([task, score]: any) => (
                          <div key={task} className="flex items-center justify-between p-3 border rounded-lg">
                            <div>
                              <div className="font-medium">{task}</div>
                              <div className="text-sm text-muted-foreground">
                                {score.main_score_name}
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="font-bold text-lg">
                                {score.main_score.toFixed(3)}
                              </div>
                              <div className="text-xs text-muted-foreground">
                                {score.dataset_revision}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="comparison" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Model Comparison
              </CardTitle>
              <CardDescription>
                Compare multiple embedding models on the same tasks
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Select Models (min 2)</label>
                  <ScrollArea className="h-64">
                    <div className="space-y-2 pr-4">
                      {models.map((model) => (
                        <div key={model} className="flex items-center space-x-2">
                          <Checkbox
                            id={`compare-${model}`}
                            checked={selectedModels.includes(model)}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setSelectedModels([...selectedModels, model]);
                              } else {
                                setSelectedModels(selectedModels.filter(m => m !== model));
                              }
                            }}
                          />
                          <label
                            htmlFor={`compare-${model}`}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                          >
                            {model}
                          </label>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Categories</label>
                  <div className="space-y-2">
                    {categories.map((category) => (
                      <div key={category.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`compare-cat-${category.value}`}
                          checked={selectedCategories.includes(category.value)}
                          onCheckedChange={(checked) => 
                            handleCategoryChange(category.value, checked as boolean)
                          }
                        />
                        <label
                          htmlFor={`compare-cat-${category.value}`}
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                        >
                          {category.label}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Selected Tasks ({selectedTasks.length})
                  </label>
                  <ScrollArea className="h-64 w-full border rounded-md p-2">
                    <div className="space-y-1">
                      {selectedTasks.map((task) => (
                        <div key={task} className="text-xs bg-muted px-2 py-1 rounded">
                          {task}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              </div>

              <Button 
                onClick={runComparison} 
                disabled={selectedModels.length < 2 || selectedTasks.length === 0 || isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Running Comparison...
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Compare Models
                  </>
                )}
              </Button>

              <Button 
                onClick={() => {
                  // Load demo data to show what the UI looks like
                  const demoData: any = {
                    models: selectedModels.length >= 2 ? selectedModels : ["openai/text-embedding-3-small", "openai/text-embedding-3-large"],
                    tasks: selectedTasks.length > 0 ? selectedTasks.slice(0, 5) : ["Banking77Classification.v2", "ImdbClassification.v2", "MSMARCO", "ArxivClusteringP2P.v2", "AskUbuntuDupQuestions"],
                    best_model: "openai/text-embedding-3-large",
                    model_results: {},
                    comparison_table: [],
                    wb_url: "https://wandb.ai/demo/embedding-benchmarks"
                  };
                  
                  // Generate demo comparison table
                  const models = demoData.models;
                  const tasks = demoData.tasks;
                  
                  for (const task of tasks) {
                    const row: any = { task };
                    for (const model of models) {
                      // Generate realistic-looking scores between 0.65 and 0.95
                      const baseScore = 0.65 + Math.random() * 0.30;
                      const modelBoost = model.includes("large") ? 0.05 : 0;
                      row[model] = (baseScore + modelBoost).toFixed(3);
                    }
                    demoData.comparison_table.push(row);
                  }
                  
                  setComparisonResults(demoData);
                }}
                variant="outline"
                className="w-full mt-2"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Load Demo Data
              </Button>
            </CardContent>
          </Card>

          {comparisonResults && (
            <Card>
              <CardHeader>
                <CardTitle>Comparison Results</CardTitle>
                <CardDescription>
                  Best performing model: <span className="font-bold text-green-600">{comparisonResults.best_model}</span>
                </CardDescription>
              </CardHeader>
              <CardContent>
                {comparisonResults.wb_url && (
                  <Alert className="mb-6">
                    <TrendingUp className="h-4 w-4" />
                    <AlertDescription>
                      Comparison logged to Weights & Biases. 
                      <Button 
                        variant="link" 
                        className="p-0 h-auto ml-2"
                        onClick={() => window.open(comparisonResults.wb_url, '_blank')}
                      >
                        View detailed analysis →
                      </Button>
                    </AlertDescription>
                  </Alert>
                )}

                <div className="overflow-x-auto">
                  <table className="w-full border-collapse border">
                    <thead>
                      <tr className="bg-muted">
                        <th className="border p-2 text-left">Task</th>
                        {selectedModels.map((model) => (
                          <th key={model} className="border p-2 text-left">{model}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonResults.comparison_table && comparisonResults.comparison_table.length > 0 ? (
                        comparisonResults.comparison_table.map((row, index) => (
                          <tr key={index}>
                            <td className="border p-2 font-medium">{row.task}</td>
                            {selectedModels.map((model) => (
                              <td key={model} className="border p-2 text-right">
                                {row[model] !== 'N/A' ? parseFloat(row[model]).toFixed(3) : 'N/A'}
                              </td>
                            ))}
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={selectedModels.length + 1} className="border p-4 text-center text-muted-foreground">
                            No comparison data available
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Benchmark History
              </CardTitle>
              <CardDescription>
                Previous benchmark runs and their results
              </CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No benchmark history available. Run your first benchmark to see results here.
                </div>
              ) : (
                <div className="space-y-4">
                  {history.map((result, index) => (
                    <Card key={index}>
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium">{result.model_name}</div>
                            <div className="text-sm text-muted-foreground">
                              {result.summary.total_tasks} tasks • {formatTime(result.total_time_seconds)}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-bold text-lg">
                              {result.summary.overall_average.toFixed(3)}
                            </div>
                            <div className="text-sm text-muted-foreground">Average Score</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tasks" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5" />
                Available MTEB Tasks
              </CardTitle>
              <CardDescription>
                Browse all available MTEB benchmark tasks by category
              </CardDescription>
            </CardHeader>
            <CardContent>
              {availableTasks.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  Loading available tasks...
                </div>
              ) : (
                <div className="space-y-6">
                  {categories.map((category) => (
                    <div key={category.value}>
                      <h3 className="font-semibold mb-3">{category.label}</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {availableTasks
                          .filter(task => task.category === category.value)
                          .map((task) => (
                            <Card key={task.name} className="p-3">
                              <div className="font-medium text-sm">{task.name}</div>
                              {task.description && (
                                <div className="text-xs text-muted-foreground mt-1">
                                  {task.description}
                                </div>
                              )}
                            </Card>
                          ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reranker" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5" />
                  MTEB Reranker Tests
                </CardTitle>
                <CardDescription>
                  Test rerankers on MTEB reranking benchmarks
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Debug Info */}
                <div className="p-3 bg-gray-100 rounded text-xs">
                  <div>Available Rerankers: {availableRerankers.length}</div>
                  <div>Selected Provider: {selectedRerankerProvider}</div>
                  <div>Selected Model: {selectedRerankerModel}</div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Reranker Provider</label>
                  <Select
                    value={selectedRerankerProvider}
                    onValueChange={(v) => {
                      setSelectedRerankerProvider(v);
                      setSelectedRerankerModel("");
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select reranker provider" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="cohere">Cohere</SelectItem>
                      <SelectItem value="alibaba">Alibaba</SelectItem>
                      <SelectItem value="jina">Jina</SelectItem>
                      <SelectItem value="bge">BGE</SelectItem>
                      <SelectItem value="zeroentropy">ZeroEntropy</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {selectedRerankerProvider && (
                  <div>
                    <label className="text-sm font-medium mb-2 block">Reranker Model</label>
                    <Select
                      value={selectedRerankerModel}
                      onValueChange={setSelectedRerankerModel}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select reranker model" />
                      </SelectTrigger>
                      <SelectContent>
                        {(() => {
                          const filteredRerankers = availableRerankers.filter(r => r.provider === selectedRerankerProvider);
                          console.log("Filtered rerankers for provider", selectedRerankerProvider, ":", filteredRerankers);
                          return filteredRerankers.map((reranker) => (
                            <SelectItem key={reranker.model} value={reranker.model}>
                              {reranker.name}
                            </SelectItem>
                          ));
                        })()}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div>
                  <label className="text-sm font-medium mb-2 block">MTEB Reranking Tasks</label>
                  <div className="space-y-2">
                    {[
                      { name: "AskUbuntuDupQuestions", desc: "Duplicate question detection on AskUbuntu" },
                      { name: "MindSmallReranking", desc: "News article reranking" },
                      { name: "SciDocsRR", desc: "Scientific document reranking" },
                      { name: "StackOverflowDupQuestions", desc: "Duplicate question detection on StackOverflow" }
                    ].map((task) => (
                      <Card key={task.name} className="p-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium text-sm">{task.name}</div>
                            <div className="text-xs text-muted-foreground">{task.desc}</div>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => runMTEBRerankerTest(task.name)}
                          >
                            Run Test
                          </Button>
                        </div>
                      </Card>
                    ))}
                  </div>
                </div>

                <div className="mt-6 p-4 bg-amber-50 rounded-lg border border-amber-200">
                  <h4 className="font-semibold text-amber-900 mb-2">Custom Reranker Test</h4>
                  <p className="text-sm text-amber-800 mb-4">
                    Test your reranker with custom query and documents
                  </p>
                  
                  <div>
                    <label className="text-sm font-medium mb-2 block">Query</label>
                    <input
                      type="text"
                      value={rerankerQuery}
                      onChange={(e) => setRerankerQuery(e.target.value)}
                      placeholder="Enter your search query..."
                      className="w-full px-3 py-2 border rounded-md"
                    />
                  </div>

                  <div className="mt-3">
                    <label className="text-sm font-medium mb-2 block">
                      Documents ({rerankerDocuments.length})
                    </label>
                    <Textarea
                      value={rerankerDocuments.join('\n\n')}
                      onChange={(e) => setRerankerDocuments(e.target.value.split('\n\n').filter(d => d.trim()))}
                      placeholder="Enter documents (one per line)..."
                      className="w-full min-h-[150px] mb-2"
                    />
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1"
                        onClick={() => {
                          const sampleDocs = [
                            "Machine learning is a subset of artificial intelligence.",
                            "Deep learning uses neural networks with multiple layers.",
                            "Natural language processing helps computers understand human language.",
                            "Computer vision enables machines to interpret visual information.",
                            "Reinforcement learning learns through trial and error."
                          ];
                          setRerankerDocuments([...rerankerDocuments, ...sampleDocs]);
                        }}
                      >
                        Add Sample Documents
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1"
                        onClick={() => setRerankerDocuments([])}
                      >
                        Clear Documents
                      </Button>
                    </div>
                  </div>

                  <Button
                    onClick={testReranker}
                    disabled={!selectedRerankerProvider || !rerankerQuery || rerankerDocuments.length === 0 || isRerankerLoading}
                    className="w-full mt-4"
                  >
                    {isRerankerLoading ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Testing Reranker...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Test Custom Reranker
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {rerankerTestResults && (
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Trophy className="w-5 h-5" />
                    Reranker Test Results
                  </CardTitle>
                  <CardDescription>
                    Results for {selectedRerankerProvider}/{selectedRerankerModel || 'default'}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {rerankerLatency !== null && (
                    <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <div className="text-sm text-blue-900">
                        <span className="font-medium">Latency:</span> {rerankerLatency.toFixed(3)}s
                      </div>
                    </div>
                  )}

                  <div className="space-y-3">
                    <h4 className="font-semibold">Reranked Documents</h4>
                    <ScrollArea className="h-96">
                      <div className="space-y-2">
                        {rerankerTestResults.map((result, index) => (
                          <div key={index} className="p-4 border rounded-lg bg-gradient-to-r from-amber-50 to-orange-50">
                            <div className="flex items-start justify-between mb-2">
                              <div className="flex-1">
                                <div className="text-xs text-muted-foreground mb-1">
                                  Rank #{index + 1} • ID: {result.id}
                                </div>
                                <div className="text-sm font-medium text-slate-900">
                                  {result.content}
                                </div>
                              </div>
                              <div className="text-right ml-4">
                                <div className="text-2xl font-bold text-amber-600">
                                  {result.relevance_score.toFixed(3)}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  Relevance Score
                                </div>
                              </div>
                            </div>
                            <div className="text-xs text-muted-foreground mt-2">
                              Original Index: {result.rerank_index}
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
