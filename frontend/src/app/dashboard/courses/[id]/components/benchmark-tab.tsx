"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
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
  TrendingUp
} from "lucide-react";

interface BenchmarkTabProps {
  readonly courseId: number;
}

interface BenchmarkTask {
  name: string;
  description?: string;
  category: string;
  languages: string[];
  main_score: string;
}

interface BenchmarkResult {
  model_name: string;
  tasks_run: string[];
  total_time_seconds: number;
  timestamp: string;
  wb_url?: string;
  summary: {
    overall_average: number;
    total_tasks: number;
    task_scores: Record<string, any>;
  };
}

interface AvailableModels {
  [category: string]: string[];
}

export function BenchmarkTab({ courseId }: BenchmarkTabProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [availableTasks, setAvailableTasks] = useState<Record<string, string[]>>({});
  const [availableModels, setAvailableModels] = useState<AvailableModels>({});
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedTasks, setSelectedTasks] = useState<string[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult | null>(null);
  const [benchmarkHistory, setBenchmarkHistory] = useState<any[]>([]);
  const [comparisonResults, setComparisonResults] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("single");

  useEffect(() => {
    loadAvailableTasks();
    loadAvailableModels();
    loadBenchmarkHistory();
  }, []);

  const loadAvailableTasks = async () => {
    try {
      const response = await fetch("/api/benchmark/tasks");
      const tasks = await response.json();
      setAvailableTasks(tasks);
    } catch (error) {
      toast.error("Failed to load available tasks");
    }
  };

  const loadAvailableModels = async () => {
    try {
      const response = await fetch("/api/benchmark/models");
      const models = await response.json();
      setAvailableModels(models);
    } catch (error) {
      toast.error("Failed to load available models");
    }
  };

  const loadBenchmarkHistory = async () => {
    try {
      const response = await fetch("/api/benchmark/history");
      const history = await response.json();
      setBenchmarkHistory(history);
    } catch (error) {
      console.error("Failed to load benchmark history:", error);
    }
  };

  const handleTaskToggle = (task: string) => {
    setSelectedTasks(prev => 
      prev.includes(task) 
        ? prev.filter(t => t !== task)
        : [...prev, task]
    );
  };

  const handleCategoryToggle = (category: string) => {
    setSelectedCategories(prev => 
      prev.includes(category) 
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const runBenchmark = async () => {
    if (!selectedModel) {
      toast.error("Please select a model");
      return;
    }

    if (selectedTasks.length === 0 && selectedCategories.length === 0) {
      toast.error("Please select tasks or categories");
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
          tasks: selectedTasks.length > 0 ? selectedTasks : undefined,
          categories: selectedCategories.length > 0 ? selectedCategories : undefined,
        }),
      });

      const result = await response.json();
      
      if (result.success) {
        setBenchmarkResults(result.data);
        toast.success("Benchmark completed successfully!");
        loadBenchmarkHistory();
      } else {
        toast.error(result.message || "Benchmark failed");
      }
    } catch (error) {
      toast.error("Failed to run benchmark");
    } finally {
      setIsLoading(false);
    }
  };

  const runComparison = async () => {
    if (selectedTasks.length === 0) {
      toast.error("Please select tasks for comparison");
      return;
    }

    setIsLoading(true);
    try {
      const modelsToCompare = Object.values(availableModels)
        .flat()
        .slice(0, 3); // Compare top 3 models for demo

      const response = await fetch("/api/benchmark/compare", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_names: modelsToCompare,
          tasks: selectedTasks,
        }),
      });

      const result = await response.json();
      
      if (result.success) {
        setComparisonResults(result.data);
        toast.success("Model comparison completed!");
      } else {
        toast.error(result.message || "Comparison failed");
      }
    } catch (error) {
      toast.error("Failed to run comparison");
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Embedding Benchmark</h2>
          <p className="text-muted-foreground">
            Test and compare embedding models using MTEB benchmarks
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={loadBenchmarkHistory}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="single">Single Model</TabsTrigger>
          <TabsTrigger value="compare">Compare Models</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        <TabsContent value="single" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Model Selection */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Model Selection
                </CardTitle>
                <CardDescription>
                  Choose the embedding model to benchmark
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(availableModels).map(([category, models]) => (
                      <div key={category}>
                        <div className="px-2 py-1.5 text-sm font-semibold text-muted-foreground">
                          {category.charAt(0).toUpperCase() + category.slice(1)}
                        </div>
                        {models.map(model => (
                          <SelectItem key={model} value={model}>
                            {model}
                          </SelectItem>
                        ))}
                      </div>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            {/* Task Categories */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Task Categories
                </CardTitle>
                <CardDescription>
                  Select categories of tasks to run
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {Object.keys(availableTasks).map(category => (
                  <div key={category} className="flex items-center space-x-2">
                    <Checkbox
                      id={category}
                      checked={selectedCategories.includes(category)}
                      onCheckedChange={() => handleCategoryToggle(category)}
                    />
                    <label
                      htmlFor={category}
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      {category.charAt(0).toUpperCase() + category.slice(1)}
                      <Badge variant="secondary" className="ml-2">
                        {availableTasks[category].length}
                      </Badge>
                    </label>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Individual Tasks */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Individual Tasks
                </CardTitle>
                <CardDescription>
                  Select specific tasks (overrides categories)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-2">
                    {Object.entries(availableTasks).map(([category, tasks]) => (
                      <div key={category}>
                        <div className="text-sm font-semibold text-muted-foreground mb-2">
                          {category.charAt(0).toUpperCase() + category.slice(1)}
                        </div>
                        {tasks.map(task => (
                          <div key={task} className="flex items-center space-x-2">
                            <Checkbox
                              id={task}
                              checked={selectedTasks.includes(task)}
                              onCheckedChange={() => handleTaskToggle(task)}
                            />
                            <label
                              htmlFor={task}
                              className="text-sm leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                            >
                              {task}
                            </label>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>

          {/* Run Benchmark */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="h-5 w-5" />
                Run Benchmark
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                <Button 
                  onClick={runBenchmark} 
                  disabled={isLoading || !selectedModel}
                  className="flex-1"
                >
                  {isLoading ? (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 mr-2" />
                      Start Benchmark
                    </>
                  )}
                </Button>
                
                {selectedModel && (
                  <div className="text-sm text-muted-foreground">
                    Model: <Badge>{selectedModel}</Badge>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Results */}
          {benchmarkResults && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="h-5 w-5" />
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

                {/* W&B Integration */}
                {benchmarkResults.wb_url && (
                  <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="h-5 w-5 text-blue-600" />
                        <div>
                          <div className="font-medium text-blue-900">
                            Results logged to Weights & Biases
                          </div>
                          <div className="text-sm text-blue-700">
                            View detailed analysis and charts
                          </div>
                        </div>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => window.open(benchmarkResults.wb_url, '_blank')}
                      >
                        <ExternalLink className="h-4 w-4 mr-2" />
                        View in W&B
                      </Button>
                    </div>
                  </div>
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
                            <div className="font-bold">{score.main_score.toFixed(3)}</div>
                            <Progress value={score.main_score * 100} className="w-20 mt-1" />
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="compare" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Model Comparison
              </CardTitle>
              <CardDescription>
                Compare multiple models on the same tasks
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={runComparison} disabled={isLoading || selectedTasks.length === 0}>
                {isLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Comparing...
                  </>
                ) : (
                  <>
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Compare Top Models
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {comparisonResults && (
            <Card>
              <CardHeader>
                <CardTitle>Comparison Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-2">Task</th>
                        {comparisonResults.models.map((model: string) => (
                          <th key={model} className="text-left p-2">{model}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonResults.comparison_table.map((row: any, index: number) => (
                        <tr key={index} className="border-b">
                          <td className="p-2">{row.task}</td>
                          {comparisonResults.models.map((model: string) => (
                            <td key={model} className="p-2">
                              {row[model] !== "N/A" ? parseFloat(row[model]).toFixed(3) : "N/A"}
                            </td>
                          ))}
                        </tr>
                      ))}
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
                <Clock className="h-5 w-5" />
                Benchmark History
              </CardTitle>
              <CardDescription>
                Previous benchmark results
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {benchmarkHistory.map((result, index) => (
                  <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                    <div>
                      <div className="font-medium">{result.model_name}</div>
                      <div className="text-sm text-muted-foreground">{result.timestamp}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold">{result.overall_average.toFixed(3)}</div>
                      <div className="text-sm text-muted-foreground">
                        {result.total_tasks} tasks • {formatTime(result.total_time_seconds)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
