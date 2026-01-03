/**
 * Type definitions for Execution Governor
 * 
 * Mirrors mandate types from bolt.diy for type safety.
 */

export interface Mandate {
  mandate_id: string;
  objectives: string[];
  constraints: MandateConstraints;
  budget: MandateBudget;
  deliverables: string[];
  governance?: MandateGovernance;
  iteration_config: IterationConfig;
  deployment?: DeploymentConfig;
  metadata?: {
    created_at?: number;
    created_by?: string;
    priority?: "low" | "medium" | "high" | "critical";
    tags?: string[];
  };
}

export interface MandateConstraints {
  language: "js" | "ts" | "python" | "html" | "css" | "json" | "yaml" | "markdown";
  maxDependencies: number;
  noNetwork: boolean;
  allowedPackages?: string[];
  maxFileSize: number;
  maxFiles: number;
  allowedFileExtensions?: string[];
  forbiddenPatterns?: string[];
}

export interface MandateBudget {
  token: number;
  time: number;
  cost: number;
}

export interface MandateGovernance {
  proposal_id?: string;
  esg_requirements?: ESGScore;
  risk_threshold?: number;
  approval_chain?: string[];
  causal_analysis_required?: boolean;
  compliance_frameworks?: string[];
  audit_required?: boolean;
}

export interface ESGScore {
  environmental_score: number;
  social_score: number;
  governance_score: number;
  overall_score: number;
  carbon_footprint?: number;
  diversity_index?: number;
  stakeholder_satisfaction?: number;
  sustainability_goals?: string[];
  last_updated?: number;
}

export interface IterationConfig {
  max_iterations: number;
  test_required: boolean;
  quality_threshold: number;
  stop_on_error?: boolean;
  retry_on_failure?: boolean;
  max_retries?: number;
}

export interface DeploymentConfig {
  enabled: boolean;
  provider: "netlify" | "vercel" | "github" | "gitlab";
  target?: string;
  auto_deploy: boolean;
  build_command?: string;
  environment_variables?: Record<string, string>;
}

export interface QueuedMandate {
  mandate: Mandate;
  priority: number;
  queuedAt: number;
  retryCount: number;
  lastError?: string;
}

export interface ActiveExecution {
  mandateId: string;
  workerId: string;
  startedAt: number;
  estimatedCompletion: number;
  resourceUsage: {
    cpu: number;
    memory: number;
  };
}

export interface Worker {
  workerId: string;
  registeredAt: number;
  lastHeartbeat: number;
  status: "idle" | "busy" | "unhealthy";
  currentMandateId?: string;
  capabilities: {
    maxConcurrent: number;
    currentExecutions: number;
  };
}

export interface ExecutionEvent {
  mandate_id: string;
  iteration: number;
  type: string;
  timestamp: number;
  data: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface ExecutionStatus {
  mandateId: string;
  status: "queued" | "executing" | "completed" | "failed" | "cancelled";
  queuePosition?: number;
  workerId?: string;
  progress?: number;
  startedAt?: number;
  completedAt?: number;
  error?: string;
  result?: any;
}

export interface GovernorState {
  queueDepth: number;
  activeExecutions: number;
  maxConcurrentExecutions: number;
  registeredWorkers: number;
  healthyWorkers: number;
  totalMandatesProcessed: number;
  totalMandatesFailed: number;
  averageExecutionTime: number;
  currentBudgetUsage: number;
  globalBudgetLimit: number;
}

