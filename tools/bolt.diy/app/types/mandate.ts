/**
 * Mandate types for LLM-native governance execution.
 * 
 * Defines structured contracts for autonomous code generation and deployment
 * without human intervention.
 */

/**
 * ESG Score structure matching CorporateSwarm format.
 */
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

/**
 * Risk Assessment structure matching CorporateSwarm format.
 */
export interface RiskAssessment {
  risk_id?: string;
  risk_category: string;
  risk_level: "low" | "medium" | "high" | "critical";
  probability: number;
  impact: number;
  risk_score: number;
  mitigation_strategies?: string[];
  owner?: string;
  status?: string;
  last_reviewed?: number;
}

/**
 * Mandate constraints for execution.
 */
export interface MandateConstraints {
  language: "js" | "ts" | "python" | "html" | "css" | "json" | "yaml" | "markdown";
  maxDependencies: number;
  noNetwork: boolean;
  allowedPackages?: string[];
  maxFileSize: number; // bytes
  maxFiles: number;
  allowedFileExtensions?: string[];
  forbiddenPatterns?: string[]; // regex patterns to disallow in code
}

/**
 * Budget limits for mandate execution.
 */
export interface MandateBudget {
  token: number; // max tokens
  time: number; // max execution time in seconds
  cost: number; // max cost in dollars
}

/**
 * Governance metadata for mandate.
 */
export interface MandateGovernance {
  proposal_id?: string;
  esg_requirements?: ESGScore;
  risk_threshold?: number;
  approval_chain?: string[];
  causal_analysis_required?: boolean;
  compliance_frameworks?: string[]; // e.g., ["SOX", "GDPR", "ISO 27001"]
  audit_required?: boolean;
}

/**
 * Iteration configuration for autonomous execution.
 */
export interface IterationConfig {
  max_iterations: number;
  test_required: boolean;
  quality_threshold: number; // 0-1 score
  stop_on_error?: boolean;
  retry_on_failure?: boolean;
  max_retries?: number;
}

/**
 * Deployment configuration for mandate.
 */
export interface DeploymentConfig {
  enabled: boolean;
  provider: "netlify" | "vercel" | "github" | "gitlab";
  target?: string; // deployment target URL or branch
  auto_deploy: boolean; // deploy automatically after successful build
  build_command?: string;
  environment_variables?: Record<string, string>;
}

/**
 * Testing configuration for mandate.
 */
export interface TestingConfig {
  enabled: boolean;
  generate_tests: boolean;
  run_tests: boolean;
  test_framework?: "vitest" | "jest" | "mocha" | "pytest";
  require_pass: boolean;
}

/**
 * I/O operations configuration for mandate.
 */
export interface IOOperationsConfig {
  file_operations?: Array<{
    type: "read" | "write" | "delete" | "list";
    path: string;
    content?: string;
  }>;
  database_operations?: Array<{
    type: "query" | "migrate" | "seed";
    database: "supabase" | "sqlite";
    query?: string;
    table?: string;
  }>;
  api_calls?: Array<{
    method: "GET" | "POST" | "PUT" | "DELETE";
    url: string;
    headers?: Record<string, string>;
    body?: any;
    timeout?: number;
  }>;
}

/**
 * Git operations configuration for mandate.
 */
export interface GitOperationsConfig {
  enabled: boolean;
  auto_commit: boolean;
  repository?: string;
  branch?: string;
  commit_message?: string;
}

/**
 * Main mandate structure for autonomous execution.
 */
export interface Mandate {
  mandate_id: string;
  objectives: string[];
  constraints: MandateConstraints;
  budget: MandateBudget;
  deliverables: string[];
  governance: MandateGovernance;
  iteration_config: IterationConfig;
  deployment?: DeploymentConfig;
  testing?: TestingConfig;
  io_operations?: IOOperationsConfig;
  git_operations?: GitOperationsConfig;
  metadata?: {
    created_at?: number;
    created_by?: string;
    priority?: "low" | "medium" | "high" | "critical";
    tags?: string[];
  };
}

/**
 * Execution event types.
 */
export type ExecutionEventType =
  | "iteration_start"
  | "iteration_end"
  | "log"
  | "error"
  | "diff"
  | "governance_check"
  | "deployment_start"
  | "deployment_end"
  | "deployment_status"
  | "budget_warning"
  | "constraint_violation"
  | "initialization_start"
  | "webcontainer_init"
  | "api_keys_loaded"
  | "provider_configured"
  | "shell_ready"
  | "executor_ready";

/**
 * Execution event data structure.
 */
export interface ExecutionEventData {
  // For iteration_start/end
  status?: "running" | "success" | "failed" | "stopped";
  iteration_number?: number;
  
  // For log
  level?: "info" | "warn" | "error" | "debug";
  message?: string;
  source?: string; // e.g., "llm", "webcontainer", "governance"
  
  // For diff
  files_changed?: string[];
  lines_added?: number;
  lines_removed?: number;
  
  // For governance_check
  esg_score?: ESGScore;
  risk_assessment?: RiskAssessment;
  budget_consumed?: {
    tokens: number;
    time: number;
    cost: number;
  };
  compliance_status?: Record<string, boolean>;
  
  // For deployment
  deployment_url?: string;
  deployment_status?: "pending" | "running" | "complete" | "failed" | "in_progress" | "success";
  deployment_provider?: string;
  
  // For constraint violations
  violation_type?: string;
  violation_details?: string;
  
  // For errors
  error?: string;
  error_message?: string;
  error_stack?: string;
  
  // For mandate details
  mandate?: Mandate;
  
  // For causal analysis
  causal_analysis?: {
    variables: string[];
    predictions: Array<{ variable: string; predicted_value: number; confidence: number }>;
    insights: string[];
    risk_implications: string[];
  };
}

/**
 * Execution event with full metadata.
 */
export interface ExecutionEvent {
  mandate_id: string;
  iteration: number;
  type: ExecutionEventType;
  timestamp: number;
  data: ExecutionEventData;
  metadata: {
    model_used?: string;
    tokens_consumed?: number;
    execution_time?: number; // milliseconds
    webcontainer_id?: string;
    action_id?: string;
    causal_analysis_required?: boolean;
    causal_variables_count?: number;
    counterfactual_scenarios_count?: number;
  };
}

/**
 * Execution result structure.
 */
export interface ExecutionResult {
  mandate_id: string;
  status: "success" | "failed" | "stopped" | "timeout" | "budget_exceeded";
  iterations_completed: number;
  final_state: {
    files_created: string[];
    files_modified: string[];
    build_successful: boolean;
    tests_passed?: boolean;
    quality_score?: number;
  };
  governance_summary: {
    esg_score?: ESGScore;
    risk_assessment?: RiskAssessment;
    compliance_status?: Record<string, boolean>;
  };
  budget_summary: {
    tokens_used: number;
    time_elapsed: number;
    cost_incurred: number;
    budget_remaining: {
      tokens: number;
      time: number;
      cost: number;
    };
  };
  deployment_result?: {
    deployed: boolean;
    provider?: string;
    url?: string;
    error?: string;
  };
  errors?: Array<{
    iteration: number;
    type: string;
    message: string;
    timestamp: number;
  }>;
  events: ExecutionEvent[];
  created_at: number;
  completed_at: number;
}

/**
 * File diff structure for tracking changes.
 */
export interface FileDiff {
  filepath: string;
  status: "created" | "modified" | "deleted";
  additions: number;
  deletions: number;
  diff?: string; // unified diff format
}

/**
 * Build output structure.
 */
export interface BuildOutput {
  success: boolean;
  output?: string;
  errors?: string[];
  warnings?: string[];
  build_time?: number;
  artifacts?: string[];
}

