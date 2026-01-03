/**
 * Type definitions for Headless Executor
 */

export interface Mandate {
  mandate_id: string;
  objectives: string[];
  constraints: any;
  budget: any;
  deliverables: string[];
  governance?: any;
  iteration_config: any;
  deployment?: any;
}

export interface ExecutionEvent {
  mandate_id: string;
  iteration: number;
  type: string;
  timestamp: number;
  data: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface ExecutionResult {
  mandate_id: string;
  status: "success" | "failed" | "stopped" | "timeout" | "budget_exceeded";
  iterations_completed: number;
  final_state: any;
  governance_summary: any;
  budget_summary: any;
  deployment_result?: any;
  errors?: any[];
  events: ExecutionEvent[];
  created_at: number;
  completed_at: number;
}

